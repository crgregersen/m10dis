"""FP8 weights for the hyper-connection low-rank matrices on the RTX (opt-in: Q38_RTX_FP8_HC=1, 2026-11-03).

Each GatedResidual (attn/mlp per layer + the final mixer) holds two BF16 low-rank matrices, 320 x 10240 and
10240 x 320 (1.2 GiB in total, the largest BF16 block left on the RTX; every decode step reads all of it).
The 128x128 block-FP8 path of qwen38_rtx_fp8_hooks does not fit (outputs 336 = 320 + 4 + pad, inputs 320),
and vLLM's fused hc_down_silu needs sm_90 (sm_86 uses the plain Linear path anyway). This module:
  1. wraps GatedResidual.__init__ so its Linear layers use Q38HcFp8LinearMethod (BF16 weights load as usual);
  2. after loading, quantizes each weight on the GPU to FP8 E4M3 with one FP32 scale per (output row,
     64-wide K group), round to nearest even, stored as uint8; the BF16 weight is freed;
  3. applies it with a Triton kernel that reads the FP8 bytes, decodes them in registers (exact: every E4M3
     value is a BF16 value), multiplies with BF16 x on the tensor cores (FP32 accumulate) and scales each
     64-wide K group in FP32. Long-K / few-tile shapes use a split-K with an in-order FP32 partial sum
     (no atomics: deterministic for a given shape).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch

GROUP_K = 64
_STATS = {"layers": 0, "bf16_bytes": 0, "fp8_bytes": 0}


def enabled() -> bool:
    return os.environ.get("Q38_RTX_FP8_HC") == "1"


def quantize_hc_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [N, K] (K % 64 == 0) -> (q uint8 [N, K] E4M3 bits, scale fp32 [N, K/64]); w ~ decode(q) * scale."""
    n, k = w.shape
    if k % GROUP_K:
        raise ValueError(f"HC FP8: K={k} not a multiple of {GROUP_K}")
    g = w.float().view(n, k // GROUP_K, GROUP_K)
    amax = g.abs().amax(-1)
    ok = torch.isfinite(amax) & (amax > 0)        # padding rows of the merged down linear may be garbage / zero
    scale = torch.where(ok, amax / 448.0, torch.ones_like(amax))
    q = (g / scale.unsqueeze(-1)).nan_to_num(0.0, 0.0, 0.0).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(n, k).view(torch.uint8).contiguous(), scale.contiguous()


def dequantize_hc_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    return (q.view(torch.float8_e4m3fn).float().view(n, k // GROUP_K, GROUP_K) * scale.unsqueeze(-1)).view(n, k)


_kernels: dict[str, Any] = {}


def _get_kernels():
    if _kernels:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _decode(b):
        # E4M3 bits -> FP16 bits (sign << 15 | low 7 bits << 7) = the exact E4M3 value * 2^-8, normals and
        # subnormals alike (exponent bias 15 vs 7); the 2^8 is folded into the scale by the caller.
        h = (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.int16)
        return h.to(tl.float16, bitcast=True)

    @triton.jit
    def _hc_fp8_gemm(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, ssn, sos, som, K_SPLIT,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT_BF16: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_s = tl.program_id(2)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mm = rm < M
        mn = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_begin = pid_s * K_SPLIT
        for kk in range(0, K_SPLIT, BLOCK_K):
            k0 = k_begin + kk
            rk = k0 + tl.arange(0, BLOCK_K)
            mk = rk < K
            x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
            b = tl.load(w_ptr + rn[None, :] * swn + rk[:, None], mask=mn[None, :] & mk[:, None], other=0).to(tl.int32)
            part = tl.dot(x, _decode(b).to(tl.bfloat16))                 # exact decode, fp32 accumulate
            sc = tl.load(s_ptr + rn * ssn + k0 // BLOCK_K, mask=mn & (k0 < K), other=0.0)
            acc += part * sc[None, :]
        optr = o_ptr + pid_s * sos + rm[:, None] * som + rn[None, :]
        if OUT_BF16:
            tl.store(optr, acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])
        else:
            tl.store(optr, acc, mask=mm[:, None] & mn[None, :])

    @triton.jit
    def _hc_fp8_gemv(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, ssn, sos, som, K_SPLIT,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT_BF16: tl.constexpr):
        # M <= 8 (decode): CUDA-core dot products, every weight byte read once for all M rows.
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        rm = tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mm = rm < M
        mn = rn < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_begin = pid_s * K_SPLIT
        for kk in range(0, K_SPLIT, BLOCK_K):
            k0 = k_begin + kk
            rk = k0 + tl.arange(0, BLOCK_K)
            mk = rk < K
            b = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int32)
            w = _decode(b).to(tl.float32)                                 # [BLOCK_N, BLOCK_K]
            x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0).to(tl.float32)
            part = tl.sum(x[:, None, :] * w[None, :, :], axis=2)          # [BLOCK_M, BLOCK_N]
            sc = tl.load(s_ptr + rn * ssn + k0 // BLOCK_K, mask=mn & (k0 < K), other=0.0)
            acc += part * sc[None, :]
        optr = o_ptr + pid_s * sos + rm[:, None] * som + rn[None, :]
        if OUT_BF16:
            tl.store(optr, acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])
        else:
            tl.store(optr, acc, mask=mm[:, None] & mn[None, :])

    @triton.jit
    def _hc_fp8_gemv1(x_ptr, w_ptr, s_ptr, o_ptr, N, K, swn, ssn, sos, K_SPLIT,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, OUT_BF16: tl.constexpr):
        # M = 1 (decode), v2: a 2D FP32 tile accumulates w * scale * x elementwise; one reduction over K at the end
        # (v1 reduced across threads every iteration). Tuned on the RTX 3070 Ti (u4): down 10.9 us, up 6.9 us vs
        # cuBLAS BF16 19.0 / 14.8 us.
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mn = rn < N
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        k_begin = pid_s * K_SPLIT
        for kk in range(0, K_SPLIT, BLOCK_K):
            rk = k_begin + kk + tl.arange(0, BLOCK_K)
            mk = rk < K
            b = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=mn[:, None] & mk[None, :], other=0).to(tl.int32)
            w = _decode(b).to(tl.float32)
            sc = tl.load(s_ptr + rn[:, None] * ssn + rk[None, :] // 64, mask=mn[:, None] & mk[None, :], other=0.0)
            x = tl.load(x_ptr + rk, mask=mk, other=0.0).to(tl.float32)
            acc += w * sc * x[None, :]
        y = tl.sum(acc, axis=1)
        if OUT_BF16:
            tl.store(o_ptr + rn, y.to(tl.bfloat16), mask=mn)
        else:
            tl.store(o_ptr + pid_s * sos + rn, y, mask=mn)

    _kernels["gemv1"] = _hc_fp8_gemv1
    _kernels["gemm"] = _hc_fp8_gemm
    _kernels["gemv"] = _hc_fp8_gemv
    return _kernels


def _launch(kern, grid, x2, q, s, M, N, K, k_split, splits, block_m, block_n, warps):
    if splits == 1:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x2.device)
        kern[grid](x2, q, s, out, M, N, K, x2.stride(0), q.stride(0), s.stride(0), 0, out.stride(0), k_split,
                   BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=GROUP_K, OUT_BF16=True, num_warps=warps)
        return out
    part = torch.empty((splits, M, N), dtype=torch.float32, device=x2.device)
    kern[grid](x2, q, s, part, M, N, K, x2.stride(0), q.stride(0), s.stride(0), part.stride(0), part.stride(1),
               k_split, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=GROUP_K, OUT_BF16=False, num_warps=warps)
    return part.sum(0).to(torch.bfloat16)          # no atomics: same shape -> same result


def hc_fp8_linear(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x [..., K] BF16 @ (decode(q) * scale)^T -> [..., N] BF16. `scale` already carries the 2^8 of the decode."""
    import triton
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M, K = x2.shape
    N = q.shape[0]
    kb = K // GROUP_K
    k = _get_kernels()
    if M == 1:
        # u4 tuning (RTX 3070 Ti): long K (the down GEMMs, K = 10240): 8 rows x 512, 16 K splits;
        # short K (the up GEMM, K = 320): 32 rows x 64, no split. 4 warps.
        bn, bk, sp = (8, 512, 16) if K > 1024 else (32, 64, 1)
        k_split = triton.cdiv(triton.cdiv(K, sp), bk) * bk
        sp = triton.cdiv(K, k_split)
        grid = (triton.cdiv(N, bn), sp)
        if sp == 1:
            out = torch.empty((1, N), dtype=torch.bfloat16, device=x2.device)
            k["gemv1"][grid](x2, q, scale, out, N, K, q.stride(0), scale.stride(0), 0, k_split,
                             BLOCK_N=bn, BLOCK_K=bk, OUT_BF16=True, num_warps=4)
        else:
            part = torch.empty((sp, N), dtype=torch.float32, device=x2.device)
            k["gemv1"][grid](x2, q, scale, part, N, K, q.stride(0), scale.stride(0), part.stride(0), k_split,
                             BLOCK_N=bn, BLOCK_K=bk, OUT_BF16=False, num_warps=4)
            out = part.sum(0, keepdim=True).to(torch.bfloat16)      # no atomics: deterministic
    elif M <= 16:
        block_m, block_n = 16, 64
        tiles = triton.cdiv(N, block_n)
        splits = 1 if tiles >= 96 else min(kb, triton.cdiv(192, tiles))
        k_split = triton.cdiv(kb, splits) * GROUP_K
        splits = triton.cdiv(K, k_split)
        out = _launch(k["gemm"], (tiles, 1, splits), x2, q, scale, M, N, K, k_split, splits, block_m, block_n, 4)
    else:
        block_m, block_n = 64, 64
        tiles = triton.cdiv(N, block_n) * triton.cdiv(M, block_m)
        splits = 1 if tiles >= 96 else min(kb, triton.cdiv(192, tiles))
        k_split = triton.cdiv(kb, splits) * GROUP_K
        splits = triton.cdiv(K, k_split)
        grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m), splits)
        out = _launch(k["gemm"], grid, x2, q, scale, M, N, K, k_split, splits, block_m, block_n, 4)
    return out.reshape(*shape[:-1], N)


def _method_class():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    class Q38HcFp8LinearMethod(UnquantizedLinearMethod):
        """BF16 weights load normally; process_weights_after_loading converts them to HC FP8."""

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            w = layer.weight.data
            q, s = quantize_hc_fp8(w)
            _STATS["layers"] += 1
            _STATS["bf16_bytes"] += w.numel() * w.element_size()
            _STATS["fp8_bytes"] += q.numel() + s.numel() * 4
            layer.weight = torch.nn.Parameter(q, requires_grad=False)
            layer.q38_hc_scale = torch.nn.Parameter(s * 256.0, requires_grad=False)   # decode yields value * 2^-8
            del w
            if _STATS["layers"] == _expected_layers():
                torch.cuda.empty_cache()
                print(f"Q38_RTX_FP8_HC,layers,{_STATS['layers']},bf16_GiB,{_STATS['bf16_bytes'] / 2**30:.3f},"
                      f"fp8_GiB,{_STATS['fp8_bytes'] / 2**30:.3f}", file=sys.stderr, flush=True)

        def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
            if bias is not None:
                raise RuntimeError("Q38 HC FP8: unexpected bias")
            return hc_fp8_linear(x, layer.weight, layer.q38_hc_scale)

    return Q38HcFp8LinearMethod


_EXPECTED = {"value": 0}


def _expected_layers() -> int:
    return _EXPECTED["value"]


def install() -> dict[str, Any]:
    if not enabled():
        return {}
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.qwen4_exp.nvidia import hyperconnection as hc_mod
    method_cls = _method_class()
    cls = hc_mod.GatedResidual
    if not getattr(cls.__init__, "_q38_hc_fp8", False):
        orig = cls.__init__

        def init(self, *a: Any, **k: Any) -> None:
            orig(self, *a, **k)
            self._use_hc_down_silu = False          # the fused kernel reads BF16 weights (sm_90 only anyway)
            for attr in ("input_mix_weight_down_block_inject", "input_mix_weight_down", "input_mix_weight_up"):
                lin = getattr(self, attr, None)
                if lin is not None and type(lin.quant_method) is UnquantizedLinearMethod:
                    lin.quant_method = method_cls()
                    _EXPECTED["value"] += 1

        init._q38_hc_fp8 = True
        cls.__init__ = init
    print("Q38_RTX_FP8_HC,installed,group_k,64,scale,fp32_per_row_group,kernel,triton_split_k_in_order",
          file=sys.stderr, flush=True)
    return {"rtx_fp8_hc": True, "group_k": GROUP_K}
