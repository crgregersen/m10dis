"""FP8 E4M3 block quantization in the checkpoint's own format (128x128 blocks, weight_scale_inv).

Convention (DeepSeek-V3 / Qwen FP8 checkpoints, vLLM Fp8LinearMethod block path):
    weight_bf16 ~= weight_fp8.float() * weight_scale_inv[block_row, block_col]
    weight_scale_inv[i, j] = bf16(amax(|w| over block i,j) / 448)
    weight_fp8 = fp8(clamp(bf16(w_bf16 / scale_bf16)))      <- division done in BF16 arithmetic
This reproduces Qwen's official Qwen3.8-Flash-Next-FP8 expert tensors byte for byte (0 differing
bytes and identical scales; test_qwen38_fp8_block_quant.py, 2026-11-03).
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0
BLOCK = 128


def quantize_block_fp8(w: torch.Tensor, block: int = BLOCK, scale_dtype: torch.dtype = torch.bfloat16
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """w [N, K] (any float dtype) -> (fp8 [N, K], weight_scale_inv [ceil(N/b), ceil(K/b)] in scale_dtype).
    With scale_dtype=bfloat16 the scale is rounded to BF16 first and that rounded scale is used for
    the division (so stored scale and quantized values are consistent)."""
    if w.ndim != 2:
        raise ValueError("expected a 2-D weight")
    n, k = w.shape
    nb, kb = -(-n // block), -(-k // block)
    wf = w.to(torch.bfloat16) if scale_dtype == torch.bfloat16 else w.float()
    if n % block or k % block:                       # pad to whole blocks (zeros do not raise amax)
        wp = torch.zeros(nb * block, kb * block, dtype=wf.dtype)
        wp[:n, :k] = wf
    else:
        wp = wf
    blocks = wp.view(nb, block, kb, block)
    amax = blocks.float().abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = (amax / FP8_MAX).to(scale_dtype)
    # Official convention: the division runs in the scale's dtype (BF16), then casts to E4M3.
    q = (blocks / scale.to(blocks.dtype)[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(nb * block, kb * block)[:n, :k].contiguous(), scale.to(scale_dtype)


def dequantize_block_fp8(q: torch.Tensor, scale_inv: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    n, k = q.shape
    s = scale_inv.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:n, :k]
    return q.float() * s
