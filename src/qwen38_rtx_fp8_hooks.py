"""Load-time FP8 for the RTX-side linear-attention projections (opt-in: Q38_RTX_FP8_LINEAR_ATTN=1).

The official Qwen3.8-Flash-Next-FP8 checkpoint is block-FP8 (128x128, weight_scale_inv) but lists
every linear_attn.in_proj_qkv / in_proj_z / out_proj in modules_to_not_convert, so vLLM keeps
them BF16 (3.96 GiB on the RTX). This hook:
  1. removes exactly those projections from Fp8Config's ignore list, so vLLM builds them with
     Fp8LinearMethod (half the memory at construction; on sm_86 the FP8 Marlin weight-only kernel);
  2. quantizes the BF16 checkpoint tensors on the CPU as they stream through the model's
     load_weights, with qwen38_fp8_block_quant (reproduces Qwen's own FP8 quantizer byte for
     byte), and passes weight + weight_scale_inv to vLLM's normal loader.
Sensitive parts stay BF16: in_proj_a/in_proj_b, conv1d, norms, routers, hyper-connections,
indexer, and all full-attention projections. The checkpoint and its index are not modified.
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

_CACHE_FILE = "linear_attn_fp8.safetensors"
_CKPT_PREFIX = "model.language_model."
_TARGET_GDN = r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
_TARGET_QSA = r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"   # QSA projections; the indexer stays BF16


def _qsa_enabled() -> bool:
    """Q38_RTX_FP8_QSA=1: also FP8 for the 12 QSA layers' q/k/v/o projections (2026-11-03; primitive-ai
    reports QSA + GDN FP8 ties the BF16-attention build). The indexer projection stays BF16, so the
    fused q/k/v/indexer linear is split (as vLLM does under LoRA)."""
    return os.environ.get("Q38_RTX_FP8_QSA") == "1"


def _target() -> "re.Pattern[str]":
    alts = _TARGET_GDN + ("|" + _TARGET_QSA if _qsa_enabled() else "")
    return re.compile(r"(^|\.)(" + alts + r")(\.weight)?$")


def _expected_tensors() -> int:
    return 3 * 36 + (4 * 12 if _qsa_enabled() else 0)


_TARGET = _target()


def _is_target(name: str) -> bool:
    """FP8 target in the main model only (the MTP head, not built here, keeps BF16)."""
    return bool(_TARGET.search(name)) and not name.startswith("mtp.") and ".mtp." not in name
_STATS = {"tensors": 0, "bytes_bf16": 0, "bytes_fp8": 0, "seconds": 0.0}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def _open_cache() -> tuple[str, list[str]] | None:
    """Q38_RTX_FP8_CACHE=DIR: use DIR/linear_attn_fp8.safetensors only if it was built from this
    checkpoint (model index sha256, from Q38_MODEL_DIR) with this quantizer (source sha256)."""
    cache_dir = os.environ.get("Q38_RTX_FP8_CACHE")
    if not cache_dir:
        return None
    from safetensors import safe_open
    path = os.path.join(cache_dir, _CACHE_FILE)
    model_dir = os.environ.get("Q38_MODEL_DIR", "")
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        keys = [k for k in f.keys() if not k.endswith("_scale_inv")]
    want_model = _sha256(os.path.join(model_dir, "model.safetensors.index.json"))
    want_quant = _sha256(os.path.join(os.path.dirname(__file__), "qwen38_fp8_block_quant.py"))
    if (meta.get("model_index_sha256") != want_model or meta.get("quantizer_sha256") != want_quant
            or len(keys) != _expected_tensors() or meta.get("qsa", "0") != ("1" if _qsa_enabled() else "0")):
        raise RuntimeError(f"FP8 cache {path} does not match this checkpoint/quantizer: {meta}")
    return path, sorted(keys)


def _enabled() -> bool:
    return os.environ.get("Q38_RTX_FP8_LINEAR_ATTN") == "1"


def _unfuse_qsa_indexer(model_mod: Any) -> None:
    """Keep q/k/v and the BF16 indexer projection in separate linears: rebuild
    Qwen4ExpQSAAttention.__init__ from vLLM's own source with exactly one edit
    (fuse_indexer_projection = False, the LoRA layout vLLM already supports) and drop the indexer
    shard from both model classes' packed q/k/v mapping (also what vLLM does under LoRA)."""
    import inspect
    import textwrap
    from vllm.models.qwen4_exp.nvidia import qsa as qsa_mod
    cls = qsa_mod.Qwen4ExpQSAAttention
    if not getattr(cls.__init__, "_q38_unfused", False):
        src = textwrap.dedent(inspect.getsource(cls.__init__))
        old = "self.fuse_indexer_projection = vllm_config.lora_config is None"
        if src.count(old) != 1 or "super()" in src:
            raise RuntimeError("Q38_RTX_FP8_QSA: QSA constructor no longer matches the expected source")
        ns = dict(vars(qsa_mod))
        exec(compile(src.replace(old, "self.fuse_indexer_projection = False"),
                     f"<q38 unfused {cls.__qualname__}.__init__>", "exec"), ns)
        ns["__init__"]._q38_unfused = True
        cls.__init__ = ns["__init__"]
    for name in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        mcls = getattr(model_mod, name)
        mcls.packed_modules_mapping = {**mcls.packed_modules_mapping, "qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    print("Q38_RTX_FP8,qsa,unfused_indexer_projection,packed_qkv,q_proj+k_proj+v_proj", file=sys.stderr, flush=True)


def install(model_mod: Any) -> dict[str, Any]:
    if not _enabled():
        return {}
    import torch
    from vllm.model_executor.layers.quantization import fp8 as fp8_mod
    from qwen38_fp8_block_quant import quantize_block_fp8
    global _TARGET
    _TARGET = _target()
    if _qsa_enabled():
        _unfuse_qsa_indexer(model_mod)

    cfg_cls = fp8_mod.Fp8Config
    if not getattr(cfg_cls, "_q38_linear_attn_patch", False):
        orig_init = cfg_cls.__init__

        def init(self, *a: Any, **k: Any) -> None:
            orig_init(self, *a, **k)
            before = list(self.ignored_layers or [])
            if self.weight_block_size != [128, 128]:
                raise RuntimeError(f"Q38 FP8 hook expects 128x128 block FP8, got {self.weight_block_size}")
            self.ignored_layers = [x for x in before if not _is_target(x)]
            print(f"Q38_RTX_FP8,ignore_list_before,{len(before)},after,{len(self.ignored_layers)},"
                  f"removed,{len(before) - len(self.ignored_layers)}", file=sys.stderr, flush=True)

        cfg_cls.__init__ = init
        cfg_cls._q38_linear_attn_patch = True

    model_cls = model_mod.Qwen4ExpModel
    prev_load = model_cls.load_weights
    cache = _open_cache()
    if cache is not None:
        # vLLM must not even read the BF16 originals (skip-before-read hook in run_stage8...).
        from vllm.model_executor.model_loader import weight_utils as _wu
        cached = set(cache[1])
        _wu._q38_extra_skip = lambda name: name in cached
        print(f"Q38_RTX_FP8,cache,{cache[0]},tensors,{len(cache[1])}", file=sys.stderr, flush=True)

    def load_weights(self, weights):
        def from_cache():
            from safetensors import safe_open
            t = time.perf_counter(); prefix = None
            for name, tensor in weights:                       # BF16 targets never arrive (skipped)
                if prefix is None and "layers." in name:
                    prefix = name[: name.index("layers.")]     # names as this loader sees them
                yield name, tensor
            prefix = prefix or ""
            with safe_open(cache[0], "pt") as f:
                for ck in cache[1]:
                    inner = prefix + ck[len(_CKPT_PREFIX):] if ck.startswith(_CKPT_PREFIX) else ck
                    q, s = f.get_tensor(ck), f.get_tensor(ck + "_scale_inv")
                    _STATS["tensors"] += 1
                    _STATS["bytes_fp8"] += q.numel() + s.numel() * s.element_size()
                    yield inner, q
                    yield inner[: -len("weight")] + "weight_scale_inv", s
            _STATS["seconds"] += time.perf_counter() - t

        def convert():
            for name, tensor in weights:
                if name.endswith(".weight") and _is_target(name[: -len(".weight")]):
                    t = time.perf_counter()
                    q, s = quantize_block_fp8(tensor)
                    _STATS["tensors"] += 1
                    _STATS["bytes_bf16"] += tensor.numel() * tensor.element_size()
                    _STATS["bytes_fp8"] += q.numel() + s.numel() * s.element_size()
                    _STATS["seconds"] += time.perf_counter() - t
                    yield name, q
                    yield name[: -len("weight")] + "weight_scale_inv", s
                else:
                    yield name, tensor
        loaded = prev_load(self, from_cache() if cache is not None else convert())
        print(f"Q38_RTX_FP8,quantized_tensors,{_STATS['tensors']},bf16_GiB,{_STATS['bytes_bf16'] / 2**30:.3f},"
              f"fp8_GiB,{_STATS['bytes_fp8'] / 2**30:.3f},cpu_seconds,{_STATS['seconds']:.1f}", file=sys.stderr, flush=True)
        if _STATS["tensors"] != _expected_tensors():
            raise RuntimeError(f"expected {_expected_tensors()} FP8 projections, quantized {_STATS['tensors']}")
        return loaded

    model_cls.load_weights = load_weights
    return {"rtx_fp8_linear_attn": True, "rtx_fp8_qsa": _qsa_enabled(), "block": [128, 128], "targets": _TARGET.pattern,
            "cache": cache[0] if cache else None}
