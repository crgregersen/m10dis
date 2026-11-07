"""Host-memory K/V tier for the QSA layers, decode only (opt-in: Q38_KV_TIER=1; docs/research/KV_HOST_TIER_PLAN.md, phase 1).

Wraps the QSA impl's forward_qsa. Each decode call first enqueues q38_kv_resolve on the current stream, so it runs in the
CUDA graph, then runs the unchanged attention kernel on a rewritten copy of the selection. Prefill runs as usual. When a
layer's last prefill chunk is done, its K/V is copied to host RAM and the tier turns on for that layer. Phase 1 leaves the
pool as it is: the first Q38_KV_TIER_TOKENS logical tokens act as the cache, and vLLM still writes new K/V at the real
position, from where the kernel copies it. The MTP layer stays out of the tier.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import mmap
import os
import re
import sys

import torch

_PREFILL_ROWS = 1024
_LAYERS: dict[int, dict] = {}


def _lib():
    lib = ctypes.CDLL(os.environ["Q38_M10_RTX_PEER_LIBRARY"])
    lib.q38_kv_tier_state_size.restype = ctypes.c_size_t
    lib.q38_kv_tier_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
    lib.q38_kv_tier_populate.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.q38_kv_verify.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.q38_kv_resolve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    return lib


def _check(rc: int, err) -> None:
    if rc:
        raise RuntimeError("q38_kv_tier: " + err.value.decode())


def _state(layer_id: int, layer, kv_cache: torch.Tensor, lib, slots: int, max_tokens: int) -> dict:
    s = _LAYERS.get(layer_id)
    if s is not None:
        return s
    key = kv_cache.transpose(1, 2).split(kv_cache.shape[-1] // 2, dim=-1)[0]
    elem = key.element_size()
    token_stride, head_stride, block_stride = key.stride(1) * elem, key.stride(2) * elem, key.stride(0) * elem
    if head_stride != 1024 or token_stride != 2048 or key.shape[2] != 2 or key.shape[3] != 256 or key.stride(3) != 1:
        raise RuntimeError(f"q38_kv_tier: unexpected K/V layout {key.shape} strides {key.stride()}")
    host = mmap.mmap(-1, max_tokens * 2048, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(host))
    dev = ctypes.c_void_p()
    err = ctypes.create_string_buffer(512)
    _check(lib.q38_kv_tier_register(host_ptr, max_tokens * 2048, ctypes.byref(dev), err, len(err)), err)
    d = key.device
    s = {"layer": layer_id, "kv": key.data_ptr(), "block_stride": block_stride, "page_tokens": key.shape[1], "token_stride": token_stride,
         "host": host, "host_ptr": host_ptr, "host_dev": dev.value, "slots": slots,
         "b2s": torch.full((max_tokens // 4 + 2,), -1, dtype=torch.int32, device=d),
         "s2b": torch.arange(slots, dtype=torch.int32, device=d),
         "epoch": torch.zeros(slots, dtype=torch.int32, device=d),
         "st": torch.zeros(lib.q38_kv_tier_state_size(), dtype=torch.uint8, device=d),
         "shadow": layer.topk_indices_buffer[:4].clone()}   # decode only: at most 4 rows
    _LAYERS[layer_id] = s
    return s


def _populate(s: dict, block_table: torch.Tensor, tokens: int, lib) -> None:
    """Copy this layer's K/V to host RAM and turn the tier on."""
    torch.cuda.synchronize()
    pages = (tokens + s["page_tokens"] - 1) // s["page_tokens"]
    table = block_table[0, :pages].cpu().numpy().astype("int32")
    err = ctypes.create_string_buffer(512)
    _check(lib.q38_kv_tier_populate(s["kv"], s["block_stride"], s["page_tokens"], s["token_stride"], table.ctypes.data, pages, s["host_ptr"], err, len(err)), err)
    s["b2s"][: s["slots"]] = torch.arange(s["slots"], dtype=torch.int32, device=s["b2s"].device)
    head = torch.tensor([1, s["slots"]], dtype=torch.uint32).numpy().tobytes()
    s["st"][: len(head)] = torch.frombuffer(bytearray(head), dtype=torch.uint8).to(s["st"].device)
    torch.cuda.synchronize()
    s["populated_tokens"] = tokens


def install() -> dict | None:
    if os.environ.get("Q38_KV_TIER") != "1":
        return None
    from vllm.forward_context import get_forward_context
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionImpl as impl

    tokens = int(os.environ.get("Q38_KV_TIER_TOKENS", "32768"))
    max_tokens = int(os.environ.get("Q38_VLLM_MAX_MODEL_LEN", "90112")) + 2048
    if tokens % 4:
        raise RuntimeError("Q38_KV_TIER_TOKENS must be a multiple of 4")
    slots = tokens // 4
    cache: dict = {}
    verify = os.environ.get("Q38_KV_TIER_VERIFY") == "1"
    original = impl.forward_qsa

    def wrapped(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config, output_gate, *a, **kw):
        m = re.search(r"layers\.(\d+)", layer.layer_name)
        layer_id = int(m.group(1)) if m and "mtp" not in layer.layer_name else None
        rows = attn_metadata.num_actual_tokens
        if layer_id is None or rows == 0:
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config, output_gate, *a, **kw)
        lib = cache.setdefault("lib", None) or cache.update(lib=_lib()) or cache["lib"]   # no CUDA in the client parent
        s = _state(layer_id, layer, kv_cache, lib, slots, max_tokens)
        if rows > 4:                                  # prefill runs as it does without the tier
            r = original(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config, output_gate, *a, **kw)
            if rows < _PREFILL_ROWS and "populated_tokens" not in s:
                metadata = get_forward_context().attn_metadata
                metadata = metadata[0] if isinstance(metadata, list) else metadata
                side = metadata[layer.indexer.raw_key_cache.prefix]
                end = int(side.logical_positions[: rows].max()) + 1
                if end > tokens + 16:
                    _populate(s, attn_metadata.block_table, end, lib)
            return r
        metadata = get_forward_context().attn_metadata
        metadata = metadata[0] if isinstance(metadata, list) else metadata
        side = metadata[layer.indexer.raw_key_cache.prefix]
        buffers = layer._buffers
        real = buffers["topk_indices_buffer"]
        err = ctypes.create_string_buffer(512)
        stream = torch.cuda.current_stream().cuda_stream
        _check(lib.q38_kv_resolve(real.data_ptr(), s["shadow"].data_ptr(), real.shape[1], rows, side.logical_positions.data_ptr(), s["kv"],
                                  s["block_stride"], s["token_stride"], s["page_tokens"], attn_metadata.block_table.data_ptr(), s["host_dev"],
                                  s["b2s"].data_ptr(), s["s2b"].data_ptr(), s["epoch"].data_ptr(), s["st"].data_ptr(), stream, err, len(err)), err)
        if verify:                                    # eager only: a K/V byte check of this row
            count = s.setdefault("verify", torch.zeros(2, dtype=torch.int64, device=real.device))
            _check(lib.q38_kv_verify(real.data_ptr(), s["shadow"].data_ptr(), real.shape[1], side.logical_positions.data_ptr(), s["kv"],
                                     s["block_stride"], s["token_stride"], s["page_tokens"], attn_metadata.block_table.data_ptr(), s["host_dev"],
                                     s["st"].data_ptr(), tokens, count.data_ptr(), count[1:].data_ptr(), stream, err, len(err)), err)
        buffers["topk_indices_buffer"] = s["shadow"]
        try:
            return original(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, use_prefill_config, output_gate, *a, **kw)
        finally:
            buffers["topk_indices_buffer"] = real

    impl.forward_qsa = wrapped
    atexit.register(_report)
    print(f"Q38_KV_TIER,installed,tokens,{tokens},max_tokens,{max_tokens}", file=sys.stderr, flush=True)
    return {"tokens": tokens}


def _report() -> None:
    path = os.environ.get("Q38_KV_TIER_REPORT")
    if not _LAYERS:
        return
    out = {}
    for lid, s in sorted(_LAYERS.items()):
        st = s["st"].cpu().numpy().tobytes()
        u32 = [int.from_bytes(st[i:i + 4], "little") for i in range(0, 32, 4)]
        u64 = [int.from_bytes(st[i:i + 8], "little") for i in range(32, 80, 8)]
        calls, selected, misses, new_tokens, max_miss, clock = u64
        out[lid] = {"enabled": u32[0], "populated_tokens": s.get("populated_tokens"), "calls": calls, "selected_blocks": selected, "misses": misses,
                    "new_tokens": new_tokens, "max_miss": max_miss, "clock_steps": clock, "hit": 1 - misses / selected if selected else None}
        if "verify" in s:
            out[lid]["verify_checked"], out[lid]["verify_bad"] = (int(x) for x in s["verify"].cpu().tolist())
    sel = sum(x["selected_blocks"] for x in out.values())
    print("Q38_KV_TIER_SUMMARY," + json.dumps({"hit": 1 - sum(x["misses"] for x in out.values()) / sel if sel else None, "layers": len(out), "verify_checked": sum(x.get("verify_checked", 0) for x in out.values()), "verify_bad": sum(x.get("verify_bad", 0) for x in out.values())}), flush=True)
    if path:
        with open(f"{path}.{os.getpid()}", "w") as f:
            json.dump(out, f, indent=1)
