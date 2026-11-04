"""Keep the token embedding and/or the LM head of Qwen3.8 in host RAM instead of RTX VRAM.

Opt-in (both default off):
  Q38_CPU_EMBEDDING=1   embed_tokens  -> CPU table, row gather on the CPU, rows copied to the RTX
  Q38_CPU_LM_HEAD=1     lm_head       -> CPU table, BF16 GEMV on the CPU, logits copied to the RTX
  Q38_CPU_HEAD_TIMING_JSON=PATH       per-process call counts and seconds, written at exit

Each table is a plain CPU tensor, not an nn.Parameter, so vLLM's post-load
device_loading_context (which moves CPU *parameters* to the GPU) never copies it to the RTX.
Weights arrive through AutoWeightsLoader, which calls a child module's load_weights().

Numerics: the embedding gather is an exact row copy (bit-identical to the RTX lookup). The
CPU LM head is a BF16 matmul with a different accumulation order than cuBLAS, so logits can
differ in the last bits; greedy tokens can change only at near-ties.

Measured on this host (i5-6400, 4 threads, 2026-11-02): LM head 62.6 ms per token, embedding
gather 7 us per token.
"""
from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

_STATS: dict[str, dict[str, float]] = {}
# Host callbacks run once per forward step inside the CPU-embedding eager break (which also runs on
# CUDA-graph replay), e.g. the GPU-driven exchange's sticky-error check when no other break remains.
STEP_HOOKS: list = []
_ZERO_COPY: dict[str, Any] = {}


def zero_copy_enabled() -> bool:
    """Q38_EMBEDDING_ZERO_COPY=1: the embedding is a GPU gather from the pinned host table (no break)."""
    return os.environ.get("Q38_EMBEDDING_ZERO_COPY") == "1"


def _zero_copy_lib():
    import ctypes
    lib = _ZERO_COPY.get("lib")
    if lib is None:
        path = os.environ.get("Q38_M10_RTX_PEER_LIBRARY")
        if not path:
            raise RuntimeError("Q38_EMBEDDING_ZERO_COPY=1 needs Q38_M10_RTX_PEER_LIBRARY (zero-copy gather kernel)")
        lib = ctypes.CDLL(path)
        lib.q38_zero_copy_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
                                               ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_size_t]
        lib.q38_zero_copy_unregister.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.q38_zero_copy_gather.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint32, ctypes.c_void_p,
                                             ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_char_p, ctypes.c_size_t]
        for f in (lib.q38_zero_copy_register, lib.q38_zero_copy_unregister, lib.q38_zero_copy_gather):
            f.restype = ctypes.c_int
        _ZERO_COPY["lib"] = lib
    return lib


def _zero_copy_close() -> None:
    import ctypes
    for entry in _ZERO_COPY.get("tables", []):
        if "host_ptr" not in entry:                  # already closed
            continue
        try:
            entry["bad_ids"] = int(entry.pop("bad").item())
        except Exception as exc:                     # report, do not mask the process exit
            entry["bad_ids_error"] = repr(exc)
        err = ctypes.create_string_buffer(512)
        entry["unregistered"] = _ZERO_COPY["lib"].q38_zero_copy_unregister(
            ctypes.c_void_p(entry.pop("host_ptr")), err, len(err)) == 0
        # Own key in the timing file: every _STATS entry must stay {calls, rows, seconds} (runner summary).
        _ZERO_COPY.setdefault("report", {})[entry["role"]] = {
            k: v for k, v in entry.items() if k not in ("device_ptr", "table")}


def _stat(name: str, seconds: float, rows: int) -> None:
    s = _STATS.setdefault(name, {"calls": 0, "rows": 0, "seconds": 0.0})
    s["calls"] += 1; s["rows"] += rows; s["seconds"] += seconds


def _dump() -> None:
    path = os.environ.get("Q38_CPU_HEAD_TIMING_JSON")
    if path and (_STATS or _ZERO_COPY.get("report")):
        p = Path(path).with_name(Path(path).name + f".{os.getpid()}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"pid": os.getpid(), "stats": _STATS,
                                 "zero_copy": _ZERO_COPY.get("report", {})}, indent=2) + "\n")


atexit.register(_dump)
atexit.register(_zero_copy_close)   # runs before _dump (atexit is LIFO)


class _HeadMethod:
    """Plain object (not an nn.Module, so it is not registered as a child) forwarding apply()."""

    def __init__(self, owner: "CpuTable"):
        self._owner = owner

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return self._owner.apply(layer, x, bias)


class CpuTable(nn.Module):
    """Stand-in for VocabParallelEmbedding / ParallelLMHead (tp_size 1) with a host-RAM table."""

    def __init__(self, num_embeddings: int, embedding_dim: int, *args: Any, role: str, **kwargs: Any):
        super().__init__()
        dtype = kwargs.get("params_dtype") or torch.get_default_dtype()
        self.role = role
        self.num_embeddings = self.org_vocab_size = self.num_embeddings_padded = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.tp_size = 1
        self.quant_method = _HeadMethod(self)  # LogitsProcessor calls lm_head.quant_method.apply(...)
        self.table = torch.empty((self.num_embeddings, self.embedding_dim), dtype=dtype, device="cpu")
        self.loaded = False

    def extra_repr(self) -> str:
        return f"role={self.role}, table={tuple(self.table.shape)}, dtype={self.table.dtype}, device=cpu"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        done = set()
        for name, tensor in weights:
            if name != "weight":
                raise KeyError(f"{self.role}: unexpected checkpoint tensor {name}")
            if tuple(tensor.shape) != tuple(self.table.shape):
                raise ValueError(f"{self.role}: shape {tuple(tensor.shape)} != {tuple(self.table.shape)}")
            self.table.copy_(tensor.to(device="cpu", dtype=self.table.dtype))
            self.loaded = True; done.add(name)
        return done

    # embedding path (an eager break under Q38_VLLM_CUDAGRAPH=1: it copies ids to the host), or with
    # Q38_EMBEDDING_ZERO_COPY=1 a GPU gather from the pinned host table (stream work, captured).
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if zero_copy_enabled():
            return self._embed_zero_copy(input_ids)
        if not hasattr(self, "_q38_break"):
            from q38_cudagraph_breaks import make_break
            object.__setattr__(self, "_q38_break", make_break(
                lambda key, mod, ids: mod._embed(ids), "embedding",
                lambda key, mod, ids: ((*ids.shape, mod.embedding_dim), mod.table.dtype, ids.device)))
        return self._q38_break(id(self), self, input_ids)

    def _zero_copy_setup(self, device: torch.device) -> dict[str, Any]:
        import ctypes
        if not self.loaded:
            raise RuntimeError(f"{self.role}: zero-copy gather before the table was loaded")
        table = self.table.contiguous()
        lib = _zero_copy_lib()
        dev = ctypes.c_void_p(); ro = ctypes.c_int(0); err = ctypes.create_string_buffer(512)
        nbytes = table.numel() * table.element_size()
        t = time.perf_counter()
        if lib.q38_zero_copy_register(ctypes.c_void_p(table.data_ptr()), nbytes, ctypes.byref(dev), ctypes.byref(ro), err, len(err)):
            raise RuntimeError(f"{self.role}: pinning the host table failed: {err.value.decode(errors='replace')}")
        entry = {"role": self.role, "table": table, "host_ptr": table.data_ptr(), "device_ptr": dev.value,
                 "bytes": nbytes, "read_only": bool(ro.value), "register_seconds": round(time.perf_counter() - t, 3),
                 "row_bytes": self.embedding_dim * table.element_size(), "rows": self.num_embeddings,
                 "bad": torch.zeros(1, dtype=torch.int32, device=device)}
        _ZERO_COPY.setdefault("tables", []).append(entry)
        object.__setattr__(self, "_q38_zero_copy", entry)
        print(f"Q38_EMBEDDING_ZERO_COPY,role,{self.role},bytes,{nbytes},read_only,{int(ro.value)},"
              f"register_s,{entry['register_seconds']}", flush=True)
        return entry

    def _embed_zero_copy(self, input_ids: torch.Tensor) -> torch.Tensor:
        import ctypes
        z = getattr(self, "_q38_zero_copy", None) or self._zero_copy_setup(input_ids.device)
        ids = input_ids.reshape(-1).to(dtype=torch.int64).contiguous()
        out = torch.empty((ids.numel(), self.embedding_dim), dtype=self.table.dtype, device=input_ids.device)
        stream = torch.cuda.current_stream(input_ids.device)
        err = ctypes.create_string_buffer(512)
        if _ZERO_COPY["lib"].q38_zero_copy_gather(ctypes.c_void_p(z["device_ptr"]), z["rows"], z["row_bytes"],
                ctypes.c_void_p(ids.data_ptr()), ids.numel(), ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(z["bad"].data_ptr()), ctypes.c_void_p(stream.cuda_stream), err, len(err)):
            raise RuntimeError(f"{self.role}: zero-copy gather enqueue failed: {err.value.decode(errors='replace')}")
        if not torch.cuda.is_current_stream_capturing():
            _stat(self.role + "_zero_copy_enqueue", 0.0, ids.numel())
        return out.view(*input_ids.shape, self.embedding_dim)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        # One call per forward step, also under CUDA-graph replay (this is an eager break), so the
        # decode-rate progress lines live here (the adapter's MoE forward is replayed from graphs).
        try:
            from qwen38_vllm_rtx_peer_adapter import _progress_note
            _progress_note(int(input_ids.numel()))
        except ImportError:
            pass
        for hook in STEP_HOOKS:
            hook()
        t = time.perf_counter()
        idx = input_ids.reshape(-1).to("cpu").long()
        rows = self.table.index_select(0, idx).view(*input_ids.shape, self.embedding_dim)
        out = rows.to(input_ids.device)
        _stat(self.role, time.perf_counter() - t, idx.numel())
        return out

    # LM head path (UnquantizedEmbeddingMethod.apply signature)
    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        del layer
        t = time.perf_counter()
        logits = torch.matmul(x.to("cpu", dtype=self.table.dtype), self.table.t())
        if bias is not None:
            logits = logits + bias.to("cpu", dtype=logits.dtype)
        out = logits.to(x.device)
        _stat(self.role, time.perf_counter() - t, x.shape[0] if x.ndim > 1 else 1)
        return out


def _module_bytes(m: nn.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in list(m.parameters(recurse=True)) + list(m.buffers(recurse=True)):
        k = f"{t.device.type}:{t.dtype}".replace("torch.", "")
        out[k] = out.get(k, 0) + t.numel() * t.element_size()
    return out


def _install_layer_memory_log(model_mod: Any) -> None:
    """Q38_LOG_LAYER_MEMORY=1: after each decoder layer / top-level model init, log parameter+buffer
    bytes by device:dtype and the change in torch.cuda.memory_allocated (one JSON line per module)."""
    path = os.environ.get("Q38_LAYER_MEMORY_LOG")
    def wrap(cls_name: str) -> None:
        cls = getattr(model_mod, cls_name, None)
        if cls is None:
            return
        orig = cls.__init__
        def init(self, *a: Any, **k: Any) -> None:
            before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            orig(self, *a, **k)
            after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            rec = {"class": cls_name, "prefix": k.get("prefix", ""), "layer_type": k.get("layer_type"),
                   "cuda_allocated_delta": after - before, "cuda_allocated": after, "bytes": _module_bytes(self)}
            line = json.dumps(rec)
            print("Q38_LAYER_MEMORY " + line, flush=True)
            if path:
                with open(path + f".{os.getpid()}.jsonl", "a") as f:
                    f.write(line + "\n")
        cls.__init__ = init
    for name in ("Qwen4ExpDecoderLayer", "Qwen4ExpModel"):
        wrap(name)


# Q38_M10_LM_HEAD=1: the LM head lives on the M10s (PLE owner slices, worker mode 7). The RTX holds no
# weight; LogitsProcessor's quant_method.apply() calls LM_HEAD_FN (set by install_model_hooks to the
# peer adapter's lm_head_logits). lm_head.weight is skipped before it is read.
LM_HEAD_FN = None


class M10LmHead(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *args: Any, **kwargs: Any):
        super().__init__()
        self.role = "lm_head_m10"
        self.num_embeddings = self.org_vocab_size = self.num_embeddings_padded = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.tp_size = 1
        self.quant_method = _HeadMethod(self)

    def extra_repr(self) -> str:
        return f"role=lm_head_m10, vocab={self.num_embeddings}, hidden={self.embedding_dim}, weights=M10"

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return {name for name, _ in weights}        # normally skipped before read; never kept

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if LM_HEAD_FN is None:
            raise RuntimeError("Q38_M10_LM_HEAD=1 but no M10 LM-head function was installed")
        if bias is not None:
            raise RuntimeError("M10 LM head does not support a logits bias")
        t = time.perf_counter()
        out = LM_HEAD_FN(x)
        _stat("lm_head_m10_enqueue", time.perf_counter() - t, int(x.shape[0]))
        if out.shape[-1] != self.num_embeddings:
            raise RuntimeError(f"M10 LM head returned vocab {out.shape[-1]}, model expects {self.num_embeddings}")
        return out


def install(model_mod: Any) -> dict[str, Any]:
    """Patch the constructors referenced by vllm.models.qwen4_exp.nvidia.model."""
    info: dict[str, Any] = {}
    if os.environ.get("Q38_LOG_LAYER_MEMORY") == "1":
        _install_layer_memory_log(model_mod)
        info["layer_memory_log"] = True
    if os.environ.get("Q38_CPU_EMBEDDING") == "1":
        model_mod.VocabParallelEmbedding = lambda n, d, *a, **k: CpuTable(n, d, *a, role="embedding", **k)
        info["embedding"] = "cpu"
    if os.environ.get("Q38_CPU_LM_HEAD") == "1" and os.environ.get("Q38_M10_LM_HEAD") == "1":
        raise RuntimeError("Q38_CPU_LM_HEAD and Q38_M10_LM_HEAD are exclusive")
    if os.environ.get("Q38_CPU_LM_HEAD") == "1":
        model_mod.ParallelLMHead = lambda n, d, *a, **k: CpuTable(n, d, *a, role="lm_head", **k)
        info["lm_head"] = "cpu"
    if os.environ.get("Q38_M10_LM_HEAD") == "1":
        model_mod.ParallelLMHead = lambda n, d, *a, **k: M10LmHead(n, d, *a, **k)
        info["lm_head"] = "m10"
    return info
