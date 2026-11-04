"""Eager breaks for vLLM breakable CUDA graphs (opt-in: Q38_VLLM_CUDAGRAPH=1).

vLLM's breakable CUDA graphs (vllm/compilation/breakable_cudagraph.py) capture the decode
forward as graph segments; a function decorated with eager_break_during_capture ends the
current segment, runs eagerly on the capture stream, is recorded, and capture resumes.
Replay calls the recorded function again but DISCARDS its return value, so downstream graph
segments must read the result from a fixed address.

make_break(fn) wraps a host-side function (our MoE exchange, PLE lookup, CPU embedding) so that
it always returns a cached static buffer (one per key + shape + dtype) holding a copy of fn's
result. During capture the graph records reads of that buffer; on replay the same buffer is
refilled before the next segment runs. Outside capture (eager prefill) the wrapper behaves the
same (one extra small device copy). The first positional argument is the cache key (for
example the layer id); non-tensor arguments pass through vLLM's weak-ref step unchanged.

Q38_VLLM_CUDAGRAPH_FULL=1 additionally captures vLLM's own GDN / QSA / indexer cores inside the
graph (see install_full_breakable); our breaks stay eager (see _always_eager_break).
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch

_STATS = {"calls": 0, "buffers": 0, "capture_skips": 0}


def enabled() -> bool:
    return os.environ.get("Q38_VLLM_CUDAGRAPH") == "1"


def make_break(fn: Callable[..., torch.Tensor], name: str,
               out_like: Callable[..., tuple[tuple[int, ...], torch.dtype, torch.device]]) -> Callable[..., torch.Tensor]:
    """out_like(key, *args) -> (shape, dtype, device) of fn's result, computed WITHOUT running fn.

    During CUDA-graph capture the preceding graph segments are only recorded, not executed, so the
    inputs of an eager break are uncomputed memory (2026-11-03: a capture-time MoE exchange sent
    garbage expert ids to the M10s and faulted every routed GM107). At capture time we therefore
    return the static buffer without calling fn; on replay (inputs valid) fn runs and refills it.
    """
    bufs: dict[tuple, torch.Tensor] = {}

    def buffer(key: Any, args: tuple) -> torch.Tensor:
        shape, dtype, device = out_like(key, *args)
        k = (name, key, tuple(shape), dtype, str(device))
        b = bufs.get(k)
        if b is None:
            b = torch.zeros(tuple(shape), dtype=dtype, device=device)
            bufs[k] = b
            _STATS["buffers"] += 1
        return b

    def run(key: Any, *args: Any) -> torch.Tensor:
        b = buffer(key, args)
        if _capturing():
            _STATS["capture_skips"] += 1
            return b                      # never run real work on uncomputed inputs
        r = fn(key, *args)
        if tuple(r.shape) != tuple(b.shape) or r.dtype != b.dtype:
            raise RuntimeError(f"q38 break {name}: result {tuple(r.shape)}/{r.dtype} does not match "
                               f"declared buffer {tuple(b.shape)}/{b.dtype}")
        b.copy_(r)
        _STATS["calls"] += 1
        return b

    run.__name__ = f"q38_break_{name}"
    if not enabled():
        return lambda key, *args: fn(key, *args)
    from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    if not is_breakable_cudagraph_enabled():
        raise RuntimeError("Q38_VLLM_CUDAGRAPH=1 needs VLLM_USE_BREAKABLE_CUDAGRAPH=1 in the engine process")
    return _always_eager_break(run)


def _always_eager_break(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Like vLLM's eager_break_during_capture, but never inlined.

    vLLM's decorator runs the function INSIDE the capture when the forward mode is FULL (its own
    breaks - GDN core, QSA, indexer - are graph-safe there). Our breaks do host work (CPU
    embedding, PLE lookup, host MoE exchange); captured inline they would become a no-op. So under
    any breakable capture we always end the segment and record fn for replay.
    """
    import functools
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture, _weak_ref_capture_arg

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        capture = BreakableCUDAGraphCapture.current()
        if capture is None or not capture._capturing:
            if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
                # A plain torch.cuda.graph capture (vLLM's monolithic FULL wrapper) would record
                # our break as a no-op: refuse instead of silently producing wrong tokens.
                raise RuntimeError(f"q38 break {fn.__name__} inside a non-breakable CUDA graph capture; "
                                   "use Q38_VLLM_CUDAGRAPH_FULL=1 (breakable FULL wrapper)")
            return fn(*args, **kwargs)
        weak_args = tuple(_weak_ref_capture_arg(a) for a in args)
        weak_kwargs = {k: _weak_ref_capture_arg(v) for k, v in kwargs.items()}
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

    return wrapper


def full_enabled() -> bool:
    return enabled() and os.environ.get("Q38_VLLM_CUDAGRAPH_FULL") == "1"


def install_full_breakable() -> bool:
    """Q38_VLLM_CUDAGRAPH_FULL=1 (with cudagraph_mode FULL_AND_PIECEWISE): capture FULL decode
    graphs with vLLM's BreakableCUDAGraphWrapper instead of the monolithic CUDAGraphWrapper.

    With FULL dispatch the GDN / QSA metadata builders stage decode metadata into their static
    graph buffers, and vLLM's own breaks (qwen_gdn_attention_core*, sparse_attn_indexer,
    qsa._run_qsa, ple _short_conv) run inline, so they are captured. Only our q38 breaks still
    split the graph. Must run in the worker before GPUModelRunner.load_model wraps the model.
    """
    if not full_enabled():
        return False
    _install_full_breakable_v2()
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker import gpu_model_runner as gmr
    orig = gmr.CUDAGraphWrapper
    if getattr(orig, "_q38_full_breakable", False):
        return True

    class _FullAsBreakable(orig):  # subclass keeps clear_all_graphs/_all_instances/isinstance
        _q38_full_breakable = True

        def __new__(cls, runnable, vllm_config, runtime_mode=None, *a, **k):
            if runtime_mode == CUDAGraphMode.FULL and not a and not k:
                _STATS["full_breakable_wrappers"] = _STATS.get("full_breakable_wrappers", 0) + 1
                return BreakableCUDAGraphWrapper(runnable, vllm_config, runtime_mode=CUDAGraphMode.FULL)
            return orig(runnable, vllm_config, runtime_mode, *a, **k)

    gmr.CUDAGraphWrapper = _FullAsBreakable
    return True


def _capturing() -> bool:
    """True while vLLM's breakable capture context is active (replay runs outside it)."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    return BreakableCUDAGraphCapture.current() is not None or (
        torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())


def _install_full_breakable_v2() -> None:
    """Same idea for vLLM's V2 model runner (vllm/v1/worker/gpu), which this engine uses.

    CudaGraphManager.capture records each FULL decode graph itself with a plain
    torch.cuda.graph(graph, pool, stream) around forward_fn(CUDAGraphMode.NONE) (2026-11-03 task 56:
    our embedding break refused that capture). We re-create that method from vLLM's own source
    with exactly two edits, each asserted so a vLLM change fails loudly:
      1. torch.cuda.graph(...)  ->  a BreakableCUDAGraphCapture bound to `graph`
         (same pool and stream; .replay() has the CUDAGraph API, so run_fullgraph is unchanged);
      2. the capture-time forward_fn(NONE) -> forward_fn(FULL), so vLLM's own breaks (GDN core,
         QSA, indexer, PLE short conv) run inline and are captured, while ours end the segment.
    The warm-up forward_fn(NONE) before the capture is left unchanged.
    """
    import contextlib
    import gc
    import inspect
    import re
    import textwrap
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.v1.worker.gpu import cudagraph_utils as cu

    mgr = cu.CudaGraphManager
    if getattr(mgr.capture, "_q38_full_breakable", False):
        return
    src = textwrap.dedent(inspect.getsource(mgr.capture))
    pat = re.compile(r"torch\.cuda\.graph\(\s*graph,\s*pool,\s*stream=self\._capture_stream\(desc\)\s*\)")
    if len(pat.findall(src)) != 1:
        raise RuntimeError("q38 FULL-breakable V2 patch: torch.cuda.graph(...) call not found exactly once")
    src = pat.sub("_q38_breakable_graph(pool, self._capture_stream(desc)) as graph", src)
    call = "forward_fn(CUDAGraphMode.NONE)"
    if src.count(call) != 2:
        raise RuntimeError(f"q38 FULL-breakable V2 patch: expected 2 x {call}, found {src.count(call)}")
    i = src.rindex(call)
    src = src[:i] + "forward_fn(CUDAGraphMode.FULL)" + src[i + len(call):]

    @contextlib.contextmanager
    def _q38_breakable_graph(pool, stream):
        # torch.cuda.graph() syncs and cleans up before capturing; mirror vLLM's breakable wrapper.
        torch.cuda.synchronize()
        if gc.isenabled():
            gc.collect()
            torch.cuda.empty_cache()
        with torch.cuda.stream(stream), BreakableCUDAGraphCapture(pool=pool) as cap:
            yield cap
        _STATS["full_graphs"] = _STATS.get("full_graphs", 0) + 1
        _STATS["full_graph_segments"] = cap.num_graphs
        _STATS["full_graph_eager_breaks"] = cap.num_eager_breaks
        print(f"Q38_VLLM_CUDAGRAPH_FULL,captured,segments,{cap.num_graphs},"
              f"eager_breaks,{cap.num_eager_breaks}", flush=True)

    ns = dict(vars(cu))
    ns["_q38_breakable_graph"] = _q38_breakable_graph
    exec(compile(src, f"<q38 patched {cu.__name__}.CudaGraphManager.capture>", "exec"), ns)
    patched = ns["capture"]
    patched._q38_full_breakable = True
    mgr.capture = patched
