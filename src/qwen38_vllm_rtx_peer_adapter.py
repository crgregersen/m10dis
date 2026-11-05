"""vLLM MoE adapter using RTX GPU kernels for resident M10 request/response."""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def discover_rtx_bdf() -> str:
    """Return the RTX PCI address from qwen38_pci_discover.py (read-only sysfs scan)."""
    script = Path(__file__).with_name("qwen38_pci_discover.py")
    out = subprocess.run(["python3", str(script), "--json"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"RTX discovery failed: {out.stderr.strip()}")
    return json.loads(out.stdout)["rtx"]["bdf"]

import torch


# Progress: every layer-0 MoE call is one forward step (rows=1 decode, rows>1 prefill chunk).
# A daemon thread prints one line every Q38_PROGRESS_SECONDS (default 10, 0 disables) to stderr.
import sys as _sys
import threading as _threading
_progress = {"decode_tokens": 0, "prefill_rows": 0, "steps": 0, "started": None, "thread": None}
_progress_lock = _threading.Lock()


def _progress_loop(period: float) -> None:
    last_tokens, last_t = 0, time.monotonic()
    while True:
        time.sleep(period)
        with _progress_lock:
            tok, pre, steps, t0 = (_progress["decode_tokens"], _progress["prefill_rows"],
                                   _progress["steps"], _progress["started"])
        now = time.monotonic()
        rate = (tok - last_tokens) / (now - last_t)
        avg = tok / (now - t0) if t0 else 0.0
        print(f"Q38_PROGRESS,utc,{time.strftime('%H:%M:%S', time.gmtime())},decode_tokens,{tok},"
              f"prefill_rows,{pre},steps,{steps},tok_per_s_last,{rate:.3f},tok_per_s_avg,{avg:.3f}",
              file=_sys.stderr, flush=True)
        last_tokens, last_t = tok, now


# Optional CUDA profiler range for nsys --capture-range=cudaProfilerApi (2026-11-04): cudaProfilerStart at
# decode token Q38_CUDA_PROFILER_START_TOKENS (0 = off), cudaProfilerStop Q38_CUDA_PROFILER_TOKENS (default
# 200) tokens later. Called from _progress_note, i.e. on the engine thread, where the RTX is CUDA device 0.
_cuprof = {"state": 0}


def _cuda_profiler_note(tok: int) -> None:
    start = int(os.environ.get("Q38_CUDA_PROFILER_START_TOKENS", "0") or 0)
    if start <= 0 or _cuprof["state"] == 2:
        return
    count = int(os.environ.get("Q38_CUDA_PROFILER_TOKENS", "200") or 200)
    if _cuprof["state"] == 0 and tok >= start:
        torch.cuda.cudart().cudaProfilerStart()
        _cuprof["state"] = 1
        print(f"Q38_CUDA_PROFILER,start,decode_tokens,{tok}", file=_sys.stderr, flush=True)
    elif _cuprof["state"] == 1 and tok >= start + count:
        torch.cuda.cudart().cudaProfilerStop()
        _cuprof["state"] = 2
        print(f"Q38_CUDA_PROFILER,stop,decode_tokens,{tok}", file=_sys.stderr, flush=True)


def _progress_note(rows: int) -> None:
    period = float(os.environ.get("Q38_PROGRESS_SECONDS", "10"))
    if period <= 0:
        return
    with _progress_lock:
        if _progress["started"] is None:
            _progress["started"] = time.monotonic()
        _progress["steps"] += 1
        if rows == 1:
            _progress["decode_tokens"] += 1
        else:
            _progress["prefill_rows"] += rows
        if _progress["thread"] is None:
            _progress["thread"] = _threading.Thread(target=_progress_loop, args=(period,), daemon=True)
            _progress["thread"].start()
        tok = _progress["decode_tokens"]
    if rows == 1:
        _cuda_profiler_note(tok)


class PeerOpenError(RuntimeError):
    def __init__(self, message: str, *, rollback_complete: bool):
        super().__init__(message)
        self.rollback_complete = rollback_complete


def _canonical_bdf(value: str) -> str:
    domain, bus, devfn = value.strip().lower().split(":")
    device, function = devfn.split(".")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(function, 16):x}"


def _write_gpu_uuid_manifest(path: Path) -> None:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader",
    ], text=True, timeout=15)
    entries: dict[str, str] = {}
    for line in raw.splitlines():
        bdf, uuid = (part.strip() for part in line.split(",", 1))
        entries[_canonical_bdf(bdf)] = uuid
    path.write_text("".join(f"{bdf}\t{uuid}\n" for bdf, uuid in sorted(entries.items())))


class RtxPeerResidentM10MoEAdapter:
    """Send CUDA tensors from the RTX directly to retained M10 RM mailboxes.

    The only host round trip here is a small status transfer after the RTX
    reducer completes. H, IDs, route weights, and the joined result stay on GPU.
    PLE remains delegated to the existing retained-BAR1 lookup adapter.
    """

    def __init__(self, library: Path | str, gpu_mappings_manifest: Path | str,
                 ple_adapter: Any | None = None,
                 rtx_bdf: str | None = None,
                 control_socket: Path | str | None = None):
        self.library_path = str(Path(library).resolve())
        self.async_window = int(os.environ.get("Q38_RTX_PEER_ASYNC_WINDOW", "0"))
        if self.async_window < 0 or self.async_window > 32:
            raise ValueError("Q38_RTX_PEER_ASYNC_WINDOW must be between 0 and 32")
        self.async_max_windows = int(os.environ.get("Q38_RTX_PEER_ASYNC_MAX_WINDOWS", "1"))
        if self.async_max_windows < 1:
            raise ValueError("Q38_RTX_PEER_ASYNC_MAX_WINDOWS must be positive")
        self.async_start_layer = int(os.environ.get("Q38_RTX_PEER_ASYNC_START_LAYER", "0"))
        if self.async_start_layer < 0 or self.async_start_layer >= 48:
            raise ValueError("Q38_RTX_PEER_ASYNC_START_LAYER must be between 0 and 47")
        self.mapping_path = str(Path(gpu_mappings_manifest).resolve())
        self.ple_adapter = ple_adapter
        # Never a hardcoded default: explicit argument, Q38_RTX_BDF, or a live scan.
        self.rtx_bdf = rtx_bdf or os.environ.get("Q38_RTX_BDF") or discover_rtx_bdf()
        self.control_socket = str(control_socket or os.environ.get("Q38_M10_CONTROL_SOCKET", ""))
        self._ingress_diagnostic = os.environ.get("Q38_RTX_PEER_INGRESS_DIAGNOSTIC") == "1"
        if not self.control_socket:
            raise RuntimeError("RTX peer mode requires the resident daemon control socket for safe mapping lifetime")
        from qwen38_resident_daemon import send_control
        self._send_control = send_control
        before = self._send_control(Path(self.control_socket), "STATUS", timeout=30)
        if before.get("status") != "READY":
            raise RuntimeError(f"resident daemon is not READY: {before}")
        self._workers_were_active = before.get("worker_activity", "ACTIVE") != "PARKED"
        if self._workers_were_active:
            parked = self._send_control(Path(self.control_socket), "PARK_WORKERS",
                                        timeout=120, clients_quiesced=True)
            if parked.get("status") != "ok" or parked.get("code") != "WORKERS_PARKED":
                raise RuntimeError(f"could not park M10 workers before peer route setup: {parked}")
        self.uuid_path = Path(os.environ.get(
            "Q38_M10_GPU_UUIDS_MANIFEST",
            str(Path(self.mapping_path).with_name(f"gpu_uuids.{os.getpid()}.tsv")),
        )).resolve()
        _write_gpu_uuid_manifest(self.uuid_path)
        self.lib = ctypes.CDLL(self.library_path)
        self.lib.q38_rtx_peer_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_open.restype = ctypes.c_int
        self.lib.q38_rtx_peer_open_rollback_complete.argtypes = []
        self.lib.q38_rtx_peer_open_rollback_complete.restype = ctypes.c_int
        self.lib.q38_rtx_peer_forward.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_forward.restype = ctypes.c_int
        self.lib.q38_rtx_peer_forward_async.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_forward_async.restype = ctypes.c_int
        self.lib.q38_rtx_peer_async_flush.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                       ctypes.c_size_t]
        self.lib.q38_rtx_peer_async_flush.restype = ctypes.c_int
        self.lib.q38_rtx_peer_close.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_close.restype = ctypes.c_int
        self.lib.q38_rtx_peer_mark_m10_quiesced.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_mark_m10_quiesced.restype = ctypes.c_int
        self.lib.q38_rtx_peer_pex_word_probe.argtypes = [ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_pex_word_probe.restype = ctypes.c_int
        self.lib.q38_rtx_peer_direct_word_probe.argtypes = [ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_direct_word_probe.restype = ctypes.c_int
        # Phase 2 GPU-driven exchange (Q38_RTX_PEER_GPU_DRIVEN=1)
        self.gpu_driven = os.environ.get("Q38_RTX_PEER_GPU_DRIVEN") == "1"
        self.lib.q38_rtx_peer_forward_gpu.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_forward_gpu.restype = ctypes.c_int
        self.lib.q38_rtx_peer_gpu_resync.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_gpu_resync.restype = ctypes.c_int
        self.lib.q38_rtx_peer_check_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_check_error.restype = ctypes.c_int
        self.lib.q38_rtx_peer_abort.argtypes = []
        self.lib.q38_rtx_peer_abort.restype = None
        self.lib.q38_rtx_peer_gpu_counters.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)]
        self.lib.q38_rtx_peer_gpu_counters.restype = ctypes.c_int
        self.lib.q38_rtx_peer_mappings_quiesced.argtypes = []
        self.lib.q38_rtx_peer_mappings_quiesced.restype = ctypes.c_int
        # PCI-cluster row gather for PLE (Q38_RTX_PEER_PLE_GPU=1, needs the GPU-driven path): the
        # PLE lookup becomes one broadcast exchange (worker mode 6) instead of the CPU-driven mode 4.
        self.ple_gpu = self.gpu_driven and os.environ.get("Q38_RTX_PEER_PLE_GPU") == "1"
        self.ple_gpu_calls = 0
        self.ple_gpu_rows = 0
        self.lib.q38_rtx_peer_configure_row_gather.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_configure_row_gather.restype = ctypes.c_int
        self.lib.q38_rtx_peer_row_gather_gpu.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                                                         ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_row_gather_gpu.restype = ctypes.c_int
        # Batched MoE (Q38_RTX_PEER_MOE_BATCH=1, needs the GPU-driven path): calls with at least
        # Q38_RTX_PEER_MOE_BATCH_MIN_ROWS rows go as one exchange per 64 rows (worker mode 8) through the
        # batch area of the shared CPU mailbox, instead of one exchange per row (mode 1).
        self.moe_batch = self.gpu_driven and os.environ.get("Q38_RTX_PEER_MOE_BATCH") == "1"
        self.moe_batch_min_rows = int(os.environ.get("Q38_RTX_PEER_MOE_BATCH_MIN_ROWS", "2"))
        # Expert-reuse study (2026-11-03): Q38_BATCH_REUSE_DIR=<dir> saves the router ids of every eager batched
        # call (int32 [rows, 10], one file per call: L<layer>_c<n>.pt) for offline grouping/chunk simulations.
        # Forces a device sync per call; off by default.
        self.reuse_dir = os.environ.get("Q38_BATCH_REUSE_DIR") or None
        self.reuse_files = 0
        self.moe_batch_calls = 0
        self.moe_batch_rows = 0
        self._batch_mailbox = None
        self._batch_view = None
        self.lib.q38_rtx_peer_configure_moe_batch.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_configure_moe_batch.restype = ctypes.c_int
        self.lib.q38_rtx_peer_moe_batch_gpu.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                                                        ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_moe_batch_gpu.restype = ctypes.c_int
        # Pipelined batched MoE (Q38_RTX_PEER_MOE_BATCH_PIPE=1, 2026-11-03): calls with at least
        # Q38_RTX_PEER_MOE_BATCH_PIPE_MIN_ROWS rows go as ONE enqueue of 32-row chunks (worker mode 9, two host slots):
        # the RTX combines chunk c and publishes chunk c+2 while the workers compute chunk c+1.
        self.moe_batch_pipe = self.moe_batch and os.environ.get("Q38_RTX_PEER_MOE_BATCH_PIPE") == "1"
        self.moe_batch_pipe_min_rows = int(os.environ.get("Q38_RTX_PEER_MOE_BATCH_PIPE_MIN_ROWS", "33"))
        self.moe_batch_pipe_calls = 0
        self.lib.q38_rtx_peer_moe_batch_pipe_gpu.argtypes = list(self.lib.q38_rtx_peer_moe_batch_gpu.argtypes)
        self.lib.q38_rtx_peer_moe_batch_pipe_gpu.restype = ctypes.c_int
        # Multi-row decode MoE (Q38_RTX_PEER_MOE_MULTIROW=1, 2026-11-04; worker mode 10, resident_multirow_abi.cuh):
        # calls of exactly 2 BF16 rows (MTP verify, batch decode) go as ONE exchange through the PEX multicast
        # request and remapped return pages, instead of 2 mode-1 exchanges or one host-staged mode-8 exchange.
        # The library reads the same variable at open (extra return / card-sum routes).
        self.moe_multirow = self.gpu_driven and os.environ.get("Q38_RTX_PEER_MOE_MULTIROW") == "1"
        # Q38_RTX_PEER_MOE_MULTIROW_MAX (2..4, default 2): the most rows one exchange takes (the library sizes its
        # return / card-sum routes for it at open); calls of 2..max rows go as one mode-10 exchange.
        self.moe_multirow_max = int(os.environ.get("Q38_RTX_PEER_MOE_MULTIROW_MAX", "2") or 2) if self.moe_multirow else 0
        # Q38_RTX_PEER_MOE_MULTIROW_MIN (1 or 2, default 1): 1 also sends single-row calls (plain decode) through mode 10,
        # whose streamed finish is faster than mode 1 (T257: 133.1 vs 141.3 us per layer; bit-identical)
        self.moe_multirow_min = int(os.environ.get("Q38_RTX_PEER_MOE_MULTIROW_MIN", "1") or 1)
        self.moe_multirow_calls = 0
        self.lib.q38_rtx_peer_moe2_gpu.argtypes = list(self.lib.q38_rtx_peer_moe_batch_gpu.argtypes)
        self.lib.q38_rtx_peer_moe2_gpu.restype = ctypes.c_int
        # LM head on the M10s (Q38_M10_LM_HEAD=1, needs the GPU-driven path and LMHEAD manifest rows): one
        # broadcast exchange per row returns every worker's exact top-K + log-sum-exp (worker mode 7).
        self.lm_head_m10 = self.gpu_driven and os.environ.get("Q38_M10_LM_HEAD") == "1"
        self.lm_topk = int(os.environ.get("Q38_M10_LM_HEAD_TOPK", "1024"))
        if not 1 <= self.lm_topk <= 2559:
            raise ValueError("Q38_M10_LM_HEAD_TOPK must be between 1 and 2559")
        self.lm_vocab = 0
        self.lm_calls = 0
        self.lib.q38_rtx_peer_lm_head_topk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                       ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p,
                                                       ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_lm_head_topk.restype = ctypes.c_int
        self.lib.q38_rtx_peer_lm_head_topk_rows.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                                            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_lm_head_topk_rows.restype = ctypes.c_int
        self.lib.q38_rtx_peer_timing_enable.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_timing_enable.restype = ctypes.c_int
        self.lib.q38_rtx_peer_timing_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                                      ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_timing_read.restype = ctypes.c_int
        self.lib.q38_rtx_peer_worker_timing_enable.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_worker_timing_enable.restype = ctypes.c_int
        self.lib.q38_rtx_peer_worker_timing_read.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                                             ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_rtx_peer_worker_timing_read.restype = ctypes.c_int
        # Route ownership: register a crash-safe route journal with the daemon BEFORE the library
        # acquires any patched-UVM record, so RECOVER_PEER_CLIENT can release them if this
        # process dies (docs/RESIDENT_DAEMON_INTERFACE.md).
        journal = os.environ.get("Q38_RTX_PEER_ROUTE_JOURNAL")
        if not journal:
            base = Path(os.environ.get("Q38_M10_RTX_PEER_TIMING_JSON", "/tmp/q38_rtx_peer"))
            journal = str(base.with_name(f"rtx_peer_routes.{os.getpid()}.journal"))
            os.environ["Q38_RTX_PEER_ROUTE_JOURNAL"] = journal
        self.route_journal = Path(journal)
        self.peer_run_id = f"rtx-peer-{os.getpid()}-{time.time_ns()}"
        reg = self._send_control(Path(self.control_socket), "REGISTER_PEER_CLIENT", timeout=30,
                                 run_id=self.peer_run_id, pid=os.getpid(), journal=journal)
        if reg.get("status") != "ok":
            if self._workers_were_active:
                self._send_control(Path(self.control_socket), "WAKE_WORKERS", timeout=120, clients_quiesced=True)
            raise PeerOpenError(f"daemon refused peer-client registration: {reg}", rollback_complete=True)
        error = ctypes.create_string_buffer(1024)
        rc = self.lib.q38_rtx_peer_open(self.rtx_bdf.encode(), self.mapping_path.encode(),
                                        str(self.uuid_path).encode(),
                                        error, len(error))
        if rc:
            rollback_complete = bool(self.lib.q38_rtx_peer_open_rollback_complete())
            if rollback_complete:
                # The library released everything it acquired; record that and hand ownership back.
                with open(self.route_journal, "a") as jf:
                    jf.write("CLOSED\trollback\n"); jf.flush(); os.fsync(jf.fileno())
                self._send_control(Path(self.control_socket), "UNREGISTER_PEER_CLIENT", timeout=30,
                                   run_id=self.peer_run_id)
            if self._workers_were_active:
                self._send_control(Path(self.control_socket), "WAKE_WORKERS",
                                   timeout=120, clients_quiesced=True)
            raise PeerOpenError(
                f"RTX→M10 peer mapping setup failed: {error.value.decode(errors='replace')}",
                rollback_complete=rollback_complete)
        if self.gpu_driven:
            self.gpu_resync()
        # A client needs live workers to service requests even when the daemon
        # parked them during idle. Wake after installing peer routes, then
        # restore the original activity state in close().
        try:
            if self.ple_gpu or self.lm_head_m10:
                # Workers are still parked: install the per-worker table sizes (saved, verified,
                # restored at close), the shard table and the LM-head slices before any request.
                manifest = os.environ.get("Q38_M10_DIRECT_PLE_MANIFEST", "")
                if not manifest:
                    raise RuntimeError("Q38_RTX_PEER_PLE_GPU / Q38_M10_LM_HEAD need Q38_M10_DIRECT_PLE_MANIFEST")
                if self.lib.q38_rtx_peer_configure_row_gather(manifest.encode(), error, len(error)):
                    raise RuntimeError(f"row-gather configuration failed: {error.value.decode(errors='replace')}")
            if self.moe_batch:
                self._configure_moe_batch(error)
            if self.lm_head_m10:
                vocab = ctypes.c_uint32(0)
                if self.lib.q38_rtx_peer_lm_head_topk(None, 2, None, None, 1, ctypes.byref(vocab), None, error, len(error)):
                    raise RuntimeError(f"M10 LM head not configured: {error.value.decode(errors='replace')}")
                self.lm_vocab = int(vocab.value)
            if not self._ingress_diagnostic:
                awake = self._send_control(Path(self.control_socket), "WAKE_WORKERS",
                                           timeout=120, clients_quiesced=True)
                if awake.get("status") != "ok" or awake.get("code") != "WORKERS_ACTIVE":
                    raise RuntimeError(f"could not start M10 workers with peer routes installed: {awake}")
        except Exception as wake_error:
            # A control timeout does not prove that WAKE was not applied.
            # Re-establish worker quiescence before releasing any route.
            try:
                status = self._send_control(Path(self.control_socket), "STATUS", timeout=30)
                if status.get("worker_activity", "ACTIVE") != "PARKED":
                    parked = self._send_control(Path(self.control_socket), "PARK_WORKERS",
                                                timeout=120, clients_quiesced=True)
                    if parked.get("status") != "ok" or parked.get("code") != "WORKERS_PARKED":
                        raise RuntimeError(f"workers did not park after ambiguous WAKE: {parked}")
                self.lib.q38_rtx_peer_mark_m10_quiesced(error, len(error))
                close_rc = self.lib.q38_rtx_peer_close(error, len(error))
                if close_rc:
                    raise RuntimeError(error.value.decode(errors="replace"))
                if self._workers_were_active:
                    self._send_control(Path(self.control_socket), "WAKE_WORKERS",
                                       timeout=120, clients_quiesced=True)
            except Exception as cleanup_error:
                raise RuntimeError(
                    f"worker wake failed ({wake_error}); safe route cleanup also failed "
                    f"({cleanup_error}); retain mappings and worker allocations") from wake_error
            raise RuntimeError(f"worker wake failed; peer mappings restored: {wake_error}") from wake_error
        self._closed = False
        self.calls = 0
        self.rows = 0
        self.timings: list[dict[str, Any]] = []
        self._async_pending_layers: list[int] = []
        self._async_stream: int | None = None
        self.async_windows: list[dict[str, Any]] = []
        atexit.register(self.close)

    @property
    def supports_direct_ple(self) -> bool:
        return self.ple_adapter is not None and self.ple_adapter.supports_direct_ple

    @torch.inference_mode()
    def forward(self, layer_id: int, hidden: torch.Tensor,
                expert_ids: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("RTX peer adapter is closed")
        if hidden.ndim != 2 or tuple(hidden.shape[1:]) != (2560,):
            raise ValueError(f"hidden must be [N,2560], got {tuple(hidden.shape)}")
        rows = int(hidden.shape[0])
        if layer_id == 0 and os.environ.get("Q38_CPU_EMBEDDING") != "1":
            _progress_note(rows)          # with the CPU embedding, its break counts steps instead
        if expert_ids.shape != (rows, 10) or route_weights.shape != (rows, 10):
            raise ValueError("expert IDs and route weights must both be [N,10]")
        if hidden.device.type != "cuda" or hidden.device.index != 0:
            raise ValueError("vLLM MoE tensors must reside on the RTX CUDA device 0")
        if hidden.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError(f"unsupported RTX→M10 hidden dtype: {hidden.dtype}")
        if rows == 0:
            return torch.empty_like(hidden)
        # vLLM invokes the MoE path with multi-row synthetic batches while it
        # profiles memory during EngineCore startup. Keep that setup path on
        # the established synchronous implementation. The async experiment is
        # specifically for one-row autoregressive decode calls. If a profiling
        # call follows a partially queued async window, drain that window first
        # so the sync path cannot overtake it on a different stream/order.
        if self.async_window and rows != 1 and self._async_pending_layers:
            self.flush_async(reason="non-single-row call")

        if self.gpu_driven:
            return self._forward_gpu(layer_id, hidden, expert_ids, route_weights, rows)
        h = hidden.contiguous()
        ids = expert_ids.to(dtype=torch.int32).contiguous()
        routes = route_weights.to(dtype=torch.float32).contiguous()
        joined = torch.empty((rows, 2560), device=hidden.device, dtype=torch.float32)
        stream_obj = torch.cuda.current_stream(hidden.device)
        stream = stream_obj.cuda_stream
        # Native launches are not visible to PyTorch's allocator. Keep every
        # temporary alive on this stream until the queued peer work consumes it.
        for tensor in (h, ids, routes, joined):
            tensor.record_stream(stream_obj)
        h_type = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[hidden.dtype]
        call_rows = []
        started = time.perf_counter()
        for row in range(rows):
            error = ctypes.create_string_buffer(1024)
            use_async = (rows == 1 and self.async_window > 0 and
                         int(layer_id) >= self.async_start_layer and
                         len(self.async_windows) < self.async_max_windows)
            forward_fn = (self.lib.q38_rtx_peer_forward_async if use_async
                          else self.lib.q38_rtx_peer_forward)
            if use_async:
                if self._async_stream is not None and self._async_stream != int(stream):
                    raise RuntimeError("async peer window cannot span vLLM CUDA streams")
                self._async_pending_layers.append(int(layer_id))
                if self._async_stream is None:
                    self._async_stream = int(stream)
            rc = forward_fn(
                int(layer_id), ctypes.c_void_p(h.data_ptr() + row * 2560 * h.element_size()),
                h_type,
                ctypes.c_void_p(ids.data_ptr() + row * 10 * ids.element_size()),
                ctypes.c_void_p(routes.data_ptr() + row * 10 * routes.element_size()),
                ctypes.c_void_p(joined.data_ptr() + row * 2560 * joined.element_size()),
                ctypes.c_void_p(stream), error, len(error),
            )
            if rc:
                if use_async:
                    try:
                        self.flush_async()
                    except Exception as drain_error:
                        raise RuntimeError(
                            f"RTX→M10 async layer {layer_id} enqueue failed and drain failed; "
                            f"retain mappings/resources: {error.value.decode(errors='replace')}; "
                            f"{drain_error}") from drain_error
                raise RuntimeError(f"RTX→M10 layer {layer_id} row {row} failed: {error.value.decode(errors='replace')}")
            call_rows.append(row)
        result = joined.to(dtype=hidden.dtype)
        if self._async_pending_layers and len(self._async_pending_layers) >= self.async_window:
            self.flush_async(reason="window full")
        self.timings.append({"layer": int(layer_id), "rows": rows,
            "gpu_peer_exchange_seconds": time.perf_counter() - started,
            "host_payload_relay_bytes": 0, "rows_completed": call_rows})
        self.calls += 1
        self.rows += rows
        return result

    def lookup_ple(self, global_rows: list[int]) -> bytes:
        # PLE uses a separate CPU publication path and shares the worker
        # command sequence, so it is a deliberate async-window boundary.
        self.flush_async(reason="PLE sequence boundary")
        if not self.supports_direct_ple:
            raise RuntimeError("no PLE adapter was configured for the RTX peer path")
        return self.ple_adapter.lookup_ple(global_rows)

    def flush_async(self, reason: str = "explicit") -> None:
        if not self._async_pending_layers:
            return
        if self._async_stream is None:
            raise RuntimeError("async peer window has pending layers but no CUDA stream")
        error = ctypes.create_string_buffer(1024)
        started = time.perf_counter()
        rc = self.lib.q38_rtx_peer_async_flush(
            ctypes.c_void_p(self._async_stream), error, len(error))
        elapsed = time.perf_counter() - started
        layers = self._async_pending_layers
        if rc:
            raise RuntimeError(f"RTX→M10 asynchronous window {layers} failed: "
                               f"{error.value.decode(errors='replace')}")
        self._async_pending_layers = []
        self._async_stream = None
        self.async_windows.append({"layers": layers, "layer_count": len(layers),
                                  "host_wait_seconds": elapsed, "flush_reason": reason})

    def timing_snapshot(self) -> dict[str, Any]:
        pex = os.environ.get("Q38_RTX_PEER_PEX_MULTICAST") == "1"
        return {"transport": ("RTX CUDA → six card-local PEX multicast ingress windows → resident M10 workers → RTX CUDA reduce"
                              if pex else "RTX CUDA fanout → M10 BAR1 mailboxes → RTX CUDA reduce"),
                "request_ingress": "PEX8747 hardware multicast" if pex else "24 direct RTX-to-worker writes",
                "calls": self.calls, "rows": self.rows, "host_payload_relay_bytes": 0,
                "gpu_driven": self.gpu_driven, "gpu_counters": getattr(self, "_gpu_counters_at_close", {}),
                "moe_batch": self.moe_batch, "moe_batch_reuse_files": self.reuse_files, "moe_batch_eager_calls": self.moe_batch_calls, "moe_batch_eager_rows": self.moe_batch_rows, "moe_batch_pipe": self.moe_batch_pipe, "moe_batch_pipe_eager_calls": self.moe_batch_pipe_calls,
                "moe_multirow": self.moe_multirow, "moe_multirow_eager_calls": self.moe_multirow_calls,
                "ple_gpu": self.ple_gpu, "ple_gpu_eager_calls": self.ple_gpu_calls, "ple_gpu_eager_rows": self.ple_gpu_rows,
                "lm_head_m10": self.lm_head_m10, "lm_topk": self.lm_topk, "lm_vocab": self.lm_vocab, "lm_calls": self.lm_calls,
                "async_window": self.async_window,
                "async_max_windows": self.async_max_windows,
                "async_start_layer": self.async_start_layer,
                "async_windows": list(self.async_windows),
                "timings": list(self.timings)}

    def probe_pex_word(self, marker: int) -> dict[str, list[int]]:
        """Send one bounded RTX store through group-0 PEX and verify its four receivers."""
        if self._closed:
            raise RuntimeError("RTX peer adapter is closed")
        before = (ctypes.c_uint32 * 4)()
        observed = (ctypes.c_uint32 * 4)()
        restored = (ctypes.c_uint32 * 4)()
        error = ctypes.create_string_buffer(1024)
        rc = self.lib.q38_rtx_peer_pex_word_probe(
            int(marker), before, observed, restored, error, len(error))
        if rc:
            raise RuntimeError(f"RTX PEX one-word probe failed: {error.value.decode(errors='replace')}")
        return {"before": list(before), "observed": list(observed),
                "restored": list(restored)}

    def probe_direct_word(self, marker: int) -> dict[str, int]:
        """Store one word directly to the first owner's BAR1 mapping."""
        if self._closed:
            raise RuntimeError("RTX peer adapter is closed")
        before = (ctypes.c_uint32 * 4)()
        observed = (ctypes.c_uint32 * 4)()
        restored = (ctypes.c_uint32 * 4)()
        error = ctypes.create_string_buffer(1024)
        rc = self.lib.q38_rtx_peer_direct_word_probe(
            int(marker), before, observed, restored, error, len(error))
        if rc:
            raise RuntimeError(f"RTX direct one-word probe failed: {error.value.decode(errors='replace')}")
        return {"before": before[0], "observed": observed[0], "restored": restored[0]}

    # ---- Phase 2: GPU-driven exchange --------------------------------------------------------
    def _forward_gpu(self, layer_id: int, hidden: torch.Tensor, expert_ids: torch.Tensor,
                     route_weights: torch.Tensor, rows: int) -> torch.Tensor:
        """Enqueue-only exchange (capturable). Dummy rows -> zeros on the GPU; errors surface via
        check_error() (called by the PLE break every layer and here when not capturing)."""
        h = hidden.contiguous()
        ids = expert_ids.to(dtype=torch.int32).contiguous()
        routes = route_weights.to(dtype=torch.float32).contiguous()
        joined = torch.empty((rows, 2560), device=hidden.device, dtype=torch.float32)
        stream_obj = torch.cuda.current_stream(hidden.device)
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            for tensor in (h, ids, routes, joined):
                tensor.record_stream(stream_obj)
        h_type = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[hidden.dtype]
        error = ctypes.create_string_buffer(1024)
        multi = self.moe_multirow and self.moe_multirow_min <= rows <= self.moe_multirow_max and h_type == 2
        batch = not multi and self.moe_batch and rows >= self.moe_batch_min_rows
        if multi:
            rc = self.lib.q38_rtx_peer_moe2_gpu(
                int(layer_id), ctypes.c_void_p(h.data_ptr()), h_type, ctypes.c_void_p(ids.data_ptr()),
                ctypes.c_void_p(routes.data_ptr()), rows, ctypes.c_void_p(joined.data_ptr()),
                ctypes.c_void_p(stream_obj.cuda_stream), error, len(error))
            if rc:
                raise RuntimeError(f"multi-row MoE enqueue layer {layer_id} rows {rows}: "
                                   f"{error.value.decode(errors='replace')}")
            if not capturing:
                self.moe_multirow_calls += 1
        if batch:
            self._forward_gpu_batch(layer_id, h, h_type, ids, routes, joined, rows, stream_obj.cuda_stream)
        for row in range(0 if (batch or multi) else rows):
            rc = self.lib.q38_rtx_peer_forward_gpu(
                int(layer_id), ctypes.c_void_p(h.data_ptr() + row * 2560 * h.element_size()), h_type,
                ctypes.c_void_p(ids.data_ptr() + row * 10 * ids.element_size()),
                ctypes.c_void_p(routes.data_ptr() + row * 10 * routes.element_size()),
                ctypes.c_void_p(joined.data_ptr() + row * 2560 * joined.element_size()),
                ctypes.c_void_p(stream_obj.cuda_stream), error, len(error))
            if rc:
                raise RuntimeError(f"RTX→M10 GPU-driven enqueue layer {layer_id} row {row}: "
                                   f"{error.value.decode(errors='replace')}")
        if not capturing:
            self.check_error()
            self.calls += 1
            self.rows += rows
        return joined.to(dtype=hidden.dtype)

    MOE_BATCH_MAX_ROWS = 64           # per exchange (Q38_BATCH_MAX_ROWS in resident_cpu_mailbox_abi.cuh)
    MOE_BATCH_PIPE_ROWS = 32          # per pipelined chunk (Q38_BATCH_PIPE_ROWS)

    def _configure_moe_batch(self, error) -> None:
        """Map the client's shared CPU mailbox (registered in every worker by the daemon) on the RTX."""
        from qwen38_cpu_mailbox_manifest import TOTAL_BYTES, attach_mailbox
        manifest = os.environ.get("Q38_M10_RTX_PEER_MAILBOX_MANIFEST", "")
        if not manifest:
            raise RuntimeError("Q38_RTX_PEER_MOE_BATCH needs Q38_M10_RTX_PEER_MAILBOX_MANIFEST")
        mailbox, _ = attach_mailbox(Path(manifest))
        view = (ctypes.c_char * TOTAL_BYTES).from_buffer(mailbox.mapping)
        if self.lib.q38_rtx_peer_configure_moe_batch(ctypes.addressof(view), TOTAL_BYTES, error, len(error)):
            del view
            mailbox.mapping.close(); os.close(mailbox.fd)
            raise RuntimeError(f"batched MoE configuration failed: {error.value.decode(errors='replace')}")
        self._batch_mailbox, self._batch_view = mailbox, view

    def _release_moe_batch(self) -> None:
        """After q38_rtx_peer_close (which unregisters the mapping from CUDA)."""
        if self._batch_mailbox is not None:
            self._batch_view = None
            self._batch_mailbox.mapping.close()
            os.close(self._batch_mailbox.fd)
            self._batch_mailbox = None

    def _forward_gpu_batch(self, layer_id: int, h: torch.Tensor, h_type: int, ids: torch.Tensor,
                           routes: torch.Tensor, joined: torch.Tensor, rows: int, stream) -> None:
        error = ctypes.create_string_buffer(1024)
        if self.reuse_dir and not torch.cuda.is_current_stream_capturing():
            os.makedirs(self.reuse_dir, exist_ok=True)
            torch.save(ids[:rows].to("cpu"), os.path.join(self.reuse_dir, f"L{int(layer_id)}_c{self.reuse_files}.pt"))
            self.reuse_files += 1
        if self.moe_batch_pipe and rows >= self.moe_batch_pipe_min_rows:
            rc = self.lib.q38_rtx_peer_moe_batch_pipe_gpu(
                int(layer_id), ctypes.c_void_p(h.data_ptr()), h_type, ctypes.c_void_p(ids.data_ptr()),
                ctypes.c_void_p(routes.data_ptr()), rows, ctypes.c_void_p(joined.data_ptr()), ctypes.c_void_p(stream),
                error, len(error))
            if rc:
                raise RuntimeError(f"pipelined batched MoE enqueue layer {layer_id} rows {rows}: "
                                   f"{error.value.decode(errors='replace')}")
            if not torch.cuda.is_current_stream_capturing():
                self.moe_batch_pipe_calls += 1
                self.moe_batch_calls += (rows + self.MOE_BATCH_PIPE_ROWS - 1) // self.MOE_BATCH_PIPE_ROWS
                self.moe_batch_rows += rows
            return
        for start in range(0, rows, self.MOE_BATCH_MAX_ROWS):
            count = min(self.MOE_BATCH_MAX_ROWS, rows - start)
            rc = self.lib.q38_rtx_peer_moe_batch_gpu(
                int(layer_id), ctypes.c_void_p(h.data_ptr() + start * 2560 * h.element_size()), h_type,
                ctypes.c_void_p(ids.data_ptr() + start * 10 * 4), ctypes.c_void_p(routes.data_ptr() + start * 10 * 4),
                count, ctypes.c_void_p(joined.data_ptr() + start * 2560 * 4), ctypes.c_void_p(stream), error, len(error))
            if rc:
                raise RuntimeError(f"batched MoE enqueue layer {layer_id} rows {start}..{start + count}: "
                                   f"{error.value.decode(errors='replace')}")
            if not torch.cuda.is_current_stream_capturing():
                self.moe_batch_calls += 1
                self.moe_batch_rows += count

    ROW_GATHER_MAX_ROWS = 64          # per exchange (kPciClusterRowGatherMaxRows in the library)
    ROW_BYTES = 160

    def ple_gather_gpu(self, rows: torch.Tensor) -> torch.Tensor:
        """PLE rows by global row id, gathered by the M10 workers that own them (broadcast exchange,
        worker mode 6). Enqueue-only and capturable; returns uint8 [n, 160] on rows.device. Failures
        produce zeros and surface through check_error()."""
        if not self.ple_gpu:
            raise RuntimeError("PLE GPU gather is not enabled (Q38_RTX_PEER_PLE_GPU=1)")
        flat = rows.reshape(-1).to(dtype=torch.int64).contiguous()
        n = int(flat.numel())
        out = torch.empty((n, self.ROW_BYTES), device=flat.device, dtype=torch.uint8)
        stream_obj = torch.cuda.current_stream(flat.device)
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            for tensor in (flat, out):
                tensor.record_stream(stream_obj)
        error = ctypes.create_string_buffer(1024)
        for start in range(0, n, self.ROW_GATHER_MAX_ROWS):
            count = min(self.ROW_GATHER_MAX_ROWS, n - start)
            rc = self.lib.q38_rtx_peer_row_gather_gpu(
                ctypes.c_void_p(flat.data_ptr() + start * 8), count,
                ctypes.c_void_p(out.data_ptr() + start * self.ROW_BYTES),
                ctypes.c_void_p(stream_obj.cuda_stream), error, len(error))
            if rc:
                raise RuntimeError(f"PLE GPU gather enqueue rows {start}..{start + count}: "
                                   f"{error.value.decode(errors='replace')}")
        if not capturing:
            self.ple_gpu_calls += 1
            self.ple_gpu_rows += n
        return out

    def lm_head_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [n, vocab] (BF16) from the M10 LM-head slices: exact top-K candidates per worker,
        -inf elsewhere. Enqueue-only; failures give zero logits and surface through check_error()."""
        if not self.lm_head_m10:
            raise RuntimeError("M10 LM head is not enabled (Q38_M10_LM_HEAD=1)")
        h = x.contiguous()
        n = int(h.shape[0])
        out = torch.empty((n, self.lm_vocab), device=h.device, dtype=torch.bfloat16)
        lse = torch.empty((n,), device=h.device, dtype=torch.float32)
        stream_obj = torch.cuda.current_stream(h.device)
        if not torch.cuda.is_current_stream_capturing():
            for tensor in (h, out, lse):
                tensor.record_stream(stream_obj)
        h_type = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[h.dtype]
        error = ctypes.create_string_buffer(1024)
        # Q38_M10_LM_HEAD_ROWS2=1 (2026-11-04): 2..4 BF16 rows (MTP verify) in ONE LM-head exchange (worker mode 7 with
        # N rows: one slice read, per row the same top-K and log-sum-exp). The record holds N(K+1) <= 2560 - N words,
        # so K is capped at (2560 - N) // N - 1 for N rows (638 for 4; still far above any sampling top-k).
        if (2 <= n <= 4 and h_type == 2 and os.environ.get("Q38_M10_LM_HEAD_ROWS2") == "1"):
            k_rows = min(self.lm_topk, (2560 - n) // n - 1)
            rc = self.lib.q38_rtx_peer_lm_head_topk_rows(
                ctypes.c_void_p(h.data_ptr()), h_type, ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(lse.data_ptr()),
                k_rows, n, None, ctypes.c_void_p(stream_obj.cuda_stream), error, len(error))
            if rc:
                raise RuntimeError(f"M10 LM-head {n}-row enqueue: {error.value.decode(errors='replace')}")
            self.lm_calls += 1
            self.last_lm_lse = lse
            return out
        for row in range(n):
            rc = self.lib.q38_rtx_peer_lm_head_topk(
                ctypes.c_void_p(h.data_ptr() + row * h.shape[1] * h.element_size()), h_type,
                ctypes.c_void_p(out.data_ptr() + row * self.lm_vocab * 2), ctypes.c_void_p(lse.data_ptr() + row * 4),
                self.lm_topk, None, ctypes.c_void_p(stream_obj.cuda_stream), error, len(error))
            if rc:
                raise RuntimeError(f"M10 LM-head enqueue row {row}: {error.value.decode(errors='replace')}")
        self.lm_calls += 1
        self.last_lm_lse = lse
        return out

    def timing_enable(self, capacity: int) -> None:
        """Per-exchange %globaltimer records (device ring); call with the device idle."""
        error = ctypes.create_string_buffer(512)
        if self.lib.q38_rtx_peer_timing_enable(int(capacity), error, len(error)):
            raise RuntimeError(f"timing enable: {error.value.decode(errors='replace')}")

    def timing_read(self, max_records: int) -> list[dict]:
        """Synchronize, return the newest records in order (layer: 0-47 MoE, 256 PLE row gather,
        512 LM head), disable timing. Times in ns (GPU %globaltimer)."""
        import numpy as np
        dt = np.dtype([("layer", "<u4"), ("seq", "<u4"), ("skip", "<u4"), ("pad", "<u4"), ("t_begin", "<u8"),
                       ("t_pub", "<u8"), ("t_seq", "<u8"), ("t_ack", "<u8"), ("t_done", "<u8")])
        buf = np.zeros(int(max_records), dtype=dt)
        n = ctypes.c_uint32(0); error = ctypes.create_string_buffer(512)
        if self.lib.q38_rtx_peer_timing_read(ctypes.c_void_p(buf.ctypes.data), int(max_records), ctypes.byref(n), error, len(error)):
            raise RuntimeError(f"timing read: {error.value.decode(errors='replace')}")
        return [{k: int(r[k]) for k in dt.names} for r in buf[: n.value]]   # pad: ack profile (Q38_RTX_PEER_ACK_PROFILE)

    def worker_timing_enable(self) -> None:
        """M10-side MoE stamps (pci_cluster_exchange.cuh ring); needs the row-gather configuration."""
        error = ctypes.create_string_buffer(512)
        if self.lib.q38_rtx_peer_worker_timing_enable(error, len(error)):
            raise RuntimeError(f"worker timing enable: {error.value.decode(errors='replace')}")

    def worker_timing_read(self) -> list[dict]:
        """Disable and return every worker's records (M10 %globaltimer ns; worker = kind >> 8)."""
        import numpy as np
        dt = np.dtype([("seq", "<u4"), ("layer", "<u4"), ("role", "<u4"), ("kind", "<u4"), ("t_seen", "<u8"),
                       ("t_checked", "<u8"), ("t_gateup", "<u8"), ("t_down", "<u8"), ("t_done", "<u8"), ("t_extra", "<u8")])
        cap = 24 * 512
        buf = np.zeros(cap, dtype=dt); n = ctypes.c_uint32(0); error = ctypes.create_string_buffer(512)
        if self.lib.q38_rtx_peer_worker_timing_read(ctypes.c_void_p(buf.ctypes.data), cap, ctypes.byref(n), error, len(error)):
            raise RuntimeError(f"worker timing read: {error.value.decode(errors='replace')}")
        out = []
        for r in buf[: n.value]:
            d = {k: int(r[k]) for k in dt.names}
            d["worker"] = d["kind"] >> 8; d["kind"] = d["kind"] & 255
            out.append(d)
        return out

    def check_error(self) -> None:
        error = ctypes.create_string_buffer(512)
        if self.lib.q38_rtx_peer_check_error(error, len(error)):
            raise RuntimeError(error.value.decode(errors="replace"))

    def gpu_resync(self) -> None:
        error = ctypes.create_string_buffer(512)
        if self.lib.q38_rtx_peer_gpu_resync(error, len(error)):
            raise RuntimeError(f"GPU sequence resync failed: {error.value.decode(errors='replace')}")

    def gpu_counters(self) -> dict:
        calls, skips = ctypes.c_uint32(0), ctypes.c_uint32(0)
        if self.lib.q38_rtx_peer_gpu_counters(ctypes.byref(calls), ctypes.byref(skips)):
            return {}
        return {"gpu_calls": calls.value, "gpu_dummy_skips": skips.value}

    def mappings_quiesced(self) -> bool:
        return bool(self.lib.q38_rtx_peer_mappings_quiesced())

    def close(self) -> None:
        if self._closed:
            return
        self.flush_async()
        if self.gpu_driven:
            # GPU-driven exchanges complete on their own (reply, timeout, deadline or abort); wait for
            # every queued one before parking the M10s and restoring routes.
            torch.cuda.synchronize()
        status = self._send_control(Path(self.control_socket), "STATUS", timeout=30)
        if status.get("status") != "READY":
            raise RuntimeError(f"resident daemon is not READY during peer teardown: {status}")
        if status.get("worker_activity", "ACTIVE") != "PARKED":
            parked = self._send_control(Path(self.control_socket), "PARK_WORKERS",
                                        timeout=120, clients_quiesced=True)
            if parked.get("status") != "ok" or parked.get("code") != "WORKERS_PARKED":
                raise RuntimeError(f"could not quiesce M10 writers before peer teardown: {parked}")
        gpu_counters_at_close = self.gpu_counters() if self.gpu_driven else {}
        error = ctypes.create_string_buffer(1024)
        rc = self.lib.q38_rtx_peer_mark_m10_quiesced(error, len(error))
        if rc:
            raise RuntimeError(f"could not mark M10 return routes quiesced: {error.value.decode(errors='replace')}")
        rc = self.lib.q38_rtx_peer_close(error, len(error))
        if rc:
            raise RuntimeError(f"RTX peer route teardown failed: {error.value.decode(errors='replace')}")
        self._release_moe_batch()
        unreg = self._send_control(Path(self.control_socket), "UNREGISTER_PEER_CLIENT", timeout=30,
                                   run_id=self.peer_run_id)
        if unreg.get("status") != "ok":
            raise RuntimeError(f"daemon did not accept peer-client unregistration after a clean close: {unreg}")
        if self.ple_adapter is not None:
            self.ple_adapter.close()
        if self._workers_were_active:
            awake = self._send_control(Path(self.control_socket), "WAKE_WORKERS",
                                       timeout=120, clients_quiesced=True)
            if awake.get("status") != "ok" or awake.get("code") != "WORKERS_ACTIVE":
                raise RuntimeError(f"M10 workers did not return to their initial active state: {awake}")
        self._closed = True
        self._gpu_counters_at_close = gpu_counters_at_close
        summary_base = os.environ.get("Q38_M10_RTX_PEER_TIMING_JSON")
        if summary_base:
            path = Path(summary_base).with_name(
                Path(summary_base).name + f".{os.getpid()}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(self.timing_snapshot(), indent=2) + "\n")
            temp.replace(path)
