"""vLLM MoE boundary adapter using direct PCIe multicast/BAR1 mailboxes.

The resident service owns weights, CUDA contexts, RM mappings and persistent
kernels. This caller maps the PEX apertures and owner BAR1 result mailboxes
once, then sends every row straight to the resident workers. No per-request
Unix socket, pipe, filesystem request, or host shared-memory relay is used.
The first prototype requires privileges to open the WC mapper and PCI resource
files; a later production handoff can pass pre-opened mapping FDs at startup.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import mmap
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


class DirectResidentM10MoEAdapter:
    def __init__(self, library: Path | str, groups_tsv: Path | str,
                 gpu_mappings_manifest: Path | str, ple_manifest: Path | str | None = None):
        self.library_path = str(Path(library).resolve())
        self.groups_path = str(Path(groups_tsv).resolve())
        self.gpu_mappings_manifest = str(Path(gpu_mappings_manifest).resolve())
        self.lib = ctypes.CDLL(self.library_path)
        self.lib.q38_direct_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_direct_open.restype = ctypes.c_int
        self.lib.q38_direct_configure_ple.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                                       ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_direct_configure_ple.restype = ctypes.c_int
        self.lib.q38_direct_ple_lookup.argtypes = [ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_direct_ple_lookup.restype = ctypes.c_int
        self.lib.q38_direct_ple_lookup_diagnostic.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.lib.q38_direct_ple_lookup_diagnostic.restype = ctypes.c_int
        self.lib.q38_direct_run.argtypes = [ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_direct_run.restype = ctypes.c_int
        self.lib.q38_direct_set_cpu_mailbox.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                                        ctypes.c_char_p, ctypes.c_size_t]
        self.lib.q38_direct_set_cpu_mailbox.restype = ctypes.c_int
        self.lib.q38_direct_close.argtypes = []
        self.lib.q38_direct_close.restype = None
        self._closed = False
        self.calls = 0
        self.rows = 0
        self.timings: list[dict[str, float]] = []
        self.ple_calls = 0
        self.ple_rows = 0
        self.ple_elapsed_seconds = 0.0
        self.ple_timings: list[dict[str, float]] = []
        self.supports_direct_ple = ple_manifest is not None
        summary_base = os.environ.get("Q38_M10_DIRECT_TIMING_JSON")
        self.summary_path = (str(Path(summary_base).with_name(
            Path(summary_base).name + f".{os.getpid()}.json")) if summary_base else None)
        error = ctypes.create_string_buffer(1024)
        rc = self.lib.q38_direct_open(self.groups_path.encode(), self.gpu_mappings_manifest.encode(),
                                      error, len(error))
        if rc:
            raise RuntimeError(f"direct M10 mailbox initialization failed: {error.value.decode(errors='replace')}")
        self._cpu_mailbox_fd = None
        self._cpu_mailbox_mapping = None
        self._cpu_mailbox_buffer = None
        self.cpu_mailbox_manifest = os.environ.get("Q38_M10_DIRECT_CPU_MAILBOX_MANIFEST")
        self.uses_cpu_mailbox = False
        if self.cpu_mailbox_manifest:
            try:
                from qwen38_cpu_mailbox_manifest import validate_manifest
                info = validate_manifest(Path(self.cpu_mailbox_manifest))
                self._cpu_mailbox_fd = os.open("/dev/shm" + info["shm_name"], os.O_RDWR)
                self._cpu_mailbox_mapping = mmap.mmap(
                    self._cpu_mailbox_fd, info["bytes"], access=mmap.ACCESS_WRITE)
                self._cpu_mailbox_buffer = (ctypes.c_char * info["bytes"]).from_buffer(
                    self._cpu_mailbox_mapping)
                rc = self.lib.q38_direct_set_cpu_mailbox(
                    ctypes.addressof(self._cpu_mailbox_buffer), info["bytes"], error, len(error))
                if rc:
                    raise RuntimeError(error.value.decode(errors="replace"))
                self.uses_cpu_mailbox = True
            except BaseException:
                self.lib.q38_direct_close()
                self._close_cpu_mailbox_mapping()
                raise
        if ple_manifest is not None:
            manifest = Path(ple_manifest).resolve()
            if not manifest.is_file():
                self.lib.q38_direct_close()
                raise FileNotFoundError(f"resident PLE manifest is missing: {manifest}")
            rc = self.lib.q38_direct_configure_ple(self.groups_path.encode(),
                                                    str(manifest).encode(), error, len(error))
            if rc:
                self.lib.q38_direct_close()
                raise RuntimeError(f"direct PLE manifest setup failed: {error.value.decode(errors='replace')}")
        atexit.register(self.close)

    @torch.inference_mode()
    def forward(self, layer_id: int, hidden: torch.Tensor,
                expert_ids: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("direct M10 adapter is closed")
        if hidden.ndim != 2 or tuple(hidden.shape[1:]) != (2560,):
            raise ValueError(f"hidden must have shape [N,2560], got {tuple(hidden.shape)}")
        rows = int(hidden.shape[0])
        if expert_ids.shape != (rows, 10) or route_weights.shape != (rows, 10):
            raise ValueError("expert_ids and route_weights must both have shape [N,10]")
        if hidden.device.type != "cuda":
            raise ValueError("vLLM adapter input must be a CUDA tensor")
        if rows == 0:
            return torch.empty_like(hidden)

        t0 = time.perf_counter()
        h = hidden.detach().to(torch.float32).cpu().contiguous().numpy()
        ids = expert_ids.detach().to(torch.int32).cpu().contiguous().numpy()
        routes = route_weights.detach().to(torch.float32).cpu().contiguous().numpy()
        d2h = time.perf_counter() - t0
        if not np.isfinite(h).all() or not np.isfinite(routes).all():
            raise ValueError("MoE request contains non-finite H or route values")
        if np.any(ids < 0) or np.any(ids >= 512) or np.any(routes < 0):
            raise ValueError("MoE request contains out-of-range IDs or negative routes")
        if np.any(np.abs(routes.sum(axis=1, dtype=np.float32) - 1.0) > 1e-5):
            raise ValueError("route weights must sum to one per row")
        if any(len(np.unique(row)) != 10 for row in ids):
            raise ValueError("each row must contain ten distinct expert IDs")

        output = np.empty((rows, 2560), dtype=np.float32)
        native_rows: list[dict[str, float]] = []
        for row in range(rows):
            pub = ctypes.c_double(); ack = ctypes.c_double()
            gather = ctypes.c_double(); join = ctypes.c_double()
            error = ctypes.create_string_buffer(1024)
            rc = self.lib.q38_direct_run(int(layer_id),
                h[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ids[row].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                routes[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output[row].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.byref(pub), ctypes.byref(ack), ctypes.byref(gather),
                ctypes.byref(join), error, len(error))
            if rc:
                raise RuntimeError(f"direct M10 request row {row} failed: {error.value.decode(errors='replace')}")
            native_rows.append({"publish_us": pub.value, "ack_us": ack.value,
                                "bar1_gather_us": gather.value, "cpu_join_us": join.value})
        t1 = time.perf_counter()
        result = torch.from_numpy(output).to(device=hidden.device, dtype=hidden.dtype)
        t2 = time.perf_counter()
        self.calls += 1; self.rows += rows
        self.timings.append({"rows": float(rows), "d2h_seconds": d2h,
            "native_seconds": sum(x["publish_us"]+x["ack_us"]+
                                  x["bar1_gather_us"]+x["cpu_join_us"]
                                  for x in native_rows)*1e-6,
            "h2d_seconds": t2-t1, "forward_seconds": t2-t0,
            "native_rows": native_rows})
        return result

    def close(self) -> None:
        if not self._closed:
            snapshot = self.timing_snapshot()
            self.lib.q38_direct_close()
            self._closed = True
            self._close_cpu_mailbox_mapping()
            if self.summary_path:
                path = Path(self.summary_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + f".{os.getpid()}.tmp")
                temp.write_text(json.dumps(snapshot, indent=2) + "\n")
                temp.replace(path)

    def _close_cpu_mailbox_mapping(self) -> None:
        self._cpu_mailbox_buffer = None
        if self._cpu_mailbox_mapping is not None:
            self._cpu_mailbox_mapping.close()
            self._cpu_mailbox_mapping = None
        if self._cpu_mailbox_fd is not None:
            os.close(self._cpu_mailbox_fd)
            self._cpu_mailbox_fd = None

    def lookup_ple(self, global_rows: list[int]) -> bytes:
        if self._closed:
            raise RuntimeError("direct M10 adapter is closed")
        if not self.supports_direct_ple:
            raise RuntimeError("this direct adapter was opened without a resident PLE manifest")
        rows = np.ascontiguousarray(global_rows, dtype=np.uint64)
        if len(rows) > 65536:
            raise ValueError(f"PLE lookup exceeds the 65536-row bound: {len(rows)}")
        output = np.empty((len(rows), 160), dtype=np.uint8)
        elapsed = ctypes.c_double()
        phases = (ctypes.c_double * 9)()
        counters = (ctypes.c_uint64 * 3)()
        error = ctypes.create_string_buffer(1024)
        row_ptr = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)) if len(rows) else None
        out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if len(rows) else None
        rc = self.lib.q38_direct_ple_lookup_diagnostic(row_ptr, len(rows), out_ptr,
            ctypes.byref(elapsed), phases, counters, error, len(error))
        if rc:
            raise RuntimeError(f"direct BAR1 PLE lookup failed: {error.value.decode(errors='replace')}")
        self.ple_calls += 1
        self.ple_rows += len(rows)
        self.ple_elapsed_seconds += elapsed.value * 1e-6
        phase_names = ("resolve", "sequence_check", "command_store", "command_fence",
                       "sequence_publish", "ack_wait", "result_status_check", "result_copy",
                       "native_total")
        timing = {"rows": float(len(rows)), "seconds": elapsed.value * 1e-6,
                  **{name + "_us": phases[i] for i, name in enumerate(phase_names)},
                  "batches": int(counters[0]), "poll_passes": int(counters[1]),
                  "poll_reads": int(counters[2])}
        self.ple_timings.append(timing)
        return output.tobytes()

    def timing_snapshot(self) -> dict[str, Any]:
        return {"calls": self.calls, "rows": self.rows, "forward_timings": list(self.timings),
                "ple_calls": self.ple_calls, "ple_rows": self.ple_rows,
                "ple_elapsed_seconds": self.ple_elapsed_seconds,
                "ple_timings": list(self.ple_timings),
                "transport": ("direct CPU stores through PEX WC windows; results via shared CPU mailbox"
                              if self.uses_cpu_mailbox else
                              "direct CPU stores through PEX WC windows; results via PCI resource BAR1 mappings"),
                "cpu_mailbox_manifest": self.cpu_mailbox_manifest}
