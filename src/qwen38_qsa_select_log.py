"""Log the QSA indexer's decode selections (opt-in: Q38_QSA_SELECT_LOG=FILE; measurement only).

Wraps Qwen4ExpQSAAttention._run_qsa. After each decode call (at most Q38_QSA_SELECT_MAX_ROWS rows) it appends one
record per row: int32 layer, position, count, then count int32 request-relative token indices, little endian.
With Q38_QSA_UNION_LOG=CSV it also records, per prefill call, the distinct blocks selected across all rows.
Needs eager decode: inside a CUDA graph the wrapper would not run, so install() refuses Q38_VLLM_CUDAGRAPH=1.
Reader: analyze_qsa_selection.py.
"""
from __future__ import annotations

import atexit
import os
import re
import struct
import sys


def install() -> dict | None:
    path = os.environ.get("Q38_QSA_SELECT_LOG")
    if not path:
        return None
    if os.environ.get("Q38_VLLM_CUDAGRAPH") == "1":
        raise RuntimeError("Q38_QSA_SELECT_LOG needs eager decode (Q38_VLLM_CUDAGRAPH=0)")
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention as cls

    max_rows = int(os.environ.get("Q38_QSA_SELECT_MAX_ROWS", "4"))
    # One file per process: the client parent also installs the hook, and two writers corrupt a shared file.
    out = open(f"{path}.{os.getpid()}", "wb", buffering=1 << 20)
    atexit.register(out.flush)
    union_path = os.environ.get("Q38_QSA_UNION_LOG")
    union = open(f"{union_path}.{os.getpid()}", "w", buffering=1) if union_path else None
    if union:
        union.write("layer,pos_first,pos_last,rows,union_blocks\n")
    original = cls._run_qsa
    state = {"records": 0}

    def logged(self, projected_qk, positions, *args, **kw):
        result = original(self, projected_qk, positions, *args, **kw)
        rows = projected_qk.shape[0]
        if rows <= max_rows:
            layer = int(re.search(r"layers\.(\d+)", self.layer_name).group(1))
            pos = positions.reshape(-1, rows)[0].tolist()
            sel = self.topk_indices_buffer[:rows].cpu()
            for r in range(rows):
                n = int(sel[r, -1])
                out.write(struct.pack("<3i", layer, pos[r], n))
                out.write(sel[r, :n].numpy().astype("<i4").tobytes())
                state["records"] += 1
            if state["records"] % 240 == 0:
                out.flush()
        elif union is not None:
            # Prefill call: distinct indexer blocks selected across all rows of this call.
            layer = int(re.search(r"layers\.(\d+)", self.layer_name).group(1))
            pos = positions.reshape(-1, rows)[0]
            sel = self.topk_indices_buffer[:rows, :-1]
            blocks = (sel[sel >= 0] // 4).unique().numel()
            union.write(f"{layer},{int(pos[0])},{int(pos[-1])},{rows},{int(blocks)}\n")
            union.flush()
        return result

    cls._run_qsa = logged
    print(f"Q38_QSA_SELECT_LOG,installed,{path},max_rows,{max_rows}", file=sys.stderr, flush=True)
    return {"path": path, "max_rows": max_rows}
