#!/usr/bin/env python3
"""Pack all 512 experts of one Qwen layer into persistent GM107 slice files."""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_m10_expert_sim as qwen


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("M10DIS_MODEL", "/mnt/nvme/models/Qwen3.8-Flash-Next-FP8")))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 0 <= args.layer <= qwen.MTP_LAYER:
        raise ValueError("layer must be in 0..48 (48 = the MTP layer)")
    model = args.model.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    qwen.MODEL = model
    index_path = model / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    prefix = qwen.layer_prefix(args.layer)
    workers = qwen.WORKERS
    width = qwen.SLICE
    hidden = qwen.HIDDEN
    experts = 512
    groups = hidden // 128
    started = time.monotonic()
    readers: dict[str, qwen.TensorReader] = {}
    bundles = []
    try:
        for shard in range(workers):
            base = out / f"gpu_{shard:02d}"
            base.mkdir(exist_ok=True)
            arrays = {
                "gate": np.memmap(base / "gate.u8", mode="w+", dtype=np.uint8,
                                  shape=(experts, width, hidden)),
                "up": np.memmap(base / "up.u8", mode="w+", dtype=np.uint8,
                                shape=(experts, width, hidden)),
                "down": np.memmap(base / "down.u8", mode="w+", dtype=np.uint8,
                                  shape=(experts, width, hidden)),
                "gate_scale": np.memmap(base / "gate_scale.f32", mode="w+", dtype=np.float32,
                                        shape=(experts, width, groups)),
                "up_scale": np.memmap(base / "up_scale.f32", mode="w+", dtype=np.float32,
                                      shape=(experts, width, groups)),
                "down_scale": np.memmap(base / "down_scale.f32", mode="w+", dtype=np.float32,
                                        shape=(experts, hidden)),
            }
            bundles.append(arrays)

        for expert in range(experts):
            values = {}
            for projection in qwen.PROJECTIONS:
                tensor = prefix + f"experts.{expert}.{projection}"
                values[projection] = (
                    qwen.load_tensor(tensor + ".weight", weight_map, readers),
                    qwen.load_tensor(tensor + ".weight_scale_inv", weight_map, readers))
            g, gs = values["gate_proj"]
            u, us = values["up_proj"]
            d, ds = values["down_proj"]
            if g.shape != (qwen.INTERMEDIATE, hidden) or u.shape != g.shape or d.shape != (hidden, qwen.INTERMEDIATE):
                raise ValueError(f"unexpected matrix shape at layer {args.layer}, expert {expert}")
            if gs.shape != (5, groups) or us.shape != gs.shape or ds.shape != (groups, 5):
                raise ValueError(f"unexpected scale shape at layer {args.layer}, expert {expert}")
            for shard, arrays in enumerate(bundles):
                first = shard * width
                last = first + width
                arrays["gate"][expert] = g[first:last]
                arrays["up"][expert] = u[first:last]
                arrays["down"][expert] = d[:, first:last].T
                for row in range(width):
                    scale_row = (first + row) // 128
                    arrays["gate_scale"][expert, row] = gs[scale_row]
                    arrays["up_scale"][expert, row] = us[scale_row]
                arrays["down_scale"][expert] = ds[np.arange(hidden) // 128, first // 128]
            if expert % 64 == 63:
                print(f"PACKED_EXPERTS,{expert + 1}/512", flush=True)

        worker_records = []
        for shard, arrays in enumerate(bundles):
            for array in arrays.values():
                array.flush()
            files = []
            for filename in ("gate.u8", "up.u8", "down.u8", "gate_scale.f32", "up_scale.f32", "down_scale.f32"):
                path = out / f"gpu_{shard:02d}" / filename
                files.append({"path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256(path)})
            worker_records.append({"shard": shard, "slice_start": shard * width,
                                   "slice_width": width, "files": files,
                                   "payload_bytes": sum(f["bytes"] for f in files)})
        manifest = {
            "status": "PACKED_LAYER_RESIDENT_EXPERTS", "layer": args.layer,
            "expert_count": experts, "format": "official E4M3 FP8 + BF16 inverse scales expanded exactly to FP32",
            "workers": workers, "slice_width": width, "hidden": hidden,
            "total_payload_bytes": sum(w["payload_bytes"] for w in worker_records),
            "elapsed_seconds": time.monotonic() - started,
            "model_index_sha256": sha256(index_path), "workers_data": worker_records,
        }
        (out / "resident_layer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"PASS_LAYER_PACKED,{args.layer},bytes,{manifest['total_payload_bytes']},seconds,{manifest['elapsed_seconds']:.3f}", flush=True)
    finally:
        for arrays in bundles:
            for array in arrays.values():
                array.flush()
                array._mmap.close()
        for reader in readers.values():
            reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
