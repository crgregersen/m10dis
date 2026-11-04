#!/usr/bin/env python3
"""Pack the MTP layer's 512 routed experts and shared expert as resident layer 48 (2026-11-04).

Same formats as the 48 model layers (prepare_resident_expert_layer.py + qwen38_m10_expert_sim.prepare_shared),
read from the checkpoint's mtp.layers.0.mlp.* tensors. Writes <bundles>/layer_48/{routed,shared}; the daemon's
--persistent-bundle-root then loads it like any other layer (Q38_DAEMON_LAYERS=49). Refuses to overwrite.
CPU only. Queue: pueue add --group default --label <label> -- python3 <this> [--bundles DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_m10_expert_sim as qwen
from qwen38_resident_daemon import valid_expert_bundle, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("M10DIS_MODEL", "/mnt/nvme/models/Qwen3.8-Flash-Next-FP8")))
    ap.add_argument("--bundles", type=Path, default=Path(os.environ.get("M10DIS_DATA", "/mnt/nvme")) / "qwen38-stage9-preserved-bundles")
    args = ap.parse_args()
    layer = qwen.MTP_LAYER
    root = args.bundles.resolve() / f"layer_{layer:02d}"
    if root.exists():
        raise SystemExit(f"ERROR,{root} exists; refusing to overwrite")
    route_dir, shared_dir = root / "routed", root / "shared"
    route_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_resident_expert_layer.py")),
                    "--model", str(args.model.resolve()), "--layer", str(layer), "--out", str(route_dir)], check=True)
    qwen.MODEL = args.model.resolve()
    index = args.model.resolve() / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    readers: dict = {}
    try:
        qwen.prepare_shared(layer, np.zeros(qwen.HIDDEN, dtype=np.float32), weight_map, readers, shared_dir)
    finally:
        for r in readers.values():
            r.close()
    ok = valid_expert_bundle(root, layer, sha256_file(index))
    print(f"MTP_BUNDLE,{root},valid,{int(ok)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
