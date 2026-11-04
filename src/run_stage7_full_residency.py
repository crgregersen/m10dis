#!/usr/bin/env python3
"""Stream-pack, load, and validate all Qwen3.8 routed/shared layers on six M10s.

One routed layer bundle is staged at a time. The CUDA service retains loaded
weights; staging is deleted only after both routed and shared LOAD commands
have completed. Every layer is exercised against the CPU reference before the
next bundle is packed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_m10_expert_sim as q
from qwen38_resident_service_manager import ResidentServiceManager


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def free_mib() -> list[int]:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    return [int(x.strip()) for x in rows]


def check_array(got: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    err = np.abs(got - expected)
    badmask = ~np.isfinite(got) | (err > atol + rtol * np.abs(expected))
    return {"bad": int(np.count_nonzero(badmask)),
            "max_abs": float(err.max(initial=0.0)),
            "expected_sha256": hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest(),
            "observed_sha256": hashlib.sha256(np.ascontiguousarray(got).tobytes()).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("M10DIS_MODEL", "/mnt/nvme/models/Qwen3.8-Flash-Next-FP8")))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--routed-service", type=Path, required=True)
    ap.add_argument("--shared-service", type=Path, required=True)
    ap.add_argument("--bundle-root", type=Path,
                    help="reuse preserved layer_XX/routed and layer_XX/shared bundles")
    ap.add_argument("--first-layer", type=int, default=0)
    ap.add_argument("--last-layer", type=int, default=47, help="inclusive")
    ap.add_argument("--min-free-mib", type=int, default=512)
    ap.add_argument("--min-disk-free-gib", type=float, default=4.0,
                    help="stop before packing a layer if output filesystem has less free space")
    ap.add_argument("--keep-staging", action="store_true")
    a = ap.parse_args()
    if not (0 <= a.first_layer <= a.last_layer < 48):
        ap.error("layer range must be within 0..47")
    model, out = a.model.resolve(), a.out.resolve()
    bundle_root = a.bundle_root.resolve() if a.bundle_root else None
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "staging"
    stage.mkdir(exist_ok=True)
    q.MODEL = model
    index = model / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"]
    model_index_hash = sha256(index)
    rng = np.random.default_rng(0x038)  # fixed run-independent validation seed
    lut = q.fp8_lut()
    route_bin = a.routed_service.resolve()
    shared_bin = a.shared_service.resolve()
    manager = ResidentServiceManager(route_bin, shared_bin, out / "service_logs")
    layers: list[dict] = []
    readers: dict[str, q.TensorReader] = {}
    overall_start = time.monotonic()
    ok = True
    try:
        for layer in range(a.first_layer, a.last_layer + 1):
            disk_free = shutil.disk_usage(out).free
            if disk_free < a.min_disk_free_gib * (1 << 30):
                raise RuntimeError(
                    f"disk-space guard before layer {layer}: "
                    f"{disk_free / (1 << 30):.2f} GiB free, "
                    f"need at least {a.min_disk_free_gib:.2f} GiB")
            t0 = time.monotonic()
            if bundle_root:
                layer_bundle = bundle_root / f"layer_{layer:02d}"
                route_dir = layer_bundle / "routed"
                shared_dir = layer_bundle / "shared"
                if not (route_dir / "resident_layer_manifest.json").is_file():
                    raise FileNotFoundError(f"missing preserved route bundle for layer {layer}: {route_dir}")
                if not (shared_dir / "shared_manifest.json").is_file():
                    raise FileNotFoundError(f"missing preserved shared bundle for layer {layer}: {shared_dir}")
                shared_bundle_manifest = json.loads((shared_dir / "shared_manifest.json").read_text())
                if shared_bundle_manifest.get("layer") != layer:
                    raise RuntimeError(f"shared bundle manifest mismatch at layer {layer}")
            else:
                route_dir = stage / f"route_{layer:02d}"
                shared_dir = stage / f"shared_{layer:02d}"
                route_dir.mkdir(parents=True, exist_ok=True)
                shared_dir.mkdir(parents=True, exist_ok=True)
            ref_dir = out / "validation" / f"layer_{layer:02d}"
            ref_dir.mkdir(parents=True, exist_ok=True)
            pack_log = out / "logs" / f"pack_{layer:02d}.log"
            pack_log.parent.mkdir(exist_ok=True)

            # Route weights from the real checkpoint router; H is a reproducible
            # synthetic activation because no preceding RTX layer is wired yet.
            x = rng.standard_normal(q.HIDDEN, dtype=np.float32) * np.float32(0.2)
            ids, probs, logits, _, route_source = q.selected_routes(layer, x, weight_map, readers)
            req = ref_dir / "request.bin"
            with req.open("wb") as f:
                f.write(struct.pack("<I", layer))
                f.write(np.asarray(x, dtype="<f4").tobytes())
                f.write(np.asarray(ids, dtype="<i4").tobytes())
                f.write(np.asarray(probs, dtype="<f4").tobytes())

            if not bundle_root:
                pack_cmd = [sys.executable, str(Path(__file__).with_name("prepare_resident_expert_layer.py")),
                            "--model", str(model), "--layer", str(layer), "--out", str(route_dir)]
                with pack_log.open("w") as log:
                    p = subprocess.run(pack_cmd, stdout=log, stderr=subprocess.STDOUT)
                if p.returncode:
                    raise RuntimeError(f"route packing failed at layer {layer}; see {pack_log}")
                manifest = json.loads((route_dir / "resident_layer_manifest.json").read_text())
            else:
                manifest = json.loads((route_dir / "resident_layer_manifest.json").read_text())
            if manifest.get("layer") != layer or manifest.get("model_index_sha256") != model_index_hash:
                raise RuntimeError(f"route manifest mismatch at layer {layer}")

            shared_reference_dir = (ref_dir / "shared_refs") if bundle_root else shared_dir
            shared_expected, shared_gate, shared_manifest = q.prepare_shared(
                layer, x, weight_map, readers, shared_reference_dir)
            route_expected, route_manifest = q.prepare_layer(
                layer, x, ids, probs, weight_map, readers, ref_dir / "route_refs", lut)

            load_start = time.monotonic()
            future = manager.load_layer_async(layer, route_dir, shared_dir)
            load_response = future.result(timeout=300)
            load_seconds = time.monotonic() - load_start
            if load_response.get("state") != "READY":
                raise RuntimeError(f"layer {layer} load failed: {load_response}")

            # Store a full request artifact for exact replay of this layer's gate.
            response_dir = out / "validation" / f"layer_{layer:02d}"
            routed_out, routed_parts = response_dir / "routed.f32", response_dir / "routed_parts.f32"
            shared_out, shared_parts = response_dir / "shared.f32", response_dir / "shared_parts.f32"
            response = manager.infer(f"stage7-{layer}", layer, req,
                                     routed_out, routed_parts, shared_out, shared_parts)
            if response.get("status") != "ok":
                raise RuntimeError(f"layer {layer} inference rejected: {response}")
            actual_route = np.fromfile(routed_out, dtype=np.float32)
            actual_shared = np.fromfile(shared_out, dtype=np.float32)
            actual_parts = np.fromfile(routed_parts, dtype=np.float32).reshape(20, q.HIDDEN)
            actual_shared_parts = np.fromfile(shared_parts, dtype=np.float32).reshape(4, q.HIDDEN)
            checks = {
                "routed_combined": check_array(actual_route, route_expected, 2e-7, 1e-3),
                "shared_combined": check_array(actual_shared, shared_expected, 2e-5, 1e-3),
                "combined": check_array(actual_route + actual_shared,
                                         route_expected + shared_expected, 3e-5, 1e-3),
                "routed_slices": [], "shared_slices": []}
            for g in range(20):
                ref = np.fromfile(ref_dir / "route_refs" / f"official_fp8.{g}.ref.bin", dtype=np.float32)
                checks["routed_slices"].append(check_array(actual_parts[g], ref, 2e-7, 1e-3))
            for g in range(4):
                ref = np.fromfile(shared_reference_dir / f"shared.{g}.ref.bin", dtype=np.float32)
                checks["shared_slices"].append(check_array(actual_shared_parts[g], ref, 2e-5, 1e-3))
            checks["routed_bad_total"] = checks["routed_combined"]["bad"] + sum(x["bad"] for x in checks["routed_slices"])
            checks["shared_bad_total"] = checks["shared_combined"]["bad"] + sum(x["bad"] for x in checks["shared_slices"])
            checks["combined_bad"] = checks["combined"]["bad"]
            layer_ok = checks["routed_bad_total"] == checks["shared_bad_total"] == checks["combined_bad"] == 0
            ok &= layer_ok
            free = free_mib()
            if min(free[:24]) < a.min_free_mib:
                raise RuntimeError(f"free VRAM guard tripped after layer {layer}: min={min(free[:24])} MiB")

            rec = {"layer": layer, "status": "PASS" if layer_ok else "FAIL",
                   "route_source": route_source, "expert_ids": ids.tolist(),
                   "route_weights": probs.tolist(), "router_logit_checksum": float(logits.sum()),
                   "routed_bundle_bytes": manifest["total_payload_bytes"],
                   "pack_seconds": manifest["elapsed_seconds"],
                   "resident_load_seconds": load_seconds,
                   "service_load_response": load_response,
                   "shared_gate": shared_gate, "shared_manifest": shared_manifest,
                   "route_manifest": route_manifest, "checks": checks,
                   "min_free_mib_after_load": min(free[:24]),
                   "elapsed_seconds": time.monotonic() - t0}
            layers.append(rec)
            (out / "stage7_progress.json").write_text(json.dumps({
                "status": "RUNNING", "completed_layers": len(layers),
                "last_layer": layer, "layers": layers}, indent=2) + "\n")
            print(f"STAGE7_LAYER,{layer},status,{rec['status']},pack_s,{rec['pack_seconds']:.2f},load_s,{load_seconds:.2f},min_free_mib,{min(free[:24])}", flush=True)

            # Service has synchronously consumed both files. It now owns device
            # copies, so bounded staging can be removed before next layer.
            if not a.keep_staging:
                if not bundle_root:
                    shutil.rmtree(route_dir)
                    shutil.rmtree(shared_dir)
            shutil.rmtree(ref_dir / "route_refs", ignore_errors=True)
            if not bundle_root:
                shutil.rmtree(shared_dir, ignore_errors=True)
            else:
                shutil.rmtree(shared_reference_dir, ignore_errors=True)

        if a.first_layer == 0 and a.last_layer == 47:
            free = free_mib()
            result = {"status": "PASS_STAGE7_ALL_48_LAYERS" if ok else "FAIL_STAGE7_VALIDATION",
                      "model": str(model), "model_index_sha256": model_index_hash,
                      "layer_count": len(layers), "routed_cards": 5, "shared_cards": 1,
                      "routed_gm107_count": 20, "shared_gm107_count": 4,
                      "minimum_free_mib_by_device": free[:24],
                      "elapsed_seconds": time.monotonic() - overall_start,
                      "layers": layers}
        else:
            result = {"status": "PASS_STAGE7_PARTIAL_RANGE" if ok else "FAIL_STAGE7_VALIDATION",
                      "first_layer": a.first_layer, "last_layer": a.last_layer,
                      "completed_layer_count": len(layers), "elapsed_seconds": time.monotonic() - overall_start,
                      "layers": layers}
        (out / "stage7_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["status"], "layers", len(layers), "seconds", round(result["elapsed_seconds"], 2), flush=True)
        return 0 if ok else 1
    except Exception as e:
        result = {"status": "BLOCKED_STAGE7" if not layers else "PARTIAL_STAGE7",
                  "reason": f"{type(e).__name__}: {e}", "completed_layer_count": len(layers),
                  "elapsed_seconds": time.monotonic() - overall_start,
                  "minimum_free_mib_by_device": free_mib()[:24], "layers": layers}
        (out / "stage7_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["status"], result["reason"], flush=True)
        return 2
    finally:
        for reader in readers.values():
            reader.close()
        manager.close()


if __name__ == "__main__":
    raise SystemExit(main())
