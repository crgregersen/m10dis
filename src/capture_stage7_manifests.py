#!/usr/bin/env python3
"""Capture per-layer builder manifests before the streaming loader deletes them."""
import hashlib, json, sys, time
from pathlib import Path

stage, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
done = set()
progress = stage.parent / "stage7_progress.json"
try:
    first_layer = int(json.loads(progress.read_text())["last_layer"]) + 1
except Exception:
    first_layer = 0
target = max(0, 48 - first_layer)
deadline = time.monotonic() + 4 * 60 * 60
while time.monotonic() < deadline and len(done) < target:
    for layer in range(first_layer, 48):
        if layer in done:
            continue
        route = stage / f"route_{layer:02d}"
        shared = stage / f"shared_{layer:02d}"
        rp, sp = route / "resident_layer_manifest.json", shared / "shared_manifest.json"
        if not rp.exists() or not sp.exists():
            continue
        rdoc, sdoc = json.loads(rp.read_text()), json.loads(sp.read_text())
        shared_files = []
        payloads = [p for p in shared.glob("shared.*.bin")
                    if p.name.endswith((".gate.bin", ".up.bin", ".down.bin"))]
        for p in sorted(payloads) + [shared / "shared_gate_weight.bin"]:
            if not p.is_file():
                continue
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            shared_files.append({"name": p.name, "bytes": p.stat().st_size,
                                 "sha256": h.hexdigest()})
        if len(shared_files) != 13:
            continue
        record = {"layer": layer, "routed_manifest": rdoc,
                  "shared_manifest": sdoc, "shared_files": shared_files}
        (output / f"layer_{layer:02d}_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
        done.add(layer)
        print(f"CAPTURED_LAYER_MANIFEST,{layer}", flush=True)
    time.sleep(0.25)
print(f"MANIFEST_CAPTURE_COUNT,{len(done)},TARGET,{target}", flush=True)
