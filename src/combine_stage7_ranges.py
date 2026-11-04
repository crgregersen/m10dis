#!/usr/bin/env python3
"""Combine disjoint Stage 7 progress/result JSON files after checking coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("layers")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing layers array")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("inputs", nargs="+", type=Path)
    a = ap.parse_args()
    by_layer: dict[int, dict] = {}
    for path in a.inputs:
        for row in read_records(path):
            layer = int(row["layer"])
            if layer in by_layer:
                raise ValueError(f"duplicate layer {layer} in {path}")
            by_layer[layer] = row
    expected = set(range(48))
    missing, extra = sorted(expected - by_layer.keys()), sorted(by_layer.keys() - expected)
    failed = [i for i, row in by_layer.items() if row.get("status") != "PASS"]
    if missing or extra or failed:
        raise SystemExit(f"coverage failure: missing={missing}, extra={extra}, failed={failed}")
    ordered = [by_layer[i] for i in range(48)]
    for row in ordered:
        checks = row.get("checks", {})
        if any(int(checks.get(k, 0)) != 0 for k in
               ("routed_bad_total", "shared_bad_total", "combined_bad")):
            raise SystemExit(f"layer {row['layer']} has nonzero numerical mismatches")
    result = {"status": "PASS_STAGE7_ALL_48_LAYERS", "layer_count": 48,
              "validated_ranges": [str(p) for p in a.inputs], "layers": ordered}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS_STAGE7_ALL_48_LAYERS", "layers", len(ordered), "out", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
