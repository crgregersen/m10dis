#!/usr/bin/env python3
"""Extract lm_head.weight (BF16, [vocab, hidden]) from the checkpoint into one raw row-major file for
the PLE owner (PLE_LM_HEAD_FILE), which loads one row slice per M10 for worker mode 7.

Writes OUT_DIR/lm_head_bf16.bin (vocab * hidden * 2 bytes, no header), SHA256SUMS and
build-report.json (model index sha256, shape, dtype, file sha256). Usage: MODEL_DIR OUT_DIR
"""
import hashlib, json, sys, time
from pathlib import Path
import torch
from safetensors import safe_open


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    model, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    idx_path = model / "model.safetensors.index.json"
    wmap = json.loads(idx_path.read_text())["weight_map"]
    t0 = time.monotonic()
    with safe_open(model / wmap["lm_head.weight"], "pt") as f:
        w = f.get_tensor("lm_head.weight")
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise SystemExit(f"unexpected lm_head.weight {w.dtype} {tuple(w.shape)}")
    w = w.contiguous()
    tmp = out / "lm_head_bf16.bin.tmp"
    w.view(torch.int16).numpy().tofile(tmp)
    tmp.replace(out / "lm_head_bf16.bin")
    digest = sha256_file(out / "lm_head_bf16.bin")
    (out / "SHA256SUMS").write_text(f"{digest}  lm_head_bf16.bin\n")
    report = {"model_index_sha256": sha256_file(idx_path), "tensor": "lm_head.weight", "dtype": "bfloat16",
              "rows": int(w.shape[0]), "hidden": int(w.shape[1]), "row_bytes": int(w.shape[1]) * 2,
              "bytes": (out / "lm_head_bf16.bin").stat().st_size, "file_sha256": digest,
              "seconds": round(time.monotonic() - t0, 1),
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("LM_HEAD_RAW_BUILT," + ",".join(f"{k},{v}" for k, v in report.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
