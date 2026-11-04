#!/usr/bin/env python3
"""Precompute the FP8 linear-attention tensors once (instead of quantizing at every vLLM load).

Reads the 108 BF16 linear_attn.{in_proj_qkv,in_proj_z,out_proj}.weight tensors of the official
checkpoint, quantizes them with qwen38_fp8_block_quant (byte-identical to Qwen's FP8 quantizer),
and writes OUT_DIR/linear_attn_fp8.safetensors with, per tensor, `<name>` (F8_E4M3) and
`<name>_scale_inv` (BF16), plus metadata binding the file to this exact checkpoint and quantizer:
  model_index_sha256      sha256 of the checkpoint's model.safetensors.index.json
  quantizer_sha256        sha256 of qwen38_fp8_block_quant.py
  format                  fp8_e4m3 block 128x128, scale bf16, bf16-division (official)
qwen38_rtx_fp8_hooks uses the file only if both hashes match (Q38_RTX_FP8_CACHE=OUT_DIR).
Also writes SHA256SUMS and build-report.json. Usage: build_qwen38_fp8_cache.py [--qsa] MODEL_DIR OUT_DIR
--qsa also quantizes the 48 QSA q/k/v/o projections (metadata qsa=1; use with Q38_RTX_FP8_QSA=1).
"""
import hashlib, json, re, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).parent))
from qwen38_fp8_block_quant import quantize_block_fp8

QSA = "--qsa" in sys.argv                     # also the 48 QSA q/k/v/o projections (Q38_RTX_FP8_QSA=1)
if QSA:
    sys.argv.remove("--qsa")
TARGET = re.compile(r"\.(linear_attn\.(in_proj_qkv|in_proj_z|out_proj)" + (r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)" if QSA else "") + r")\.weight$")
EXPECTED = 108 + (48 if QSA else 0)


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
    names = sorted(n for n in wmap if TARGET.search(n) and not n.startswith("mtp."))   # MTP head stays BF16
    if len(names) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} projections, found {len(names)}")
    meta = {"model_index_sha256": sha256_file(idx_path),
            "quantizer_sha256": sha256_file(Path(__file__).parent / "qwen38_fp8_block_quant.py"),
            "format": "fp8_e4m3 block 128x128, scale bf16, bf16-division (official Qwen FP8)",
            "tensors": str(len(names)), "qsa": "1" if QSA else "0", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.monotonic(); tensors = {}; by_file = {}
    for n in names:
        by_file.setdefault(wmap[n], []).append(n)
    for fname, group in sorted(by_file.items()):
        with safe_open(model / fname, "pt") as f:
            for n in group:
                q, s = quantize_block_fp8(f.get_tensor(n))
                tensors[n] = q; tensors[n + "_scale_inv"] = s
    quant_s = time.monotonic() - t0
    tmp = out / "linear_attn_fp8.safetensors.tmp"
    save_file(tensors, str(tmp), metadata=meta)
    tmp.replace(out / "linear_attn_fp8.safetensors")
    digest = sha256_file(out / "linear_attn_fp8.safetensors")
    (out / "SHA256SUMS").write_text(f"{digest}  linear_attn_fp8.safetensors\n")
    report = {**meta, "file_sha256": digest, "bytes": (out / "linear_attn_fp8.safetensors").stat().st_size,
              "quantize_seconds": round(quant_s, 1), "total_seconds": round(time.monotonic() - t0, 1)}
    (out / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FP8_CACHE_BUILT," + ",".join(f"{k},{v}" for k, v in report.items() if k != "format"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
