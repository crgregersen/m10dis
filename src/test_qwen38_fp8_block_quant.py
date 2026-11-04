#!/usr/bin/env python3
"""CPU-only validation of qwen38_fp8_block_quant against Qwen's own FP8 quantizer.

1. Fetch BF16 expert weights from Qwen/Qwen3.8-Flash-Next (HTTP range, only those tensors),
   quantize them with quantize_block_fp8, and compare bit-for-bit with the FP8 weight and
   weight_scale_inv of the same tensors in the local official Qwen3.8-Flash-Next-FP8 checkpoint.
   This proves both the scale convention and that our quantizer reproduces Qwen's.
2. Quantize real linear-attention weights (local BF16): every element within the E4M3 rounding
   bound, GEMV relative error on random activations, negative control (a wrong scale must fail).
Usage: CUDA_VISIBLE_DEVICES= python -I test_qwen38_fp8_block_quant.py LOCAL_FP8_MODEL_DIR
"""
import json, struct, sys, urllib.request
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0, str(Path(__file__).parent))
from qwen38_fp8_block_quant import quantize_block_fp8, dequantize_block_fp8

local = Path(sys.argv[1])
BF16_REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
lidx = json.loads((local / "model.safetensors.index.json").read_text())["weight_map"]

def get(url, a=None, b=None):
    r = urllib.request.Request(url)
    if a is not None: r.add_header("Range", f"bytes={a}-{b}")
    with urllib.request.urlopen(r, timeout=180) as f: return f.read()

ridx = json.loads(get(BF16_REPO + "model.safetensors.index.json"))["weight_map"]
hcache = {}
def remote_expert(layer, expert, proj):
    """BF16 rows of one expert from the fused tensors: gate_up_proj [512,1280,2560], down_proj [512,2560,640]."""
    name = f"model.language_model.layers.{layer}.mlp.experts." + ("down_proj" if proj == "down" else "gate_up_proj")
    fn = ridx[name]
    if fn not in hcache:
        n = struct.unpack("<Q", get(BF16_REPO + fn, 0, 7))[0]
        hcache[fn] = (json.loads(get(BF16_REPO + fn, 8, 8 + n - 1)), 8 + n)
    h, off = hcache[fn]; e = h[name]; a, _ = e["data_offsets"]
    _, rows, cols = e["shape"]; per = rows * cols * 2
    t = torch.frombuffer(bytearray(get(BF16_REPO + fn, off + a + expert * per, off + a + (expert + 1) * per - 1)),
                         dtype=torch.bfloat16).reshape(rows, cols)
    if proj == "gate": return t[: rows // 2]
    if proj == "up": return t[rows // 2:]
    return t

def loc(name):
    with safe_open(local / lidx[name], "pt") as f: return f.get_tensor(name)

checks = {}
P = "model.language_model.layers."
for layer, expert, proj in ((0, 0, "gate"), (0, 0, "up"), (17, 300, "down")):
    name = f"{P}{layer}.mlp.experts.{expert}.{proj}_proj.weight"
    w = remote_expert(layer, expert, proj)
    q_ref, s_ref = loc(name), loc(name + "_scale_inv")
    for sdt in (torch.bfloat16,):
        q, s = quantize_block_fp8(w, scale_dtype=sdt)
        same_q = torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        same_s = torch.equal(s.to(s_ref.dtype), s_ref)
        diff = int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
        print(f"OFFICIAL_MATCH,{name},scale_dtype,{str(sdt)[6:]},fp8_bytes_identical,{int(same_q)},scale_identical,{int(same_s)},differing_bytes,{diff}/{q.numel()}")
        if sdt == s_ref.dtype:
            checks[f"official_quantizer_reproduced:{layer}.{expert}.{proj}"] = same_q and same_s

g = torch.Generator().manual_seed(0)
for name in (P + "0.linear_attn.in_proj_qkv.weight", P + "0.linear_attn.in_proj_z.weight", P + "40.linear_attn.out_proj.weight"):
    w = loc(name); q, s = quantize_block_fp8(w); d = dequantize_block_fp8(q, s); r = w.float()
    bound = 2.0 ** -4 * r.abs() + s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: r.shape[0], : r.shape[1]] * 2.0 ** -9
    within = float(((d - r).abs() <= bound + 1e-12).float().mean())
    x = torch.randn(8, w.shape[1], generator=g)
    y_ref, y = x @ r.t(), x @ d.t()
    gemv_rel = float(torch.linalg.norm(y - y_ref) / torch.linalg.norm(y_ref))
    bad = x @ (q.float() * 2.0 * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[: r.shape[0], : r.shape[1]]).t()
    neg = float(torch.linalg.norm(bad - y_ref) / torch.linalg.norm(y_ref))
    print(f"LINEAR_ATTN,{name},shape,{tuple(w.shape)},scale_shape,{tuple(s.shape)},within_e4m3_bound,{within:.6f},gemv_rel_err,{gemv_rel:.4f},negative_control_rel_err,{neg:.3f}")
    checks[f"bound:{name.split('layers.')[1]}"] = within == 1.0
    checks[f"gemv:{name.split('layers.')[1]}"] = gemv_rel < 0.05
    checks[f"negative_control:{name.split('layers.')[1]}"] = neg > 0.5
for k, v in checks.items(): print("CHECK", k, "OK" if v else "FAIL")
ok = all(checks.values()); print("FP8_BLOCK_QUANT_TEST," + ("PASS" if ok else "FAIL")); sys.exit(0 if ok else 1)
