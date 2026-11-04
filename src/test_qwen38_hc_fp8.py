#!/usr/bin/env python3
"""Offline test of qwen38_hc_fp8 on the RTX (2026-11-03): the HC shapes at M = 1/7/64/1024.
Checks: kernel vs FP32 reference of the dequantized weight (kernel error), run-to-run bit identity, FP8 vs the
original BF16 weight (quantization error, informational), and timing vs BF16 F.linear (cuBLAS).
Uses real HC weights of layer 0 when Q38_MODEL_DIR is set, random ones otherwise. Prints Q38_HC_FP8_TEST lines."""
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_hc_fp8 import dequantize_hc_fp8, hc_fp8_linear, quantize_hc_fp8

dev = next((i for i in range(torch.cuda.device_count()) if "RTX" in torch.cuda.get_device_name(i)), None)
if dev is None:
    raise SystemExit("no RTX device found")
torch.cuda.set_device(dev)
print(f"Q38_HC_FP8_TEST,device,{torch.cuda.get_device_name(dev)},cc,{torch.cuda.get_device_capability(dev)}")


def real_weights():
    md = os.environ.get("Q38_MODEL_DIR")
    if not md:
        return None
    from safetensors import safe_open
    wmap = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model.layers.0.mlp_hyper_connection."
    out = {}
    for name in ("input_mix_weight_down.weight", "block_inject_weight.weight", "input_mix_weight_up.weight"):
        with safe_open(os.path.join(md, wmap[pre + name]), "pt") as f:
            out[name] = f.get_tensor(pre + name)
    down = torch.cat([out["input_mix_weight_down.weight"], out["block_inject_weight.weight"],
                      torch.zeros(12, 10240, dtype=torch.bfloat16)])           # merged + padded to 336 rows
    return {"down336": down, "up": out["input_mix_weight_up.weight"], "down320": out["input_mix_weight_down.weight"]}


g = torch.Generator().manual_seed(0)
rw = real_weights()
shapes = {"down336": (336, 10240), "up": (10240, 320), "down320": (320, 10240)}
ok_all = True
for key, (n, k) in shapes.items():
    w = (rw[key] if rw else (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)).cuda()
    q, s = quantize_hc_fp8(w)
    wd = dequantize_hc_fp8(q, s)
    for m in (1, 7, 64, 1024):
        x = (torch.randn(m, k, generator=g)).to(torch.bfloat16).cuda()
        s8 = s * 256.0
        y = hc_fp8_linear(x, q, s8)
        y2 = hc_fp8_linear(x, q, s8)
        ref = (x.float() @ wd.T).to(torch.bfloat16).float()
        bf = F.linear(x, w).float()
        kern_err = (y.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-30)
        quant_err = (y.float() - bf).abs().max().item() / max(bf.abs().max().item(), 1e-30)
        same = torch.equal(y, y2)
        # GPU time from CUDA-graph replays (decode runs in graphs; eager Triton launches are CPU-bound)
        def graph_us(fn):
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(st)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(20):
                    fn()
            gr.replay(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(5):
                gr.replay()
            e1.record(); torch.cuda.synchronize()
            return e0.elapsed_time(e1) / 100 * 1e3
        t8 = graph_us(lambda: hc_fp8_linear(x, q, s8))
        t16 = graph_us(lambda: F.linear(x, w))
        good = same and kern_err < 1e-2
        ok_all &= good
        print(f"Q38_HC_FP8_TEST,shape,{key},M,{m},weights,{'real' if rw else 'random'},kernel_rel_err,{kern_err:.2e},"
              f"fp8_vs_bf16_rel_err,{quant_err:.2e},repeat_identical,{int(same)},fp8_us,{t8:.1f},bf16_us,{t16:.1f},"
              f"{'OK' if good else 'BAD'}")
print(f"Q38_HC_FP8_TEST_DONE,{'OK' if ok_all else 'BAD'}")
