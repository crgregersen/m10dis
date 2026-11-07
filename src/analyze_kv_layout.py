#!/usr/bin/env python3
"""Try K/V layouts offline from a Q38_KV_CONFIG_DUMP file, no GPU. Usage: analyze_kv_layout.py DUMP [MAX_MODEL_LEN...]"""
import collections, copy, dataclasses, logging, os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import cloudpickle
d = cloudpickle.load(open(sys.argv[1], "rb"))
from vllm.v1.core import kv_cache_utils as U
from vllm.v1.kv_cache_interface import FullAttentionSpec
specs = d["specs"][0]
print("layers", len(specs), collections.Counter(type(s).__name__ for s in specs.values()))
for k, s in list(specs.items()):
    if "self_attn" in k and "layers.3." in k or "linear_attn" in k and "layers.0." in k or "mtp" in k.lower():
        print(" ", k, type(s).__name__, {f: getattr(s, f, None) for f in ("block_size", "num_kv_heads", "head_size", "head_size_v", "page_size_bytes", "tokens_per_state")})
def run(name, sp, max_len, kv_bytes):
    v = copy.deepcopy(d["vllm_config"])
    v.model_config.max_model_len = max_len
    v.cache_config.kv_cache_memory_bytes = kv_bytes
    try:
        cfg = U.get_kv_cache_configs(v, [sp], [kv_bytes])[0]
    except Exception as e:
        print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}"); return
    gs = collections.Counter((g.kv_cache_spec.block_size, g.kv_cache_spec.page_size_bytes, len(g.layer_names)) for g in cfg.kv_cache_groups)
    print(f"{name} max_len {max_len}: blocks {cfg.num_blocks}, concurrency {U.get_max_concurrency_for_kv_cache_config(v, cfg):.2f}, groups (block_tokens, page_bytes, layers) x n: {dict(gs)}")
def shrink(sp, head=8, kv=1, mtp=False):
    out = {}
    for k, s in sp.items():
        if type(s) is FullAttentionSpec and (mtp or "mtp" not in k):
            s = dataclasses.replace(s, num_kv_heads=kv, head_size=head, head_size_v=head)
        out[k] = s
    return out
for ml in [int(x) for x in sys.argv[2:]] or [64000]:
    run("today", specs, ml, 2523293286)
    for pool in (2523293286, 1 << 30, 700 << 20, 400 << 20):
        run(f"main shrunk, pool {pool >> 20} MiB", shrink(specs), ml, pool)
        run(f"main+mtp shrunk, pool {pool >> 20} MiB", shrink(specs, mtp=True), ml, pool)
