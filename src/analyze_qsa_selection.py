#!/usr/bin/env python3
"""Analyze a Q38_QSA_SELECT_LOG file: selection shape, block balance over GPUs, token-to-token overlap, LRU hit rate.
Usage: analyze_qsa_selection.py LOG [OUT_JSON]. Caches are per layer and hold blocks of 4 tokens; WARMUP tokens are
excluded from the hit rate. Striping: 'mod' = block mod 4; 'perm' = a fixed 256-entry random permutation of the block id mod 4."""
import collections
import json
import pathlib
import struct
import sys

import numpy as np

RATIO, GPUS, WARMUP = 4, 4, 50
CACHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]  # tokens per layer


def records(path):
    b = pathlib.Path(path).read_bytes()
    o = 0
    while o + 12 <= len(b):
        layer, pos, n = struct.unpack_from("<3i", b, o)
        o += 12
        if o + 4 * n > len(b):
            break  # a truncated tail
        yield layer, pos, np.frombuffer(b, "<i4", n, o)
        o += 4 * n


def lru(seq, entries):
    c, hit, tot = collections.OrderedDict(), 0, 0
    for i, blocks in enumerate(seq):
        h = 0
        for x in blocks:
            if x in c:
                c.move_to_end(x)
                h += 1
            else:
                c[x] = 1
                if len(c) > entries:
                    c.popitem(last=False)
        if i >= WARMUP:
            hit += h
            tot += len(blocks)
    return hit / tot if tot else None


def main():
    per = collections.defaultdict(list)
    for layer, pos, idx in records(sys.argv[1]):
        per[layer].append((pos, idx))
    perm = np.random.default_rng(38).integers(0, GPUS, 256)
    res = {"layers": {}}
    all_bal = {"mod": [], "perm": []}
    for layer in sorted(per):
        rows = per[layer]
        seq, counts, jac, age, bal = [], [], [], [], {"mod": [], "perm": []}
        for pos, idx in rows:
            blocks = np.unique(idx // RATIO)
            seq.append(blocks.tolist())
            counts.append(len(idx))
            age.append(float(np.median(pos - idx)))
            for name, gpu in (("mod", blocks % GPUS), ("perm", perm[blocks % 256])):
                load = np.bincount(gpu, minlength=GPUS)
                bal[name].append(load.max() / load.mean())
        for a, b in zip(seq, seq[1:]):
            s, t = set(a), set(b)
            jac.append(len(s & t) / max(len(s), 1))
        hits = {str(c): lru(seq, c // RATIO) for c in CACHES}
        res["layers"][layer] = {
            "tokens": len(rows), "pos_first": rows[0][0], "pos_last": rows[-1][0],
            "selected_median": float(np.median(counts)), "median_age": float(np.median(age)),
            "kept_from_previous_token": float(np.mean(jac)) if jac else None,
            "balance_mod_mean": float(np.mean(bal["mod"])), "balance_mod_max": float(np.max(bal["mod"])),
            "balance_perm_mean": float(np.mean(bal["perm"])), "balance_perm_max": float(np.max(bal["perm"])),
            "lru_hit": hits}
        for k in all_bal:
            all_bal[k] += bal[k]
    L = res["layers"].values()
    res["all"] = {
        "layers": len(res["layers"]),
        "kept_from_previous_token": float(np.mean([x["kept_from_previous_token"] for x in L if x["kept_from_previous_token"] is not None])),
        "balance_mod_mean": float(np.mean(all_bal["mod"])), "balance_mod_p99": float(np.percentile(all_bal["mod"], 99)), "balance_mod_max": float(np.max(all_bal["mod"])),
        "balance_perm_mean": float(np.mean(all_bal["perm"])), "balance_perm_p99": float(np.percentile(all_bal["perm"], 99)), "balance_perm_max": float(np.max(all_bal["perm"])),
        "lru_hit": {str(c): float(np.mean([x["lru_hit"][str(c)] for x in L])) for c in CACHES}}
    print(json.dumps(res["all"], indent=1))
    for k, x in res["layers"].items():
        print(k, x["tokens"], x["pos_first"], x["pos_last"], x["selected_median"], x["median_age"], round(x["kept_from_previous_token"], 3), {c: round(v, 3) for c, v in x["lru_hit"].items()})
    if len(sys.argv) > 2:
        pathlib.Path(sys.argv[2]).write_text(json.dumps(res, indent=1))


main()
