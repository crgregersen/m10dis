#!/usr/bin/env python3
"""Replay a Q38_QSA_SELECT_LOG file through per-layer block caches, per token: misses per token over all layers, tail,
LRU against CLOCK. Usage: analyze_qsa_bursts.py LOG [OUT_JSON]. Time per miss assumes 8 KB per block (4 tokens x 2 KB) at 3.2 GB/s."""
import collections
import json
import sys

import numpy as np

sys.argv, argv = [sys.argv[0]], sys.argv
import importlib.util

spec = importlib.util.spec_from_file_location("sel", __file__.replace("analyze_qsa_bursts", "analyze_qsa_selection"))
src = open(spec.origin).read().replace("\nmain()\n", "\n")
ns = {}
exec(compile(src, spec.origin, "exec"), ns)
records = ns["records"]
WARMUP = 50
SIZES = [16384, 32768, 65536]
MS_PER_BLOCK = 8192 / 3.2e9 * 1e3


class Lru:
    def __init__(self, n):
        self.c, self.n = collections.OrderedDict(), n

    def access(self, x):
        if x in self.c:
            self.c.move_to_end(x)
            return 0
        self.c[x] = 1
        if len(self.c) > self.n:
            self.c.popitem(last=False)
        return 1


class Clock:
    def __init__(self, n):
        self.slot, self.ref, self.tag, self.hand, self.n = {}, [0] * n, [None] * n, 0, n

    def access(self, x):
        s = self.slot.get(x)
        if s is not None:
            self.ref[s] = 1
            return 0
        while self.ref[self.hand]:
            self.ref[self.hand] = 0
            self.hand = (self.hand + 1) % self.n
        s = self.hand
        self.hand = (s + 1) % self.n
        if self.tag[s] is not None:
            del self.slot[self.tag[s]]
        self.tag[s], self.slot[x], self.ref[s] = x, s, 1
        return 1


per = collections.defaultdict(list)
for layer, pos, idx in records(argv[1]):
    per[pos].append((layer, np.unique(idx // 4).tolist()))
res = {}
for size in SIZES:
    for name, cls in (("lru", Lru), ("clock", Clock)):
        caches = collections.defaultdict(lambda: cls(size // 4))
        miss = []
        for i, pos in enumerate(sorted(per)):
            m = sum(caches[l].access(b) for l, blocks in per[pos] for b in blocks)
            if i >= WARMUP:
                miss.append(m)
        a = np.array(miss)
        ms = a * MS_PER_BLOCK
        res[f"{name}_{size}"] = {"mean_blocks": float(a.mean()), "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()), "tokens_over_2ms": int((ms > 2).sum())}
        print(name, size, {k: round(v, 2) for k, v in res[f"{name}_{size}"].items()})
if len(argv) > 2:
    json.dump(res, open(argv[2], "w"), indent=1)
