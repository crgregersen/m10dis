#!/usr/bin/env python3
"""Fix the evaluation order of a benchmark item file (build_bench_items.py output), deterministically.

  order_bench_items.py shuffle    IN OUT [--seed 20261104]   random order (MMLU: a deadline-cut run stays an
                                                               unbiased sample of all subjects)
  order_bench_items.py interleave IN OUT [--seed 20261104]   LCB: shuffle inside each difficulty, then merge the
                                                               difficulties proportionally, so any prefix of the
                                                               run has the subset's difficulty mix
Reproduces the item files of the 2026-11-04 runs (sha256 prefixes 7d35a54f and b00550b2).
"""
import argparse, json, random

ap = argparse.ArgumentParser()
ap.add_argument("op", choices=("shuffle", "interleave"))
ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=20261104)
a = ap.parse_args()
items = [json.loads(l) for l in open(a.src)]
if a.op == "shuffle":
    random.Random(a.seed).shuffle(items)
    ordered = items
else:
    groups = {}
    for x in items:
        groups.setdefault(x["meta"]["difficulty"], []).append(x)
    rng = random.Random(a.seed)
    keyed = []
    for d, g in groups.items():
        rng.shuffle(g)
        keyed += [((i + 0.5) / len(g), d, x) for i, x in enumerate(g)]
    keyed.sort(key=lambda k: (k[0], k[1]))
    ordered = [x for _, _, x in keyed]
with open(a.out, "w") as f:
    for x in ordered:
        f.write(json.dumps(x) + "\n")
print(f"ORDERED,{a.op},{len(ordered)},{a.out}")
