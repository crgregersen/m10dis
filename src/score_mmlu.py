#!/usr/bin/env python3
"""Score MMLU-style choice results of the Q38 benchmark mode (2026-11-04).

  score_mmlu.py ITEMS_JSONL RESULTS_JSONL [--out summary.json]

pred = the choice letter with the highest logprob among the top-N first-token candidates (no letter in the top N
-> wrong). Reports accuracy with its binomial standard error, the macro average over subjects, how many items had
no letter in the candidates, and the prediction distribution (an all-equal distribution flags a broken run).
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items"); ap.add_argument("results"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    items = {}
    for line in open(a.items):
        it = json.loads(line); items[it["id"]] = it
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "choice"]
    ids = [r["id"] for r in rows]
    dup = len(ids) - len(set(ids)); unknown = sum(1 for i in ids if i not in items)
    subj_ok, subj_n = defaultdict(int), defaultdict(int)
    preds = Counter(); no_letter = 0; ok_total = 0
    for r in rows:
        it = items[r["id"]]
        s = it["meta"]["subject"]
        ok = r.get("pred") == it["answer"]
        no_letter += r.get("pred") is None
        preds[r.get("pred")] += 1
        subj_n[s] += 1; subj_ok[s] += ok; ok_total += ok
    n = len(rows)
    acc = ok_total / max(n, 1)
    se = math.sqrt(acc * (1 - acc) / max(n, 1))
    macro = sum(subj_ok[s] / subj_n[s] for s in subj_n) / max(len(subj_n), 1)
    summary = {"items_scored": n, "items_total": len(items), "accuracy": round(100 * acc, 2),
               "stderr": round(100 * se, 2), "macro_subject_avg": round(100 * macro, 2), "subjects": len(subj_n),
               "no_letter": no_letter, "pred_distribution": dict(preds), "duplicate_ids": dup, "unknown_ids": unknown}
    print("MMLU_SCORE," + ",".join(f"{k},{v}" for k, v in summary.items() if k != "pred_distribution")
          + ",preds," + "/".join(f"{k}:{v}" for k, v in sorted(preds.items(), key=str)))
    if a.out:
        summary["per_subject"] = {s: {"n": subj_n[s], "acc": round(100 * subj_ok[s] / subj_n[s], 1)} for s in sorted(subj_n)}
        json.dump(summary, open(a.out, "w"), indent=1)
    return 0 if n and not dup and not unknown else 1


if __name__ == "__main__":
    sys.exit(main())
