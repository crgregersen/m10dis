#!/usr/bin/env python3
"""Score the HumanEval pass of one or two vLLM client results (run_vllm_resident_m10_client.py with
Q38_HUMANEVAL_JSONL) and, with two, compare them.

Per completion the official HumanEval program is executed: prompt + completion + test +
check(entry_point), in its own temporary directory, `python3 -I`, 15 s timeout. Run as a normal user
(not root): this executes model-generated code.

Usage: score_humaneval.py --dataset HumanEval.jsonl.gz --out OUT.json RESULT_A.json [RESULT_B.json]
Prints HUMANEVAL_SCORE lines and, with two results, HUMANEVAL_COMPARE lines.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def load_problems(path: Path) -> dict[str, dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return {p["task_id"]: p for p in (json.loads(line) for line in fh if line.strip())}


def sanitize_body(completion: str) -> str:
    """Keep only the function body: cut at the first non-empty line that starts at column 0 (the
    model's own top-level asserts, prints, prose or new definitions after the body), as standard
    HumanEval post-processing does. --sanitize enables it."""
    out = []
    for line in completion.split("\n"):
        if line.strip() and not line[0].isspace():
            break
        out.append(line)
    return "\n".join(out)


def run_one(problem: dict, completion: str, timeout: float = 15.0) -> tuple[bool, str]:
    program = problem["prompt"] + completion + "\n\n" + problem["test"] + "\n" + f"check({problem['entry_point']})\n"
    with tempfile.TemporaryDirectory(prefix="q38_he_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program)
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True, text=True,
                                  timeout=timeout, env={"PATH": "/usr/bin:/bin", "HOME": tmp})
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode == 0:
            return True, "passed"
        last = (proc.stderr.strip().splitlines() or ["failed"])[-1]
        return False, last[:200]


SANITIZE = False


def score(result_path: Path, problems: dict[str, dict]) -> dict:
    res = json.loads(result_path.read_text())
    he = res.get("humaneval") or {}
    items = he.get("items") or []
    rows = {}
    for it in items:
        prob = problems[it["task_id"]]
        if it.get("mode", "").startswith("chat"):
            # chat answers carry the whole function: keep the prompt's imports/helpers, then the answer
            # (its definition replaces the prompt's stub)
            ok, why = run_one(dict(prob, prompt=prob["prompt"] + "\n    pass\n\n"), it["completion"])
        else:
            ok, why = run_one(prob, sanitize_body(it["completion"]) if SANITIZE else it["completion"])
        rows[it["task_id"]] = {"passed": ok, "detail": why, "tokens": len(it["generated_token_ids"]),
                               "ids": it["generated_token_ids"], "finish": it.get("finish_reason")}
    n = len(rows); p = sum(r["passed"] for r in rows.values())
    out = {"result": str(result_path), "generated": n, "passed": p, "pass_at_1": round(p / n, 4) if n else None,
           "skipped": len(he.get("skipped") or []), "seconds": he.get("seconds"), "rows": rows}
    print(f"HUMANEVAL_SCORE,{result_path},generated,{n},passed,{p},pass@1,{out['pass_at_1']},skipped,{out['skipped']}")
    return out


def compare(a: dict, b: dict) -> dict:
    common = sorted(set(a["rows"]) & set(b["rows"]))
    both = sum(a["rows"][t]["passed"] and b["rows"][t]["passed"] for t in common)
    only_a = [t for t in common if a["rows"][t]["passed"] and not b["rows"][t]["passed"]]
    only_b = [t for t in common if b["rows"][t]["passed"] and not a["rows"][t]["passed"]]
    identical, first_div = 0, []
    for t in common:
        x, y = a["rows"][t]["ids"], b["rows"][t]["ids"]
        if x == y:
            identical += 1
        else:
            n = min(len(x), len(y))
            first_div.append(next((i for i in range(n) if x[i] != y[i]), n))
    first_div.sort()
    out = {"common": len(common), "both_pass": both, "only_a": only_a, "only_b": only_b,
           "token_identical": identical, "diverged": len(first_div),
           "first_divergence_median": first_div[len(first_div) // 2] if first_div else None,
           "first_divergence_min": first_div[0] if first_div else None}
    print(f"HUMANEVAL_COMPARE,common,{out['common']},a_pass,{sum(a['rows'][t]['passed'] for t in common)},"
          f"b_pass,{sum(b['rows'][t]['passed'] for t in common)},both,{both},only_a,{len(only_a)},only_b,{len(only_b)},"
          f"token_identical,{identical},diverged,{out['diverged']},first_div_median,{out['first_divergence_median']}")
    print("HUMANEVAL_ONLY_A," + ",".join(only_a))
    print("HUMANEVAL_ONLY_B," + ",".join(only_b))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sanitize", action="store_true", help="keep only the function body (cut at the first column-0 line)")
    ap.add_argument("results", type=Path, nargs="+")
    args = ap.parse_args()
    global SANITIZE
    SANITIZE = args.sanitize
    if os.geteuid() == 0:
        raise SystemExit("refusing to execute model-generated code as root")
    problems = load_problems(args.dataset)
    scored = [score(r, problems) for r in args.results[:2]]
    report = {"dataset": str(args.dataset), "sanitize": SANITIZE, "scores": [{k: v for k, v in s.items() if k != "rows"} for s in scored],
              "per_task": {s["result"]: {t: {k: v for k, v in r.items() if k != "ids"} for t, r in s["rows"].items()}
                           for s in scored}}
    if len(scored) == 2:
        report["compare"] = compare(*scored)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
