#!/usr/bin/env python3
"""Score LiveCodeBench code-generation results of the Q38 benchmark mode (2026-11-04).

  score_lcb.py LCB_JSONL RESULTS_JSONL [--timeout 6] [--out scored.jsonl]

Code: the last ```python block after </think> (an answer without </think> was cut by the output cap: no code ->
fail, reported as "capped"). Tests: public_test_cases (JSON) + private_test_cases (base64 -> zlib -> pickle -> JSON;
the pickle holds a plain string and is loaded with an unpickler that refuses every class). stdin problems: the
program reads stdin, output compared line by line after stripping (numbers compared with 1e-6 tolerance when both
lines parse as floats). functional (LeetCode, metadata func_name): Solution().<func_name>(*args), args one JSON value
per input line, compared with == after JSON round-trip. Each test runs in a fresh python3 subprocess with a wall
timeout and a 2 GiB address-space limit; a problem passes only if every test passes (pass@1).
Runs the generated code on this machine (unprivileged, temporary directory).
"""
import argparse
import base64
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zlib
from collections import Counter


class _StrOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}")


def private_tests(s):
    try:
        return json.loads(s)
    except Exception:
        return json.loads(_StrOnly(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load())


def extract(text):
    if "</think>" not in text:
        return None
    answer = text.split("</think>", 1)[1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S)
    return blocks[-1] if blocks else None


FUNC_HARNESS = r'''
import json, sys
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
import math, string, re
{code}
_args = [json.loads(l) for l in sys.stdin.read().strip().split("\n")]
_res = Solution().{func}(*_args)
print(json.dumps(_res))
'''


def _limits():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))


def run(src, stdin, timeout):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sol.py")
        open(path, "w").write(src)
        try:
            p = subprocess.run([sys.executable, "-I", path], input=stdin, capture_output=True, text=True,
                               timeout=timeout, cwd=d, preexec_fn=_limits)
            return p.returncode, p.stdout, p.stderr[-300:]
        except subprocess.TimeoutExpired:
            return "timeout", "", ""


def same_stdout(got, want):
    g = [l.strip() for l in got.strip().splitlines()]
    w = [l.strip() for l in want.strip().splitlines()]
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for a, b in zip(g, w):
        if a == b:
            continue
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if x == y:
                continue
            try:
                if abs(float(x) - float(y)) > 1e-6 * max(1.0, abs(float(y))):
                    return False
            except ValueError:
                return False
    return True


def score(prob, code, timeout):
    tests = json.loads(prob["public_test_cases"]) + private_tests(prob["private_test_cases"])
    meta = json.loads(prob.get("metadata") or "{}")
    func = meta.get("func_name")
    src = FUNC_HARNESS.format(code=code, func=func) if func else code
    for i, t in enumerate(tests):
        rc, out, err = run(src, t["input"], timeout)
        if rc != 0:
            return False, f"test {i}/{len(tests)}: {'timeout' if rc == 'timeout' else 'error ' + str(rc)} {err.strip()[-120:]}"
        if func:
            try:
                ok = json.loads(out.strip().splitlines()[-1]) == json.loads(t["output"])
            except Exception:
                ok = False
        else:
            ok = same_stdout(out, t["output"])
        if not ok:
            return False, f"test {i}/{len(tests)}: wrong answer"
    return True, f"{len(tests)} tests"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lcb"); ap.add_argument("results")
    ap.add_argument("--timeout", type=float, default=6.0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    probs = {}
    for line in open(a.lcb):
        p = json.loads(line); probs[p["question_id"]] = p
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    rows = [r for r in rows if r.get("kind") == "generate"]
    by = Counter(); tot = Counter(); out = []
    for r in rows:
        p = probs[r["id"]]
        code = extract(r.get("text", ""))
        if code is None:
            ok, why = False, "capped (no </think>)" if r.get("finish_reason") == "length" else "no code block"
        else:
            ok, why = score(p, code, a.timeout)
        d = p["difficulty"]
        tot[d] += 1; by[d] += ok
        out.append({"id": r["id"], "difficulty": d, "pass": ok, "why": why, "output_tokens": r.get("output_tokens"),
                    "thinking_tokens": r.get("thinking_tokens"), "finish_reason": r.get("finish_reason")})
        print(f"LCB_SCORE,{r['id']},{d},{'PASS' if ok else 'FAIL'},{why},tokens,{r.get('output_tokens')}", flush=True)
    n = len(out)
    capped = sum(1 for o in out if o["why"].startswith("capped"))
    print(f"LCB_TOTAL,pass,{sum(by.values())},of,{n},pct,{100 * sum(by.values()) / max(n, 1):.1f},capped,{capped},"
          + ",".join(f"{d},{by[d]}/{tot[d]}" for d in ("easy", "medium", "hard")))
    if a.out:
        with open(a.out, "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
