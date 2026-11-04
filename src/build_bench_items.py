#!/usr/bin/env python3
"""Build Q38_BENCH_JSONL item files (2026-11-04) for run_vllm_resident_m10_client.py's benchmark mode.

  build_bench_items.py lcb  LCB_JSONL OUT [--subset N] [--ids a,b]   LiveCodeBench code generation (official
        lcb_runner chat prompt: system + question + format block, stdin or starter code). --subset N picks N
        problems stratified by difficulty and contest month (deterministic: problems sorted by date and id, then
        every k-th inside each difficulty, proportional to the full set).
  build_bench_items.py mmlu MMLU_DIR OUT [--shots 5] [--subjects a,b]   MMLU-style 4-choice questions from the
        parquet / csv files of a HF MMLU-Redux 2.0 or cais/mmlu download: few-shot examples (the dev split of each
        subject, or the first questions of the subject when no dev split exists) as chat turns, answer letter only.

Data files are only read (json / csv / parquet); nothing from them is executed or unpickled here.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

LCB_SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will "
              "generate a correct Python program that matches the specification and passes all tests.")
LCB_FMT_STARTER = ("### Format: You will use the following starter code to write the solution to the problem and "
                   "enclose your code within delimiters.\n```python\n{starter}\n```\n\n")
LCB_FMT_STDIN = ("### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not "
                 "directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that "
                 "when the python program runs, it reads the inputs, runs the algorithm and writes output to "
                 "STDOUT.\n```python\n# YOUR CODE HERE\n```\n\n")


def lcb(args):
    probs = [json.loads(line) for line in open(args.src) if line.strip()]
    probs.sort(key=lambda p: (p["contest_date"], p["question_id"]))
    if args.ids:
        want = set(args.ids.split(","))
        probs = [p for p in probs if p["question_id"] in want]
    elif args.subset:
        by_diff = defaultdict(list)
        for p in probs:
            by_diff[p["difficulty"]].append(p)
        total = len(probs)
        picked = []
        for diff, group in sorted(by_diff.items()):
            n = max(1, round(args.subset * len(group) / total))
            step = len(group) / n
            picked += [group[int(i * step + step / 2)] for i in range(n)]
        probs = sorted(picked, key=lambda p: (p["contest_date"], p["question_id"]))
    with open(args.out, "w") as f:
        for p in probs:
            starter = (p.get("starter_code") or "").strip()
            user = ("### Question:\n" + p["question_content"] + "\n\n"
                    + (LCB_FMT_STARTER.format(starter=starter) if starter else LCB_FMT_STDIN)
                    + "### Answer: (use the provided format with backticks)\n\n")
            f.write(json.dumps({"id": p["question_id"], "kind": "generate",
                                "messages": [{"role": "system", "content": LCB_SYSTEM}, {"role": "user", "content": user}],
                                "meta": {"platform": p["platform"], "difficulty": p["difficulty"],
                                         "contest_date": p["contest_date"], "starter": bool(starter)}}) + "\n")
    print(f"LCB_ITEMS,{len(probs)},{args.out}," + ",".join(f"{d}:{sum(1 for p in probs if p['difficulty'] == d)}"
                                                         for d in ("easy", "medium", "hard")))


def _rows(path):
    if path.suffix == ".arrow":                      # HF datasets save_to_disk (MMLU-Redux 2.0)
        import pyarrow.ipc as ipc
        return ipc.open_stream(str(path)).read_all().to_pylist()
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))


def mmlu(args):
    """Expects per-subject files <subject>/{dev,test}*.parquet (cais/mmlu layout) or MMLU-Redux 2.0 <subject>/data
    files with question / choices / answer (int) and, for Redux, error_type (rows marked as wrong are dropped)."""
    root = Path(args.src)
    subjects = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects.split(",")]
    letters = "ABCD"
    n_items = 0
    dev_by = defaultdict(list)                       # --dev: cais/mmlu all/dev parquet (5 per subject, 'subject')
    if args.dev:
        for r in _rows(Path(args.dev)):
            dev_by[r["subject"]].append(r)
    with open(args.out, "w") as f:
        for subj in subjects:
            files = (sorted((root / subj).rglob("*.parquet")) or sorted((root / subj).rglob("*.arrow"))
                     or sorted((root / subj).rglob("*.csv")))
            test = [r for fp in files if "dev" not in fp.name and "validation" not in fp.name for r in _rows(fp)]
            dev = dev_by.get(subj) or [r for fp in files if "dev" in fp.name for r in _rows(fp)]
            # MMLU-Redux: keep only questions whose original answer is correct (error_type "ok"), as in the paper
            if test and "error_type" in test[0]:
                test = [r for r in test if r.get("error_type") == "ok"]
            shots = (dev or test)[: args.shots]
            if not dev:
                test = test[args.shots:]
            name = subj.replace("_", " ")
            head = f"The following are multiple choice questions (with answers) about {name}.\n\n"

            def fmt(r):
                ch = r["choices"]
                if not isinstance(ch, list):
                    import ast
                    ch = ast.literal_eval(ch)              # Redux stores the list as its Python repr (literal only)
                return r["question"].strip() + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(ch)) + "\nAnswer:"
            msgs = []
            for i, r in enumerate(shots):
                msgs.append({"role": "user", "content": (head if i == 0 else "") + fmt(r)})
                msgs.append({"role": "assistant", "content": letters[int(r["answer"])]})
            for k, r in enumerate(test):
                q = {"role": "user", "content": (head if not msgs else "") + fmt(r)}
                f.write(json.dumps({"id": f"{subj}/{k}", "kind": "choice", "choices": list(letters),
                                    "answer": letters[int(r["answer"])], "messages": msgs + [q],
                                    "meta": {"subject": subj}}) + "\n")
                n_items += 1
    print(f"MMLU_ITEMS,{n_items},subjects,{len(subjects)},{args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lcb"); a.add_argument("src"); a.add_argument("out")
    a.add_argument("--subset", type=int, default=0); a.add_argument("--ids", default="")
    b = sub.add_parser("mmlu"); b.add_argument("src"); b.add_argument("out")
    b.add_argument("--shots", type=int, default=5); b.add_argument("--subjects", default="")
    b.add_argument("--dev", default="", help="cais/mmlu all/dev parquet for the few-shot examples")
    args = ap.parse_args()
    lcb(args) if args.cmd == "lcb" else mmlu(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
