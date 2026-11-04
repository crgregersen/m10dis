# Results

| Path | Content |
|---|---|
| `experiment-log.json` | every hardware and analysis run of the batched-MoE work and the benchmark night (92+ entries): label, what, goal, pre-registered expectation, runner, status, result. Paths are relative to the original workspace (`runs/…` = run folders, not included) |
| `benchmarks/mmlu_redux_5shot/` | MMLU-Redux 2.0, 5-shot: `results.jsonl` (2,628 rows: id, choice logprobs, prediction, seed, timing), `score.json`, `score.log`, `items.sha256`, `run-summary.txt` |
| `benchmarks/lcb_v6_subset40/` | LiveCodeBench v6 subset of 40: `results.jsonl` (full generated text incl. thinking, token counts, finish reason, seed), `scored.jsonl` / `score.log` (per-problem test results) |
| `benchmarks/lcb_v6_probe3/` | the 3-problem smoke run before the night (cap 32,768) |

The item files are not included (they contain the benchmark prompts). Rebuild them with `src/build_bench_items.py` and `src/order_bench_items.py` ([docs/BENCHMARKS.md](../docs/BENCHMARKS.md)); `items.sha256` lets you check that you got the same files.

Summary and interpretation: [docs/RESULTS.md](../docs/RESULTS.md).
