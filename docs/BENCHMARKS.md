# Benchmarks

The benchmark path checks that the system's changes (FP8 weights, M10 expert slices, M10 LM head, batched prefill) haven't hurt the model's accuracy. The approach is to run official prompts and scoring against the published numbers, rather than A/B against another deployment.

## Pieces

| Piece | File |
|---|---|
| Item builder (LiveCodeBench, MMLU-style) | `src/build_bench_items.py` |
| Fixed evaluation order | `src/order_bench_items.py` |
| Benchmark mode in the client (`Q38_BENCH_*`) | `src/run_vllm_resident_m10_client.py` |
| Runner with the frozen serving configuration | `src/run_qwen38_bench.sh` |
| Scoring (CPU) | `src/run_bench_score_v2.sh`, `src/score_lcb.py`, `src/score_mmlu.py` |

## 1. Build the item files

Datasets (downloaded by you, read only): LiveCodeBench `test6.jsonl` (release v6, 175 problems from 2025-01 to 2025-04), MMLU-Redux 2.0 and the `cais/mmlu` `all/dev` split for the few-shot examples.

```bash
python3 src/build_bench_items.py lcb  "$M10DIS_BENCH_DATA/lcb/test6.jsonl" lcb_v6_subset40.jsonl --subset 40
python3 src/order_bench_items.py interleave lcb_v6_subset40.jsonl lcb_v6_subset40_interleaved.jsonl
python3 src/build_bench_items.py mmlu "$M10DIS_BENCH_DATA/mmlu-redux" mmlu_redux_5shot.jsonl --shots 5 --dev "$M10DIS_BENCH_DATA/mmlu/all/dev-00000-of-00001.parquet"
python3 src/order_bench_items.py shuffle mmlu_redux_5shot.jsonl mmlu_redux_5shot_shuffled.jsonl
```

- **LCB** uses the official `lcb_runner` chat prompt: system message, question, then the starter-code or stdin format block. `--subset 40` picks problems stratified by difficulty and contest month (10 easy, 12 medium, 18 hard). `interleave` orders them so that any prefix has the same difficulty mix.
- **MMLU-Redux** keeps only questions marked `error_type == "ok"` (5,330). The 5-shot examples come from the subject's MMLU dev split, as chat turns, and the answer is a letter. `shuffle` makes a run that stops at a deadline an unbiased sample.

The 2026-11-04 item files have SHA-256 prefixes `b00550b2` (LCB) and `7d35a54f` (MMLU-Redux). The scripts above reproduce them exactly from the same downloads.

## 2. Run

```bash
pueue add --group gpu --label bench_lcb -- sudo -n env Q38_BENCH_ITEMS=lcb_v6_subset40_interleaved.jsonl \
  Q38_BENCH_NAME=lcb_v6_subset40 Q38_BENCH_THINKING=1 Q38_BENCH_MAX_TOKENS=81920 bash src/run_qwen38_bench.sh
pueue add --group gpu --label bench_mmlu -- sudo -n env Q38_BENCH_ITEMS=mmlu_redux_5shot_shuffled.jsonl \
  Q38_BENCH_NAME=mmlu_redux_5shot Q38_BENCH_THINKING=0 Q38_BENCH_DEADLINE=$(date -d '02:30' +%s) bash src/run_qwen38_bench.sh
```

The serving configuration is frozen in the runner: pipelined batched MoE (chunk 1024), FP8 GDN + QSA + hyper-connections, M10 LM head (top-K 1024), prefix caching off, KV 2.5 GiB (94,631 tokens), `max_model_len` 90,112, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, Triton autotuning off. One sequence at a time.

Benchmark-mode settings:

| Variable | Meaning |
|---|---|
| `Q38_BENCH_JSONL` / `Q38_BENCH_OUT` | item file / results file (set by the runner; an existing results file resumes: finished ids are skipped) |
| `Q38_BENCH_THINKING`, `Q38_BENCH_EFFORT` | chat-template thinking switch and effort (template default `xhigh`) |
| `Q38_BENCH_MAX_TOKENS` | output cap per item (default 32,768; the LCB run used 81,920) |
| `Q38_BENCH_TEMPERATURE`, `_TOP_P`, `_TOP_K` | sampling (default 1.0 / 0.95 / 20, as recommended for the model) |
| `Q38_BENCH_CHOICE_LOGPROBS` | choice items: number of top logprobs to read the answer letter from (default 20) |
| `Q38_BENCH_LIMIT`, `Q38_BENCH_IDS` | first N items / explicit ids |
| `Q38_BENCH_DEADLINE` | unix time after which no new item starts |

Each item's seed is derived from `sha256(id)`, so a resumed or repeated run samples each item the same way, up to the RTX's non-determinism. Choice items take the highest-logprob answer letter from one forward pass. Generate items write the full text, token counts and finish reason. Results are appended one JSON line per item, so a crash loses at most the current item.

The runner ends with an artifact check (`BENCH_VERIFY`): results parse, no duplicate ids, every id is in the item set, and not all choice predictions are equal.

## 3. Score

```bash
pueue add --group default --label score_lcb  -- bash src/run_bench_score_v2.sh lcb_v6_subset40 lcb
pueue add --group default --label score_mmlu -- bash src/run_bench_score_v2.sh mmlu_redux_5shot mmlu
```

- `score_lcb.py` takes the last fenced Python block after `</think>`. It runs public and private tests in a subprocess with a 2 GiB address-space limit and 6 s per test, in a temporary folder, as the unprivileged scoring user. Private tests are decoded with a restricted unpickler that accepts strings only. A capped answer (no `</think>`) counts as a fail.
- `score_mmlu.py` reports accuracy with its binomial standard error, the macro average over subjects and the prediction distribution.

Scores are written to `$M10DIS_RUNS/night-runs/scores/<run folder>/`.

## Unattended runs

Long benchmark chains run overnight. The rules that kept those runs trustworthy:

1. **Approve the chain in advance**: label, runner, item set, parameters and a pre-registered expectation for every run. Nothing outside the chain gets queued.
2. **Clean start**: daemon `READY`, 48 layers, `PARKED`, no peer client; fan control active; the expected UVM module loaded.
3. **Everything through Pueue**, chained with `pueue add --after`. GPU runs in the one-slot group, scoring in another.
4. **Frozen parameters**: only the knob under test changes. KV size is set in bytes, never by utilization.
5. **Flush per item** and print a diagnostic line for the first item, so a broken setup shows within minutes.
6. **A supervisor checks every 15 minutes**: progress lines, daemon status, `dmesg` for Xid/UVM errors, fans, disk. It may kill only a provably hung task (no progress and no GPU activity for 30 minutes), through the runner's own cleanup.
7. **No automatic retries.** A requeue needs an explicit retry budget, a named root cause, a new label and a fresh folder, and it resumes from a copy of the finished results. A low score or a capped answer is a result, not a reason to retry.
8. **"Implausibly fast" counts as failed** until the artifacts prove otherwise.
9. **In the morning**: verify every landing (completion marker, score line, row counts, item-file hash), record scores with their uncertainty, write a digest.

Results of the 2026-11-04 night: [RESULTS.md](RESULTS.md#accuracy).
