# Results

All numbers come from runs on the test machine (2026-11-02 to 2026-11-05). Every run has an entry in [`results/experiment-log.json`](../results/experiment-log.json), with its label, goal, pre-registered expectation where there was one, and result. Labels like `t20d` or `i15` below refer to that log.

## Throughput

| Phase | Rate | Conditions |
|---|---|---|
| Decode | **52.5 tok/s** (steady 52.4–52.8) | one sequence, CUDA graphs, FP8 GDN/QSA/HC, M10 LM head; the 2026-11-04 benchmark night |
| Prefill | **≈ 277 tok/s** (261–275 in floor runs) | pipelined batched MoE (mode 9), vLLM steps of 1024 tokens, 4,097-token prompt |
| Daemon start | 181 s to ready | 51.2 GB of n-gram table (1.33 GB/s) + 128.8 GB of expert slices (0.92 GB/s), from NVMe |

### How prefill got there

| Step | Prefill | What changed |
|---|---|---|
| design start | ≈ 51 tok/s | 8-token eager steps, one exchange per row per layer |
| `i3` | 115 tok/s | per-row path, larger steps (batched path at chunk 64 was slower: 88) |
| `i5`, `i6` | 182, 197 tok/s | batched mode 8, 1024-token steps, BF16 tiles and requests |
| `i7` | 274 tok/s | pipelined mode 9: two 32-row slots, RTX publish/combine hidden |

### How decode got there

| Run | Decode | What changed |
|---|---|---|
| `r27` | 46.0–46.5 tok/s | per-row exchange, before the BF16 request change (2026-11-03 morning) |
| `d3` | 48.0–48.5 tok/s | BF16 requests in mode 1: 168 → 152 µs per MoE layer |
| `d4b` | 50.9–51.0 tok/s | FP8 hyper-connection matrices with tuned M = 1 GEMV (down 19.0 → 10.9 µs, up 14.6 → 6.4 µs per call) |
| benchmark night | 52.4–52.8 tok/s | same configuration, longer runs |

## MTP speculative decoding

Current defaults (2026-11-05): `max_model_len` 32,768, 1000 generated tokens per run, MTP layer resident as layer 48, multi-row exchange up to 4 rows (`Q38_RTX_PEER_MOE_MULTIROW_MAX=4`), N-row LM head, outer multicast group, streamed card-sum finish and one-kernel publish+go. Rates include the short prefill (115 tokens natural, 265 humaneval). One run per point, runs `ksweep_N_k*` / `ksweep_H_k*` (pueue tasks 311-318). Acceptance is the share of drafts accepted at each draft position.

| Draft tokens | Natural long-form prompt (`prompts/mtp_eval_prompt.txt`) | Accepted per position | Tokens per step | Humaneval prompt, `ignore_eos` | Accepted per position | Tokens per step |
|---|---|---|---|---|---|---|
| 0 | 56.3 tok/s | – | 1 | 54.6 tok/s | – | 1 |
| 1 | **69.8** | 70 % | 1.70 | 74.1 | 89 % | 1.89 |
| 2 | 69.0 | 70 / 46 % | 2.16 | 88.3 | 97 / 93 % | 2.90 |
| 3 | 64.3 | 67 / 45 / 26 % | 2.38 | **96.3** | 97 / 96 / 92 % | 3.84 |

Single runs vary by up to about 1.4 %, so 1 and 2 drafts are level on natural text. Suggested: 2 draft tokens by default (1 % behind 1 draft on natural text, 19 % ahead on code), 3 for code-heavy work.

### How MTP decode got there

Each row includes the changes above it.

| Change | Natural, 0 / 1 / 2 / 3 drafts (tok/s) | Humaneval, 3 drafts | Runs |
|---|---|---|---|
| N-row exchange, faster multi-row down, N-row LM head (2026-11-04) | 53.4 / 65.6 / 65.4 / 60.1 | 89.6 | `final_*`, `fastdown_*` |
| + outer multicast group | 54.9 / 67.8 / 68.0 / – | 93.7 | `outer_final_*` |
| + streamed card-sum finish (exchange 141.3 → 133.1 µs per layer for 1 row, 400.4 → 379.9 µs for 4; runs `stream_poc_off` / `stream_poc_on`) | 55.7 / 69.5 / 69.8 / – | 95.8 | `stream_final_*` |
| + one-kernel publish+go (2026-11-05) | 56.3 / 69.8 / 69.0 / 64.3 | 96.3 | `ksweep_*` |

With begin, publish and go in one RTX kernel (default since 2026-11-05, `Q38_RTX_PEER_PUBLISH_ORDER=4`; exchange 133.1 → 128.0 µs per layer for 1 row, 204.8 → 200.7 for 2, 380.9 → 374.8 for 4, runs `pubgo_ref` / `pubgo_merge`), mean of four alternating pairs on the natural prompt (runs `pubgo_tps_*`; identical tokens in all runs):

| Drafts | Three kernels | One kernel | Δ |
|---|---|---|---|
| 0 | 55.63 (sd 0.35) | **56.10** (sd 0.27) | +0.8 % |
| 2 | 69.75 (sd 0.13) | **69.84** (sd 0.66) | +0.1 % |

The harness saving (4–6 µs per layer) shrinks inside vLLM's CUDA graph, where kernel boundaries are already cheap.

The outer group (`Q38_RTX_PEER_PEX_OUTER=1`, [TRANSPORT.md](TRANSPORT.md#8-optional-one-outer-multicast-group-q38_rtx_peer_pex_outer1)) publishes each request once instead of six times: 16.4 → 7.2 µs per MoE layer for one row, 45.1 → 11.3 µs for four, with identical results.

The humaneval column overstates acceptance: with `ignore_eos` the model keeps going after its answer, and that continuation repeats. The best draft length is lower than on a single GPU, because every verify row costs M10 time: one multi-row exchange takes 150 / 239 / 337 / 432 µs per layer for 1 / 2 / 3 / 4 rows (bandwidth-bound: each row reads its own experts). Speculative decoding takes KV-pool space; see [Memory and context](#memory-and-context).

What did not help: unrolling the gate/up lane loop (gate/up 62 → 81 µs for 2 rows) and reading experts shared by consecutive tokens once (3.7 of 10 are shared on average; less load overlap made it slower). Both reverted. Records `mtp_*`, `mr_*`, `lm2_*`, `nrow_*`, `fastdown_*`, `final_*` in the run log.

## Where the time goes in one MoE layer

### Decode (1 token, mode 10, defaults of 2026-11-05)

| Step | Time | Run |
|---|---|---|
| RTX begin + publish + go (one kernel, outer multicast) | 4.1 µs | `pubgo_merge` |
| Wait for the 24 worker headers (M10 round trip) | 117.8 µs | `pubgo_merge` |
| RTX combine (9 result rows: 5 card-leader sums of the routed experts + 4 shared-expert rows, summed on the RTX) | 6.1 µs | `pubgo_merge` |
| **Whole layer** | **128.0 µs** (+ 5.1 µs to the next layer) | `pubgo_merge` |

Inside the round trip, the slowest worker is a card leader (critical path 91.1 µs, worker clocks, runs `ackprof_D_outer`, `diag_ref_newkern` v2, `prof_on`):

| Phase (routed GPU) | Time |
|---|---|
| Validate the request | 4.1 µs |
| x staging + gate/up (1.64 MB FP8) | 33.8 µs |
| Down (0.82 MB FP8) | 21.5 µs |
| Leader: wait for the three peers' pushes (push + fence + flag ≈ 11 µs on the peer) | 13.3 µs |
| Leader: own sum (2.0) + write + system fence (4.1) | 7.2 µs |
| Leader: grid barrier + ack | 5.1 µs |

The rest of the 117.8 µs (≈ 27 µs) is the request reaching the workers and the last leader's data and header crossing the RTX's x4 link. That link carries all nine result rows (≈ 90 KB per layer per row; the shared card has no on-card sum in decode, unlike prefill mode 9, so its four workers each send a row), so it is the limit from 2 rows on: without the result writes (diagnostic `diag_nortx` v2) a layer is 2 / 20 / 54 µs faster at 1 / 2 / 4 rows.

The compute is close to the M10's practical bandwidth: in `bench_m10_expert_compute` (runs `m10_compute_bench` v1/v2) the deployed gate/up + down take 56.3 µs, and the best variants (2 rows per warp, transposed down layout, up to 40 warps per SM) 50.3 µs. Gate/up stays at ≈ 30 µs (≈ 60 GB/s of the 83 GB/s peak) at any occupancy. [TRANSPORT.md §9](TRANSPORT.md#9-what-limits-the-decode-exchange) lists the transport experiments.

Before the card-leader sums and mode 10 (mode 1, 2026-11-02, `t20d`): publish 29 µs, round trip 132 µs, combine 9 µs, ≈ 150 µs per layer.

### Prefill (mode 9, per 32-row chunk on the M10s)

| Phase | Time |
|---|---|
| copy rows from host memory | 0.21 ms |
| gate/up | 0.67 ms |
| down | 0.54 ms |
| finish (card sum, flags, acks) | 0.72 ms |
| **Chunk period** | **2.21 ms**, the M10 critical path; the RTX side is fully hidden |

| Rows per layer call | Pipelined (mode 9) | Batched (mode 8) | One row at a time (mode 1) |
|---|---|---|---|
| 32 | 3.12 ms | 3.11 ms | |
| 64 | 5.79 ms | 6.16 ms | |
| 256 | 19.5 ms | 24.5 ms | |
| 1024 | 72.9 ms | 98.4 ms | 157.2 ms |

Fit over the batched compute (µs per call, R rows, distinct experts D): gate/up ≈ 18 + 1.2·D + 2.1·(10R), down ≈ 18 + 1.75·D + 1.2·(10R). The per-(row, expert) work outweighs the weight reads, so larger batches gain less than expert reuse alone predicts. The kernel optimization steps behind these numbers (t1–t19) are in [research/BATCHED_MOE_EXCHANGE.md](research/BATCHED_MOE_EXCHANGE.md).

## Memory and context

| Item | Size |
|---|---|
| M10 per GPU | ≈ 6.8–7.2 GB of 8 GB used: expert slices, n-gram shard, LM-head slice, worker scratch |
| RTX model weights in vLLM | 3.62 GiB with FP8 hyper-connections (4.22 GiB with BF16) |
| Hyper-connection matrices | 194 matrices, 1.213 GiB BF16 → 0.645 GiB FP8 |
| KV cache | 2.5 GiB → **94,631 tokens**; `max_model_len` 90,112 |
| Context in the MTP runs | `max_model_len` 32,768; the model supports 262,144 (`max_position_embeddings`) |

KV capacity per configuration, measured by the vLLM start (runs `s1`–`s3`):

| KV bytes | Prefix caching | Tokens |
|---|---|---|
| 1 GiB | on | 14,745 |
| 1 GiB | off | 34,028 |
| 2 GiB | off | 75,296 |
| 2.5 GiB | off | 94,631 (needs `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`: the default 512 MiB indexer workspace ran out of memory in the first prefill) |

KV capacity at 2.5 GiB in the MTP runs (`max_model_len` 32,768), from the vLLM start of runs `ksweep_*` (decode-profile run directories 20261105T002529Z-20261105T003758Z). The MTP layer's attention KV and the draft slots take part of the pool; one request can still use the full 32,768 tokens in every row.

| Draft tokens | Tokens |
|---|---|
| 0 (no speculative config) | 84,440 |
| 1 | 66,628 |
| 2 | 58,789 |
| 3 | 52,601 |

With prefix caching on, vLLM's Mamba "align" mode stores a GDN state snapshot for every 1,568-token block. That costs 72.8 KB per token instead of 31.6 KB.

## Accuracy

Benchmarks ran on 2026-11-04 with the frozen benchmark configuration ([BENCHMARKS.md](BENCHMARKS.md)). Each was compared with the model card's published number, against a threshold registered before the run.

| Benchmark | Result | Published | Threshold (pre-registered) |
|---|---|---|---|
| MMLU-Redux 2.0, 5-shot (base-model protocol) | **90.98 ± 0.56** (2,628 of 5,330 questions in shuffled order; macro over 57 subjects 90.67) | 90.68 (base model) | ≥ 88.7 → met |
| LiveCodeBench v6, 40-problem stratified subset, thinking | **35/40 = 87.5 % ± 5.2** (easy 10/10, medium 12/12, hard 13/18) | 91.9 | ≥ 80 % → met |

LCB details:

- The cap was 81,920 output tokens. Two hard problems hit it without finishing their thinking (`arc190_c`, `arc193_d`). Of the 38 that finished, 35 passed (92.1 %).
- Two wrong answers (`abc397_d`, `arc192_b`) and one genuine runtime error in the generated code (`abc392_f`, `IndexError`).
- Output length per problem: 104 to 81,920 tokens, median 30,440. The run took 6.1 hours: 1.15 M output tokens at 52.4–52.8 tok/s.
- The MMLU run stopped at a fixed start deadline. Every question got an answer letter within its top-20 logprobs. Predictions were spread A 585 / B 651 / C 688 / D 704.

Per-item outputs: [`results/benchmarks/`](../results/benchmarks/).

## Determinism

Repeated identical runs don't give bit-identical logits. The jitter comes from the RTX side: vLLM's GPU kernels, attention in particular, reduce in a different order from run to run. The M10 path is bit-reproducible: batched and pipelined outputs are bit-identical to the per-row path (0 mismatching rows in the 20-pass stress test `t22b`), because every summation runs in a fixed order.

Token hashes are therefore the wrong test for "did a change hurt". The method used instead is the **floor run**:

1. Repeat an unchanged configuration N times with `Q38_VLLM_LOGPROBS=5`.
2. For every pair of runs and every generated position with an identical token prefix, take the maximum |Δ logprob| over the tokens in both top-5 lists. The maximum over all of them is the **floor**.
3. A candidate configuration is "similar" if its cross-difference to the reference stays within the floor (`src/lm_head_floor.py`).

| Runs | Floor max | Median |
|---|---|---|
| `i13`: pipelined, BF16 hyper-connections, 10 runs | 2.13 | 0.31 |
| `i14`: per row, 5 runs | 2.29 | 0.25 |
| cross `i14` × `i13` | 2.25 | 0.31 |
| `i15`: pipelined, FP8 hyper-connections, 5 runs | 1.25 | 0.30 |
| cross `i15` × `i13` | 1.875 | 0.375 |

So batched/pipelined MoE adds no jitter of its own, and FP8 hyper-connections stay inside the BF16 floor. Earlier "token flips" (different text from token 6 or 11 on) are near-ties in the top-2 (margins of 0–0.6), exposed by this RTX-side variation. Setting `VLLM_TRITON_FORCE_FIRST_CONFIG=1` (no Triton autotuning) did not remove them (`i12`).

## Release-tree verification

The release tree was checked on the same machine before publishing:

- the five UVM patches rebuild a module with the srcversion of the module used for all results (`38263E1067A39300ABB966D`);
- the daemon was restarted from the release tree: services, cubins and the PLE owner were built from its sources, and it loaded 48 layers in 181 s;
- the benchmark item orderings reproduce the item files of the benchmark night byte for byte;
- the CPU-side unit tests pass (PCI discovery 17/17, FP8 block quantization, CPU head hooks, daemon peer recovery);
- the `i15` floor run was repeated from the release tree, with the same settings and the same prompt (byte-identical). All 5 runs passed every runner check and produced the same tokens as `i15`, at 272.8–276.0 tok/s prefill. Its own floor is 1.19 max / 0.31 median (`i15`: 1.25 / 0.30). The cross-difference to `i15` is 1.50 max / 0.31 median: inside the 10-run floor of `i13` (2.13) and below the accepted `i15` × `i13` cross (1.875), so the release tree behaves like the development tree.

Two small source changes came after that hardware check. The standalone entry point of `run_stage8_stage9_vllm_generation.py` now discovers the RTX instead of naming its UUID; the client path already did. And the header comments of the worker kernels were corrected.
