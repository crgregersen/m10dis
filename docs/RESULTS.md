# Results

All numbers come from runs on the test machine (2026-11-02 to 2026-11-04). Every run has an entry in [`results/experiment-log.json`](../results/experiment-log.json), with its label, goal, pre-registered expectation where there was one, and result. Labels like `t20d` or `i15` below refer to that log.

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

## Where the time goes in one MoE layer

### Decode (1 token, mode 1)

| Step | Time |
|---|---|
| RTX publishes the request (6 multicast writes) | 29 µs |
| M10 round trip: compute, card sum, return, acks | 132 µs |
| RTX combine | 9 µs |
| **Whole layer** | **≈ 150 µs** (154 µs per row measured in `t20d`) |

Inside the round trip, for the slowest routed GPU (2026-11-02 profile): gate/up 39 µs, down 18–23 µs, the rest is the return over PCIe and the card-level sum. The slowest worker per request dropped from 146 µs to 103 µs with card-leader sums. The weights a GPU reads per token are only 2.5 MB (≈ 31 µs at bandwidth), so decode is **latency-bound**. 48 layers × ≈ 150 µs ≈ 7.4 ms of a 19 ms token.

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

KV capacity per configuration, measured by the vLLM start (runs `s1`–`s3`):

| KV bytes | Prefix caching | Tokens |
|---|---|---|
| 1 GiB | on | 14,745 |
| 1 GiB | off | 34,028 |
| 2 GiB | off | 75,296 |
| 2.5 GiB | off | 94,631 (needs `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`: the default 512 MiB indexer workspace ran out of memory in the first prefill) |

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
