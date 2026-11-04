# Lessons

What didn't work, what surprised us and what the fix was. Labels refer to `results/experiment-log.json`.

## Hardware and boot

- **The limit at boot is I/O port space, not memory space.** The firmware has 64 KiB of legacy port space and hands out 4 KiB per bridge. Six M10 cards with their internal switches and an I/O BAR per GM107 don't fit. The GPUs don't need port I/O at all: a kernel quirk now drops BAR5, and the late cards get bridge windows with `hpiosize=0`.
- **Powering cards on after boot and rescanning isn't enough.** Link disable, secondary-bus reset and retraining gave 2 or 3 of a late card's 4 GPUs. A deep S3 suspend/resume with the second PSU switched on during the sleep initializes all four reliably.
- **Clocks.** The M10s run capped by the application-clocks setting (1137–1189 of 1202 MHz). Parked workers drop to 135/405 MHz, so every timing run needs a warm-up and a clock check first, or the first numbers are meaningless.
- **Swapping the UVM module needs every `/dev/nvidia*` user gone.** That includes the fan controller (it reads GPU temperatures) and any `nvtop` left open.

## Page-table remapping

- **64 KiB leaves don't work.** The UVM query path frees the big-page table before the PTE write, so writes through a remapped 64 KiB leaf never arrive. Everything uses 4 KiB leaves. The external owner window still has to be at least 64 KiB.
- **The record table is the real budget.** 512 remapped PTEs in total, 141 used by a decode session. Batched prefill would need about 1.3 pages per row per returning worker, so it moves its bulk data through pinned host memory and sends only a doorbell over the remapped pages.
- **Processes die with routes applied.** vLLM kills its engine 5 s after shutdown starts, which left 141 records behind once. The fixes: close the adapter inside the engine process, raise `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` to 120, journal every route so the daemon can restore it (`recover-peer-client`), and have UVM release records whose VA space is gone (lifecycle patch, `REAP_DEAD`).
- **Every worker must acknowledge, including on error.** An early row-gather mode acknowledged only in its local record. The RTX waited 2 s for an acknowledgement that went elsewhere. Every exit path now writes `status`, the exit bits and `seq` into the RTX-side slot.

## Worker kernels

- **The cost was parallelism, not synchronization.** The first profile blamed barriers and polling. The measurement showed barriers at about 13 µs and polling at 4 µs, against 174 µs of gate/up running 10 blocks on a 5-SM GPU. Restructuring gate/up (v2) brought it to 39 µs.
- **The shared expert was the critical path first**: 303 µs, then 76 µs with v2. After that the routed workers were.
- **Card-leader sums beat everything else for the return.** The slowest worker dropped from 146 µs to 103 µs. Summing all 24 partials on the RTX instead pushes about 13.8 MB per 64-row call through its x4 link and was slower (0.83×, `t6`).
- **Prefetching the next chunk's rows didn't pay.** Copying during the finish phase (`k19`) cut the stage from 212 µs to 41 µs, but finish grew by 190 µs: the extra host reads slowed the credit polling that finish depends on. Moving the copy into the down phase (`k20`) found the next doorbell not there yet, and the extra code spilled registers (mode-1 down 22.5 → 42 µs). Reverted.
- **A leader sum in place with one bulk host copy (`k17`)** was slower than streaming each row (864 vs 727 µs per chunk). Reverted.
- **Derive shared slot numbers from the sequence, on both sides.** The first pipelined version took the slot from the chunk index on the RTX and from `seq & 1` on the workers. An odd base sequence made all 24 workers read the wrong header (`t20b`).

## vLLM side

- **Token hashes don't tell you whether a change hurt.** Identical runs produce different logits, because the RTX kernels (attention in particular) aren't reduction-order deterministic. A near-tie at one position then flips the rest of the text. Turning off Triton autotuning didn't change that (`i12`). Comparing runs by their LM-head logprob jitter against a floor of identical repeats does work. See [RESULTS.md](RESULTS.md#determinism).
- **Prefix caching costs context.** With the GDN layers, vLLM's Mamba "align" mode stores a state snapshot every 1,568 tokens: 72.8 KB per token against 31.6 KB without. 1 GiB of KV holds 14,745 tokens with prefix caching and 34,028 without.
- **The sparse-attention indexer reserves 512 MiB by default.** At 2.5 GiB of KV that ran out of memory in the first prefill. It needs only rows × ⌈context/4⌉ × 4 B (about 80 MiB at 82k context), so `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` fixes it.
- **Time small kernels inside a CUDA graph.** The FP8 hyper-connection kernels looked 2× slower than BF16 when timed eagerly: the timings were launch-bound. Graph-replay timing plus a tuned M = 1 GEMV made them about 2× faster than BF16 instead.

## Measurement hygiene

- **The first progress window lies.** A decode rate taken from the first 10-second window includes the prefill and showed 25 tok/s against a steady 52.8.
- **Pre-register the expectation.** Every benchmark had its threshold written down before it ran, so a score couldn't be explained after the fact.
- **Root-owned run folders.** The runners need root, so their folders are root-owned and an unprivileged scorer couldn't write into them. Scores now go to a separate folder.
- **No automatic retries.** Every failure in this project carried information. A retry loop would have hidden at least the vLLM shutdown, the indexer workspace and the slot-numbering bugs.
