> Design record written during development (2026-11-03). Numbers and run labels refer to `results/experiment-log.json`. The current system is described in [ARCHITECTURE.md](../ARCHITECTURE.md) and [TRANSPORT.md](../TRANSPORT.md).

# Batched MoE exchange — research (2026-11-03)

Goal: one RTX→M10 MoE exchange that carries many token rows (prefill chunks, MTP verification),
instead of one exchange per token per layer. Today prefill runs at ≈51 tok/s because every prompt
token pays a full exchange in each of the 48 layers (≈170 µs: publish 29 + round trip 132 + combine 9).

## Facts that constrain the design (measured or read from code)

| Fact | Value | Source |
|---|---|---|
| Expert placement | every expert is **sliced across all 20 routed GPUs**: each holds W=32 of the expert's intermediate rows for all E=512 experts (gate/up/down FP8 + scales) | `bench_resident_multilayer_fp8.cu` (`H=2560, W=32, K=10, E=512`) |
| Routed weights per GPU per layer | 512 × 32 × 2560 × 3 B ≈ **126 MB** (≈246 KB per expert slice) | same |
| Per token per layer per GPU | 10 experts × 246 KB ≈ **2.5 MB** read (≈31 µs at ~80 GB/s) — the 132 µs round trip is latency, not bandwidth | r28 + arithmetic |
| Down projection | partial sums over the intermediate dim: the 20 routed partials must be **summed** (today: on-card leader aggregation, then the RTX sums leaders + shared) | `resident_dynamic_routed_v3.cuh` |
| Request window (RTX→M10) | PEX multicast, 4 × 4 KiB pages per card group (page 0 + 0x8000–0xB000), one row of FP32 x (10 KB) + ids + routes | `qwen38_vllm_rtx_peer.cu` |
| Return window (M10→RTX) | per worker one `Q38CpuMailboxSlot` (10,260 B) remapped onto RTX BAR1, 3 × 4 KiB leaves | same (`kReturnPages = 3`) |
| PTE route records | **141 of 512** used (24 request + 117 return/aggregation); the 512 limit must not be raised (handover rule) | runner logs, `uvm_ioctl.h` |
| 64 KiB leaves | unusable: QUERY_REQUESTER_PTE frees the big-page table before the PTE write (extension bug, memory `rtx-64k-leaf-freed-table`) | code reading 2026-11-02 |
| RTX link | **PCIe Gen3 x4** (LnkCap x16; shows 2.5 GT/s when idle) ≈ 3.5 GB/s per direction — every byte to/from the RTX shares it | `lspci` |
| M10 card links | each M10 card connects to the outer PEX8748 by **Gen3 x4** (switch port LnkSta 8 GT/s x4); host ↔ outer switch x16 | `lspci -vv`, 2026-11-03 |
| Host-memory read from M10s (task 89, h1) | 1 GPU 3.2 GB/s; 4 GPUs of one card 3.35 GB/s in total; 6 GPUs (one per card) 11.7–12.0 GB/s; all 24 GPUs 9.4–10.0 GB/s | `run_qwen38_m10_host_read_bw.sh` |
| Shared host mailbox | 24 × 16 KiB slots in shm, `cudaHostRegister`ed in every worker (size compiled into the services: `Q38_CPU_MAILBOX_BYTES`) | `resident_cpu_mailbox*.cuh` |

Consequence: growing the PTE-remapped windows with 4 KiB leaves does not scale. R rows of BF16
returns need 1.3·R pages per returning worker: R = 32 already exceeds the 512-record budget.

## Compute model on the M10s (per routed GPU, per layer)

Distinct experts touched by R rows (top-10 of 512): D(R) = 512·(1 − (1 − 10/512)^R).

| Rows R | Distinct experts | Weight bytes per GPU | Time at ~80 GB/s | Per token |
|---|---|---|---|---|
| 1 | 10 | 2.5 MB | 31 µs (today 132 µs incl. latency) | 132 µs |
| 4 (MTP verify) | 39 | 9.6 MB | 0.12 ms | 30 µs |
| 32 | 239 | 59 MB | 0.73 ms | 23 µs |
| 64 | 365 | 90 MB | 1.12 ms | 18 µs |
| 256 | 508 | 125 MB | 1.56 ms | 6 µs |

Weight reuse only pays off with an **expert-major** kernel: for each touched expert, read its slice
once and apply it to every row routed to it (today's kernel is row-major: one row, its 10 experts).

## Transport options

| Option | Request (RTX→M10) | Results (M10→RTX) | New PTE records | Changes |
|---|---|---|---|---|
| **A. Host-staged bulk + PEX doorbell** | RTX writes R rows once into a shared pinned host buffer (5.2 KB/row BF16); the existing multicast carries only {seq, layer, mode, R, offset} | workers write partials to host result areas; reduction before the RTX (see below); RTX reads the reduced rows from host memory | **0** | new shared buffer registered in all workers (service change → daemon restart), worker mode, RTX op |
| B. Larger PEX windows, 4 KiB leaves | multicast pages per row | remapped return pages | ≈ 6·1.3R + 9·1.3R… | only viable for **R ≤ 4** (MTP): +3 request pages/group, returns via leaders only |
| C. Fix the 64 KiB-leaf extension bug | as B with 16× fewer records | as B | moderate | patched-UVM rebuild + module swap; risky, separate project |

Byte budget per layer for option A (R rows, BF16, ≈5.2 KB request and 5 KB result per row):

| Path | Volume | Link | R = 256 |
|---|---|---|---|
| RTX → host (request) | 5.2R KB | RTX Gen3 x4 | 1.3 MB → ≈0.4 ms |
| host → 24 M10 (each reads all rows) | 24 × 5.2R KB | PEX8748 uplink (Gen3 x16 ≈ 12 GB/s) | 32 MB → ≈2.7 ms |
| M10 → host (partials, after on-card sum: 5 routed leaders + 1 shared) | 6 × 5R KB | uplink | 7.7 MB → ≈0.6 ms |
| host → RTX (reduced results) | 6 × 5R KB (or 1 × with a full M10-side reduction) | RTX Gen3 x4 | 7.7 MB → ≈2.2 ms (0.4 ms if reduced to 1) |

Estimated per layer at R = 256: transport ≈3–6 ms + compute ≈1.6 ms (partly overlapped) →
≈ 48 × 5 ms ≈ 0.25 s per 256 tokens → **≈1,000 tok/s prefill (≈20× today)**, before optimizing the
host-read fan-out (one leader read per card + on-card P2P, or an outer-switch multicast group).
RTX side must batch too: prefill chunks of 256 (vLLM `max_num_batched_tokens`) so its weights are read
once per chunk; today's 8-token eager chunks cost ≈11 ms per token on the RTX alone.

## Recommended design

1. **Transport A** (host-staged bulk, PEX doorbell) for R ≥ 8 — no new PTE records, no UVM change.
2. **Worker mode 8 "MoE batch"** (routed v3 + shared v2): read R rows (BF16 x, ids, routes) from the host
   request area; build per-expert row lists; expert-major gate/up → act[row][k][32]; down projection
   accumulating per-row partials; on-card leader sum (existing aggregation pattern, R rows); leaders and
   the shared leader write R × 2560 BF16 to their host result areas; acknowledge via the existing slot
   header (count, status, seq).
3. **RTX op `PciClusterMoeBatchOp`** on the `pci_cluster` skeleton: write rows to the host area, doorbell,
   wait, sum the 6 result areas (+ shared gate per row) into the [R, 2560] output. Capturable.
4. **Client**: `RemoteMoEProxy` uses the batch op when rows > 1; `max_num_batched_tokens` 256 (memory
   check against the KV budget); decode (R = 1) keeps the low-latency PEX path.
5. **Daemon**: a second shared region (request ring R_max × 5.2 KB + 6 result areas R_max × 5 KB ≈ 9 MB
   for R_max = 256) registered by the services at load (ABI version bump) → one daemon restart.
6. **MTP later**: R = 4 can use either A or a small option-B window; measure both.

Consequence of the link measurements: the card's x4 link, not the host x16, limits a fan-out read. All 4
GPUs of a card reading the same R rows cost 4× the card link; R = 64 FP32 rows (655 KB) take ≈0.2 ms per
GPU at 3.3 GB/s shared. v1 accepts this (prefill only); a later step reads the rows once per card
(leader read + on-card copy).

## Implementation v1 (2026-11-03)

| Part | Where | Notes |
|---|---|---|
| ABI | `resident_cpu_mailbox_abi.cuh`, `qwen38_cpu_mailbox_manifest.py` | batch area after the 24 slots; FP32 rows (not BF16) so the worker input equals mode 1's; R ≤ 64 (BF16 values since k8, FP32 row stride) |
| Routed worker | `resident_routed_batch.cuh`, `resident_dynamic_routed_v3.cuh` | stage, counting-sort expert lists, gate/up one warp per (expert, row r of the slice), down terms per (row, k), k-ordered sum, leader adds peers in peer order; 96 registers, no spills in v1 (k16: 128 registers, 16 B spill of per-request scalars) |
| Shared worker | `resident_dynamic_shared_v2.cuh` | per row as mode 2/1; worker 20 sums 20..23 from 0 and writes gate[row]; 87 registers in v1 (k16: 123) |
| Services | `resident_*_persistent_multicast_experimental.cu` | `batch_area` at mailbox load / unload, routed device scratch |
| RTX op | `PciClusterMoeBatchOp`, `q38_rtx_peer_configure_moe_batch`, `q38_rtx_peer_moe_batch_gpu` in `qwen38_vllm_rtx_peer.cu` | rows to host, doorbell, wait 24 acks, check reducer flags, combine; capturable |
| Client | `qwen38_vllm_rtx_peer_adapter.py` (`_forward_gpu_batch`), `run_vllm_resident_m10_client.py` | `Q38_RTX_PEER_MOE_BATCH=1`, rows ≥ `Q38_RTX_PEER_MOE_BATCH_MIN_ROWS` |

Summation orders equal mode 1's, so every output row is bit-identical to a mode-1 call for that row
(verified offline: `run_q38_moe_batch_test.sh`, b2 task 91, R = 1/17/64, 0 mismatches of up to 655,360
values per GPU; invalid requests rejected with 16384 / 8192).

Hardware integration followed the same day (section below).

## Results on hardware (2026-11-03)

All numbers come from the experiment log (`results/experiment-log.json`). Every timing run compared
each path's output with the per-row mode-1 path: **bit-identical in every run** (diff 0.0).

### Integration fixes before the first batched run

| Run | Observed | Fix |
|---|---|---|
| i1 | mode 8 exit 0x2000 (config): `batch_area` = 0 | services skipped the control write while workers were parked; now written on mailbox load in any state |
| i2 | still 0x2000; `batch_scratch` 0 on worker 0 | the CPU-driven warm-up INFER copies the whole control struct and zeroed the appended fields; the payload copy now stops before them |
| i3 | tokens identical; batched 30% **slower** at chunk 64 (88.4 vs 115.4 tok/s) | led to the timing harness and the kernel work below |

### Kernel optimization (R = 64, ms per layer call, real router ids from L12 from t8 on)

| Step | Change | Batch | Per row | Batch gain | Note |
|---|---|---|---|---|---|
| t1 | first version | 14.75 | 10.78 | 0.73x | stage + gate/up 6.2 ms |
| t4 (k1) | gate/up tiles | 13.29 | 10.81 | 0.81x | serial list builder: R = 1 stage 166 µs |
| t5 (k2) | stage in shared memory | 11.34 | 10.76 | 0.95x | |
| t6 (k3) | reduce on the RTX instead of the card | 12.97 | 10.76 | 0.83x | 13.8 MB through RTX Gen3 x4: card reduce stays |
| t7 (k4) | streamed on-card reduce | 8.07 | 10.77 | 1.33x | finish ≈ 20 µs per row |
| t9 (k5) | fast FP8 decode | 7.27 | 10.79 | 1.48x | gate/up ≈ 43 GB/s: weight-read bound |
| t11 (k7) | BF16 tiles | 6.92 | 10.72 | 1.55x | gate/up 1.11 ms |
| t12 (k8) | BF16 request x | 6.54 | 10.70 | 1.64x | stage latency bound (≈ 6 GB/s) |
| t14 (k10) | format word once per block | 6.53 | 10.74 | 1.64x | R = 1 stage 126 → 64 µs |
| t16 (k12) | parallel expert lists | 6.56 | 10.71 | 1.63x | list build 140 → 5 µs |
| t18 (k14) | mode 1 BF16 request | 6.51 | 9.66 | 1.49x | per-row path 10% faster too (decode +4%, d3) |
| t19 (k15) | shared expert BF16 tiles | 6.38 | 9.65 | 1.51x | shared busy 2.84 ms, off the critical path |

Tried and reverted: contiguous gate/up unit ranges (k6, no gain), 16-byte `__ldcv` stage loads (k9, within
noise), leader in-place sum with a bulk host copy in finish (k17: 864 vs 727 µs per 32-row chunk).

Fit over t7 + t8 (µs, entries = 10R): gate/up ≈ 18 + 1.2·distinct + 2.1·entries, down ≈ 18 + 1.75·distinct +
1.2·entries. The per-entry work outweighs the weight reads, so bigger batches gain less than the expert-reuse
model above predicted. Real routing reuses more than random: 3.65 rows per expert at chunk 64 (random 1.75).

### Pipelining (mode 9, k16)

R = 64 is the batch-area limit, so a 1024-row vLLM step was 16 serial mode-8 exchanges with the RTX publish
and combine on the critical path. Mode 9 splits the area into two 32-row slots and keeps two chunks in flight
(layout and protocol: `../INTERFACE.md`, mode 9).

| R | Pipe (ms per layer call) | Batch | Per row | Pipe vs batch |
|---|---|---|---|---|
| 32 | 3.12 | 3.11 | | 1.00x |
| 64 | 5.79 | 6.16 | | 1.06x |
| 256 | 19.54 | 24.54 | | 1.26x |
| 1024 (t20d) | 72.89 | 98.40 | 157.23 | 1.35x |

At R = 1024 the chunk period (2213 µs) equals the worker critical path: the RTX side is fully hidden. Per
32-row chunk on the M10s: gate/up 0.67 ms, down 0.54, finish 0.72, x copy 0.21.

### End-to-end prefill (4097-token prompt, vLLM steps of 1024)

| Run | Per row (mode 1) | Batched (mode 8) | Pipelined (mode 9) |
|---|---|---|---|
| i3 (chunk 64) | 115.4 tok/s | 88.4 | |
| i4 (chunk 1024) | 118.6 | 164.7 | |
| i5 (BF16 tiles) | 118.7 | 182.4 | |
| i6 (BF16 request) | 119.8 | 197.1 | |
| i7 (k16) | | 204.8 (20.69 s) | **273.7** (15.65 s) |
| i11 (autotune off) | | 203.8 / 204.8 | 272.6 / 273.7 |

Prefill went from ≈ 51 tok/s (8-token eager steps, design start) to ≈ 274 tok/s. Decode keeps mode 1.

### Determinism

- The M10 exchange is deterministic by construction (fixed summation orders) and in test. Stress run t22b:
  20 passes × 48 layers × 1024 rows, pipe (30,720 chunks) and batch (15,360 exchanges), each layer with a
  different input. **0 mismatching rows** against mode 1 (64-bit row checksums of the BF16 output).
- End-to-end prompts gave two token sequences: eb7d0aab4915 and, in 2 of 5 pipelined runs (i8, i10),
  1744d27ab3a0. It happened with k16 and k17 alike, so the finish change did not cause it.
- Triton autotuning is not the cause: with `VLLM_TRITON_FORCE_FIRST_CONFIG=1` i12 still flipped 4 of 10 runs, all at
  generated token 6. Logprobs (i13, `Q38_VLLM_LOGPROBS=5`) show why: token 6 is a near-tie (margin 0-0.625, exact BF16
  ties in two runs) and token 11 nearly one (0.375-1.25), while every run's logits differ, already at token 0
  (prefill only, margin 2.25-3.125).
- Floor runs (operator method: the LM-head jitter of identical repeats is the tolerance; `src/lm_head_floor.py`): pipelined (i13, 10 runs) max 2.13 / median 0.31, per row (i14,
  5 runs) max 2.29 / median 0.25, per row x pipelined max 2.25 / median 0.31. The batched MoE adds no jitter of its
  own; the variation is in the RTX vLLM kernels (attention). End-to-end comparisons use the floor, not token hashes.

### Open levers

| Lever | Expected | Blocker |
|---|---|---|
| Finish handshake (0.72 ms per chunk, latency bound) | part of 0.72 ms per chunk | a multi-row leader window needs more PTE records (the window holds one FP32 row) |
| Prefetch x of chunk n + 1 during chunk n | tried, no gain | k19 (copy during finish): stage 212 → 41 µs but finish +190 µs (host reads slow the credit polling); k20 (copy inside down): the next doorbell is not there yet at down start and the extra code spilled (mode 1 down 22.5 → 42 µs). Reverted to k16 (k21). Worth retrying with k19 once finish no longer polls host memory |
| Larger slots (raise `Q38_BATCH_MAX_ROWS`) | fewer chunks, more expert reuse | ABI size change, daemon restart |

## Risks

- Host uplink fan-out (24 GPUs reading the same rows) may dominate; mitigation: per-card leader read +
  on-card P2P copy (uses existing aggregation routes) or an outer PEX8748 multicast group.
- Expert-major kernel on GM107 (5 SMs, 10-block persistent grid): needs care to keep all SMs busy with
  uneven per-expert row counts.
- RTX Gen3 x4 link is shared by request, results and the zero-copy embedding.
- Service ABI change → daemon restart (operator approval).
