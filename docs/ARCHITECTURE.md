# Architecture

## The model

Qwen3.8-Flash-Next-FP8 (`model_type qwen4_exp_text`), 173 GB of FP8 checkpoint:

| Property | Value |
|---|---|
| Layers | 48: 36 linear-attention (Gated DeltaNet), 12 full attention (every 4th, with a sparse top-k indexer, "QSA") |
| Hidden size | 2560 |
| MoE | 512 routed experts per layer, 10 per token, intermediate size 640; one shared expert (intermediate 640) |
| Hyper-connections | 4 streams, low-rank 320 (`GatedResidual`, 2 per layer + 1) |
| N-gram embeddings ("PLE") | hashed 3-gram table, about 51 GB, used at layer 2 |
| Vocabulary | 248,320 tokens, untied LM head (2560 × 248,320) |

Almost all the bytes are in the experts: about 121 GB of routed weights against a few GB of everything else. The experts are also the part that is cheap per token: 10 of 512 per layer.

## What runs where

| Component | Device | Notes |
|---|---|---|
| Token embedding | CPU RAM | read by the RTX zero-copy (`Q38_CPU_EMBEDDING=1`, `Q38_EMBEDDING_ZERO_COPY=1`) |
| GDN and attention layers, QSA indexer, router, norms | RTX 3070 Ti | vLLM 0.32, CUDA graphs; GDN, QSA and hyper-connection weights in FP8 (block-scaled, `qwen38_fp8_block_quant.py`, `qwen38_hc_fp8.py`) |
| KV cache and GDN state | RTX | 2.5 GiB, 94,631 tokens, prefix caching off |
| Routed experts | 20 GM107 (cards 0–4 in worker order) | each GPU holds 32 of the 640 intermediate rows of every expert in every layer: 512 × 32 × 2560 × 3 B ≈ 126 MB per layer, ≈ 6 GB per GPU |
| Shared expert | the 4 GM107 of the sixth card | worker 20 holds the card's reduced result and the shared gate |
| N-gram embedding table | all 24 GM107 | 128 shards spread by PCI address; rows fetched per token with worker mode 6 (row gather) |
| LM head | all 24 GM107 | one row slice of 10,346–10,347 vocabulary rows (≈ 53 MB) per GPU; worker mode 7 returns each slice's top-K candidates and its log-sum-exp |
| Sampling | RTX | over the 24 × K candidates (exact for greedy and top-k ≤ K) |

The split follows the hardware:

- The RTX has fast tensor cores and 8 GB. It gets the dense, latency-critical work.
- The GM107s have 192 GB together but no FP16/FP8 math, about 1.67 TFLOPS FP32 and about 43 GB/s achieved memory bandwidth each. They get the large, sparse, bandwidth-bound work, and they never talk to each other except within a card.

## Why the experts are sliced, not placed

The obvious layout puts whole experts on GPUs: 512 experts / 20 GPUs ≈ 26 each. A token's 10 experts would then land on at most 10 GPUs, and some GPUs would get several of them, so the slowest GPU sets the pace.

m10dis slices every expert across all 20 routed GPUs instead: GPU *g* holds rows 32·g … 32·g + 31 of the gate, up and down projections of **every** expert. Every token then costs every GPU the same work: 10 experts × 32 rows. The price is that the down projection produces 20 partial sums per token that have to be added up. That's what the card leaders do (see [TRANSPORT.md](TRANSPORT.md)).

## The resident daemon

`src/qwen38_resident_daemon.py` owns the M10s for the whole session:

- It loads the 48 layers of expert slices (from pre-packed bundles), the n-gram shards and the LM-head slices once, and keeps them in M10 memory. Loading takes several minutes. After that, vLLM clients can start and stop without reloading anything.
- It runs one routed service per routed GPU and one shared service per shared GPU (`resident_routed_service.cu`, `resident_shared_service.cu`, built from `resident_*_persistent_multicast_experimental.cu`). Each service keeps a **persistent kernel** resident that polls its request window.
- It parks the workers between clients (idle clocks), wakes them for a session and can **reload the worker kernels** without touching the weights (`RELOAD_KERNELS`).
- It journals every route and register change and can recover after a client crash (`RECOVER_PEER_CLIENT`).

Commands and state: [INTERFACE.md](INTERFACE.md).

## The vLLM side

`src/run_vllm_resident_m10_client.py` starts vLLM with hooks loaded through `sitecustomize.py`:

- `qwen38_vllm_rtx_peer_adapter.py` replaces the MoE layer's forward with a call into `libqwen38_vllm_rtx_peer.so` (built from `qwen38_vllm_rtx_peer.cu`). That library publishes requests, waits for acknowledgements and combines, all as CUDA kernels on vLLM's stream, so the whole step can be captured in a CUDA graph.
- `qwen38_rtx_fp8_hooks.py` and `qwen38_hc_fp8.py` swap the GDN, QSA and hyper-connection weights for FP8 versions and their kernels.
- `qwen38_cpu_head_hooks.py` moves the embedding to CPU memory and the LM head to the M10s.

## One decode token through one MoE layer

1. The RTX computes attention or GDN, then the router: 10 expert ids and 10 weights.
2. **Publish** (≈ 29 µs): a kernel writes the BF16 hidden state, ids and weights into each card's 4 KiB request pages. These pages are the M10s' own memory, mapped into the RTX's address space. The write goes RTX → PEX8748 → the card's PEX8747, which **multicasts** it to the card's four GPUs.
3. **Compute**: each routed worker sees the new sequence number, runs gate/up (≈ 39 µs) and down (≈ 18–23 µs) for its 32-row slices of the 10 experts. The shared workers run the shared expert.
4. **Card sum**: the three peers of each card push their partial rows to the card leader over on-card P2P. The leader adds them in a fixed order.
5. **Return**: leaders and shared workers write their result rows straight into RTX memory, through UVM page-table entries remapped onto the RTX's BAR1. Every worker then writes its acknowledgement (`seq`) into its return slot. Steps 3 to 5 take about 132 µs.
6. **Combine** (≈ 9 µs): the RTX waits for 24 acknowledgements, then computes Σ 5 routed sums + gate · shared.

At about 150 µs per layer, the 48 MoE layers take about 7.4 ms of the 19 ms decode step. The rest is the RTX's own layers and the LM-head exchange.

## Prefill

One row per exchange would make prefill as slow as decode: about 51 tok/s at the start of the project. Prefill therefore uses **batched exchanges** (worker modes 8 and 9):

- The RTX writes up to 64 rows (hidden states, ids, weights) into a pinned **host-memory batch area** that every worker has registered. Only a small doorbell goes through the multicast windows. That keeps the number of remapped page-table entries constant, and the patched UVM limits those to 512.
- The workers process the rows expert-major (all rows that picked expert *e* together), reduce on the card and leave the reduced rows in host memory.
- Mode 9 pipelines this: two 32-row slots, so the RTX publishes chunk *n + 1* and combines chunk *n − 1* while the M10s compute chunk *n*.

That brings prefill to about 277 tok/s. The outputs are bit-identical to the single-row path. Design record: [research/BATCHED_MOE_EXCHANGE.md](research/BATCHED_MOE_EXCHANGE.md).
