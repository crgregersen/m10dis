# Plan: host-memory K/V tier for the full-attention layers

**Status: Phases 0 and 1 done. Phase 2 not started.** Goal: decode at up to 262k tokens on the 8 GB RTX, with K/V in host RAM and a VRAM cache of the blocks in use. Numbers marked *(estimate)* are calculations. Background: [KV_ATTENTION_ON_M10.md](KV_ATTENTION_ON_M10.md), runs `qsa_select_*` (T319–T322).

## The idea

The vLLM attention kernel reads K/V through two levels: the selected token index, then a page table with pages of 1,568 tokens (`ops/qsa.py`). Pages are far too coarse for a tier, because 512 scattered blocks touch most pages. Token indices are not:

```text
 indexer ─► selected positions ─► [RESOLVE] ─► cache positions ─► unchanged attention kernel
             1040 1041 1042 1043               208  209  210  211
             52000 ...                         12   13   14   15
                                  │
                 hit: rewrite the index only
                 miss: copy the 4-token block from host RAM into a cache slot, then rewrite
```

Only one kernel is new. The attention kernel stays as it is and reads what looks like a short context. `RESOLVE` runs in the CUDA graph between steps 3 and 5 of the attention layer, the same way the zero-copy embedding gather already reads pinned host RAM without a graph break (`q38_zero_copy_gather`).

## What does not change

No M10 kernel, daemon restart, register, PEX, ACS, route or PTE change is needed. The RTX library is rebuilt per run, and vLLM is patched by hooks (`qwen38_*_hooks.py`), not edited. So a failure ends in a client error, not in M10 cleanup.

## New pieces

| Piece | Where | Does |
|---|---|---|
| Host K/V store | pinned RAM, `q38_zero_copy_register` | K and V for every token, 12 layers, 24 KB per token |
| Tag table | VRAM, per layer | block → cache slot, slot → block, CLOCK bits |
| `q38_kv_resolve` | CUDA kernel in `qwen38_vllm_rtx_peer.cu` | per row: look up 512 blocks, fill misses from host, pick victims, rewrite indices |
| K/V write hook | Python wrapper on `_run_qsa` | steers the new token's K/V to its cache slot and writes it through to host RAM |
| Settings | env | `Q38_KV_TIER=1`, `Q38_KV_TIER_TOKENS` (cache per layer) |

The newest group's tail tokens must stay in the cache. They are pinned against eviction until their group is complete.

## Phase 0 results

All on the 78–81k prompts unless noted. Times are for misses alone, at 8 KB per block and 3.2 GB/s.

| Cache per layer | Mean per token | p99 | Worst token | Tokens over 2 ms | Source |
|---|---|---|---|---|---|
| 16k tokens, CLOCK | 0.86–1.1 ms | ≈ 3.7 ms | ≈ 5 ms | 68–107 of 949 | replay of T321, T322 |
| 32k tokens, CLOCK | 0.44–0.52 ms | 2.3–2.4 ms | 2.9–3.8 ms | 17–18 of 949 | replay of T321, T322 |
| 32k tokens at 28k context, CLOCK | 0.12–0.13 ms | ≈ 1 ms | 2.2–3.0 ms | 1–2 of 949 | replay of T319, T320 |

CLOCK is within about 5 % of LRU, so the kernel can use it.

Prefill union, distinct tokens selected per 1,024-row call, per layer (`qsa_prefill_union`, T323):

| Context at the call | Mean union | Largest | Share of context |
|---|---|---|---|
| 10–20k | 15.4k | 20.2k | 97 % |
| 30–40k | 30.8k | 37.7k | 85 % |
| 50–60k | 42.5k | 54.6k | 76 % |
| 70–78k | 53.7k | 69.3k | 72 % |

A prefill call touches most of the context, so **a decode-sized cache cannot hold one call**. This does not affect Phase 1, because prefill stays as it is up to the pool size.

Page layout: main K/V and compressed index keys share pool blocks (the 1,568-token attention group, packed per block). Capping main K/V would also cap the keys.

## Phase 1 results

The tier ran end to end in vLLM on the 12 main attention layers, in the CUDA graph. The pool stayed unchanged, so this is real tier behavior at context the pool could also hold.

| Context, drafts | Cache per layer | Hit | Without tier | With tier | Cost | Runs |
|---|---|---|---|---|---|---|
| 78k, 0 | 16k tokens (384 MB) | 92.7 % | 55.0 tok/s | 50.4 tok/s | +1.66 ms per token | T344 vs T327, T329, T331 |
| 78k, 0 | 32k tokens (768 MB) | 97.5 % | 55.0 tok/s | 53.3 tok/s | +0.59 ms per token | T326, T328, T330 |
| 78k, 0 | 65k tokens (1.5 GB) | 99.4 % | 55.0 tok/s | 54.4 tok/s | +0.20 ms per token | T345 |
| 60k, 2 | 32k tokens | 98.5 % | 33.1 steps/s | 32.0 steps/s | +1.04 ms per step, ≈ +0.35 ms per row | T340–T343 |

Correctness: 0 bad of 24,569,400 selected entries, comparing the 2 KB of K/V against the true data on every token and layer (`kv_tier_verify` v2, T333). The logprob floor is not usable here: two untiered runs share only 20–30 tokens, and the maximum already reaches 4.1 (T327, T329, T331).

Long-context acceptance at 2 drafts stayed high on this prompt: 87 / 74 % per position, 2.6 tokens per step, about 85 tok/s. The prompt's review task quotes its own text, so this is not a general figure.

Costs run a little above the simulation: misses move at about 2.2 GB/s rather than 3.2 GB/s. The copy cost dominates: `RESOLVE` itself is small.

## Phases

Each phase has a gate. Each hardware run needs a go-ahead, gets a PIPELINE record, and has no automatic retry.

### Phase 0: data, no kernel work (≈ 1 session)

1. **Prefill union.** Extend the logger to record, per prefill call and layer, the number of distinct blocks selected across the 1,024 rows. Run at 80k.
2. **Burst simulation.** Replay T319–T322 per token: misses per token, worst token, CLOCK against LRU.
3. **Read the group layout.** Find which `kv cache groups` hold main K/V, and whether they can be capped without touching the GDN state pages (see Phase 2).

**Gate:** the union decides prefill (see below), and the simulation decides whether the tail is tolerable.

### Phase 1: decode tier at ≤ 90k, pool unchanged (the real test)

Everything fits the pool, so allocator work is avoided. Prefill runs as today. At prefill end, K/V is copied to host RAM (≈ 1.9 GB, under 1 s *(estimate)*), and the first `C` tokens of each layer's K/V act as the cache. The rest of the pool sits unused, so this behaves exactly like a tier without changing vLLM's accounting.

| Step | Check |
|---|---|
| 1. `q38_kv_resolve` plus a unit bench on the RTX with synthetic K/V | correct bytes after hits, misses, evictions; time per call |
| 2. Eager decode with the hook | hit rate against the replay; no errors |
| 3. CUDA graph decode | floor method against untiered repeats; tok/s pairs A/B at 80k |
| 4. Sizes and drafts | `C` = 16k, 32k, 65k; drafts 0 and 2 |

**Gates:**

- Results within the floor of untiered repeats (`src/lm_head_floor.py`). Token identity is the wrong test: two untiered runs already diverge at token 21 (T321, T323).
- Measured hit rate matches the simulation within 1 point.
- Cost at 80k, `C` = 32k: at most +1 ms per row. That is +0.55 ms expected, plus `RESOLVE`, which I can't yet size (+0.1–0.3 ms *(estimate)*).

If these fail, we stop here, having spent no effort on the allocator.

### Phase 2: allocator cap, context above the pool

vLLM reserves pages for every token of a request, so 262k needs more than the pool. A cap can't work, because main K/V shares blocks with the compressed keys. The approach instead:

- Shrink the QSA layers' `FullAttentionSpec` so pool blocks carry only the compressed keys.
- The tier owns K/V in its own tensor and hands the attention kernel its own identity page table.
- vLLM then picks a new block size, because it matches attention and GDN page sizes. A startup test shows whether that works, without generating.
- HiSparse (`v1/hisparse/layout.py`) stays the fallback reference.

**Gate:** a start at `max_model_len` 131072, then a 120k run. Above the pool there is no untiered reference, so correctness rests on the phase 1 floor and the kernel bench.

### Phase 3: prefill and 262k

Phase 0 settled this: the union is far above `C`, about 65 % of context by 262k *(estimate)*. So prefill attention must run in sub-chunks of rows, each resolved separately, inside `forward_qsa`.

- Each call must move its union at least once. At 262k that is about 170k tokens × 24 KB ≈ 4 GB per 1,024 tokens, ≈ 1.3 s over x4 *(estimate)*, against 3.7 s per call today.
- Late prefill therefore runs at about 200 tok/s at best *(estimate)*, and slower where sub-chunks evict each other. A full 262k prefill would take about 20–25 minutes.

Then a full run with `max_model_len` 262144.

**Gate:** decode cost and acceptance at 262k, compared with the table below.

## Expected result *(estimates)*

| Context | Cache per layer | VRAM | Decode, 0 drafts | Decode, 2 drafts |
|---|---|---|---|---|
| Up to ≈ 80k | 32k tokens | 768 MB | ≈ 54.6 tok/s | ≈ 65.6 tok/s |
| 262k | 45–55k tokens | 1.0–1.25 GiB | ≈ 53.3 tok/s | ≈ 63.0 tok/s |

References today: 56.3 and 69.0 tok/s at short context (`ksweep_N_k0`, `ksweep_N_k2`, T311, T313). `RESOLVE` isn't included.

## Risks

| Risk | Phase | If it happens |
|---|---|---|
| Shrinking the K/V spec breaks vLLM's page layout | 2 | Tier stays useful only up to the pool, ≈ 95k: stop or use HiSparse layout |
| Prefill union far above the cache | 0: **confirmed** | Sub-chunks in Phase 3; prefill slows above the pool size; decode unaffected |
| `RESOLVE` too slow on one kernel | 1 | Parallelize lookups; misses copied by a second kernel |
| Tail latency on topic jumps | 1 | CPU attends the misses and LSE-merges, as a later step |
| x4 contention with M10 traffic | 1 | Shows up in the A/B; size the cache larger |
| MTP acceptance falls at long context | 1, 3 | Lower the draft count at long context |

## Effort *(estimate)*

| Phase | Size | Comment |
|---|---|---|
| 0 | small | logger and simulation |
| 1 | medium | one kernel, one hook, careful ordering in the graph |
| 2 | unknown until read | the real risk |
| 3 | small to large | depends on the prefill union |

Phases 0 and 1 are easy to medium and answer the main question: does a tier hold its numbers on real hardware, with identical tokens?
