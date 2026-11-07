# K/V tier: sparse attention split between VRAM and host RAM

The 12 full-attention layers of Qwen3.8-Flash-Next keep K and V for every token, but each token attends to only about 2,048 of them. The K/V tier uses that: **every token's K/V lives in host RAM, and VRAM holds a cache of the blocks in use.** A context of 261k tokens then runs on the 8 GB RTX 3070 Ti, where the pool held about 95k before.

The tier is off by default and experimental. It needs no M10, daemon, switch, register, route or PTE change: it is a vLLM hook plus one CUDA kernel in the RTX library. Background and measurements: [research/KV_HOST_TIER_PLAN.md](research/KV_HOST_TIER_PLAN.md), [research/KV_ATTENTION_ON_M10.md](research/KV_ATTENTION_ON_M10.md).

## 1. Why it works

A full-attention layer (QSA) has an indexer that scores finished groups of 4 tokens and keeps the top 512 groups, plus the open group's 0–3 tail tokens. Attention then reads only those 2,048–2,051 tokens, exactly. The rest of the context is never read in that layer and step.

What a token reads changes slowly. Measured on 28k–81k prompts (`qsa_select_*`, pueue tasks 319–322):

| Measure | Result |
|---|---|
| Blocks kept from the previous token | 59–65 % |
| LRU hit, 16k tokens cached per layer | 98 % at 28k context, 93–95 % at 80k |
| LRU hit, 32k tokens cached per layer | 99 % at 28k, 97 % at 80k |
| Median distance of a selected token | 9k–74k tokens back |

So a VRAM cache of 32k–48k tokens per layer serves almost every read. The rest comes over PCIe from host RAM.

## 2. Where everything lives

```text
                 RTX 3070 Ti, 8 GB                                       host RAM
 ┌───────────────────────────────────────────────────────┐   ┌──────────────────────────────┐
 │ vLLM pool (small)                                     │   │ K/V store, pinned            │
 │   GDN state, compressed index keys, raw key ring,     │   │   every token, 12 layers     │
 │   MTP layer K/V                                       │   │   2 KB per token per layer   │
 │                                                       │   │   ≈ 6.5 GB at 262k           │
 │ K/V tier, per layer                                   │   │                              │
 │   block cache: C tokens × 2 KB  (C = 32k–48k)         │◄──┤  misses: 8 KB per block,     │
 │   tag tables: block → slot, slot → block, epoch       │   │  read by the RTX over PCIe   │
 │ shared: inbox (1,024 rows × 2 KB), selection shadow   │──►│  new K/V written through     │
 └───────────────────────────────────────────────────────┘   └──────────────────────────────┘
```

| Data | Where | Size at 262k |
|---|---|---|
| Main K/V of the 12 QSA layers | host RAM, every token | 24 KB per token, ≈ 6.5 GB pinned |
| Cache of the blocks in use | VRAM, one tensor per layer | 0.8 GB at 32k tokens per layer, 1.2 GB at 48k |
| Tag tables and state | VRAM | ≈ 3 MB |
| Inbox and selection shadow | VRAM, shared by all layers | ≈ 10 MB |
| GDN state, compressed index keys, raw key ring | vLLM pool, VRAM | ≈ 0.7 GiB at 0 drafts |
| MTP layer K/V and the extra GDN state copies for drafts | vLLM pool, VRAM | ≈ 0.8 GiB more at 2 drafts |
| Indexer scan | RTX | unchanged |

The pool shrinks because the QSA spec becomes a 16-dimension stub: vLLM then reserves almost nothing for main K/V, but keeps its page layout. At 262k, vLLM asks for 7.24 GiB today, against about 1.5 GiB with the stub at 2 drafts (`analyze_kv_layout.py`).

## 3. One attention call

The tier wraps two methods of vLLM's QSA attention: the K/V write and `forward_qsa`.

```text
 projections, indexer (unchanged)
      │
 1  K/V write ─────────► inbox (VRAM), one 2 KB row per token of this call
      │
 2  write-through ─────► host RAM at the token's position, before anything can select it
      │
 3  per sub-chunk of 16 rows (one sub-chunk in decode):
      q38_kv_resolve
        a. hits: look up each selected block, rewrite its index to a cache position
        b. misses: CLOCK picks a slot, the block is copied from host RAM
        c. the new token: from the inbox into its slot
      attention kernel (unchanged) on the rewritten indices and the tier's own page table
      │
 4  output gate, o_proj (unchanged)
```

The attention kernel does not change. It reads K/V through a token index and a page table, and the tier hands it rewritten indices and an identity page table over its own cache. To the kernel the cache looks like a short context.

Rules that keep it exact:

- **Pinning.** Every slot selected in a call carries the call's epoch, so CLOCK never evicts it in that call, across rows and sub-chunks. If nothing is evictable, `overflow` counts it: none occurred in any run.
- **Write before read.** All new K/V goes to host RAM in step 2, so any row can select a token written earlier in the same call.
- **Inert rows.** A graph captured for 4 rows replays a 3-row step with one inert row (count 0, stale position). Write-through and resolve skip it.
- **Immutable K/V.** A token's K/V never changes once written, so evictions need no write-back. A rejected draft's position is simply written again.
- **Sub-chunks in prefill.** A 1,024-row call selects most of the context (72–97 %, `qsa_prefill_union`), far more than the cache. Sixteen consecutive rows overlap enough to fit.

`q38_kv_resolve` is one 1,024-thread block per call. Hits are found in parallel. Allocation runs on one thread, because misses average about 3 % of 512 blocks per row. Copies are parallel again.

## 4. Using it

The tier needs graph decode or eager decode, any draft count up to the tested 2, and a single sequence.

| Setting | Meaning |
|---|---|
| `Q38_KV_TIER=1` + `Q38_KV_TIER_OWN=1` | turn the tier on, with K/V out of the pool |
| `Q38_KV_TIER_TOKENS` | cache per layer, multiple of 4 (default 32768) |
| `Q38_KV_TIER_SUBROWS` | rows per prefill sub-chunk (default 16) |
| `Q38_VLLM_MAX_MODEL_LEN` | up to 262144 |
| `Q38_VLLM_KV_CACHE_BYTES` | **set it small**: 0.8 GiB at 262k and 0 drafts, 1.55 GiB at 2 drafts. A large pool leaves no room for graph capture and runs out of memory |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=48` | with drafts in graph mode: the default 128 MiB workspace fails on fragmentation |
| `Q38_KV_TIER_VERIFY=1` | eager only: compares the 2 KB behind every checked entry with the true K/V |
| `Q38_KV_TIER_REPORT=FILE` | per-layer hits, misses, largest miss count, overflow |

Settings that fit, as tested:

| Context | Drafts | Cache per layer | Pool | Run |
|---|---|---|---|---|
| 120k | 0 | 32768 | 0.6 GiB | `kv_own_120k` v3 |
| 120k | 2 | 32768 | 1.15 GiB | `kv_own_k2_120k` v2 |
| 255k–261k | 0 | 49152 | 0.8 GiB | `kv_own_255k_speed` |
| 255k–261k | 2 | 32768 | 1.55 GiB | `kv_own_k2_255k` v2, `kv_own_chat_261k` |

Example, a 261k chat request with 2 drafts:

```bash
pueue add --group gpu --label <label> -- sudo -n env \
  Q38_RTX_PEER_PEX_OUTER=1 Q38_RTX_PEER_MOE_MULTIROW=1 Q38_RTX_PEER_MOE_MULTIROW_MAX=4 Q38_M10_LM_HEAD_ROWS2=1 \
  Q38_MTP_SPEC_TOKENS=2 Q38_KV_TIER=1 Q38_KV_TIER_OWN=1 Q38_KV_TIER_TOKENS=32768 \
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=48 Q38_VLLM_MAX_MODEL_LEN=262144 Q38_VLLM_KV_CACHE_BYTES=1664299827 \
  Q38_LONG_PROMPT=<prompt file> bash src/run_qwen38_decode_profile.sh
```

## 5. Results

| Context | Drafts | Prefill | Decode | Hit | Runs |
|---|---|---|---|---|---|
| short, no tier (reference) | 0 / 2 | 277 tok/s | 56.3 / 69.0 tok/s | – | T311, T313 |
| 120k | 0 | ≈ 245 tok/s | 52.7 tok/s | 98.6 % | T353 |
| 120k | 2 | ≈ 240 tok/s | ≈ 72 tok/s | 98.6 % | T360 |
| 255k | 0 | ≈ 237 tok/s | 51.4 tok/s | 97.9 % | T356 |
| 255k | 2 | ≈ 217 tok/s | ≈ 66 tok/s | 96.3 % | T361 |
| 261k, chat template | 2 | ≈ 210 tok/s | ≈ 68 tok/s | 95.7 % | T362 |

The decode rates with drafts are one 10-second window × tokens per step, from one run each. Prefill counts the whole run before the first token. Acceptance is prompt-dependent: 74–86 % at the first position, 52–74 % at the second.

Within the old pool limit, the tier's cost against the same context without it (78k, 0 drafts, 3 pairs, `kv_tier_graph_*`): +0.59 ms per token at 32k tokens per layer, +0.20 ms at 65k, +1.66 ms at 16k.

**Correctness.** The verify mode compares the 2 KB of K/V behind each checked entry with the true bytes:

| Run | Entries compared | Bad |
|---|---|---|
| 78k, cache inside the pool, all decode rows (`kv_tier_verify` v2) | 24,569,400 | 0 |
| 28k, K/V out of the pool, prefill and decode (`kv_own_verify` v3) | 46,411,800 | 0 |
| 120k, 2 drafts, all decode rows (`kv_own_k2_verify`) | 192,342,684 | 0 |
| 255k (`kv_own_255k_verify`) | 397,725,744 | 0 |

At 261k with the chat template, the model named the first and last file of the context and quoted the first line exactly (`kv_own_chat_261k`). Token identity is not a usable test here: two runs without the tier already diverge by token 21 at long context.

## 6. Limits and open points

- **Token-77 split.** On the short humaneval prompt, both tier runs diverge from the reference runs at the same token (`kv_own_H_k2`, `_k3`). The byte checks rule out wrong K/V. The likely cause is the unfused QSA path that the tier forces, tipping a near-tie, but a floor test has not been run.
- **Pool sized by hand.** A pool too large for the setting runs out of memory in graph capture.
- **First call is slow.** Warm-up compiles the stub shape, so the real attention kernel compiles at first use: about 3 s once per client start.
- **MTP K/V stays in the pool.** It is the main VRAM cost with drafts at 262k, and the reason the cache drops to 32k tokens there.
- **One sequence.** Tags, inbox and host store are per process, not per request.
- **Prefill past 128k** runs at about 210–245 tok/s; at 261k that is about 21 minutes, mostly the M10 MoE.
- **Two modes.** Without `Q38_KV_TIER_OWN`, the tier runs inside the normal pool (phase 1, decode only). That mode is now mainly a test harness.

## 7. Code

| File | Role |
|---|---|
| `src/qwen38_kv_tier.cuh` | `q38_kv_resolve`, write-through and verify kernels, tier state |
| `src/qwen38_vllm_rtx_peer.cu` | C exports: register host memory, resolve, write-through, verify |
| `src/qwen38_kv_tier_hook.py` | vLLM hooks: stub spec, inbox, sub-chunks, report; `Q38_KV_CONFIG_DUMP` |
| `src/bench_kv_tier.cu` | unit bench: replays logged selections with synthetic K/V and checks every entry |
| `src/qwen38_qsa_select_log.py`, `src/analyze_qsa_selection.py`, `src/analyze_qsa_bursts.py` | log real selections, analyze hit rate, balance and miss bursts |
| `src/analyze_kv_layout.py` | try vLLM K/V layouts offline from a config dump |
