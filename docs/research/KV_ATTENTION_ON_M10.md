# K/V cache and attention on the M10s

**Status: design study, nothing here is implemented or measured as a system.** Numbers marked *(estimate)* are calculations from tensor shapes and from measured exchange times. Numbers with a run label or a file are measured or read from that source.

## Updates since this was written

The sections below keep the first estimates. Later work changed three of them:

- **Cost.** Rebuilt on measured phases, a layer is about 95–100 µs, not 80 µs, or +0.65–0.70 ms per token *(estimate)*.
- **×3 replication.** It may not be needed: 8 GPUs at ×1 give the same context and the same read per GPU, at about +5–10 µs per layer *(estimate)*. A 12-head kernel benchmark would settle this.
- **Rival.** A host-memory K/V tier with a VRAM cache of 16k tokens per layer (384 MB) reached 97.8–98.5 % hits at 28k context (T319, T320). That costs about 0.2–0.3 ms per token *(estimate)* and needs no M10 kernels. At 78–81k context (T321, T322) 16k tokens fall to 93.4–94.9 %, and 32k tokens (768 MB) reach 96.9–97.3 %, about 0.4–0.5 ms per token *(estimate)*. The needed cache grows with context, but more slowly than the context. The tier is now built and measured: +0.59 ms per token at 78k with a 32k cache, 53.3 against 55.0 tok/s (T326–T331). See [KV_HOST_TIER_PLAN.md](KV_HOST_TIER_PLAN.md).

Details: `reports/Parallel attention across GPUs.md` in the planning workspace.

## Summary

- Move only the QK/softmax/PV kernel of the 12 full-attention layers and their K/V cache to the M10s. The projections, the indexer and the output gate stay on the RTX.
- Split K/V two ways: **by KV head across cards** (each head on 3 cards) and **by indexer block inside a card** (4 GPUs).
- Each query head is computed on exactly one card, so no merge is needed between cards, and the shared RTX x4 link carries 24 KB per layer instead of 144 KB.
- A layer costs **≈ 75–85 µs** *(estimate)* against **≈ 42 µs** on the RTX today (`QSA ~0.5` ms per token over 12 layers, run `decode_profile_nsys` v4, T200).
- The result is about **+0.45 ms per token (+3 %)** *(estimate)*: slower, not faster. What it buys is context: **≈ 200k tokens on the 8 GB RTX** *(estimate)*.

## 1. Where things are today

```text
 CPU RAM                 RTX 3070 Ti, 8 GB                           24 GM107, 8 GB each
┌─────────────┐         ┌──────────────────────────────┐            ┌──────────────────────┐
│ token       │  read   │ GDN + QSA projections (FP8)  │  x4 link   │ routed experts (FP8) │
│ embedding   │────────►│ hyper-connections (FP8)      │◄══════════►│ shared expert (BF16) │
│ (BF16)      │         │ router, norms, indexer       │  shared by │ n-gram table (FP8)   │
└─────────────┘         │ K/V CACHE 2.5 GiB (BF16)     │  all M10   │ LM head (BF16)       │
                        │ GDN state (FP32)             │  traffic   │                      │
                        │ attention math (BF16)        │            │ ≈ 0.8 GB free each   │
                        └──────────────────────────────┘            └──────────────────────┘
```

Per-token time today, 0 drafts, ≈ 17.7 ms:

| Part | Time | Source |
|---|---|---|
| RTX dense work | ≈ 10.0 ms | T200 |
| of which QSA kernels (12 layers) | ≈ 0.5 ms, ≈ 42 µs per layer | T200 |
| of which GDN and QSA projection GEMMs (FP8) | 5.39 ms (96 calls × 56 µs) | T200 |
| M10 MoE exchange | ≈ 6.4 ms (133 µs × 48 layers) | `pubgo_merge`, T293 |
| LM head | ≈ 1 ms | estimate |

The QSA kernels are about 3 % of a token. That is why moving them cannot help speed much either way.

## 2. One attention layer, today

Shapes are from the checkpoint (`layers.11.self_attn.*`). The head mapping is from `vllm/models/qwen4_exp/nvidia/ops/qsa.py` (`first_head = kv_head * GROUP_SIZE`).

| Item | Shape | Size |
|---|---|---|
| Hidden input | 2560 | |
| Q + gate (`q_proj`) | 12288 outputs: 24 heads × 256 (Q) + 24 × 256 (gate) | 31.5 MB weights |
| K, V (`k_proj`, `v_proj`) | 512 outputs each: 2 KV heads × 256 | 1.3 MB weights each |
| Output (`o_proj`) | 6144 inputs: 24 heads × 256 | 15.7 MB weights |
| Query heads per KV head | 12: heads 0–11 use KV 0, heads 12–23 use KV 1 | |
| Indexer | 512 blocks of 4 tokens → 2,048 tokens + up to 3 tail tokens | `indexer_budget`, `indexer_compress_ratio` |
| K + V per token, per layer | 2 KV heads × (K 512 B + V 512 B) | 2 KB (BF16) |

```text
 hidden (1 token)
   │
   ├─► q_proj ──► Q (24 heads × 256) ───────────────┐
   │        └───► gate (24 × 256) ───────────────┐  │
   ├─► k_proj, v_proj ──► new K, V ─► written    │  │
   │                                  to cache   │  │
   └─► index_qk_proj ──► scan compressed keys    │  │
                             │                   │  │
                             ▼                   │  ▼
                       top 512 blocks ───────► ┌─────────────────────────────┐
                       (2,048 tokens)          │  ATTENTION KERNEL           │
                                               │  read ≈ 4 MB of K/V         │
              K/V cache ─────────────────────► │  QK → softmax → PV          │
                                               │  ≈ 50 Mflop (estimate)      │
                                               └──────────────┬──────────────┘
                                                              ▼
                                           output × gate ─► o_proj ─► hidden
```

The projections can't move: they need about 50 MB of weights per layer, and they would read them every token. Only the box can.

## 3. How a new token enters the caches

This section follows one token through one full-attention layer, in execution order. It comes from `vllm/models/qwen4_exp/nvidia/qsa.py` (`forward`, `_run_qsa`), `indexer_qsa.py` (`forward`) and `ops/qsa_indexer.py`.

### 3.1 The four stores of a full-attention layer

| Store | Written per token | Holds | Size per token, per layer | Read by |
|---|---|---|---|---|
| Main K cache | every token | 2 KV heads × 256, after RMSNorm and RoPE | 1 KB (BF16) | attention (QK), selected tokens only |
| Main V cache | every token | 2 KV heads × 256, no norm, no RoPE | 1 KB (BF16) | attention (PV), selected tokens only |
| Raw key ring | every token | the last few raw index keys, 128 dims | fixed size, per sequence | the compressor |
| Compressed key cache | every 4th token | 1 row per finished 4-token group, 128 dims | 256 B per group, 64 B per token | indexer scan, **every visible row, every token** |

K is stored already rotated for its own position, so reading it needs no RoPE. V is never rotated.

```text
 token position   ...  20  21  22  23 | 24  25  26  27 | 28  29  30
                       ── group 5 ────|── group 6 ─────|── open ──
 main K, V             ■   ■   ■   ■  | ■   ■   ■   ■  | ■   ■   ■      one entry per token
 compressed key        ■■■■■■■■■■■■■■ | ■■■■■■■■■■■■■■ |  (none yet)   one entry per finished group
 raw key ring                                            ■   ■   ■      only the open group, overwritten
```

### 3.2 Before the cache: from hidden state to Q, K, V

```text
 HC mix (FP8)  ─► block input, 2560 values
                    │
   qkv_proj         │  index_qk_proj (640 outputs)
   (FP8, 13.3k      │  BF16
    outputs)        │
     │              └─► index Q: 4 heads × 128     raw index K: 1 × 128
     ▼
   Q + gate (24 × 256 × 2)     K (2 × 256)     V (2 × 256)
     │                          │               │
   RMSNorm(q_norm) + RoPE     RMSNorm(k_norm)   │ nothing
     │    gate stays raw        + RoPE          │
     ▼                          ▼               ▼
   Q, gate                   K ready         V ready
```

RoPE touches a quarter of each head's dimensions (`partial_rotary_factor` 0.25, 64 of 256) and uses the token's absolute position.

### 3.3 Inside `_run_qsa`: the order of the cache operations

```text
 STEP  OPERATION                                                    TOUCHES
 ────  ──────────────────────────────────────────────────────────   ─────────────────────
  1    Compress: if this token finishes a group, average the 4      raw ring (read, write)
       raw keys, RMSNorm(k_layernorm), RoPE at the group's first    compressed cache (write)
       position, and store one row. Store the raw key in the ring.

  2    Select: score every visible compressed row against the       compressed cache (read
       4 index Q heads:                                              all rows)
          score(row) = Σ over heads of max(Q_h · key_row, 0)
       Keep the top 512 rows.

  3    Expand: 512 blocks × 4 tokens = 2,048 token positions,       none, indices only
       plus the open group's tail (0-3 tokens, always included).

  4    Write the new token's K and V at its position.               main K, V (write)

  5    Attend, for each query head h:                               main K, V (read
          s_i  = softmax_scale · Q_h · K_i                           2,048-2,051 tokens)
          out_h = Σ softmax(s) · V_i      FP32 accumulate

  6    out_h × sigmoid(gate_h)  ─►  o_proj  ─►  MoE stage            none
```

Three consequences of this order:

- **The token sees itself only through the tail.** K and V are written in step 4, before attention in step 5, and the tail contains the token unless it just finished a group. When it finishes one, its group is a normal candidate in step 2 and may not be selected.
- **A token joins the search late.** Tail tokens are always attended. Once its group finishes, it is reachable only by being selected. Median selected age was 9.0k–21.5k tokens in `qsa_select_docs` and `qsa_select_code` (T319, T320), so selection does reach far back.
- **Step 2 is the part that grows with context.** It scans N/4 rows on every token and in every layer. Step 5 stays at 2,051 tokens or fewer at any context.

vLLM can also run steps 1 and 4 together in one fused kernel (`qsa_prepare`), so K and V may be written before selection instead of after it. The result is the same, because both happen before step 5.

Some sizes for a 32k context, one token, one layer *(estimate)*: the scan reads 8k rows × 256 B = 2 MB, the attention reads 4 MB of K/V.

With drafts, each row has its own selection. Later MTP steps reuse the rows frozen at the first step (`skip_topk`), and a rejected draft's entries are overwritten at the same position.

### 3.4 What moves to the M10s

```text
 STEP                         TODAY            WITH THE DESIGN
 ───────────────────────────  ───────────────  ─────────────────────────────────────────────
 Projections, norm, RoPE      RTX              RTX
 1 Compress                   RTX              RTX
 2 Select (scan, top-k)       RTX              RTX: 12 × N/4 × 256 B of keys stay in VRAM
 3 Expand                     RTX              RTX sends block ids; each GPU keeps its own
 4 Write K, V                 RTX              RTX sends K, V in the request; the owner stores
 5 Attend                     RTX              M10s, then on-card merge
 6 Gate, o_proj               RTX              RTX, after the result returns
```

Step 4 is the only cache write that crosses PCIe. The request carries the new token's K and V (2 KB per row, per layer) by multicast:

```text
 RTX request for position 26, layer 11 (block 6, GPU slot 2)
 ┌───────────┬─────────┬─────────────────────┬───────────────┬───────────────┐
 │ header    │ Q       │ selected block ids  │ K0 V0  (1 KB) │ K1 V1  (1 KB) │
 │ layer,    │ 12 KB   │ 512 × 4 B = 2 KB    │               │               │
 │ position  │         │ + tail from position│               │               │
 └───────────┴─────────┴─────────────────────┴───────────────┴───────────────┘
        │  multicast to all 24 GPUs
        ▼
 each GPU, in this order:
   1. block 6 mod 4 == my slot, and my card owns this KV head?  store K, V at layer base + offset
   2. take my 4 query heads from Q
   3. keep the selected blocks with block mod 4 == my slot, plus the tail if it is mine
   4. attend over those blocks        (the new K, V are already in place)
```

The write must finish before the read. In one request that is plain ordering on the GPU that owns the block, and no other GPU needs that data, because the same GPU reads what it stored. The tail is included in this rule: all tail tokens share one block, so one GPU per card serves it.

**At the M10 level, the cache is simple.** One sequence means no block table: address = layer base + position-derived offset. vLLM's allocator would still reserve main K/V on the RTX, so a hook has to shrink or skip that allocation (see the open questions).

**Prefill follows the same order with up to 1,024 rows per call.** All rows' K and V are stored before attention, because a row can select earlier tokens from its own chunk. 1,024 rows × 2 KB is 2 MB per layer, far more than the 28 KiB request window. The likely path is mode 9's existing host-memory staging, which copies 32 rows in 0.21 ms (`docs/RESULTS.md`, prefill breakdown). The volume itself is small: about 25 MB per 1,024 tokens over 12 layers *(estimate)*.

## 4. Choosing the split

The box takes Q + selected positions and returns 24 heads × 256 values. The kernel on the RTX already works in splits: each split computes a partial output and a log-sum-exp (LSE), and they are merged exactly afterwards. That is what makes a split across GPUs possible.

| Split | Replication | x4 traffic, per layer per row | Problem |
|---|---|---|---|
| Every GPU holds everything | ×24 | small | memory: 576 KB per token |
| Across the 256 K/V dimensions | ×1 | huge | all 2,048 × 24 scores must be gathered before the softmax |
| Positions across 24 GPUs | ×1 | 6 cards × 24 KB = 144 KB, ≈ 45 µs | every card returns all 24 heads |
| Query heads by card, positions inside a card (**chosen**) | ×3 | 6 cards × 4 KB = 24 KB, ≈ 7.5 µs | K/V stored three times |

*(all estimates, at ≈ 3.2 GB/s on the x4 link)*

## 5. K/V placement

**Rule 1, KV head → cards.** Each KV head is stored on 3 cards, and each of those cards serves 4 of its 12 query heads:

```text
                KV head 0                                  KV head 1
        (K0, V0 stored three times)                (K1, V1 stored three times)

   ┌────────┐ ┌────────┐ ┌────────┐         ┌────────┐ ┌────────┐ ┌────────┐
   │ card 0 │ │ card 1 │ │ card 2 │         │ card 3 │ │ card 4 │ │ card 5 │
   ├────────┤ ├────────┤ ├────────┤         ├────────┤ ├────────┤ ├────────┤
   │ Q 0-3  │ │ Q 4-7  │ │ Q 8-11 │         │ Q12-15 │ │ Q16-19 │ │ Q20-23 │
   └────────┘ └────────┘ └────────┘         └────────┘ └────────┘ └────────┘
```

**Rule 2, indexer block → GPU inside the card.** Block number = token position ÷ 4; GPU = block mod 4. All four tokens of a block stay on one GPU, so a selected block is one contiguous read. All cards use the same rule.

```text
 token position   0  1  2  3 | 4  5  6  7 | 8  9 10 11 |12 13 14 15 |16 17 18 19 | ...
 block            ---- 0 ----|---- 1 -----|---- 2 -----|---- 3 -----|---- 4 -----|
 GPU on the card     GPU 0   |   GPU 1    |   GPU 2    |   GPU 3    |   GPU 0    |
```

**Together, for one attention layer:**

```text
                      GPU 0            GPU 1            GPU 2            GPU 3
                 blocks 0,4,8..   blocks 1,5,9..   blocks 2,6,10..  blocks 3,7,11..
              ┌────────────────┬────────────────┬────────────────┬────────────────┐
 card 0  Q0-3 │  K0 V0         │  K0 V0         │  K0 V0         │  K0 V0         │
 card 1  Q4-7 │  K0 V0         │  K0 V0         │  K0 V0         │  K0 V0         │  KV head 0
 card 2  Q8-11│  K0 V0         │  K0 V0         │  K0 V0         │  K0 V0         │
              ├────────────────┼────────────────┼────────────────┼────────────────┤
 card 3 Q12-15│  K1 V1         │  K1 V1         │  K1 V1         │  K1 V1         │
 card 4 Q16-19│  K1 V1         │  K1 V1         │  K1 V1         │  K1 V1         │  KV head 1
 card 5 Q20-23│  K1 V1         │  K1 V1         │  K1 V1         │  K1 V1         │
              └────────────────┴────────────────┴────────────────┴────────────────┘
```

Example: position 26 is in block 6, and 6 mod 4 = 2, so its K0/V0 is on the third GPU of cards 0, 1 and 2, and its K1/V1 on the third GPU of cards 3, 4 and 5.

**Inside one GPU**, one array per attention layer, with no lookup table:

```text
 layer 3 array on card 0, GPU 2 (holds blocks 2, 6, 10, ...)
 ┌──────────────────────────────┬──────────────────────────────┬─────
 │ block 2                      │ block 6                      │ ...
 │ K pos 8..11  V pos 8..11     │ K pos 24..27  V pos 24..27   │
 │ 4 × 512 B    4 × 512 B       │ 4 × 512 B     4 × 512 B      │
 └──────────────────────────────┴──────────────────────────────┴─────
 address = layer base + (block ÷ 4) × 4 KB
```

**What stays on the RTX:** the compressed key cache, the raw key ring, GDN state, and everything in section 3.4 marked RTX.

## 6. How the computation runs

```text
  RTX                                    x4 link          M10 cards
  ───                                    ───────          ─────────
  project Q, gate, new K/V
  indexer picks 512 blocks
       │
  1 ───┴─ multicast: Q (12 KB) ─────────────────────────► all 24 GPUs receive
          block ids (2 KB)                                 Q, blocks, new K/V
          new K/V (2 KB)
                                                          each GPU:
  2                                                       ─ keep its own Q heads (4)
                                                          ─ keep its blocks (≈ 128)
                                                          ─ store new K/V if the block
                                                            is its own

  3                                                       read ≈ 512 KB of K/V
                                                          for each of 4 heads:
                                                            s = softmax_scale · Q·K
                                                            L = log2 Σ 2^s
                                                            O = Σ 2^(s − L) · V

  4                                                       3 peers push O + L to the
                                                          card leader (one PEX8747,
                                                          nothing on the x4 link)

  5                                                       leader merges per head:
                                                            L* = log2 Σ 2^L_p
                                                            O* = Σ 2^(L_p − L*) · O_p

  6   ◄──────────────────────── 6 × 4 KB + headers ────── leaders write O* to RTX VRAM

  7   assemble 24 heads (each from one card, no sum)
      output × gate ─► o_proj ─► next layer
```

Step 5 gives the same result as the softmax over all selected tokens. It is the merge that the RTX kernel does between its splits, and the kernel uses base 2 (scores are scaled by 1.44269), so both sides must agree on the base. Between cards nothing is merged, because a query head lives on one card only.

Compute per GPU, per layer, per row *(estimate)*: 128 blocks × 4 tokens = 512 tokens × 1 KB = 512 KB read, and 4 heads × 512 × 256 × 2 dot products ≈ 2 Mflop. Reading dominates, as it does for the experts.

## 7. Time per layer

| Phase | Time *(estimate)* | Scaled from |
|---|---|---|
| Publish Q, blocks, new K/V | ≈ 9 µs | 4.1 µs for 5 KB (`pubgo_merge`) |
| Validate + request delivery | ≈ 15 µs | decode breakdown (`diag_ref_newkern`, `prof_on`) |
| Read, dots, softmax per GPU | ≈ 15–20 µs | gate/up 1.64 MB in 33.8 µs (`prof_on`) |
| Peer push + flag | ≈ 12 µs | 13.3 µs (`prof_on`) |
| Leader merge + write + ack | ≈ 12 µs | 7.2 + 5.1 µs (`prof_on`) |
| x4 crossing | ≈ 10–15 µs | decode breakdown |
| **Whole layer, 1 row** | **≈ 75–85 µs** | |
| RTX today | ≈ 42 µs | T200 |

```text
 RTX today  │████████████████████                                    42 µs
 On M10s    │█████████████████████████████████████████               ≈ 80 µs
             publish│ deliver │ compute │ push │ merge │ x4
```

Per token: 12 layers × ≈ +38 µs ≈ **+0.45 ms**, or about +3 % on 17.7 ms *(estimate)*. Almost all of the extra time is transport, not attention math.

With drafts, Q, the output and the compute all scale with rows. Expect ≈ +30 µs per extra row *(estimate)*, the same order as the RTX, which also pays per row. That needs a measurement.

Decode speed stops depending on context length, apart from the indexer scan: the selection is 2,048 tokens at any context.

## 8. Memory

K/V per token: 2 KB × 12 layers = 24 KB once, **72 KB** with three copies. That is 3 KB per GPU on average.

| Context | Per GPU | All M10s | Fits *(estimate)* |
|---|---|---|---|
| 32k tokens | ≈ 100 MB | ≈ 2.3 GB | yes |
| 100k tokens | ≈ 300 MB | ≈ 7 GB | yes |
| 200k tokens | ≈ 600 MB | ≈ 14 GB | yes, ≈ 0.8 GB free per GPU |
| 262k tokens (model limit) | ≈ 790 MB | ≈ 19 GB | no safe margin |

FP8 K/V would halve this, but only after the accuracy question in the FP8 discussion is settled.

On the RTX, the 2 GiB or so of attention K/V goes away. What remains is the GDN state (≈ 164 MiB per copy, 1 + drafts copies, fit from `ksweep_*`) and the compressed keys (64 B per token per layer, 0.77 KB over 12 layers, ≈ 200 MB at 262k *(estimate)*), plus the small raw key ring.

| Way to get context | Context | Decode cost | Work |
|---|---|---|---|
| Raise `max_model_len` | ≈ 67–95k | none | a config change and a test |
| 16 GB RTX | up to the 262k model limit | none | a new card |
| FP8 K/V | ≈ 134–190k | ≈ none | accuracy testing; Ampere support is open |
| **This design** | **≈ 200k** | **≈ +3 %** | **large** |

*(all estimates)*

## 9. Open questions

1. **Prefill.** Each 32-row slot must carry K/V for all rows, and the request pages may not fit that. New K/V rides in the request so that no new routes are needed: the 512 PTE and 64-route limits stay untouched.
2. **Worker kernels.** A new mode in the register-sensitive persistent kernels, probably a separate noinline path, as the profile code had to be.
3. **Balance.** Measured on real selections: block mod 4 gives 1.06× on average and 1.22× at worst (`qsa_select_docs`, `qsa_select_code`, T319, T320). This is no longer a risk.
4. **Softmax on GM107.** `exp` in FP32 over 512 tokens × 4 heads needs a benchmark.
5. **MTP.** A rejected draft's K/V is overwritten at the same position, so rollback should be free.
6. **Shared card.** Card 5 takes part like the others. Its free memory needs checking.
7. **vLLM allocation.** Main K/V pages are packed beside the indexer pages in each pool block. Keeping the indexer caches while dropping main K/V needs a hook in the cache setup.

## 10. Cheapest next step

A benchmark on one M10 card, in the style of `bench_m10_expert_compute`: 4 GPUs, the partial QK/softmax/PV kernel on synthetic K/V, and the on-card merge. It would replace the two largest guesses in section 7. No RTX or daemon change is needed.
