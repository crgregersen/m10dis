// Host-memory K/V tier for the QSA layers (docs/research/KV_HOST_TIER_PLAN.md).
// q38_kv_resolve runs between the K/V write and the attention kernel. It maps the selected real token
// positions to positions inside a block cache (indexer blocks of 4 tokens), so the unchanged attention kernel
// reads what looks like a short context. Misses are copied from pinned host RAM; new tokens are written through.
//
// One 1024-thread block, per row: find the hits in parallel, allocate misses on thread 0 (CLOCK with a call
// epoch: anything selected in this call is never evicted in this call), copy misses, then store the new token.
#pragma once
#include <stdint.h>

#define Q38_KV_BLOCK_TOKENS 4u
#define Q38_KV_MAX_MISS 516u          // 512 selected blocks + the open group
#define Q38_KV_TOKEN_VEC 128u         // 2 KB per token in 16-byte vectors

struct Q38KvTierState {
    uint32_t enabled;                 // 0: copy the indices unchanged
    uint32_t slots;                   // cache blocks; slot s is the virtual token range 4s..4s+3
    uint32_t hand;
    uint32_t epoch;
    uint32_t nmiss;
    uint32_t overflow;                // CLOCK found nothing evictable: everything in the cache is pinned
    uint32_t pad[2];
    uint64_t calls, selected, misses, new_tokens, max_miss, clock_steps;
    int32_t miss_block[Q38_KV_MAX_MISS];
    int32_t miss_entry[Q38_KV_MAX_MISS];   // first selection entry, -1 for the open group
    int32_t miss_slot[Q38_KV_MAX_MISS];
};

struct Q38KvTierArgs {
    const int32_t *in;                // selection, row stride `width`, last column = count
    int32_t *out;                     // rewritten selection
    uint32_t width;                   // packed width (entries + count column)
    uint32_t rows;
    const int64_t *positions;         // row -> real position
    uint8_t *kv;                      // main K/V cache of this layer
    uint64_t block_stride;            // bytes between pool blocks
    uint32_t token_stride;            // bytes between tokens of one page
    uint32_t page_tokens;             // tokens per pool block
    const int32_t *block_table;       // logical page -> pool block
    uint8_t *host;                    // token position * 2 KB
    int32_t *block_to_slot;
    int32_t *slot_to_block;
    uint32_t *slot_epoch;             // 0 = evictable, current epoch = pinned, older = second chance
    Q38KvTierState *st;
    const uint8_t *inbox;             // phase 2: new K/V, row * 2 KB, already in host RAM; null: at the real position
};

__device__ __forceinline__ uint8_t *q38_kv_addr(const Q38KvTierArgs &a, uint64_t token) {
    const uint32_t page = uint32_t(token / a.page_tokens), off = uint32_t(token % a.page_tokens);
    return a.kv + uint64_t(a.block_table[page]) * a.block_stride + uint64_t(off) * a.token_stride;
}

__global__ static void q38_kv_resolve_kernel(Q38KvTierArgs a) {
    Q38KvTierState *st = a.st;
    const uint32_t t = threadIdx.x, T = blockDim.x;
    if (!st->enabled) {
        for (uint32_t r = 0; r < a.rows; r++) {
            const int32_t *in = a.in + uint64_t(r) * a.width;
            int32_t *out = a.out + uint64_t(r) * a.width;
            for (uint32_t j = t; j < a.width; j += T) out[j] = in[j];
        }
        return;
    }
    if (t == 0) { st->epoch++; st->calls++; }
    __syncthreads();
    const uint32_t epoch = st->epoch;
    for (uint32_t r = 0; r < a.rows; r++) {
        const int32_t *in = a.in + uint64_t(r) * a.width;
        int32_t *out = a.out + uint64_t(r) * a.width;
        const uint32_t entries = a.width - 1;
        const int32_t count = in[entries];
        const int64_t p = a.positions[r];
        if (t == entries % T) out[entries] = count;
        if (count <= 0) {                     // an inert row
            for (uint32_t j = t; j < entries; j += T) out[j] = -1;
            continue;
        }
        const uint32_t tail = uint32_t((p + 1) % Q38_KV_BLOCK_TOKENS);
        const int32_t expanded = count - int32_t(tail);
        const int32_t open_block = int32_t(p / Q38_KV_BLOCK_TOKENS);
        if (t == 0) {
            st->nmiss = 0;
            st->new_tokens++;
            st->selected += uint64_t(expanded) / Q38_KV_BLOCK_TOKENS;
        }
        __syncthreads();
        // Hits, in parallel. The epoch pins them for this whole call, across rows.
        for (uint32_t j = t; j < entries; j += T) {
            if (j >= uint32_t(count)) { out[j] = -1; continue; }
            const int32_t tok = in[j];
            const int32_t s = a.block_to_slot[tok / int32_t(Q38_KV_BLOCK_TOKENS)];
            if (s >= 0) {
                a.slot_epoch[s] = epoch;
                out[j] = s * int32_t(Q38_KV_BLOCK_TOKENS) + (tok & 3);
            } else if (int32_t(j) < expanded && (j & 3u) == 0) {
                const uint32_t m = atomicAdd(&st->nmiss, 1u);
                st->miss_block[m] = tok / int32_t(Q38_KV_BLOCK_TOKENS);
                st->miss_entry[m] = int32_t(j);
            }
        }
        __syncthreads();
        // Allocation, sequential.
        if (t == 0) {
            uint32_t n = st->nmiss;
            const int32_t os = a.block_to_slot[open_block];
            bool listed = false;              // a block completed by this token can also be selected
            for (uint32_t m = 0; m < n; m++) listed |= st->miss_block[m] == open_block;
            if (os >= 0) a.slot_epoch[os] = epoch;
            else if (!listed) { st->miss_block[n] = open_block; st->miss_entry[n] = -1; n++; }
            uint32_t hand = st->hand; uint64_t steps = 0;
            for (uint32_t m = 0; m < n; m++) {
                for (;;) {
                    steps++;
                    if (steps > 3ull * st->slots) { st->overflow++; break; }
                    const uint32_t e = a.slot_epoch[hand];
                    if (e == epoch) { hand = hand + 1 == st->slots ? 0 : hand + 1; continue; }
                    if (e != 0) { a.slot_epoch[hand] = 0; hand = hand + 1 == st->slots ? 0 : hand + 1; continue; }
                    break;
                }
                const int32_t old = a.slot_to_block[hand];
                if (old >= 0) a.block_to_slot[old] = -1;
                a.slot_to_block[hand] = st->miss_block[m];
                a.block_to_slot[st->miss_block[m]] = int32_t(hand);
                a.slot_epoch[hand] = epoch;
                st->miss_slot[m] = int32_t(hand);
                hand = hand + 1 == st->slots ? 0 : hand + 1;
            }
            st->hand = hand; st->nmiss = n;
            st->misses += n; st->clock_steps += steps;
            if (n > st->max_miss) st->max_miss = n;
        }
        __syncthreads();
        // The tail entries of an open block that just got a slot, and the missed selection entries.
        const uint32_t nmiss = st->nmiss;
        for (uint32_t m = t; m < nmiss; m += T) {
            const int32_t e = st->miss_entry[m], s = st->miss_slot[m];
            if (e >= 0) for (int32_t k = 0; k < int32_t(Q38_KV_BLOCK_TOKENS); k++) out[e + k] = s * int32_t(Q38_KV_BLOCK_TOKENS) + k;
        }
        const uint32_t vec = Q38_KV_BLOCK_TOKENS * Q38_KV_TOKEN_VEC;
        for (uint32_t i = t; i < nmiss * vec; i += T) {
            const uint32_t m = i / vec, v = i % vec;
            const uint64_t real = uint64_t(st->miss_block[m]) * Q38_KV_BLOCK_TOKENS + v / Q38_KV_TOKEN_VEC;
            const uint64_t virt = uint64_t(st->miss_slot[m]) * Q38_KV_BLOCK_TOKENS + v / Q38_KV_TOKEN_VEC;
            const uint32_t w = v % Q38_KV_TOKEN_VEC;
            reinterpret_cast<uint4 *>(q38_kv_addr(a, virt))[w] = reinterpret_cast<const uint4 *>(a.host + real * 2048u)[w];
        }
        __syncthreads();
        // Selected tail entries, then the new token: from where vLLM wrote it to its slot and to host RAM.
        const int32_t ns = a.block_to_slot[open_block];
        for (int32_t j = expanded + int32_t(t); j < count; j += int32_t(T))
            out[j] = ns * int32_t(Q38_KV_BLOCK_TOKENS) + (in[j] & 3);
        const uint64_t virt = uint64_t(ns) * Q38_KV_BLOCK_TOKENS + uint64_t(p & 3);
        const uint4 *from = reinterpret_cast<const uint4 *>(a.inbox ? a.inbox + uint64_t(r) * 2048u : q38_kv_addr(a, uint64_t(p)));
        uint4 *to = reinterpret_cast<uint4 *>(q38_kv_addr(a, virt));
        uint4 *host = reinterpret_cast<uint4 *>(a.host + uint64_t(p) * 2048u);
        for (uint32_t w = t; w < Q38_KV_TOKEN_VEC; w += T) {
            const uint4 x = from[w];
            to[w] = x;
            if (!a.inbox) host[w] = x;
        }
        __syncthreads();
    }
    __threadfence_system();
}

// Phase 1 check (Q38_KV_TIER_VERIFY=1): every selected entry of row 0 must read the same 2 KB through the cache as the true
// K/V of its real token: the pool at the real position, or host RAM below cache_tokens, where the pool now holds the cache.
__global__ static void q38_kv_verify_kernel(Q38KvTierArgs a, uint32_t cache_tokens, unsigned long long *checked, unsigned long long *bad) {
    if (!a.st->enabled) return;
    for (uint32_t r = 0; r < a.rows; r++) {
    const int32_t *in = a.in + uint64_t(r) * a.width, *out = a.out + uint64_t(r) * a.width;
    const int32_t count = in[a.width - 1];
    for (int32_t j = int32_t(blockIdx.x * blockDim.x + threadIdx.x); j < count; j += int32_t(gridDim.x * blockDim.x)) {
        const int32_t tok = in[j], v = out[j];
        const uint4 *x = reinterpret_cast<const uint4 *>(q38_kv_addr(a, uint64_t(v)));
        const uint4 *y = reinterpret_cast<const uint4 *>(uint32_t(tok) < cache_tokens ? a.host + uint64_t(tok) * 2048u : q38_kv_addr(a, uint64_t(tok)));
        uint32_t diff = v < 0 ? 1u : 0u;
        for (uint32_t w = 0; w < Q38_KV_TOKEN_VEC; w++) diff |= (x[w].x ^ y[w].x) | (x[w].y ^ y[w].y) | (x[w].z ^ y[w].z) | (x[w].w ^ y[w].w);
        atomicAdd(checked, 1ull);
        if (diff) atomicAdd(bad, 1ull);
    }
    }
}

// Phase 2: all rows' new K/V from the inbox to host RAM, before any row of this call can select them.
// A graph captured for more rows than the step has replays with inert rows (count 0, stale position): skip them.
__global__ static void q38_kv_write_through_kernel(const uint4 *inbox, uint8_t *host, const int64_t *positions, uint32_t rows,
        const int32_t *in, uint32_t width) {
    const uint32_t n = rows * Q38_KV_TOKEN_VEC;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint32_t r = i / Q38_KV_TOKEN_VEC, w = i % Q38_KV_TOKEN_VEC;
        if (in[uint64_t(r) * width + width - 1] <= 0) continue;
        reinterpret_cast<uint4 *>(host + uint64_t(positions[r]) * 2048u)[w] = inbox[i];
    }
    __threadfence_system();
}
