// Routed worker, mode 8: batched MoE (2026-11-03; docs/BATCHED_MOE_EXCHANGE_RESEARCH.md).
//
// One request carries R <= Q38_BATCH_MAX_ROWS token rows (x, 10 expert ids, 10 routes per row) in the
// host batch area of the shared CPU mailbox. The worker computes its W=32-row slice of every expert
// EXPERT-MAJOR: each expert slice touched by the batch is read once and applied to every (row, k) that
// routes to it. The arithmetic per (row, k) is exactly mode 1's (same per-lane loop order, same warp
// reduction, same fmaf chain, same k-ordered final sum), so every row is bit-identical to a mode-1
// request with that row.
//
// Phases (grid barriers between): stage (rows -> device scratch, per-expert lists) | gate/up | down terms
// | per-row k sum | reduction (card leader: own + peers in peer order, like the mode-1 aggregation).
// Results: every worker writes its rows to its own part area, then a flag (u32 seq). A leader waits for
// its peers' flags, adds their rows and leaves the sum in its own area. Workers without aggregation are
// their own reducers. The RTX reads only the reducers' areas.
#pragma once
#include "resident_batch_layout.cuh"


struct Q38BatchView {
    char *batch;          // host batch area (worker's device pointer)
    unsigned slot;        // request slot: 0 for mode 8, seq & 1 for mode 9 (resident_cpu_mailbox_abi.cuh)
    float *x_stage, *act, *pd;
    int32_t *ids; float *routes;               // device copies [R][K]
    unsigned *count, *start, *expert_of, *entry, *n_distinct;   // per-expert lists
    unsigned *pstart;     // gate/up work units (pairs of entries) per distinct expert, prefix [nd+1]
    unsigned *pstart4;    // the same for 4-row units (BF16 x tiles), prefix [nd+1]
    unsigned *x_inexact;  // seq of the last request whose x was not exactly BF16 (then 2-row FP32 tiles)
};

template <class Control>
__device__ __forceinline__ Q38BatchView q38_batch_view(const Control *c, unsigned slot = 0) {
    Q38BatchView b;
    b.batch = q38_batch_area(c);
    b.slot = slot;
    char *s = reinterpret_cast<char *>(static_cast<const volatile Control *>(c)->batch_scratch);
    b.x_stage = reinterpret_cast<float *>(s + Q38_BATCH_SCRATCH_X);
    b.act = reinterpret_cast<float *>(s + Q38_BATCH_SCRATCH_ACT);
    b.pd = reinterpret_cast<float *>(s + Q38_BATCH_SCRATCH_PD);
    unsigned *l = reinterpret_cast<unsigned *>(s + Q38_BATCH_SCRATCH_LISTS);
    b.count = l; b.start = l + 512; b.expert_of = l + 1025; b.entry = l + 1537;
    b.n_distinct = l + 1537 + Q38_BATCH_MAX_ROWS * Q38_BATCH_K;
    b.ids = reinterpret_cast<int32_t *>(b.n_distinct + 1);
    b.routes = reinterpret_cast<float *>(b.ids + Q38_BATCH_MAX_ROWS * Q38_BATCH_K);
    b.pstart = reinterpret_cast<unsigned *>(b.routes + Q38_BATCH_MAX_ROWS * Q38_BATCH_K);
    b.pstart4 = b.pstart + 513;
    b.x_inexact = b.pstart4 + 513;
    return b;
}

// Every thread, before the validation barrier: returns reason bits (0 = safe to run). R and its bound come from
// control->ple_count (mode 8, <= MAX_ROWS) or the slot header (mode 9, <= PIPE_ROWS).
template <class Control>
__device__ unsigned q38_batch_validate_routed(const Control *c, unsigned layer, int E, unsigned R, unsigned max_rows, unsigned slot) {
    const volatile Control *v = c;
    if (R < 1 || R > max_rows || !v->batch_area || v->cpu_mailbox_worker >= Q38_CPU_MAILBOX_WORKERS ||
        !(v->cpu_mailbox_reserved & Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN) || layer >= 48 ||
        !v->batch_scratch || v->batch_scratch_bytes < Q38_BATCH_SCRATCH_BYTES)
        return Q38_BATCH_REASON_CONFIG;
    const char *bt = q38_batch_area(c);
    const volatile int32_t *ids = reinterpret_cast<const volatile int32_t *>(bt + q38_batch_req_ids_off(slot));
    const volatile float *routes = reinterpret_cast<const volatile float *>(bt + q38_batch_req_routes_off(slot));
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < R * Q38_BATCH_K) {
        const int e = ids[t]; const float r = routes[t];
        if (e < 0 || e >= E || !(r == r) || r > 3.0e38f || r < -3.0e38f) return Q38_BATCH_REASON_ROWS;
    }
    return 0;
}

// List area: count 512, start 513, expert_of 512, entry/ids/routes R*K each, n_distinct 1, pstart 513,
// pstart4 513, x_inexact 1 words.
static_assert((512 + 513 + 512 + Q38_BATCH_MAX_ROWS * Q38_BATCH_K * 3 + 1 + 513 + 513 + 1) * 4 <= 64 * 1024, "batch lists exceed the scratch list area");
// Phase 1 (all threads): stage x rows into device scratch; block 0 builds the expert lists. v2 (2026-11-03):
// block 0 first copies ids/routes from the host request into shared memory in parallel (sm, >= E + 2*R*K
// words; the kernel's shared x buffer, unused until gate/up), then thread 0 builds the lists from shared
// memory. v1 did 2*R*K serial host reads and E serial global reads on one thread (166 us at R=1).
// Block 0, all threads: the per-expert lists from sid[0..n) (2026-11-03, parallel build). Counts by shared-memory
// atomics; one block-wide exclusive scan over the experts gives each expert's entry offset, its distinct index and
// both unit prefixes (2-row pstart, 4-row pstart4); entries are then placed with atomics. The order of the
// entries inside one expert is not fixed (it only changes which rows share a gate/up tile; every entry's
// arithmetic is independent of its tile neighbours, so results are bit-identical). cnt: E words of shared memory.
// The v1 build ran on one thread: ~30 us + ~1.7 us per row (140 us at R=64).
__device__ void q38_batch_build_lists(const Q38BatchView &b, unsigned n, int E, unsigned *cnt, const unsigned *sid) {
    __shared__ unsigned wsum[32][4];
    const unsigned tid = threadIdx.x, nt = blockDim.x, lane = tid & 31, warp = tid >> 5, nwarp = (nt + 31) >> 5;
    for (unsigned e = tid; e < unsigned(E); e += nt) cnt[e] = 0;
    __syncthreads();
    for (unsigned t = tid; t < n; t += nt) atomicAdd(&cnt[sid[t]], 1u);
    __syncthreads();
    const unsigned per = (unsigned(E) + nt - 1) / nt, e0 = min(tid * per, unsigned(E)), e1 = min(e0 + per, unsigned(E));
    unsigned v[4] = {0u, 0u, 0u, 0u};                // entries, distinct experts, 2-row units, 4-row units
    for (unsigned e = e0; e < e1; e++) { const unsigned c = cnt[e]; v[0] += c; v[1] += c ? 1u : 0u; v[2] += (c + 1) >> 1; v[3] += (c + 3) >> 2; }
    unsigned inc[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        inc[k] = v[k];
        for (unsigned o = 1; o < 32; o <<= 1) { const unsigned y = __shfl_up_sync(0xffffffffu, inc[k], o); if (lane >= o) inc[k] += y; }
    }
    if (lane == 31) for (int k = 0; k < 4; k++) wsum[warp][k] = inc[k];
    __syncthreads();
    if (tid < 4) { unsigned run = 0; for (unsigned w = 0; w < nwarp; w++) { const unsigned x = wsum[w][tid]; wsum[w][tid] = run; run += x; } }
    __syncthreads();
    unsigned pos = wsum[warp][0] + inc[0] - v[0], d = wsum[warp][1] + inc[1] - v[1];
    unsigned u2 = wsum[warp][2] + inc[2] - v[2], u4 = wsum[warp][3] + inc[3] - v[3];
    for (unsigned e = e0; e < e1; e++) {
        const unsigned c = cnt[e];
        if (c) { b.expert_of[d] = e; b.start[d] = pos; b.pstart[d] = u2; b.pstart4[d] = u4; d++; }
        cnt[e] = pos; pos += c; u2 += (c + 1) >> 1; u4 += (c + 3) >> 2;
    }
    if (tid == nt - 1) { b.start[d] = pos; b.pstart[d] = u2; b.pstart4[d] = u4; *b.n_distinct = d; }   // totals
    __syncthreads();                                  // every cnt[e] holds its start offset
    for (unsigned t = tid; t < n; t += nt) b.entry[atomicAdd(&cnt[sid[t]], 1u)] = t;   // (row * K + k), grouped by expert
}

// Returns (block 0 thread 0) the list-build duration in ns, for the worker timing record. Block 0 builds the
// lists first (they need only the ids) and then takes its share of the x copy.
__device__ unsigned long long q38_batch_stage(const Q38BatchView &b, unsigned R, int E, int H, unsigned *sm, unsigned seq) {
    unsigned long long lists_ns = 0;
    if (!blockIdx.x) {
        const volatile int32_t *hid = reinterpret_cast<const volatile int32_t *>(b.batch + q38_batch_req_ids_off(b.slot));
        const volatile float *hr = reinterpret_cast<const volatile float *>(b.batch + q38_batch_req_routes_off(b.slot));
        const unsigned n = R * Q38_BATCH_K;
        unsigned *cnt = sm, *sid = sm + E;
        float *srt = reinterpret_cast<float *>(sm + E + n);
        for (unsigned t = threadIdx.x; t < n; t += blockDim.x) { sid[t] = unsigned(hid[t]); srt[t] = hr[t]; }
        __syncthreads();
        for (unsigned t = threadIdx.x; t < n; t += blockDim.x) { b.ids[t] = int32_t(sid[t]); b.routes[t] = srt[t]; }
        const unsigned long long tl0 = pci_cluster_gtimer();
        q38_batch_build_lists(b, n, E, cnt, sid);
        if (!threadIdx.x) lists_ns = pci_cluster_gtimer() - tl0;
        __syncthreads();   // sm is the shared x buffer: free it before gate/up (a grid barrier follows anyway)
    }
    // BF16 check (2026-11-03): x comes from BF16 hidden states widened to FP32, so its low 16 bits are 0 and
    // gate/up may hold it as BF16 (4-row tiles, lossless). Any other value marks the request (seq) and gate/up
    // uses the 2-row FP32 tiles; the marker needs no reset (older seqs never match).
    // A BF16 request (Q38_BATCH_FMT_BF16) is exact by construction; an FP32 request is checked value by value.
    if (!q38_batch_stage_x(b.batch, R, H, b.x_stage, b.slot)) {
        bool inexact = false;
        for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < R * unsigned(H); i += gridDim.x * blockDim.x)
            inexact |= (__float_as_uint(b.x_stage[i]) & 0xFFFFu) != 0;
        if (inexact) *reinterpret_cast<volatile unsigned *>(b.x_inexact) = seq;
    }
    __threadfence();
    return lists_ns;
}

// q38_fp8dot16 with the fast decode: same fmaf chain, bit-identical.
__device__ __forceinline__ float q38_fp8dot16_fast(uint4 w, const float *x) {
    const float4 *x4 = reinterpret_cast<const float4 *>(x);
    float acc = 0.0f;
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 xv = x4[i];
        acc = fmaf(q38_fp8_fast(words[i], 0), xv.x, acc);
        acc = fmaf(q38_fp8_fast(words[i], 1), xv.y, acc);
        acc = fmaf(q38_fp8_fast(words[i], 2), xv.z, acc);
        acc = fmaf(q38_fp8_fast(words[i], 3), xv.w, acc);
    }
    return acc;
}

// Two-row FP8 dot: decodes each weight once and applies it to rows a and b. Per row the fmaf chain is
// exactly q38_fp8dot16's, so each result is bit-identical to q38_fp8dot16(w, x_row).
__device__ __forceinline__ void q38_fp8dot16x2(uint4 w, const float *xa, const float *xb, float &ra, float &rb) {
    const float4 *a4 = reinterpret_cast<const float4 *>(xa), *b4 = reinterpret_cast<const float4 *>(xb);
    float sa = 0.0f, sb = 0.0f;
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 av = a4[i], bv = b4[i];
        const float f0 = q38_fp8_fast(words[i], 0), f1 = q38_fp8_fast(words[i], 1), f2 = q38_fp8_fast(words[i], 2), f3 = q38_fp8_fast(words[i], 3);
        sa = fmaf(f0, av.x, sa); sa = fmaf(f1, av.y, sa); sa = fmaf(f2, av.z, sa); sa = fmaf(f3, av.w, sa);
        sb = fmaf(f0, bv.x, sb); sb = fmaf(f1, bv.y, sb); sb = fmaf(f2, bv.z, sb); sb = fmaf(f3, bv.w, sb);
    }
    ra = sa; rb = sb;
}

// Four-row FP8 dot against BF16 x (packed 2 per word: element 2m low half, 2m+1 high half). Per row the fmaf
// chain is exactly q38_fp8dot16's, and BF16 -> FP32 widening is exact, so each result is bit-identical to
// q38_fp8dot16(w, x_row) whenever x_row is exactly BF16 (checked in q38_batch_stage).
__device__ __forceinline__ void q38_fp8dot16x4_bf16(uint4 w, const uint32_t *x0, const uint32_t *x1, const uint32_t *x2,
                                                    const uint32_t *x3, float (&out)[4]) {
    const uint32_t *xr[4] = {x0, x1, x2, x3};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float f0 = q38_fp8_fast(words[i], 0), f1 = q38_fp8_fast(words[i], 1), f2 = q38_fp8_fast(words[i], 2), f3 = q38_fp8_fast(words[i], 3);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint2 xv = reinterpret_cast<const uint2 *>(xr[r])[i];
            acc[r] = fmaf(f0, __uint_as_float(xv.x << 16), acc[r]);
            acc[r] = fmaf(f1, __uint_as_float(xv.x & 0xFFFF0000u), acc[r]);
            acc[r] = fmaf(f2, __uint_as_float(xv.y << 16), acc[r]);
            acc[r] = fmaf(f3, __uint_as_float(xv.y & 0xFFFF0000u), acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) out[r] = acc[r];
}

// Phase 2, BF16 variant (2026-11-03): units of up to 4 entries (pstart4); the shared x buffer holds 4 BF16 rows
// (4*H*2 bytes = the 2*H floats of the FP32 variant). Rows past the unit's count repeat its last row and are not
// written. Weight reads per batch drop to ~0.67x of the 2-row variant on real routing (i4 router ids).
__device__ void q38_batch_gate_up_bf16(const Q38BatchView &b, const uint32_t *g, const uint32_t *u, const float *gs, const float *us,
                                       float *xs, int H, int W, int GROUP) {
    const int lane = threadIdx.x & 31, warp = int(threadIdx.x >> 5), nw = int(blockDim.x >> 5);
    uint32_t *xb = reinterpret_cast<uint32_t *>(xs);          // [4][H/2] packed BF16 pairs
    const unsigned nd = *reinterpret_cast<volatile unsigned *>(b.n_distinct);
    const unsigned nu = nd ? b.pstart4[nd] : 0u;
    for (unsigned unit = blockIdx.x; unit < nu; unit += gridDim.x) {
        unsigned lo = 0, hi = nd - 1;
        while (lo < hi) { const unsigned mid = (lo + hi + 1) >> 1; if (b.pstart4[mid] <= unit) lo = mid; else hi = mid - 1; }
        const unsigned d = lo, p = b.start[d] + 4u * (unit - b.pstart4[d]);
        const unsigned n = min(4u, b.start[d + 1] - p);
        unsigned t[4];
#pragma unroll
        for (int i = 0; i < 4; i++) t[i] = b.entry[p + min(unsigned(i), n - 1)];
        __syncthreads();                              // the previous unit's warps are done with xb
        for (unsigned rr = 0; rr < n; rr++) {
            const float2 *src = reinterpret_cast<const float2 *>(b.x_stage + size_t(t[rr] / Q38_BATCH_K) * H);
            for (int i = threadIdx.x; i < H / 2; i += blockDim.x) {
                const float2 v = src[i];
                xb[rr * unsigned(H / 2) + i] = (__float_as_uint(v.x) >> 16) | (__float_as_uint(v.y) & 0xFFFF0000u);
            }
        }
        __syncthreads();
        const unsigned rs = unsigned(H / 2);
        const uint32_t *x0 = xb, *x1 = xb + min(1u, n - 1) * rs, *x2 = xb + min(2u, n - 1) * rs, *x3 = xb + min(3u, n - 1) * rs;
        const int e = int(b.expert_of[d]);
        for (int r = warp; r < W; r += nw) {
            const size_t row = size_t(e) * W + r;
            const uint4 *g4 = reinterpret_cast<const uint4 *>(g + row * (H / 4));
            const uint4 *u4 = reinterpret_cast<const uint4 *>(u + row * (H / 4));
            const float *gsr = gs + row * (H / GROUP), *usr = us + row * (H / GROUP);
            float a[4] = {0.0f, 0.0f, 0.0f, 0.0f}, c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int v = lane; v < H / 16; v += 32) {
                const int grp = (v * 16) / GROUP;
                const float gsv = __ldg(gsr + grp), usv = __ldg(usr + grp);
                float gd[4], ud[4];
                q38_fp8dot16x4_bf16(__ldg(g4 + v), x0 + v * 8, x1 + v * 8, x2 + v * 8, x3 + v * 8, gd);
                q38_fp8dot16x4_bf16(__ldg(u4 + v), x0 + v * 8, x1 + v * 8, x2 + v * 8, x3 + v * 8, ud);
#pragma unroll
                for (int i = 0; i < 4; i++) { a[i] = fmaf(gsv, gd[i], a[i]); c[i] = fmaf(usv, ud[i], c[i]); }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) { a[i] = q38_routed_warp_sum(a[i]); c[i] = q38_routed_warp_sum(c[i]); }
            if (!lane) {
#pragma unroll
                for (int i = 0; i < 4; i++) if (unsigned(i) < n) b.act[size_t(t[i]) * W + r] = (a[i] / (1.0f + expf(-a[i]))) * c[i];
            }
        }
    }
    __threadfence();
}

// Phase 2 (all threads): act[(row*K+k)*W + r] = silu(gate . x_row) * (up . x_row) for every (row, k)
// routed to each expert. v2 (2026-11-03, step 2 of docs/BATCHED_MOE_EXCHANGE_RESEARCH.md): one work unit is
// (expert, up to 2 of its entries); units are spread over the blocks via pstart. A block copies the unit's
// x rows into shared memory once (xs, 2*H floats, the kernel's shared x buffer); each warp then streams its
// gate/up weight rows once and applies them to both rows. v1 read x from global memory for every weight
// row, R*K*W*H*4 bytes per exchange (210 MB at R=64). Per row the per-lane loop order, fmaf chain and warp
// sum are those of q38_routed_v2_gate_up, so every act is bit-identical to mode 1.
__device__ void q38_batch_gate_up(const Q38BatchView &b, const uint32_t *g, const uint32_t *u, const float *gs, const float *us,
                                  float *xs, int H, int W, int GROUP) {
    const int lane = threadIdx.x & 31, warp = int(threadIdx.x >> 5), nw = int(blockDim.x >> 5);
    const unsigned nd = *reinterpret_cast<volatile unsigned *>(b.n_distinct);
    const unsigned nu = nd ? b.pstart[nd] : 0u;
    for (unsigned unit = blockIdx.x; unit < nu; unit += gridDim.x) {
        unsigned lo = 0, hi = nd - 1;                 // last d with pstart[d] <= unit
        while (lo < hi) { const unsigned mid = (lo + hi + 1) >> 1; if (b.pstart[mid] <= unit) lo = mid; else hi = mid - 1; }
        const unsigned d = lo, p = b.start[d] + 2u * (unit - b.pstart[d]);
        const bool two = p + 1 < b.start[d + 1];
        const unsigned t0 = b.entry[p], t1 = two ? b.entry[p + 1] : t0;
        __syncthreads();                              // the previous unit's warps are done with xs
        const float4 *s0 = reinterpret_cast<const float4 *>(b.x_stage + size_t(t0 / Q38_BATCH_K) * H);
        const float4 *s1 = reinterpret_cast<const float4 *>(b.x_stage + size_t(t1 / Q38_BATCH_K) * H);
        float4 *x4 = reinterpret_cast<float4 *>(xs);
        for (int i = threadIdx.x; i < H / 4; i += blockDim.x) { x4[i] = s0[i]; if (two) x4[H / 4 + i] = s1[i]; }
        __syncthreads();
        const int e = int(b.expert_of[d]);
        for (int r = warp; r < W; r += nw) {
            const size_t row = size_t(e) * W + r;
            const uint4 *g4 = reinterpret_cast<const uint4 *>(g + row * (H / 4));
            const uint4 *u4 = reinterpret_cast<const uint4 *>(u + row * (H / 4));
            const float *gsr = gs + row * (H / GROUP), *usr = us + row * (H / GROUP);
            float a0 = 0.0f, c0 = 0.0f, a1 = 0.0f, c1 = 0.0f;
            if (two) {
                for (int v = lane; v < H / 16; v += 32) {
                    const int grp = (v * 16) / GROUP;
                    const float gsv = __ldg(gsr + grp), usv = __ldg(usr + grp);
                    float ga, gb, ua, ub;
                    q38_fp8dot16x2(__ldg(g4 + v), xs + v * 16, xs + H + v * 16, ga, gb);
                    q38_fp8dot16x2(__ldg(u4 + v), xs + v * 16, xs + H + v * 16, ua, ub);
                    a0 = fmaf(gsv, ga, a0); c0 = fmaf(usv, ua, c0);
                    a1 = fmaf(gsv, gb, a1); c1 = fmaf(usv, ub, c1);
                }
                a1 = q38_routed_warp_sum(a1); c1 = q38_routed_warp_sum(c1);
            } else {
                for (int v = lane; v < H / 16; v += 32) {
                    const int grp = (v * 16) / GROUP;
                    a0 = fmaf(__ldg(gsr + grp), q38_fp8dot16_fast(__ldg(g4 + v), xs + v * 16), a0);
                    c0 = fmaf(__ldg(usr + grp), q38_fp8dot16_fast(__ldg(u4 + v), xs + v * 16), c0);
                }
            }
            a0 = q38_routed_warp_sum(a0); c0 = q38_routed_warp_sum(c0);
            if (!lane) {
                b.act[size_t(t0) * W + r] = (a0 / (1.0f + expf(-a0))) * c0;
                if (two) b.act[size_t(t1) * W + r] = (a1 / (1.0f + expf(-a1))) * c1;
            }
        }
    }
    __threadfence();
}

// Phase 3 (all threads): pd[(row*K+k)*H + h] = (sum_q down[e][q][h] * act) * ds[e][h], the exact
// `part * ds` of mode 1, for every (row, k) of every touched expert; the down column of an expert is
// read once per output h. Phase 4 then sums the k terms per row in k order. v2 (2026-11-03, FP8 decode
// amortization): the expert's entries are taken in groups of up to 4; each weight is decoded once (fast
// decode) and applied to every entry of the group. Each entry's fmaf chain (q, j order) is unchanged.
#ifndef Q38_DOWN_GROUP
#define Q38_DOWN_GROUP 4   // entries per decoded down weight (1..4)
#endif
static_assert(Q38_DOWN_GROUP >= 1 && Q38_DOWN_GROUP <= 4, "Q38_DOWN_GROUP");
template <int N>
__device__ __forceinline__ void q38_batch_down_group(const Q38BatchView &b, const uint32_t (&w)[8], float sc,
                                                     unsigned p, int H, int W, int h) {
    unsigned t[N], off[N]; float part[N];        // 32-bit offsets into act (R*K*W < 2^32): fewer registers
    const float *act = b.act;
#pragma unroll
    for (int i = 0; i < N; i++) { t[i] = b.entry[p + i]; off[i] = t[i] * unsigned(W); part[i] = 0.0f; }
#pragma unroll
    for (int q = 0; q < 8; q++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float f = q38_fp8_fast(w[q], j);
#pragma unroll
            for (int i = 0; i < N; i++) part[i] = fmaf(f, act[off[i] + unsigned(q * 4 + j)], part[i]);
        }
#pragma unroll
    for (int i = 0; i < N; i++) b.pd[size_t(t[i]) * H + h] = part[i] * sc;
}
__device__ void q38_batch_down_terms(const Q38BatchView &b, const uint32_t *dw, const float *ds, int H, int W) {
    const unsigned nd = *reinterpret_cast<volatile unsigned *>(b.n_distinct);
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < H; h += gridDim.x * blockDim.x) {
        for (unsigned d = 0; d < nd; d++) {
            const int e = int(b.expert_of[d]);
            uint32_t w[8];
#pragma unroll
            for (int q = 0; q < 8; q++) w[q] = dw[(size_t(e) * (W / 4) + q) * H + h];
            const float sc = ds[size_t(e) * H + h];
            const unsigned p1 = b.start[d + 1];
            unsigned p = b.start[d];
            for (; p + Q38_DOWN_GROUP <= p1; p += Q38_DOWN_GROUP) q38_batch_down_group<Q38_DOWN_GROUP>(b, w, sc, p, H, W, h);
            const unsigned rest = p1 - p;
            if (Q38_DOWN_GROUP > 3 && rest == 3) q38_batch_down_group<(Q38_DOWN_GROUP > 3 ? 3 : 1)>(b, w, sc, p, H, W, h);
            else if (Q38_DOWN_GROUP > 2 && rest == 2) q38_batch_down_group<(Q38_DOWN_GROUP > 2 ? 2 : 1)>(b, w, sc, p, H, W, h);
            else if (rest == 1) q38_batch_down_group<1>(b, w, sc, p, H, W, h);
        }
    }
    __threadfence();
}

// Phase 4 (all threads): own[row][h] = fmaf chain over k (mode 1's `sum`), written to `dst` rows.
__device__ void q38_batch_sum_rows(const Q38BatchView &b, unsigned R, int H, float *dst) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < R * unsigned(H); i += gridDim.x * blockDim.x) {
        const unsigned rowi = i / H, h = i % H;
        float sum = 0.0f;
        for (int k = 0; k < int(Q38_BATCH_K); k++) {
            const unsigned t = rowi * Q38_BATCH_K + k;
            sum = fmaf(b.pd[size_t(t) * H + h], b.routes[t], sum);
        }
        dst[i] = sum;
    }
    __threadfence_system();
}

// Phase 5 (v2 step 3, 2026-11-03): on-card reduce through the mode-1 aggregation windows, one row at a time.
// Each card peer owns one 12-KiB window into its leader's memory (push_va -> leader control + slot_offset +
// (peer_index-1) * slot_stride; data H floats, mode-1 flag word at +H*4, mode-8 row token at +H*4+4). A
// window holds one row, so rows stream: peer writes row r, fences, writes token(seq, r); the leader waits
// for all peers' tokens, writes own + peer 1 + peer 2 + peer 3 (the mode-1 order) to its host part area,
// then grants credit token(seq, r+1) in each peer's host flag page (word 16). No PTE is added or changed;
// the windows are the ones mode 1 uses, and the mode-1 flag word is not touched. Block 0 only; the other
// blocks wait at the grid barrier that follows. Returns false on the aggregation deadline (error bit 64).
__device__ __forceinline__ unsigned q38_batch_token(unsigned seq, unsigned r) { return (seq << 7) | r; }   // r <= 64
__device__ __forceinline__ volatile unsigned *q38_batch_credit(char *batch, unsigned w, unsigned slot = 0) { return q38_batch_flag(batch, w, slot) + 16; }

__device__ bool q38_batch_card_peer(const Q38BatchView &b, unsigned R, int H, unsigned seq, unsigned me, volatile float *push) {
    __shared__ unsigned fail;
    volatile unsigned *credit = q38_batch_credit(b.batch, me, b.slot);
    volatile unsigned *tok = reinterpret_cast<volatile unsigned *>(push + H) + 1;
    if (!threadIdx.x) fail = 0;
    __syncthreads();
    const unsigned long long t0 = clock64();
    for (unsigned r = 0; r < R; r++) {
        // row 0: the window is free (the previous request completed before this one was published)
        if (r && !threadIdx.x) while (*credit != q38_batch_token(seq, r)) if (clock64() - t0 > Q38_AGG_DEADLINE) { fail = 1; break; }
        __syncthreads();
        if (fail) return false;
        const float *src = b.x_stage + size_t(r) * H;
        for (int i = threadIdx.x; i < H; i += blockDim.x) push[i] = src[i];
        __threadfence_system();
        __syncthreads();
        if (!threadIdx.x) { *tok = q38_batch_token(seq, r); __threadfence_system(); }
    }
    // Mode 9 (2026-11-03): wait for the leader's last credit, i.e. until it has read the last row. The window is
    // then free for the next request's row 0 even when that request is already published (pipelining); before,
    // the RTX round trip between requests guaranteed it.
    if (!threadIdx.x) while (*credit != q38_batch_token(seq, R)) if (clock64() - t0 > Q38_AGG_DEADLINE) { fail = 1; break; }
    __syncthreads();
    return !fail;
}

__device__ bool q38_batch_card_leader(const Q38BatchView &b, unsigned R, int H, unsigned seq, unsigned me,
                                      const char *slots, unsigned stride, unsigned np, float *mine) {
    __shared__ unsigned fail;
    if (!threadIdx.x) fail = 0;
    __syncthreads();
    const unsigned long long t0 = clock64();
    for (unsigned r = 0; r < R; r++) {
        if (!threadIdx.x)
            for (unsigned k = 0; k < np && !fail; k++) {
                const volatile unsigned *tok = reinterpret_cast<const volatile unsigned *>(slots + size_t(k) * stride + size_t(H) * 4) + 1;
                while (*tok != q38_batch_token(seq, r)) if (clock64() - t0 > Q38_AGG_DEADLINE) { fail = 1; break; }
            }
        __syncthreads();
        if (fail) return false;
        __threadfence();
        const float *own = b.x_stage + size_t(r) * H;
        float *dst = mine + size_t(r) * H;
        for (int i = threadIdx.x; i < H; i += blockDim.x) {
            float value = own[i];
            for (unsigned k = 0; k < np; k++) value += reinterpret_cast<const volatile float *>(slots + size_t(k) * stride)[i];
            dst[i] = value;
        }
        __syncthreads();                       // every read of the windows is done before the credit
        if (!threadIdx.x) { for (unsigned k = 0; k < np; k++) *q38_batch_credit(b.batch, me + 1 + k, b.slot) = q38_batch_token(seq, r + 1); __threadfence_system(); }
    }
    __threadfence_system();
    return true;
}
