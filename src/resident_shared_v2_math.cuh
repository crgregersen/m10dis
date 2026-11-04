// Shared-expert worker math, v2 (2026-11-02).
//
// Used by resident_dynamic_shared_v2.cuh (the resident GM107 kernel) and by
// qwen38_shared_v2_math_test.cu (a standalone check against a CPU reference).
//
// v1 profile (benchmark-results/qwen38-worker-phase-profile/20261102T174038Z):
// compute+gate 263 us, reduce+copy 31 us on a 10-block x 256-thread grid. v1
// used 2-byte loads per lane (80 per row), strided down-projection loads with
// half the lanes idle, and a single-block reduction over 10 partial vectors.
// v2 keeps the grid and buffers, and changes only the arithmetic layout:
//   hidden: one warp per intermediate row, 16-byte loads (8 BF16 per lane)
//   down:   one warp per 32 consecutive output rows, one 16-byte load per lane
//           (S/8 = 20 lanes), so results leave in coalesced 128-byte lines
#pragma once
#include <cstdint>

// Dot product of 8 BF16 weights (packed in a uint4, little-endian) with 8 floats.
__device__ __forceinline__ float q38_dot8(uint4 v, const float *x, float acc) {
    const float4 x0 = reinterpret_cast<const float4 *>(x)[0];
    const float4 x1 = reinterpret_cast<const float4 *>(x)[1];
    acc = fmaf(__uint_as_float(v.x << 16), x0.x, acc); acc = fmaf(__uint_as_float(v.x & 0xffff0000u), x0.y, acc);
    acc = fmaf(__uint_as_float(v.y << 16), x0.z, acc); acc = fmaf(__uint_as_float(v.y & 0xffff0000u), x0.w, acc);
    acc = fmaf(__uint_as_float(v.z << 16), x1.x, acc); acc = fmaf(__uint_as_float(v.z & 0xffff0000u), x1.y, acc);
    acc = fmaf(__uint_as_float(v.w << 16), x1.z, acc); acc = fmaf(__uint_as_float(v.w & 0xffff0000u), x1.w, acc);
    return acc;
}
__device__ __forceinline__ float q38_warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// hidden[row] = silu(g[row,:].x) * (u[row,:].x) for rows gwarp, gwarp+nwarps, ...
// xs: H floats, 16-byte aligned (shared memory). H must be a multiple of 8.
__device__ __forceinline__ void q38_shared_v2_hidden(const uint16_t *g, const uint16_t *u, const float *xs,
                                                     float *hidden, int gwarp, int nwarps, int lane,
                                                     int H, int S) {
    for (int row = gwarp; row < S; row += nwarps) {
        const uint4 *g4 = reinterpret_cast<const uint4 *>(g + size_t(row) * H);
        const uint4 *u4 = reinterpret_cast<const uint4 *>(u + size_t(row) * H);
        float a = 0.0f, b = 0.0f;
        for (int v = lane; v < H / 8; v += 32) {
            const float *xv = xs + v * 8;
            a = q38_dot8(__ldg(g4 + v), xv, a);
            b = q38_dot8(__ldg(u4 + v), xv, b);
        }
        a = q38_warp_sum(a); b = q38_warp_sum(b);
        if (!lane) hidden[row] = (a / (1.0f + expf(-a))) * b;
    }
}
// Mode 8 (batched, 2026-11-03): q38_shared_v2_hidden for two x rows at once (xb may be null): each
// weight vector is loaded once and applied to both rows; per row the arithmetic is exactly
// q38_shared_v2_hidden's (same per-lane order, warp sum and silu), so the results are bit-identical.
__device__ __forceinline__ void q38_shared_v2_hidden2(const uint16_t *g, const uint16_t *u, const float *xa,
                                                      const float *xb, float *ha, float *hb, int gwarp,
                                                      int nwarps, int lane, int H, int S) {
    for (int row = gwarp; row < S; row += nwarps) {
        const uint4 *g4 = reinterpret_cast<const uint4 *>(g + size_t(row) * H);
        const uint4 *u4 = reinterpret_cast<const uint4 *>(u + size_t(row) * H);
        float a0 = 0.0f, b0 = 0.0f, a1 = 0.0f, b1 = 0.0f;
        for (int v = lane; v < H / 8; v += 32) {
            const uint4 gv = __ldg(g4 + v), uv = __ldg(u4 + v);
            a0 = q38_dot8(gv, xa + v * 8, a0); b0 = q38_dot8(uv, xa + v * 8, b0);
            if (xb) { a1 = q38_dot8(gv, xb + v * 8, a1); b1 = q38_dot8(uv, xb + v * 8, b1); }
        }
        a0 = q38_warp_sum(a0); b0 = q38_warp_sum(b0);
        if (xb) { a1 = q38_warp_sum(a1); b1 = q38_warp_sum(b1); }
        if (!lane) { ha[row] = (a0 / (1.0f + expf(-a0))) * b0; if (xb) hb[row] = (a1 / (1.0f + expf(-a1))) * b1; }
    }
}
// Per-thread partial of gate[:].x; the caller reduces across the block.
// BF16-x variants (mode 8, 2026-11-03): x packed as BF16 pairs (4 words = 8 values, element 2m in the low
// half). Widening BF16 -> FP32 is exact and the fmaf chain is q38_dot8's, so each result is bit-identical to
// the FP32-x function whenever x is exactly BF16 (BF16 requests, Q38_BATCH_FMT_BF16).
__device__ __forceinline__ float q38_dot8_bfx(uint4 v, const uint32_t *xw, float acc) {
    const uint4 w = reinterpret_cast<const uint4 *>(xw)[0];
    acc = fmaf(__uint_as_float(v.x << 16), __uint_as_float(w.x << 16), acc); acc = fmaf(__uint_as_float(v.x & 0xffff0000u), __uint_as_float(w.x & 0xffff0000u), acc);
    acc = fmaf(__uint_as_float(v.y << 16), __uint_as_float(w.y << 16), acc); acc = fmaf(__uint_as_float(v.y & 0xffff0000u), __uint_as_float(w.y & 0xffff0000u), acc);
    acc = fmaf(__uint_as_float(v.z << 16), __uint_as_float(w.z << 16), acc); acc = fmaf(__uint_as_float(v.z & 0xffff0000u), __uint_as_float(w.z & 0xffff0000u), acc);
    acc = fmaf(__uint_as_float(v.w << 16), __uint_as_float(w.w << 16), acc); acc = fmaf(__uint_as_float(v.w & 0xffff0000u), __uint_as_float(w.w & 0xffff0000u), acc);
    return acc;
}
// q38_shared_v2_hidden for up to 4 BF16 x rows (x[r] = row r, H/2 words; rows >= n repeat row n-1 and are not
// written): each gate/up weight vector is loaded once for all rows. Per row identical to q38_shared_v2_hidden.
__device__ __forceinline__ void q38_shared_v2_hidden4_bfx(const uint16_t *g, const uint16_t *u, const uint32_t *const (&x)[4],
                                                          unsigned n, float *const (&h)[4], int gwarp, int nwarps,
                                                          int lane, int H, int S) {
    for (int row = gwarp; row < S; row += nwarps) {
        const uint4 *g4 = reinterpret_cast<const uint4 *>(g + size_t(row) * H);
        const uint4 *u4 = reinterpret_cast<const uint4 *>(u + size_t(row) * H);
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f}, b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int v = lane; v < H / 8; v += 32) {
            const uint4 gv = __ldg(g4 + v), uv = __ldg(u4 + v);
#pragma unroll
            for (int r = 0; r < 4; r++) { a[r] = q38_dot8_bfx(gv, x[r] + v * 4, a[r]); b[r] = q38_dot8_bfx(uv, x[r] + v * 4, b[r]); }
        }
#pragma unroll
        for (int r = 0; r < 4; r++) { a[r] = q38_warp_sum(a[r]); b[r] = q38_warp_sum(b[r]); }
        if (!lane)
#pragma unroll
            for (int r = 0; r < 4; r++) if (unsigned(r) < n) h[r][row] = (a[r] / (1.0f + expf(-a[r]))) * b[r];
    }
}
__device__ __forceinline__ float q38_shared_v2_gate_partial_bfx(const uint16_t *gate, const uint32_t *xw,
                                                                int tid, int nthreads, int H) {
    const uint4 *g4 = reinterpret_cast<const uint4 *>(gate);
    float acc = 0.0f;
    for (int v = tid; v < H / 8; v += nthreads) acc = q38_dot8_bfx(__ldg(g4 + v), xw + v * 4, acc);
    return acc;
}
__device__ __forceinline__ float q38_shared_v2_gate_partial(const uint16_t *gate, const float *xs,
                                                            int tid, int nthreads, int H) {
    const uint4 *g4 = reinterpret_cast<const uint4 *>(gate);
    float acc = 0.0f;
    for (int v = tid; v < H / 8; v += nthreads) acc = q38_dot8(__ldg(g4 + v), xs + v * 8, acc);
    return acc;
}
// Mode 8: out[t][base + r] = d[base+r, :].hs[t] for t < nt rows of hs (hs: nt x S floats in shared
// memory, 16-byte aligned). Each d vector is loaded once per tile; the per-row value is
// q38_shared_v2_down_chunk's (same dot8, same warp sum; lane 0 holds the sum).
__device__ __forceinline__ void q38_shared_v2_down_tile(const uint16_t *d, const float *hs, unsigned nt,
                                                        float *out, int base, int lane, int H, int S) {
    for (int r = 0; r < 32; ++r) {
        uint4 dv = make_uint4(0u, 0u, 0u, 0u);
        if (lane < S / 8) dv = __ldg(reinterpret_cast<const uint4 *>(d + size_t(base + r) * S) + lane);
        for (unsigned t = 0; t < nt; ++t) {
            float s = 0.0f;
            if (lane < S / 8) s = q38_dot8(dv, hs + t * S + lane * 8, 0.0f);
            s = q38_warp_sum(s);
            if (!lane) out[size_t(t) * H + base + r] = s;
        }
    }
}
// Returns out[base + lane] = d[base+lane, :].hs for one chunk of 32 rows.
// hs: S floats, 16-byte aligned. S must be a multiple of 8 and <= 256.
__device__ __forceinline__ float q38_shared_v2_down_chunk(const uint16_t *d, const float *hs, int base,
                                                          int lane, int S) {
    float mine = 0.0f;
    for (int r = 0; r < 32; ++r) {
        float s = 0.0f;
        if (lane < S / 8)
            s = q38_dot8(__ldg(reinterpret_cast<const uint4 *>(d + size_t(base + r) * S) + lane), hs + lane * 8, 0.0f);
        s = q38_warp_sum(s);
        s = __shfl_sync(0xffffffffu, s, 0);
        if (lane == r) mine = s;
    }
    return mine;
}
// Mode 10 (multi-row decode, 2026-11-04): q38_shared_v2_down_chunk for two hs rows, each d vector loaded once;
// per row the same dot8, warp sum and broadcast, so m0 / m1 are bit-identical to two down_chunk calls.
__device__ __forceinline__ void q38_shared_v2_down_chunk2(const uint16_t *d, const float *hs0, const float *hs1,
                                                          int base, int lane, int S, float &m0, float &m1) {
    m0 = 0.0f; m1 = 0.0f;
    for (int r = 0; r < 32; ++r) {
        float s0 = 0.0f, s1 = 0.0f;
        if (lane < S / 8) {
            const uint4 dv = __ldg(reinterpret_cast<const uint4 *>(d + size_t(base + r) * S) + lane);
            s0 = q38_dot8(dv, hs0 + lane * 8, 0.0f);
            s1 = q38_dot8(dv, hs1 + lane * 8, 0.0f);
        }
        s0 = q38_warp_sum(s0); s1 = q38_warp_sum(s1);
        s0 = __shfl_sync(0xffffffffu, s0, 0); s1 = __shfl_sync(0xffffffffu, s1, 0);
        if (lane == r) { m0 = s0; m1 = s1; }
    }
}
