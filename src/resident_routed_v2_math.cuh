// Routed-expert worker math, v2 (2026-11-02).
//
// Used by resident_dynamic_routed_v2.cuh (the resident GM107 kernel) and by
// qwen38_routed_v2_math_test.cu (a standalone check against a CPU reference).
//
// v1 profile (benchmark-results/qwen38-worker-phase-profile/20261102T174038Z and
// 20261102T180400Z): gate/up 174 us of 277 us. v1 walks the 320 (expert, row)
// tasks one block at a time (32 per block, serially), each task a 640-word loop
// over 256 threads with a block-wide reduction and two __syncthreads.
// v2 gives each warp whole tasks: 320 tasks / 80 warps = 4 per warp, 16 FP8
// weights per 16-byte load (5 loads per row per matrix), x from shared memory,
// one warp reduction per task and no block-wide synchronisation.
// The FP8 decode is identical to fp8e4m3() in bench_resident_multilayer_fp8.cu.
#pragma once
#include <cstdint>

__device__ __forceinline__ float q38_fp8e4m3(uint32_t x) {
    const unsigned b = x & 255, a = b & 127;
    const float v = (a < 8) ? float(a) * (1.0f / 512.0f) : __uint_as_float((a << 20) + (120u << 23));
    return (b & 128) ? -v : v;
}
// FP8 E4M3 decode in ~4 instructions (2026-11-03, docs/BATCHED_MOE_EXCHANGE_RESEARCH.md): the byte's 7 magnitude
// bits go to FP32 bits 20..26 (exponent field 0..15 + 3 mantissa bits), the sign to bit 31, then * 2^120.
// Exponent 0 gives an FP32 subnormal, a * 2^-129, and the power-of-two multiply is exact, so the result is
// bit-identical to q38_fp8e4m3() for all 256 bytes (checked exhaustively, incl. -0). REQUIRES FP32 subnormals:
// the kernels are built with plain -O2 (no -ftz=true / --use_fast_math); a flush-to-zero build would change
// tiny weights to 0 and show up as a mismatch in the mode-8 vs mode-1 bit-identity checks.
__device__ __forceinline__ float q38_fp8_fast(uint32_t w, int j) {
    const uint32_t b = w >> (8 * j);
    return __uint_as_float(((b & 0x7Fu) << 20) | ((b & 0x80u) << 24)) * __uint_as_float(247u << 23);   // 2^120
}
// 16 FP8 values (4 little-endian words, byte j of word i = element 4i+j) dotted with 16 floats. Since
// 2026-11-03 with q38_fp8_fast (bit-identical to q38_fp8e4m3; used by mode-1 gate/up).
__device__ __forceinline__ float q38_fp8dot16(uint4 w, const float *x) {
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
__device__ __forceinline__ float q38_routed_warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// act[task] = silu(gate[e,r,:].x) * (up[e,r,:].x) for tasks gwarp, gwarp+nwarps, ...
// task = k*W + r, e = ids[k]. g/u: packed FP8 words, (e*W + r)*(H/4) word rows.
// gs/us: one scale per GROUP inputs, (e*W + r)*(H/GROUP) per row. xs: H floats in
// shared memory, 16-byte aligned. Requires H % 16 == 0 and GROUP % 16 == 0.
__device__ __forceinline__ void q38_routed_v2_gate_up(const uint32_t *g, const uint32_t *u, const float *gs,
                                                      const float *us, const int32_t *ids, const float *xs,
                                                      float *act, int gwarp, int nwarps, int lane,
                                                      int H, int W, int K, int GROUP) {
    for (int task = gwarp; task < K * W; task += nwarps) {
        const int k = task / W, r = task % W, e = ids[k];
        const size_t row = size_t(e) * W + r;
        const uint4 *g4 = reinterpret_cast<const uint4 *>(g + row * (H / 4));
        const uint4 *u4 = reinterpret_cast<const uint4 *>(u + row * (H / 4));
        const float *gsr = gs + row * (H / GROUP), *usr = us + row * (H / GROUP);
        float a = 0.0f, b = 0.0f;
        for (int v = lane; v < H / 16; v += 32) {
            const float *xv = xs + v * 16;
            const int grp = (v * 16) / GROUP;
            a = fmaf(__ldg(gsr + grp), q38_fp8dot16(__ldg(g4 + v), xv), a);
            b = fmaf(__ldg(usr + grp), q38_fp8dot16(__ldg(u4 + v), xv), b);
        }
        a = q38_routed_warp_sum(a); b = q38_routed_warp_sum(b);
        if (!lane) act[task] = (a / (1.0f + expf(-a))) * b;
    }
}
