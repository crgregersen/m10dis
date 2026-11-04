// Standalone check of resident_routed_v2_math.cuh (routed gate/up) against a
// double-precision CPU reference at the live routed-worker shapes:
// H=2560, W=32 rows per expert slice, K=10 selected experts, E=512, GROUP=128,
// 10 x 256 grid. Runs on any CUDA GPU (the RTX here); it does not touch the M10s.
// Weights are random FP8 bytes in the same packed word layout as pack_gate().
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>
#include "resident_routed_v2_math.cuh"

constexpr int H = 2560, W = 32, K = 10, E = 512, GROUP = 128, BLOCKS = 10, T = 256;

__global__ void gate_up(const uint32_t *g, const uint32_t *u, const float *gs, const float *us,
                        const int32_t *ids, const float *x, float *act) {
    __shared__ __align__(16) float xs[H];
    for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, gwarp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    q38_routed_v2_gate_up(g, u, gs, us, ids, xs, act, gwarp, gridDim.x * (blockDim.x >> 5), lane, H, W, K, GROUP);
}
static double fp8(uint8_t b) {  // identical to fp8e4m3()
    const unsigned a = b & 127; float v;
    if (a < 8) v = float(a) * (1.0f / 512.0f); else { uint32_t bits = (a << 20) + (120u << 23); memcpy(&v, &bits, 4); }
    return (b & 128) ? -v : v;
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA_ERROR,%s,%s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    std::mt19937 rng(0x0ded2026);
    std::uniform_int_distribution<int> byte(0, 255), expert(0, E - 1);
    std::normal_distribution<float> nx(0.0f, 0.5f), ns(0.0f, 0.01f);
    // Only the K selected experts are generated; the others stay zero.
    std::vector<int32_t> ids(K);
    for (int k = 0; k < K; ++k) { int e; do { e = expert(rng); } while (std::find(ids.begin(), ids.begin() + k, e) != ids.begin() + k); ids[k] = e; }
    const size_t words = size_t(E) * W * (H / 4), scales = size_t(E) * W * (H / GROUP);
    std::vector<uint32_t> g(words, 0), u(words, 0);
    std::vector<float> gs(scales, 0), us(scales, 0), x(H);
    for (int k = 0; k < K; ++k) for (int r = 0; r < W; ++r) {
        const size_t row = size_t(ids[k]) * W + r;
        for (int q = 0; q < H / 4; ++q) {
            uint32_t wg = 0, wu = 0;
            for (int j = 0; j < 4; ++j) { wg |= uint32_t(byte(rng)) << (8 * j); wu |= uint32_t(byte(rng)) << (8 * j); }
            g[row * (H / 4) + q] = wg; u[row * (H / 4) + q] = wu;
        }
        for (int s = 0; s < H / GROUP; ++s) { gs[row * (H / GROUP) + s] = fabsf(ns(rng)); us[row * (H / GROUP) + s] = fabsf(ns(rng)); }
    }
    for (auto &v : x) v = nx(rng);
    // Reference in double, plus a floating-point error bound per task. A fixed
    // tolerance is wrong here: uniformly random FP8 bytes reach +-448, so terms of
    // a dot product can be large and cancel, and any FP32 summation order (v1 or
    // v2) then differs from double by up to ~eps * sum|terms|. The bound uses
    // 2e-5 * sum|terms| per dot product (about 3x the sqrt(n)*eps estimate);
    // one wrong or missing element costs ~sum|terms|/2560 = 3.9e-4 * sum|terms|,
    // so indexing or scale bugs still fail by a wide margin.
    std::vector<double> ref(K * W), bound(K * W);
    for (int k = 0; k < K; ++k) for (int r = 0; r < W; ++r) {
        const size_t row = size_t(ids[k]) * W + r; double a = 0, b = 0, A = 0, B = 0;
        for (int c = 0; c < H; ++c) {
            const uint32_t wg = g[row * (H / 4) + c / 4], wu = u[row * (H / 4) + c / 4];
            const double ta = fp8(uint8_t(wg >> (8 * (c % 4)))) * gs[row * (H / GROUP) + c / GROUP] * x[c];
            const double tb = fp8(uint8_t(wu >> (8 * (c % 4)))) * us[row * (H / GROUP) + c / GROUP] * x[c];
            a += ta; b += tb; A += fabs(ta); B += fabs(tb);
        }
        const double sig = 1 / (1 + exp(-a)), silu = a * sig, dsilu = sig * (1 + a * (1 - sig));
        ref[k * W + r] = silu * b;
        bound[k * W + r] = fabs(dsilu * b) * 2e-5 * A + fabs(silu) * 2e-5 * B + 1e-12;
    }
    uint32_t *dg, *du; float *dgs, *dus, *dx, *dact; int32_t *dids;
    CK(cudaMalloc(&dg, words * 4)); CK(cudaMalloc(&du, words * 4)); CK(cudaMalloc(&dgs, scales * 4)); CK(cudaMalloc(&dus, scales * 4));
    CK(cudaMalloc(&dx, H * 4)); CK(cudaMalloc(&dact, K * W * 4)); CK(cudaMalloc(&dids, K * 4));
    CK(cudaMemcpy(dg, g.data(), words * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, u.data(), words * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dgs, gs.data(), scales * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dus, us.data(), scales * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), H * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dids, ids.data(), K * 4, cudaMemcpyHostToDevice));
    gate_up<<<BLOCKS, T>>>(dg, du, dgs, dus, dids, dx, dact); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> act(K * W); CK(cudaMemcpy(act.data(), dact, K * W * 4, cudaMemcpyDeviceToHost));
    double max_abs = 0, max_ref = 0, worst_ratio = 0; int bad = 0, fixed_tol_misses = 0;
    for (int i = 0; i < K * W; ++i) {
        const double e = fabs(act[i] - ref[i]); max_abs = fmax(max_abs, e); max_ref = fmax(max_ref, fabs(ref[i]));
        worst_ratio = fmax(worst_ratio, e / bound[i]);
        if (!std::isfinite(act[i]) || e > bound[i]) ++bad;
        if (e > 2e-5 + 2e-5 * fabs(ref[i])) ++fixed_tol_misses;  // the earlier fixed rule, for the record
    }
    // Negative control: an off-by-one input (x shifted by one element) must fail the bound
    // on most tasks, otherwise the bound is too loose to mean anything.
    std::vector<float> xs1(H); for (int c = 0; c < H; ++c) xs1[c] = x[(c + 1) % H];
    CK(cudaMemcpy(dx, xs1.data(), H * 4, cudaMemcpyHostToDevice));
    gate_up<<<BLOCKS, T>>>(dg, du, dgs, dus, dids, dx, dact); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> act_bad(K * W); CK(cudaMemcpy(act_bad.data(), dact, K * W * 4, cudaMemcpyDeviceToHost));
    int control_fail = 0; for (int i = 0; i < K * W; ++i) control_fail += !(fabs(act_bad[i] - ref[i]) <= bound[i]);
    const bool control_ok = control_fail > K * W * 9 / 10;
    const bool ok = bad == 0 && control_ok;
    printf("ROUTED_V2_MATH_TEST,%s,act_max_abs_err,%.3e,act_max_abs_ref,%.3e,act_bad,%d,worst_err_over_bound,%.3f,"
           "fixed_tolerance_misses,%d,negative_control_detected,%d/%d,tasks,%d\n",
           ok ? "PASS" : "FAIL", max_abs, max_ref, bad, worst_ratio, fixed_tol_misses, control_fail, K * W, K * W);
    return ok ? 0 : 1;
}
