// Standalone check of resident_shared_v2_math.cuh against a double-precision CPU
// reference, at the live shared-worker shapes (H=2560, S=160, 10 x 256 grid).
// Runs on any CUDA GPU (the RTX here); it does not touch the M10 workers.
// Prints SHARED_V2_MATH_TEST,PASS|FAIL with the worst errors.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include "resident_shared_v2_math.cuh"

constexpr int H = 2560, S = 160, BLOCKS = 10, T = 256, WARPS = T / 32;

__global__ void phase_hidden(const uint16_t *g, const uint16_t *u, const uint16_t *gate, const float *x,
                             float *hidden, float *gate_out) {
    __shared__ __align__(16) float xs[H];
    __shared__ float red[T];
    for (int i = threadIdx.x; i < H; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    q38_shared_v2_hidden(g, u, xs, hidden, blockIdx.x * WARPS + warp, gridDim.x * WARPS, lane, H, S);
    if (blockIdx.x == 0) {
        red[threadIdx.x] = q38_shared_v2_gate_partial(gate, xs, threadIdx.x, blockDim.x, H);
        __syncthreads();
        for (int s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
        if (!threadIdx.x) *gate_out = 1.0f / (1.0f + expf(-red[0]));
    }
}
__global__ void phase_down(const uint16_t *d, const float *hidden, float *out) {
    __shared__ __align__(16) float hs[S];
    for (int i = threadIdx.x; i < S; i += blockDim.x) hs[i] = hidden[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gwarp = blockIdx.x * WARPS + warp, nwarps = gridDim.x * WARPS;
    for (int base = gwarp * 32; base < H; base += nwarps * 32) out[base + lane] = q38_shared_v2_down_chunk(d, hs, base, lane, S);
}
static double bf(uint16_t v) { uint32_t b = uint32_t(v) << 16; float f; memcpy(&f, &b, 4); return f; }
static uint16_t to_bf(float f) { uint32_t b; memcpy(&b, &f, 4); return uint16_t(b >> 16); }
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA_ERROR,%s,%s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    std::mt19937 rng(0x5eed2026);
    std::normal_distribution<float> n(0.0f, 0.02f), nx(0.0f, 0.5f);
    std::vector<uint16_t> g(size_t(S) * H), u(size_t(S) * H), d(size_t(H) * S), gate(H);
    std::vector<float> x(H);
    for (auto &v : g) v = to_bf(n(rng)); for (auto &v : u) v = to_bf(n(rng));
    for (auto &v : d) v = to_bf(n(rng)); for (auto &v : gate) v = to_bf(n(rng));
    for (auto &v : x) v = nx(rng);
    // CPU reference in double
    std::vector<double> hid(S), ref(H); double gref = 0;
    for (int r = 0; r < S; ++r) { double a = 0, b = 0; for (int c = 0; c < H; ++c) { a += bf(g[size_t(r) * H + c]) * x[c]; b += bf(u[size_t(r) * H + c]) * x[c]; } hid[r] = a / (1 + exp(-a)) * b; }
    for (int r = 0; r < H; ++r) { double s = 0; for (int c = 0; c < S; ++c) s += bf(d[size_t(r) * S + c]) * hid[c]; ref[r] = s; }
    for (int c = 0; c < H; ++c) gref += bf(gate[c]) * x[c]; gref = 1 / (1 + exp(-gref));
    uint16_t *dg, *du, *dd, *dgate; float *dx, *dh, *dout, *dgo;
    CK(cudaMalloc(&dg, g.size() * 2)); CK(cudaMalloc(&du, u.size() * 2)); CK(cudaMalloc(&dd, d.size() * 2)); CK(cudaMalloc(&dgate, H * 2));
    CK(cudaMalloc(&dx, H * 4)); CK(cudaMalloc(&dh, S * 4)); CK(cudaMalloc(&dout, H * 4)); CK(cudaMalloc(&dgo, 4));
    CK(cudaMemcpy(dg, g.data(), g.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, u.data(), u.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, d.data(), d.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dgate, gate.data(), H * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), H * 4, cudaMemcpyHostToDevice));
    phase_hidden<<<BLOCKS, T>>>(dg, du, dgate, dx, dh, dgo); CK(cudaGetLastError());
    phase_down<<<BLOCKS, T>>>(dd, dh, dout); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<float> out(H), h(S); float go = 0;
    CK(cudaMemcpy(out.data(), dout, H * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h.data(), dh, S * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&go, dgo, 4, cudaMemcpyDeviceToHost));
    double max_abs = 0, max_ref = 0, hid_abs = 0; int bad = 0;
    for (int r = 0; r < H; ++r) { double e = fabs(out[r] - ref[r]); max_abs = fmax(max_abs, e); max_ref = fmax(max_ref, fabs(ref[r])); if (e > 2e-5 + 2e-5 * fabs(ref[r])) ++bad; }
    for (int r = 0; r < S; ++r) hid_abs = fmax(hid_abs, fabs(h[r] - hid[r]));
    const double gate_abs = fabs(go - gref);
    const bool ok = bad == 0 && gate_abs < 1e-6 && std::isfinite(max_abs);
    printf("SHARED_V2_MATH_TEST,%s,out_max_abs_err,%.3e,out_max_abs_ref,%.3e,out_bad,%d,hidden_max_abs_err,%.3e,gate_abs_err,%.3e\n",
           ok ? "PASS" : "FAIL", max_abs, max_ref, bad, hid_abs, gate_abs);
    return ok ? 0 : 1;
}
