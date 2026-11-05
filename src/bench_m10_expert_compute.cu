// M10 routed-expert compute microbenchmark (2026-11-04): times the deployed 1-row gate/up and down code
// (dynamic_gate_layer / dynamic_down_layer from resident_dynamic_routed_v3.cuh, the same headers the worker cubin
// is built from) against variants with more loads in flight or more blocks, on ONE M10 GM107, with synthetic FP8
// weights of one layer (512 experts x 32 rows x 2560, 126 MB) so no L2 reuse between iterations.
//
// Each variant is one kernel launch running WARM + ITERS iterations: phase 1 (x staging + gate/up), grid barrier,
// phase 2 (down), grid barrier; block 0 thread 0 stamps %globaltimer after each barrier (and before the down
// barrier). Expert sets rotate over NSETS random sets of K distinct experts. Every variant's act/out of set 0 is
// compared bit for bit with the deployed code. A variant runs only if its whole grid is co-resident (occupancy
// API), otherwise the grid barrier could not complete.
//
// Device: the first "Tesla M10" (PCI bus order) with enough free memory; nothing is hard-coded. The resident
// daemon may hold the M10s (workers PARKED): this process only adds its own context and ~180 MB.
// Build: nvcc -O2 -std=c++17 -arch=sm_50 -Isrc ... src/bench_m10_expert_compute.cu   (no -ftz: FP32 subnormals)
// Run:   bench_m10_expert_compute OUT_JSON [iters]
#define QWEN_M10_LIBRARY_ONLY
#include "bench_resident_multilayer_fp8.cu"     // H, W, K, E, GROUP, DeviceWeights, Params, fp8e4m3 (as the worker cubin)
#include "resident_dynamic_routed_v3.cuh"       // dynamic_gate_layer, dynamic_down_layer, q38_routed_xbuf (deployed code)
#include <random>
#include <algorithm>
#include <array>
#include <cstring>

constexpr int NSETS = 64, WARM = 200;
struct Bar { unsigned count, generation, error; };
__device__ __forceinline__ unsigned long long gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void gbar(Bar *b, unsigned &gen) {   // same scheme as dynamic_grid_barrier
  __syncthreads();
  if (!threadIdx.x) { ++gen; __threadfence(); unsigned t = atomicAdd(&b->count, 1u);
    if (t == gridDim.x - 1) { atomicExch(&b->count, 0u); __threadfence(); atomicExch(&b->generation, gen); }
    unsigned long long s = clock64();
    while (atomicAdd(&b->generation, 0u) != gen) if (clock64() - s > 2000000000ULL) { atomicExch(&b->error, 1u); break; } }
  else ++gen;
  __syncthreads();
}
struct Args {
  DeviceWeights w; const uint32_t *dT;   // dT: down weights transposed to [e][row][W/4 words] (32 B per (e,row))
  const uint32_t *xw;                    // NSETS x H/2 BF16 words (x rows)
  const int32_t *ids; const float *routes; float *act, *out;   // act: K*W, out: H (per run; set 0 checked)
  Bar *bar; unsigned long long *ts;      // ts: iters x 3 (after gate/up barrier, block 0 down done, after down barrier)
  int iters;
};
// ---- x staging exactly as dynamic_gate_layer (BF16 words -> FP32 in q38_routed_xbuf) ----
__device__ __forceinline__ float *stage_x(const uint32_t *xw) {
  float *xs = q38_routed_xbuf();
  for (int i = threadIdx.x; i < H / 2; i += blockDim.x) { const uint32_t w = xw[i];
    reinterpret_cast<float2 *>(xs)[i] = make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u)); }
  __syncthreads(); return xs;
}
// ---- 1-row x buffer (2026-11-05 extension): H floats (10 KB) instead of q38_routed_xbuf's 2*H (sized for mode-8 tiles),
// so more blocks fit per SM; the down split variants reuse it for their partial sums (x is dead after gate/up).
__device__ __forceinline__ float *small_xbuf() { __shared__ __align__(16) float buf[H]; return buf; }
__device__ __forceinline__ float *stage_x_small(const uint32_t *xw) {
  float *xs = small_xbuf();
  for (int i = threadIdx.x; i < H / 2; i += blockDim.x) { const uint32_t w = xw[i];
    reinterpret_cast<float2 *>(xs)[i] = make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xFFFF0000u)); }
  __syncthreads(); return xs;
}
// ---- gate/up variants (same per-task arithmetic as q38_routed_v2_gate_up: bit-identical) ----
__device__ __forceinline__ void gu_hoist(const DeviceWeights &w, const int32_t *ids, const float *xs, float *act, int gwarp, int nwarps, int lane) {
  for (int task = gwarp; task < K * W; task += nwarps) {
    const int k = task / W, r = task % W, e = ids[k]; const size_t row = size_t(e) * W + r;
    const uint4 *g4 = reinterpret_cast<const uint4 *>(w.g + row * (H / 4)), *u4 = reinterpret_cast<const uint4 *>(w.u + row * (H / 4));
    const float *gsr = w.gs + row * (H / GROUP), *usr = w.us + row * (H / GROUP);
    uint4 gv[H / 16 / 32], uv[H / 16 / 32]; float sg[H / 16 / 32], su[H / 16 / 32];
#pragma unroll
    for (int t = 0; t < H / 16 / 32; t++) { const int v = lane + 32 * t; gv[t] = __ldg(g4 + v); uv[t] = __ldg(u4 + v);
      sg[t] = __ldg(gsr + (v * 16) / GROUP); su[t] = __ldg(usr + (v * 16) / GROUP); }
    float a = 0.0f, b = 0.0f;
#pragma unroll
    for (int t = 0; t < H / 16 / 32; t++) { const float *xv = xs + (lane + 32 * t) * 16;
      a = fmaf(sg[t], q38_fp8dot16(gv[t], xv), a); b = fmaf(su[t], q38_fp8dot16(uv[t], xv), b); }
    a = q38_routed_warp_sum(a); b = q38_routed_warp_sum(b);
    if (!lane) act[task] = (a / (1.0f + expf(-a))) * b;
  }
}
__device__ __forceinline__ void gu_pair(const DeviceWeights &w, const int32_t *ids, const float *xs, float *act, int gwarp, int nwarps, int lane) {
  for (int t0 = gwarp; t0 < K * W; t0 += 2 * nwarps) {   // tasks t0 and t0 + nwarps in one warp, loads interleaved
    const int t1 = t0 + nwarps; const bool two = t1 < K * W;
    const size_t r0 = size_t(ids[t0 / W]) * W + t0 % W, r1 = two ? size_t(ids[t1 / W]) * W + t1 % W : r0;
    const uint4 *g0 = reinterpret_cast<const uint4 *>(w.g + r0 * (H / 4)), *u0 = reinterpret_cast<const uint4 *>(w.u + r0 * (H / 4));
    const uint4 *g1 = reinterpret_cast<const uint4 *>(w.g + r1 * (H / 4)), *u1 = reinterpret_cast<const uint4 *>(w.u + r1 * (H / 4));
    float a0 = 0, b0 = 0, a1 = 0, b1 = 0;
    for (int v = lane; v < H / 16; v += 32) { const float *xv = xs + v * 16; const int grp = (v * 16) / GROUP;
      const uint4 G0 = __ldg(g0 + v), U0 = __ldg(u0 + v), G1 = __ldg(g1 + v), U1 = __ldg(u1 + v);
      const float s0 = __ldg(w.gs + r0 * (H / GROUP) + grp), q0 = __ldg(w.us + r0 * (H / GROUP) + grp);
      const float s1 = __ldg(w.gs + r1 * (H / GROUP) + grp), q1 = __ldg(w.us + r1 * (H / GROUP) + grp);
      a0 = fmaf(s0, q38_fp8dot16(G0, xv), a0); b0 = fmaf(q0, q38_fp8dot16(U0, xv), b0);
      a1 = fmaf(s1, q38_fp8dot16(G1, xv), a1); b1 = fmaf(q1, q38_fp8dot16(U1, xv), b1); }
    a0 = q38_routed_warp_sum(a0); b0 = q38_routed_warp_sum(b0); a1 = q38_routed_warp_sum(a1); b1 = q38_routed_warp_sum(b1);
    if (!lane) { act[t0] = (a0 / (1.0f + expf(-a0))) * b0; if (two) act[t1] = (a1 / (1.0f + expf(-a1))) * b1; }
  }
}
// ---- down variants (same chain sum = fmaf(part * ds, route, sum) over k = 0..K-1 in order: bit-identical) ----
__device__ __forceinline__ float dn_part(const uint32_t *d, int e, int row, const float *acts, int k) {
  float part = 0;
#pragma unroll
  for (int q = 0; q < W / 4; q++) { const uint32_t v = d[(size_t(e) * (W / 4) + q) * H + row];
#pragma unroll
    for (int j = 0; j < 4; j++) part = fmaf(fp8e4m3(v >> (8 * j)), acts[k * W + q * 4 + j], part); }
  return part;
}
__device__ __forceinline__ void dn_k2(const Params &p) {   // two experts' loads issued together
  __shared__ float acts[K * W]; for (int i = threadIdx.x; i < K * W; i += blockDim.x) acts[i] = p.act[i]; __syncthreads();
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < H; row += gridDim.x * blockDim.x) { float sum = 0;
    for (int k = 0; k < K; k += 2) { const int e0 = p.ids[k], e1 = p.ids[k + 1];
      const float pa = dn_part(p.w.d, e0, row, acts, k), pb = dn_part(p.w.d, e1, row, acts, k + 1);
      sum = fmaf(pa * p.w.ds[size_t(e0) * H + row], p.routes[k], sum);
      sum = fmaf(pb * p.w.ds[size_t(e1) * H + row], p.routes[k + 1], sum); }
    p.out[row] = sum; }
}
__device__ __forceinline__ void dn_trans(const Params &p, const uint32_t *dT) {   // [e][row][8 words]: 2 x 16 B per (e,row)
  __shared__ float acts[K * W]; for (int i = threadIdx.x; i < K * W; i += blockDim.x) acts[i] = p.act[i]; __syncthreads();
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < H; row += gridDim.x * blockDim.x) { float sum = 0;
#pragma unroll 2
    for (int k = 0; k < K; k++) { const int e = p.ids[k];
      const uint4 *src = reinterpret_cast<const uint4 *>(dT + (size_t(e) * H + row) * (W / 4));
      const uint4 A = __ldg(src), B = __ldg(src + 1); const uint32_t v[8] = {A.x, A.y, A.z, A.w, B.x, B.y, B.z, B.w};
      float part = 0;
#pragma unroll
      for (int q = 0; q < W / 4; q++)
#pragma unroll
        for (int j = 0; j < 4; j++) part = fmaf(fp8e4m3(v[q] >> (8 * j)), acts[k * W + q * 4 + j], part);
      sum = fmaf(part * p.w.ds[size_t(e) * H + row], p.routes[k], sum); }
    p.out[row] = sum; }
}
// 4 threads per row: thread s computes part*ds for k = s, s+4, s+8 into shared, then s == 0 chains k = 0..K-1 in order.
// Needs blockDim / 4 rows per block (64), so H / 64 = 40 blocks cover every row once.
__device__ __forceinline__ void dn_split4(const Params &p) {
  __shared__ float acts[K * W]; __shared__ float pd[64][K];
  for (int i = threadIdx.x; i < K * W; i += blockDim.x) acts[i] = p.act[i]; __syncthreads();
  const int s = threadIdx.x & 3, lr = threadIdx.x >> 2;
  for (int rb = blockIdx.x * 64; rb < H; rb += gridDim.x * 64) { const int row = rb + lr;
    for (int k = s; k < K; k += 4) { const int e = p.ids[k]; pd[lr][k] = dn_part(p.w.d, e, row, acts, k) * p.w.ds[size_t(e) * H + row]; }
    __syncthreads();
    if (!s) { float sum = 0; for (int k = 0; k < K; k++) sum = fmaf(pd[lr][k], p.routes[k], sum); p.out[row] = sum; }
    __syncthreads(); }
}
// S threads per row (S = 2: 128 rows per block, 20 blocks cover H once): thread s computes part*ds for k = s, s+S, ...
// into pd (the x buffer), then s == 0 chains k = 0..K-1 in order (bit-identical to the deployed down).
template <int S>
__device__ __forceinline__ void dn_split(const Params &p, float *pd) {
  __shared__ float acts[K * W]; for (int i = threadIdx.x; i < K * W; i += blockDim.x) acts[i] = p.act[i]; __syncthreads();
  constexpr int RB = 256 / S; const int s = threadIdx.x % S, lr = threadIdx.x / S;
  for (int rb = blockIdx.x * RB; rb < H; rb += gridDim.x * RB) { const int row = rb + lr;
    for (int k = s; k < K; k += S) { const int e = p.ids[k]; pd[lr * K + k] = dn_part(p.w.d, e, row, acts, k) * p.w.ds[size_t(e) * H + row]; }
    __syncthreads();
    if (!s) { float sum = 0; for (int k = 0; k < K; k++) sum = fmaf(pd[lr * K + k], p.routes[k], sum); p.out[row] = sum; }
    __syncthreads(); }
}
// ---- one kernel per (gate/up variant, down variant, min blocks per SM); gu: 0 prod, 1 hoist, 2 pair (both with the
// deployed 2*H buffer), 3 deployed q38_routed_v2_gate_up with the 1-row buffer, 4 pair with the 1-row buffer;
// dn: 0 prod, 1 k2, 2 trans, 3 split4, 5 split2 (pd in the 1-row buffer), 9 none. if constexpr: a variant only
// allocates the shared memory of the paths it uses (2026-11-05: plain ifs reserved q38_routed_xbuf in every kernel). MINB = blocks per SM the grid needs (launch bounds = register cap:
// 2 -> 128 registers as the deployed kernel, 4 -> 64, 8 -> 32). Per iteration block 0 thread 0 records
// t0 (iteration start = previous barrier), t1 (after the gate/up barrier), t2 (block 0's down done), t3 (after the down barrier).
template <int GU, int DN, int MINB>
__global__ void __launch_bounds__(256, MINB) bench(Args a) {
  unsigned gen = 0; const int lane = threadIdx.x & 31, gwarp = int(blockIdx.x) * int(blockDim.x >> 5) + int(threadIdx.x >> 5),
                         nwarps = int(gridDim.x) * int(blockDim.x >> 5);
  const bool rec = !blockIdx.x && !threadIdx.x;
  for (int it = 0; it < WARM + a.iters; it++) {
    unsigned long long t0 = 0, t1 = 0, t2 = 0, t3 = 0; if (rec) t0 = gt();
    const int set = it % NSETS; const int32_t *ids = a.ids + set * K; const float *rt = a.routes + set * K;
    const uint32_t *xw = a.xw + size_t(set) * (H / 2);
    Params p{a.w, reinterpret_cast<const float *>(xw), rt, ids, a.act, a.out};
    if constexpr (GU == 0) dynamic_gate_layer(p, true);             // deployed: BF16 staging + q38_routed_v2_gate_up
    else if constexpr (GU == 1 || GU == 2) { const float *xs = stage_x(xw);
      if constexpr (GU == 1) gu_hoist(a.w, ids, xs, a.act, gwarp, nwarps, lane); else gu_pair(a.w, ids, xs, a.act, gwarp, nwarps, lane); }
    else { const float *xs = stage_x_small(xw);
      if constexpr (GU == 3) q38_routed_v2_gate_up(a.w.g, a.w.u, a.w.gs, a.w.us, ids, xs, a.act, gwarp, nwarps, lane, H, W, K, GROUP);
      else gu_pair(a.w, ids, xs, a.act, gwarp, nwarps, lane); }
    gbar(a.bar, gen); if (rec) t1 = gt();
    if constexpr (DN == 0) dynamic_down_layer(p);                    // deployed down
    else if constexpr (DN == 1) dn_k2(p); else if constexpr (DN == 2) dn_trans(p, a.dT); else if constexpr (DN == 3) dn_split4(p);
    else if constexpr (DN == 5) dn_split<2>(p, small_xbuf());
    __syncthreads(); if (rec) t2 = gt();
    gbar(a.bar, gen); if (rec) t3 = gt();
    if (rec && it >= WARM) { unsigned long long *r = a.ts + size_t(it - WARM) * 4; r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; }
  }
}
static void ck2(cudaError_t e, const char *w) { if (e != cudaSuccess) { fprintf(stderr, "ERROR,%s,%s\n", w, cudaGetErrorString(e)); std::exit(1); } }
#define C2(x) ck2((x), #x)
template <int GU, int DN, int MINB> const void *kptr() { return reinterpret_cast<const void *>(&bench<GU, DN, MINB>); }

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: bench_m10_expert_compute OUT_JSON [iters]\n"); return 2; }
  const int iters = argc > 2 ? std::atoi(argv[2]) : 2000;
  // device: first Tesla M10 in PCI order with >= 300 MB free
  int ndev = 0; C2(cudaGetDeviceCount(&ndev)); int dev = -1; char bdf[32] = {0}; size_t freeb = 0, totalb = 0;
  for (int d = 0; d < ndev && dev < 0; d++) { cudaDeviceProp pr; C2(cudaGetDeviceProperties(&pr, d));
    if (std::string(pr.name).find("M10") == std::string::npos) continue;
    C2(cudaSetDevice(d)); C2(cudaMemGetInfo(&freeb, &totalb));
    if (freeb >= (300ull << 20)) { dev = d; snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.0", pr.pciDomainID, pr.pciBusID, pr.pciDeviceID); }
    else C2(cudaDeviceReset()); }
  if (dev < 0) { fprintf(stderr, "ERROR,no Tesla M10 with 300 MB free\n"); return 3; }
  cudaDeviceProp pr; C2(cudaGetDeviceProperties(&pr, dev));
  printf("DEVICE,cuda,%d,bdf,%s,sms,%d,clock_khz,%d,mem_clock_khz,%d,bus_bits,%d,free_mb,%zu\n", dev, bdf, pr.multiProcessorCount,
         pr.clockRate, pr.memoryClockRate, pr.memoryBusWidth, freeb >> 20);
  // synthetic weights (deterministic): g/u [E*W rows][H/4 words], scales [E*W][H/GROUP]; d [E][W/4][H] + transposed [E][H][W/4]; ds [E][H]
  std::mt19937_64 rng(20261104);
  auto words = [&](size_t n) { std::vector<uint32_t> v(n); for (auto &x : v) x = uint32_t(rng()); return v; };
  auto floats = [&](size_t n, float lo, float hi) { std::uniform_real_distribution<float> u(lo, hi); std::vector<float> v(n); for (auto &x : v) x = u(rng); return v; };
  const size_t nrow = size_t(E) * W;
  auto g = words(nrow * (H / 4)), u = words(nrow * (H / 4)), d = words(size_t(E) * (W / 4) * H);
  std::vector<uint32_t> dT(d.size());
  for (int e = 0; e < E; e++) for (int q = 0; q < W / 4; q++) for (int row = 0; row < H; row++)
    dT[(size_t(e) * H + row) * (W / 4) + q] = d[(size_t(e) * (W / 4) + q) * H + row];
  auto gs = floats(nrow * (H / GROUP), 1e-3f, 2e-3f), us = floats(nrow * (H / GROUP), 1e-3f, 2e-3f), ds = floats(size_t(E) * H, 1e-3f, 2e-3f);
  std::vector<uint32_t> xw(size_t(NSETS) * (H / 2));
  { std::normal_distribution<float> nd(0.0f, 0.05f);
    for (auto &w2 : xw) { float f0 = nd(rng), f1 = nd(rng); uint32_t b0, b1; std::memcpy(&b0, &f0, 4); std::memcpy(&b1, &f1, 4); w2 = (b0 >> 16) | (b1 & 0xFFFF0000u); } }
  std::vector<int32_t> ids(size_t(NSETS) * K); std::vector<float> routes = floats(size_t(NSETS) * K, 0.02f, 0.2f);
  for (int s = 0; s < NSETS; s++) { std::vector<int32_t> all(E); for (int e = 0; e < E; e++) all[e] = e;
    for (int k = 0; k < K; k++) { std::uniform_int_distribution<int> pick(k, E - 1); std::swap(all[k], all[pick(rng)]); ids[s * K + k] = all[k]; } }
  auto up = [&](auto &v) { using T = typename std::decay_t<decltype(v)>::value_type; T *p = nullptr; C2(cudaMalloc(&p, v.size() * sizeof(T)));
    C2(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p; };
  Args a{}; a.w.g = up(g); a.w.u = up(u); a.w.d = up(d); a.w.gs = up(gs); a.w.us = up(us); a.w.ds = up(ds); a.dT = up(dT);
  a.xw = up(xw); a.ids = up(ids); a.routes = up(routes); a.iters = iters;
  C2(cudaMalloc(&a.act, K * W * sizeof(float))); C2(cudaMalloc(&a.out, H * sizeof(float))); C2(cudaMalloc(&a.bar, sizeof(Bar)));
  C2(cudaMalloc(&a.ts, size_t(iters) * 4 * sizeof(unsigned long long)));
  size_t f2 = 0; C2(cudaMemGetInfo(&f2, &totalb)); printf("ALLOC_DONE,free_mb_after,%zu\n", f2 >> 20);
  struct V { const char *name; const void *k; int gu, dn, blocks; };
  std::vector<V> vs = {
    {"prod_gu+prod_dn_b10", kptr<0, 0, 2>(), 0, 0, 10}, {"prod_gu+prod_dn_b20", kptr<0, 0, 4>(), 0, 0, 20}, {"prod_gu+prod_dn_b40", kptr<0, 0, 8>(), 0, 0, 40},
    {"hoist_gu+k2_dn_b10", kptr<1, 1, 2>(), 1, 1, 10}, {"hoist_gu+k2_dn_b20", kptr<1, 1, 4>(), 1, 1, 20}, {"hoist_gu+k2_dn_b40", kptr<1, 1, 8>(), 1, 1, 40},
    {"pair_gu+trans_dn_b10", kptr<2, 2, 2>(), 2, 2, 10}, {"pair_gu+trans_dn_b20", kptr<2, 2, 4>(), 2, 2, 20}, {"pair_gu+trans_dn_b40", kptr<2, 2, 8>(), 2, 2, 40},
    {"hoist_gu+split4_dn_b40", kptr<1, 3, 8>(), 1, 3, 40}, {"prod_gu+none_b10", kptr<0, 9, 2>(), 0, 9, 10},
    // 2026-11-05 extension: 1-row x buffer (10 KB), 15 / 20 / 25 blocks (register caps 85 / 64 / 51)
    {"prodsmall_gu+prod_dn_b10", kptr<3, 0, 2>(), 3, 0, 10}, {"prodsmall_gu+prod_dn_b15", kptr<3, 0, 3>(), 3, 0, 15},
    {"prodsmall_gu+prod_dn_b20", kptr<3, 0, 4>(), 3, 0, 20}, {"prodsmall_gu+prod_dn_b25", kptr<3, 0, 5>(), 3, 0, 25},
    {"pairsmall_gu+prod_dn_b15", kptr<4, 0, 3>(), 4, 0, 15}, {"pairsmall_gu+prod_dn_b20", kptr<4, 0, 4>(), 4, 0, 20},
    {"pairsmall_gu+prod_dn_b25", kptr<4, 0, 5>(), 4, 0, 25},
    {"prodsmall_gu+split2_dn_b20", kptr<3, 5, 4>(), 3, 5, 20}, {"pairsmall_gu+split2_dn_b20", kptr<4, 5, 4>(), 4, 5, 20},
    {"pairsmall_gu+trans_dn_b20", kptr<4, 2, 4>(), 4, 2, 20}, {"pairsmall_gu+trans_dn_b15", kptr<4, 2, 3>(), 4, 2, 15},
  };
  std::vector<float> ref_act, ref_out; FILE *jf = fopen(argv[1], "w"); if (!jf) { perror("out"); return 4; }
  fprintf(jf, "{\"device\": {\"cuda\": %d, \"bdf\": \"%s\", \"sms\": %d, \"clock_khz\": %d, \"mem_clock_khz\": %d, \"bus_bits\": %d}, \"iters\": %d, \"warm\": %d, \"nsets\": %d, \"variants\": [\n",
          dev, bdf, pr.multiProcessorCount, pr.clockRate, pr.memoryClockRate, pr.memoryBusWidth, iters, WARM, NSETS);
  bool first = true;
  for (auto &v : vs) {
    int per_sm = 0; C2(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, v.k, 256, 0));
    cudaFuncAttributes fa; C2(cudaFuncGetAttributes(&fa, v.k));
    if (per_sm * pr.multiProcessorCount < v.blocks) { printf("SKIP,%s,blocks,%d,max_coresident,%d,regs,%d,smem_b,%zu\n", v.name, v.blocks, per_sm * pr.multiProcessorCount, fa.numRegs, fa.sharedSizeBytes); continue; }
    auto run = [&](int n) { Args b = a; b.iters = n; C2(cudaMemset(a.bar, 0, sizeof(Bar))); void *args[] = {&b};
      C2(cudaLaunchKernel(v.k, dim3(v.blocks), dim3(256), args, 0, 0)); C2(cudaDeviceSynchronize());
      Bar hb; C2(cudaMemcpy(&hb, a.bar, sizeof hb, cudaMemcpyDeviceToHost)); if (hb.error) { fprintf(stderr, "ERROR,grid barrier timeout in %s\n", v.name); std::exit(5); } };
    // bit-identity: every variant runs the same n_check iterations, so its last iteration uses the same expert set
    const int n_check = NSETS - (WARM % NSETS);
    run(n_check);
    std::vector<float> act(K * W), out(H); C2(cudaMemcpy(act.data(), a.act, act.size() * 4, cudaMemcpyDeviceToHost)); C2(cudaMemcpy(out.data(), a.out, out.size() * 4, cudaMemcpyDeviceToHost));
    if (ref_act.empty()) { ref_act = act; ref_out = out; }
    const bool act_same = std::memcmp(act.data(), ref_act.data(), act.size() * 4) == 0;
    const bool out_same = v.dn == 9 || std::memcmp(out.data(), ref_out.data(), out.size() * 4) == 0;
    run(iters);
    std::vector<unsigned long long> ts(size_t(iters) * 4); C2(cudaMemcpy(ts.data(), a.ts, ts.size() * 8, cudaMemcpyDeviceToHost));
    std::vector<double> gu, dn0, dn; for (int i = 1; i < iters; i++) { const auto *r = &ts[size_t(i) * 4];
      gu.push_back((r[1] - r[0]) / 1e3); dn0.push_back((r[2] - r[1]) / 1e3); dn.push_back((r[3] - r[1]) / 1e3); }
    auto stat = [](std::vector<double> x) { std::sort(x.begin(), x.end()); double m = 0; for (double y : x) m += y; return std::array<double, 3>{x[x.size() / 2], m / x.size(), x[x.size() * 9 / 10]}; };
    const auto G = stat(gu), D0 = stat(dn0), D = stat(dn);
    const double gb_gu = (2.0 * K * W * H + 2.0 * K * W * (H / GROUP) * 4) / (G[1] * 1e3), gb_dn = (1.0 * K * W * H + K * H * 4.0) / (D[1] * 1e3);
    printf("RESULT,%s,blocks,%d,regs,%d,smem_b,%zu,local_b,%zu,max_blocks_per_sm,%d,gateup_us_med,%.2f,mean,%.2f,p90,%.2f,down_b0_us_med,%.2f,down_all_us_med,%.2f,mean,%.2f,p90,%.2f,gateup_GBps,%.1f,down_GBps,%.1f,act_identical,%d,out_identical,%d\n",
           v.name, v.blocks, fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes, per_sm, G[0], G[1], G[2], D0[0], D[0], D[1], D[2], gb_gu, gb_dn, act_same, out_same);
    fprintf(jf, "%s  {\"name\": \"%s\", \"blocks\": %d, \"regs\": %d, \"max_blocks_per_sm\": %d, \"gateup_us\": {\"median\": %.3f, \"mean\": %.3f, \"p90\": %.3f}, "
                "\"smem_b\": %zu, \"down_block0_us_median\": %.3f, \"down_us\": {\"median\": %.3f, \"mean\": %.3f, \"p90\": %.3f}, \"gateup_GBps\": %.2f, \"down_GBps\": %.2f, "
                "\"act_identical\": %s, \"out_identical\": %s}", first ? "" : ",\n", v.name, v.blocks, fa.numRegs, per_sm, G[0], G[1], G[2], fa.sharedSizeBytes, D0[0], D[0], D[1], D[2], gb_gu, gb_dn,
            act_same ? "true" : "false", out_same ? "true" : "false");
    first = false; fflush(stdout);
  }
  fprintf(jf, "\n]}\n"); fclose(jf);
  printf("BENCH_DONE,%s\n", argv[1]);
  return 0;
}
