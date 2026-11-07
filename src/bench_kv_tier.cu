// Unit bench for q38_kv_resolve: replays one layer of a Q38_QSA_SELECT_LOG file on the RTX with synthetic K/V.
// Each token's K/V is a pattern of its position. After every call, each selected entry must read the pattern of
// the real token through the page table, as the attention kernel would. Usage: LOG LAYER CACHE_TOKENS OUT_JSON
// [REPEAT]. Prefill is simulated: positions below the first logged position are in host RAM.
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "qwen38_kv_tier.cuh"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x ": ") + cudaGetErrorString(e_)); } while (0)
static const uint32_t PAGE = 1568, TOKEN = 2048;
static inline uint8_t *q38_kv_addr_host(const std::vector<int32_t> &tab, uint8_t *kv, uint64_t stride, uint64_t tok) { return kv + tab[tok / PAGE] * stride + (tok % PAGE) * TOKEN; }
__host__ __device__ static inline uint32_t pat(uint64_t tok, uint32_t w) { return uint32_t(tok * 2654435761u) ^ (w * 40503u) ^ 0x5a5a1234u; }

__global__ static void fill_pattern(uint4 *p, uint64_t tok_first, uint64_t n) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * 128; i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t tok = tok_first + i / 128; const uint32_t w = uint32_t(i % 128);
        p[i] = make_uint4(pat(tok, w), pat(tok, w + 1000), pat(tok, w + 2000), pat(tok, w + 3000));
    }
}
__global__ static void verify(Q38KvTierArgs a, const int32_t *real, uint32_t n, uint32_t *bad) {
    for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const int32_t v = a.out[j]; const int32_t tok = real[j];
        const uint4 *x = reinterpret_cast<const uint4 *>(q38_kv_addr(a, uint64_t(v)));
        for (uint32_t w = 0; w < 128; w += 37)
            if (x[w].x != pat(tok, w) || x[w].w != pat(tok, w + 3000)) { atomicAdd(bad, 1u); break; }
    }
}

int main(int argc, char **argv) {
    try {
        if (argc < 5) throw std::runtime_error("usage: LOG LAYER CACHE_TOKENS OUT_JSON [REPEAT]");
        const int layer = atoi(argv[2]); const uint32_t cache_tokens = atoi(argv[3]); const int repeat = argc > 5 ? atoi(argv[5]) : 1;
        int dev = -1, ndev = 0; CK(cudaGetDeviceCount(&ndev));
        for (int i = 0; i < ndev; i++) { cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i)); if (strstr(p.name, "3070")) dev = i; }
        if (dev < 0) throw std::runtime_error("no RTX found");
        CK(cudaSetDevice(dev));
        std::ifstream f(argv[1], std::ios::binary); std::vector<int64_t> pos; std::vector<std::vector<int32_t>> sel;
        for (;;) { int32_t h[3]; if (!f.read(reinterpret_cast<char *>(h), 12)) break; std::vector<int32_t> v(h[2]); f.read(reinterpret_cast<char *>(v.data()), 4 * h[2]); if (!f) break; if (h[0] == layer) { pos.push_back(h[1]); sel.push_back(std::move(v)); } }
        if (pos.empty()) throw std::runtime_error("no records for this layer");
        const uint64_t ctx = pos.back() + 64; const uint32_t pages = uint32_t(ctx / PAGE + 2);
        // Pool: shuffled pages, as a real block table is not sequential.
        std::vector<int32_t> tab(pages); for (uint32_t i = 0; i < pages; i++) tab[i] = i; std::shuffle(tab.begin(), tab.end(), std::mt19937(38));
        const uint64_t stride = uint64_t(PAGE) * TOKEN; uint8_t *kv; int32_t *dtab, *b2s, *s2b, *din, *dout, *dreal; uint32_t *ep, *dbad; int64_t *dpos; Q38KvTierState *st;
        CK(cudaMalloc(&kv, stride * pages)); CK(cudaMalloc(&dtab, 4 * pages));
        CK(cudaMemcpy(dtab, tab.data(), 4 * pages, cudaMemcpyHostToDevice));
        const uint32_t slots = cache_tokens / 4, nblocks = uint32_t(ctx / 4 + 2), width = 2052;
        CK(cudaMalloc(&b2s, 4 * nblocks)); CK(cudaMalloc(&s2b, 4 * slots)); CK(cudaMalloc(&ep, 4 * slots)); CK(cudaMalloc(&st, sizeof(*st)));
        CK(cudaMalloc(&din, 4 * width)); CK(cudaMalloc(&dout, 4 * width)); CK(cudaMalloc(&dreal, 4 * width)); CK(cudaMalloc(&dpos, 8)); CK(cudaMalloc(&dbad, 4));
        void *hmem = mmap(nullptr, ctx * TOKEN, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (hmem == MAP_FAILED) throw std::runtime_error("mmap");
        CK(cudaHostRegister(hmem, ctx * TOKEN, cudaHostRegisterMapped | cudaHostRegisterPortable)); void *hdev; CK(cudaHostGetDevicePointer(&hdev, hmem, 0));
        // Host store below the first logged position; the pool's first pages hold the cache.
        fill_pattern<<<1024, 256>>>(reinterpret_cast<uint4 *>(hdev), 0, uint64_t(pos[0])); CK(cudaGetLastError());
        for (uint32_t p = 0; p * PAGE < cache_tokens + PAGE; p++) { fill_pattern<<<256, 256>>>(reinterpret_cast<uint4 *>(kv + tab[p] * stride), uint64_t(p) * PAGE, PAGE); CK(cudaGetLastError()); }
        std::vector<int32_t> hb(nblocks, -1), hs(slots), he(slots, 0); for (uint32_t s = 0; s < slots; s++) { hb[s] = s; hs[s] = s; }
        CK(cudaMemcpy(b2s, hb.data(), 4 * nblocks, cudaMemcpyHostToDevice)); CK(cudaMemcpy(s2b, hs.data(), 4 * slots, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ep, he.data(), 4 * slots, cudaMemcpyHostToDevice));
        Q38KvTierState hst{}; hst.enabled = 1; hst.slots = slots; CK(cudaMemcpy(st, &hst, sizeof hst, cudaMemcpyHostToDevice));
        uint32_t zero = 0; CK(cudaMemcpy(dbad, &zero, 4, cudaMemcpyHostToDevice));
        Q38KvTierArgs a{din, dout, width, 1, dpos, kv, stride, TOKEN, PAGE, dtab, reinterpret_cast<uint8_t *>(hdev), b2s, s2b, ep, st, nullptr};
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        std::vector<float> us; uint32_t bad_max = 0;
        for (int rep = 0; rep < repeat; rep++) for (size_t i = 0; i < pos.size(); i++) {
            std::vector<int32_t> row(width, -1); std::copy(sel[i].begin(), sel[i].end(), row.begin()); row[width - 1] = int32_t(sel[i].size());
            int64_t p = pos[i];
            CK(cudaMemcpy(din, row.data(), 4 * width, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dreal, row.data(), 4 * width, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dpos, &p, 8, cudaMemcpyHostToDevice));
            // What vLLM does just before: K and V at the real position.
            fill_pattern<<<1, 256>>>(reinterpret_cast<uint4 *>(q38_kv_addr_host(tab, kv, stride, p)), uint64_t(p), 1); CK(cudaGetLastError());
            CK(cudaEventRecord(e0)); q38_kv_resolve_kernel<<<1, 1024>>>(a); CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1)); us.push_back(ms * 1000);
            verify<<<16, 128>>>(a, dreal, uint32_t(sel[i].size()), dbad); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        }
        CK(cudaMemcpy(&bad_max, dbad, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hst, st, sizeof hst, cudaMemcpyDeviceToHost));
        std::vector<float> s = us; std::sort(s.begin(), s.end());
        const double hit = 1.0 - double(hst.misses) / std::max<uint64_t>(hst.selected, 1);
        char buf[600]; snprintf(buf, sizeof buf, "{\"layer\":%d,\"cache_tokens\":%u,\"calls\":%llu,\"bad_entries\":%u,\"hit\":%.4f,\"misses\":%llu,\"max_miss\":%llu,\"clock_steps\":%llu,\"us_p50\":%.1f,\"us_p99\":%.1f,\"us_max\":%.1f}\n",
            layer, cache_tokens, (unsigned long long)hst.calls, bad_max, hit, (unsigned long long)hst.misses, (unsigned long long)hst.max_miss, (unsigned long long)hst.clock_steps, s[s.size() / 2], s[s.size() * 99 / 100], s.back());
        printf("%s", buf); std::ofstream(argv[4]) << buf;
        CK(cudaHostUnregister(hmem)); return bad_max ? 2 : 0;
    } catch (const std::exception &e) { fprintf(stderr, "ERROR,%s\n", e.what()); return 1; }
}
