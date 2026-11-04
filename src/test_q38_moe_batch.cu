// Offline hardware test of worker mode 8 (batched MoE, 2026-11-03) against mode 1 on one M10 card:
// four persistent worker grids (one per GPU of the card), random weights for one layer, the shared CPU
// mailbox (with the batch area) in mapped host memory, control blocks in mapped host memory laid out
// like the services'. For every batch the reference is mode 1 row by row WITHOUT aggregation (each
// worker's own result), then mode 8 WITH the card's aggregation roles; the batch output must be
// bit-identical:
//   routed (default): leader area == ((own0 + own1) + own2) + own3 (the mode-1 leader order), streamed
//                     through emulated aggregation windows (v2 step 3); flags == seq, slot headers acknowledged.
//   shared (-DTEST_SHARED): worker 20 area == ((0 + s20) + s21) + s22) + s23 (the RTX's mode-1 order),
//                     gate[row] == worker 0's mode-1 gate, other areas == own rows.
// Usage: test_q38_moe_batch "bdf,bdf,bdf,bdf"   (the card's four GPUs, from qwen38_pci_discover)
#ifdef TEST_SHARED
#define QWEN_M10_SHARED_LIBRARY_ONLY
#include "bench_resident_multilayer_shared.cu"
#include "resident_dynamic_shared_v2.cuh"
using Control = SharedResidentControl; using Return = SharedResidentReturn; using Grid = SharedResidentGrid;
static const char *kKernel = "resident_dynamic_shared_v2";
static constexpr unsigned kWorker0 = 20;
#else
#define QWEN_M10_LIBRARY_ONLY
#include "bench_resident_multilayer_fp8.cu"
#include "resident_dynamic_routed_v3.cuh"
using Control = ResidentDynamicControl; using Return = ResidentDynamicReturn; using Grid = ResidentDynamicGrid;
static const char *kKernel = "resident_dynamic_routed_v3";
static constexpr unsigned kWorker0 = 0;
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define T_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA,%s,%s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
static int g_fail = 0;
static void check(bool ok, const std::string &n, const std::string &d = "") {
    std::printf("CHECK,%s,%s%s%s\n", n.c_str(), ok ? "OK" : "FAIL", d.empty() ? "" : ",", d.c_str()); if (!ok) g_fail++;
}
static bool await_u32(const volatile unsigned *p, unsigned v, int ms) {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) { if (*p == v) return true; }   // spin: the test also times requests
    return false;
}
__global__ void fill_bytes(uint8_t *p, size_t n, uint32_t seed, int kind) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint32_t h = uint32_t(i) * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        if (kind == 0) p[i] = uint8_t(h & 0x77);                       // FP8 E4M3, |v| < 1 (exp <= 7)
        else if (kind == 1) reinterpret_cast<float *>(p)[i] = 0.002f + float(h & 1023) * 1e-5f;    // scales
        else reinterpret_cast<uint16_t *>(p)[i] = uint16_t(0x3c00u | (h & 0x01ff) | ((h >> 9) & 1) << 15);  // BF16 ~[-0.01,0.01]
    }
}
static std::string canon(std::string b) {
    for (auto &c : b) c = char(std::tolower(static_cast<unsigned char>(c)));
    const auto p = b.find(':'); return b.substr(p == std::string::npos ? 0 : p + 1);
}

struct Dev {
    int ord; unsigned idx;                  // CUDA ordinal, global worker index
    uint8_t *hc, *dc; Return *hr, *dr; Grid *grid; float *act, *local, *partial; void *scratch;
#ifdef TEST_SHARED
    Weights *table;
#else
    DeviceWeights *table;
#endif
    cudaStream_t st;
    Control *C() { return reinterpret_cast<Control *>(hc); }
    ResidentDirectResult *D() { return reinterpret_cast<ResidentDirectResult *>(hc + Q38_DIRECT_RESULT_OFF); }
};
static uint8_t *g_mb = nullptr;              // mailbox (host), UVA pointer valid on every GPU
static unsigned g_seq = 0;

static void launch(Dev &v) {
    T_CK(cudaSetDevice(v.ord)); T_CK(cudaMemset(v.grid, 0, sizeof(Grid))); v.hr->ready = 0; v.D()->ready = 0;
#ifdef TEST_SHARED
    resident_dynamic_shared<<<10, T, 0, v.st>>>(v.table, reinterpret_cast<Control *>(v.dc), v.grid, v.partial, v.dr, nullptr,
                                                  v.idx - 20, 30000000000ULL);
#else
    resident_dynamic_routed<<<10, THREADS, 0, v.st>>>(v.table, reinterpret_cast<Control *>(v.dc), v.grid, v.act, v.local, v.dr, nullptr, 30000000000ULL);
#endif
    T_CK(cudaGetLastError());
}
static bool publish(std::vector<Dev> &ds, unsigned mode, unsigned layer, unsigned count) {
    ++g_seq;
    for (auto &v : ds) { v.C()->layer = layer; v.C()->ple_count = count; v.C()->mode = mode; }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    for (auto &v : ds) v.C()->seq = g_seq;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    bool ok = true;
    for (auto &v : ds) ok &= await_u32(&v.D()->seq, g_seq, 30000);
    return ok;
}
static void set_agg(std::vector<Dev> &ds, bool on) {
#ifndef TEST_SHARED
    for (size_t j = 0; j < ds.size(); j++) {
        auto *a = reinterpret_cast<volatile unsigned *>(ds[j].hc + Q38_AGG_OFF);
        a[0] = on ? Q38_AGG_MAGIC : 0u; a[1] = j == 0 ? 1u : 2u; a[2] = unsigned(j); a[3] = 3u;
        // aggregation windows as the peer library lays them out: leader control + 0x28000 + (peer-1) * 0x4000;
        // here the controls are mapped host memory, so a peer's push_va is the leader's UVA pointer
        const unsigned long long push = j ? reinterpret_cast<unsigned long long>(ds[0].dc) + 0x28000ull + (j - 1) * 0x4000ull : 0ull;
        a[4] = unsigned(push); a[5] = unsigned(push >> 32); a[6] = 0x28000u; a[7] = 0x4000u;
    }
    static_assert(0x28000 + 0x10000 <= Q38_WORKER_BAR_MAP_BYTES, "aggregation windows outside the control map");
#else
    (void)ds; (void)on;
#endif
}

int main(int argc, char **argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s 'bdf,bdf,bdf,bdf'\n", argv[0]); return 2; }
    std::vector<std::string> bdfs; { std::stringstream ss(argv[1]); std::string b; while (std::getline(ss, b, ',')) bdfs.push_back(canon(b)); }
    int count = 0; T_CK(cudaGetDeviceCount(&count));
    std::map<std::string, int> ord; for (int d = 0; d < count; d++) { char b[32]; T_CK(cudaDeviceGetPCIBusId(b, sizeof(b), d)); ord[canon(b)] = d; }
    std::printf("Q38_MOE_BATCH_TEST,kernel,%s,gpus,%s\n", kKernel, argv[1]);
    T_CK(cudaSetDevice(ord.at(bdfs[0])));
    T_CK(cudaHostAlloc(&g_mb, Q38_CPU_MAILBOX_BYTES, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(g_mb, 0, Q38_CPU_MAILBOX_BYTES);
    std::vector<Dev> ds(4);
    const int layer = 0;
    for (int j = 0; j < 4; j++) {
        Dev &v = ds[j]; v.ord = ord.at(bdfs[j]); v.idx = kWorker0 + unsigned(j);
        T_CK(cudaSetDevice(v.ord)); T_CK(cudaStreamCreateWithFlags(&v.st, cudaStreamNonBlocking));
        T_CK(cudaHostAlloc(&v.hc, Q38_WORKER_BAR_MAP_BYTES, cudaHostAllocMapped | cudaHostAllocPortable)); std::memset(v.hc, 0, Q38_WORKER_BAR_MAP_BYTES);
        T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&v.dc), v.hc, 0));
        T_CK(cudaHostAlloc(&v.hr, sizeof(Return), cudaHostAllocMapped | cudaHostAllocPortable)); std::memset(v.hr, 0, sizeof(Return));
        T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&v.dr), v.hr, 0));
        T_CK(cudaMalloc(&v.grid, sizeof(Grid))); T_CK(cudaMalloc(&v.act, K * 32 * sizeof(float))); T_CK(cudaMalloc(&v.local, H * sizeof(float)));
        T_CK(cudaMalloc(&v.partial, 10 * H * sizeof(float))); T_CK(cudaMalloc(&v.scratch, Q38_BATCH_SCRATCH_BYTES));
        void *mb_dev = nullptr; T_CK(cudaHostGetDevicePointer(&mb_dev, g_mb, 0));
        v.C()->cpu_mailbox_worker = v.idx; v.C()->cpu_mailbox_reserved = Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN;
        v.C()->cpu_mailbox_device_ptr = reinterpret_cast<unsigned long long>(static_cast<char *>(mb_dev) + size_t(v.idx) * Q38_CPU_MAILBOX_SLOT_STRIDE);
        v.C()->batch_area = reinterpret_cast<unsigned long long>(static_cast<char *>(mb_dev) + Q38_BATCH_OFF);
        v.C()->batch_scratch = reinterpret_cast<unsigned long long>(v.scratch); v.C()->batch_scratch_bytes = unsigned(Q38_BATCH_SCRATCH_BYTES);
        const uint32_t seed = 0x9e37u * (j + 1);
#ifdef TEST_SHARED
        Weights w{}; const size_t gb = size_t(S) * H * 2;
        T_CK(cudaMalloc(&w.g, gb)); T_CK(cudaMalloc(&w.u, gb)); T_CK(cudaMalloc(&w.d, gb)); T_CK(cudaMalloc(&w.gate, H * 2));
        fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.g), gb / 2, seed + 1, 2); fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.u), gb / 2, seed + 2, 2);
        fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.d), gb / 2, seed + 3, 2); fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.gate), H, seed + 4, 2);
        std::vector<Weights> tab(48, w); T_CK(cudaMalloc(&v.table, 48 * sizeof(Weights)));
        T_CK(cudaMemcpy(v.table, tab.data(), 48 * sizeof(Weights), cudaMemcpyHostToDevice));
#else
        DeviceWeights w{}; const size_t wb = size_t(E) * W * H, sb = size_t(E) * W * (H / GROUP) * 4, db = size_t(E) * H * 4;
        T_CK(cudaMalloc(&w.g, wb)); T_CK(cudaMalloc(&w.u, wb)); T_CK(cudaMalloc(&w.d, wb));
        T_CK(cudaMalloc(&w.gs, sb)); T_CK(cudaMalloc(&w.us, sb)); T_CK(cudaMalloc(&w.ds, db));
        fill_bytes<<<256, 256>>>(reinterpret_cast<uint8_t *>(w.g), wb, seed + 1, 0); fill_bytes<<<256, 256>>>(reinterpret_cast<uint8_t *>(w.u), wb, seed + 2, 0);
        fill_bytes<<<256, 256>>>(reinterpret_cast<uint8_t *>(w.d), wb, seed + 3, 0);
        fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.gs), sb / 4, seed + 4, 1); fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.us), sb / 4, seed + 5, 1);
        fill_bytes<<<64, 256>>>(reinterpret_cast<uint8_t *>(w.ds), db / 4, seed + 6, 1);
        std::vector<DeviceWeights> tab(48, w); T_CK(cudaMalloc(&v.table, 48 * sizeof(DeviceWeights)));
        T_CK(cudaMemcpy(v.table, tab.data(), 48 * sizeof(DeviceWeights), cudaMemcpyHostToDevice));
#endif
        T_CK(cudaDeviceSynchronize());
    }
    for (auto &v : ds) launch(v);
    for (auto &v : ds) if (!await_u32(&v.hr->ready, 1, 15000)) { std::fprintf(stderr, "startup timeout\n"); return 3; }

    char *bt = reinterpret_cast<char *>(g_mb) + Q38_BATCH_OFF;
    std::mt19937 rng(38); std::normal_distribution<float> nx(0.f, 1.f); std::uniform_real_distribution<float> ur(0.01f, 0.3f);
    for (unsigned R : {1u, 17u, 64u}) {
        // rows: x, 10 distinct experts from a narrow range (rows share experts), routes
        std::vector<float> x(size_t(R) * H), routes(size_t(R) * 10); std::vector<int32_t> ids(size_t(R) * 10);
        for (auto &f : x) f = nx(rng);
        for (unsigned r = 0; r < R; r++) {
            std::vector<int> pool; for (int e = 0; e < 48; e++) pool.push_back((e * 37 + int(R)) % 512);
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int k = 0; k < 10; k++) { ids[r * 10 + k] = pool[k]; routes[r * 10 + k] = ur(rng); }
        }
        // reference: mode 1 per row, no aggregation -> each worker's own row
        set_agg(ds, false);
        std::vector<std::vector<float>> own(4, std::vector<float>(size_t(R) * H)); std::vector<float> gate(R, 0.f);
        bool ref_ok = true; double t_mode1_us = 0.0;
        for (unsigned r = 0; r < R; r++) {
            for (auto &v : ds) {
                for (int i = 0; i < H; i++) v.C()->x[i] = x[size_t(r) * H + i];
#ifndef TEST_SHARED
                for (int k = 0; k < 10; k++) { v.C()->ids[k] = ids[r * 10 + k]; v.C()->routes[k] = routes[r * 10 + k]; }
#endif
            }
            const auto tr0 = std::chrono::steady_clock::now();
            ref_ok &= publish(ds, 1, layer, 0);
            t_mode1_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tr0).count();
            for (int j = 0; j < 4; j++) {
                ref_ok &= ds[j].D()->status == 0;
                std::memcpy(&own[j][size_t(r) * H], const_cast<const float *>(ds[j].D()->result), H * 4);
            }
            gate[r] = ds[0].D()->gate;
        }
        check(ref_ok, "reference_mode1_R" + std::to_string(R));
        // batch: request rows into the mailbox batch area, mode 8 with aggregation roles
        std::memcpy(bt + Q38_BATCH_REQ_X, x.data(), x.size() * 4);
        std::memcpy(bt + Q38_BATCH_REQ_IDS, ids.data(), ids.size() * 4);
        std::memcpy(bt + Q38_BATCH_REQ_ROUTES, routes.data(), routes.size() * 4);
        set_agg(ds, true);
        const auto tb0 = std::chrono::steady_clock::now();
        const bool acked = publish(ds, Q38_BATCH_MODE, layer, R);
        const double t_batch_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - tb0).count();
        bool status_ok = acked;
        for (auto &v : ds) {
            status_ok &= v.D()->status == 0;
            const auto *slot = reinterpret_cast<const volatile Q38CpuMailboxSlot *>(g_mb + size_t(v.idx) * Q38_CPU_MAILBOX_SLOT_STRIDE);
            status_ok &= await_u32(&slot->seq, g_seq, 10000);   // the slot is acknowledged after `direct` (as the RTX polls it)
            status_ok &= slot->seq == g_seq && slot->status == 0 && slot->worker == 0 && slot->layer == 1;
            status_ok &= *reinterpret_cast<const volatile unsigned *>(bt + Q38_BATCH_PART0 + size_t(v.idx) * Q38_BATCH_PART_STRIDE + Q38_BATCH_PART_ROWS_BYTES) == g_seq;
        }
        check(status_ok, "batch_ack_flags_R" + std::to_string(R), "exit," + std::to_string(ds[0].D()->exit_reason));
        // expected outputs (mode-1 summation orders)
        size_t mism = 0, n = size_t(R) * H;
        for (int j = 0; j < 4; j++) {
            const float *got = reinterpret_cast<const float *>(bt + Q38_BATCH_PART0 + size_t(ds[j].idx) * Q38_BATCH_PART_STRIDE);
            for (size_t i = 0; i < n; i++) {
                float want;
#ifdef TEST_SHARED
                if (j == 0) { want = 0.0f; for (int q = 0; q < 4; q++) want += own[q][i]; } else want = own[j][i];
#else
                if (j) break;   // since v2 step 3 peers stream rows to the leader and write no part area
                want = own[0][i]; for (int q = 1; q < 4; q++) want += own[q][i];
#endif
                uint32_t a, b; std::memcpy(&a, &want, 4); std::memcpy(&b, &got[i], 4);
                if (a != b) { if (!mism) std::printf("FIRST_MISMATCH,worker,%u,index,%zu,want,%.9g,got,%.9g\n", ds[j].idx, i, want, got[i]); mism++; }
            }
        }
#ifdef TEST_SHARED
        for (unsigned r = 0; r < R; r++) {
            const float g = reinterpret_cast<const float *>(bt + Q38_BATCH_GATE)[r];
            uint32_t a, b; std::memcpy(&a, &gate[r], 4); std::memcpy(&b, &g, 4); if (a != b) mism++;
        }
#endif
        std::printf("TIMING,kernel,%s,R,%u,mode1_sum_us,%.1f,mode1_per_row_us,%.1f,batch_us,%.1f,batch_per_row_us,%.1f,speedup,%.2f\n",
                    kKernel, R, t_mode1_us, t_mode1_us / R, t_batch_us, t_batch_us / R, t_mode1_us / t_batch_us);
        check(mism == 0, "batch_bit_identical_R" + std::to_string(R), "mismatches," + std::to_string(mism) + ",values," + std::to_string(n * 4));
    }
    // error case: invalid expert id -> every routed worker exits with Q38_BATCH_REASON_ROWS (shared: R=0 config)
    {
#ifdef TEST_SHARED
        const bool acked = publish(ds, Q38_BATCH_MODE, layer, 0);
        const unsigned bit = Q38_BATCH_REASON_CONFIG;
#else
        reinterpret_cast<int32_t *>(bt + Q38_BATCH_REQ_IDS)[3] = 600;
        const bool acked = publish(ds, Q38_BATCH_MODE, layer, 2);
        const unsigned bit = Q38_BATCH_REASON_ROWS;
#endif
        bool ok = acked;
        for (auto &v : ds) ok &= v.D()->status == 1 && (v.D()->exit_reason & bit);
        for (auto &v : ds) { T_CK(cudaSetDevice(v.ord)); T_CK(cudaStreamSynchronize(v.st)); }
        check(ok, "batch_rejects_invalid_request", "exit," + std::to_string(ds[0].D()->exit_reason));
    }
    std::printf("Q38_MOE_BATCH_TEST,%s,kernel,%s,failures,%d\n", g_fail ? "FAIL" : "PASS", kKernel, g_fail);
    return g_fail ? 1 : 0;
}
