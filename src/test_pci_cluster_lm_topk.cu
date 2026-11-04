// Offline test of the PCI-cluster LM-head top-k (worker mode 7, pci_cluster_exchange.cuh) in the real
// resident worker kernels (-DTEST_SHARED: shared v2, else routed v3) on one GPU with a synthetic table:
// [PLE-like region][LM slice: R rows x 5120 B BF16][scratch: R x 4 B], ending exactly at the table end.
// Reference: host logits in double, rounded to BF16 (RNE). Checks per K: count, every returned value
// equals the host value within 1 BF16 ulp, no host value clearly above the returned minimum is missing,
// argmax, log-sum-exp; the RTX return slot carries the same words; five rejection cases.
#ifdef TEST_SHARED
#define QWEN_M10_SHARED_LIBRARY_ONLY
#include "bench_resident_multilayer_shared.cu"
#include "resident_dynamic_shared_v2.cuh"
using Control = SharedResidentControl; using Return = SharedResidentReturn; using Grid = SharedResidentGrid;
static const char *kKernel = "resident_dynamic_shared_v2";
#else
#define QWEN_M10_LIBRARY_ONLY
#include "bench_resident_multilayer_fp8.cu"
#include "resident_dynamic_routed_v3.cuh"
using Control = ResidentDynamicControl; using Return = ResidentDynamicReturn; using Grid = ResidentDynamicGrid;
static const char *kKernel = "resident_dynamic_routed_v3";
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define T_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA,%s,%s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)
static constexpr size_t kControlBytes = 0x40000;
static constexpr unsigned kMe = 9, kRows = 10347;
static constexpr size_t kPleBytes = 1u << 20;
static int g_fail = 0;
static uint8_t *hc = nullptr, *dc = nullptr; static Return *hr = nullptr, *dr = nullptr; static Grid *grid = nullptr;
static float *partial = nullptr; static uint8_t *table = nullptr; static Q38CpuMailboxSlot *slot = nullptr, *dslot = nullptr;
static unsigned seq = 0; static size_t table_bytes = 0, lm_off = 0;
static Control *C() { return reinterpret_cast<Control *>(hc); }
static ResidentDirectResult *D() { return reinterpret_cast<ResidentDirectResult *>(hc + Q38_DIRECT_RESULT_OFF); }

static bool await_u32(const volatile unsigned *p, unsigned v, int ms) {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) { if (*p == v) return true; std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    return false;
}
static void check(bool ok, const std::string &n, const std::string &d = "") {
    std::printf("CHECK,%s,%s%s%s\n", n.c_str(), ok ? "OK" : "FAIL", d.empty() ? "" : ",", d.c_str()); if (!ok) g_fail++;
}
static void launch() {
    T_CK(cudaMemset(grid, 0, sizeof(Grid))); hr->ready = 0; D()->ready = 0; std::atomic_thread_fence(std::memory_order_seq_cst);
#ifdef TEST_SHARED
    resident_dynamic_shared<<<10, T>>>(nullptr, reinterpret_cast<Control *>(dc), grid, partial, dr, table, 20u, 30000000000ULL);
#else
    resident_dynamic_routed<<<10, THREADS>>>(nullptr, reinterpret_cast<Control *>(dc), grid, nullptr, nullptr, dr, table, 30000000000ULL);
#endif
    T_CK(cudaGetLastError());
    if (!await_u32(&hr->ready, 1, 10000)) { std::fprintf(stderr, "startup timeout\n"); std::exit(3); }
}
static void configure(unsigned long long tb) {
    C()->cpu_mailbox_device_ptr = reinterpret_cast<unsigned long long>(dslot);
    C()->cpu_mailbox_worker = kMe; C()->cpu_mailbox_reserved = Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN;
    C()->ple_reserved[1] = unsigned(tb); C()->ple_reserved[2] = unsigned(tb >> 32);
}
static void set_request(unsigned k, unsigned long long off, unsigned long long rows, unsigned me = kMe) {
    C()->ple_count = k;
    for (unsigned w = 0; w < 24; w++) { C()->ple_offsets[2 * w] = 0xdead0000ull + w; C()->ple_offsets[2 * w + 1] = 7; }
    C()->ple_offsets[2 * me] = off; C()->ple_offsets[2 * me + 1] = rows;
}
static bool request() {
    C()->mode = PCI_CLUSTER_MODE_LM_TOPK; std::atomic_thread_fence(std::memory_order_seq_cst);
    C()->seq = ++seq; std::atomic_thread_fence(std::memory_order_seq_cst);
    return await_u32(&D()->seq, seq, 20000) && await_u32(&slot->seq, seq, 20000);
}
static unsigned bf16_rne(double v) { float f = float(v); unsigned u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return u >> 16; }
static float bf16_f(unsigned b) { unsigned u = b << 16; float f; std::memcpy(&f, &u, 4); return f; }

int main() {
    cudaDeviceProp prop{}; T_CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("PCI_CLUSTER_LM_TOPK_TEST,kernel,%s,device,%s\n", kKernel, prop.name);
    T_CK(cudaHostAlloc(&hc, kControlBytes, cudaHostAllocMapped | cudaHostAllocPortable)); std::memset(hc, 0, kControlBytes);
    T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dc), hc, 0));
    T_CK(cudaHostAlloc(&hr, sizeof(Return), cudaHostAllocMapped | cudaHostAllocPortable)); std::memset(hr, 0, sizeof(Return));
    T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dr), hr, 0));
    T_CK(cudaHostAlloc(&slot, sizeof(Q38CpuMailboxSlot), cudaHostAllocMapped | cudaHostAllocPortable)); std::memset(slot, 0, sizeof(*slot));
    T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dslot), slot, 0));
    T_CK(cudaMalloc(&grid, sizeof(Grid))); T_CK(cudaMalloc(&partial, 10 * H * sizeof(float)));
    lm_off = (kPleBytes + 4095) & ~size_t(4095);
    table_bytes = lm_off + size_t(kRows) * (PCI_CLUSTER_LM_ROW_BYTES + 4);
    T_CK(cudaMalloc(&table, table_bytes));
    // weights: random BF16 in [-0.05, 0.05] plus a few planted large rows; x: random BF16 values
    std::mt19937 rng(38); std::uniform_real_distribution<float> uw(-0.05f, 0.05f), ux(-2.f, 2.f);
    std::vector<uint16_t> w(size_t(kRows) * PCI_CLUSTER_LM_HIDDEN); std::vector<float> x(PCI_CLUSTER_LM_HIDDEN);
    for (auto &v : w) v = uint16_t(bf16_rne(uw(rng)));
    for (unsigned r : {17u, 5000u, 10346u}) for (unsigned i = 0; i < PCI_CLUSTER_LM_HIDDEN; i++) w[size_t(r) * PCI_CLUSTER_LM_HIDDEN + i] = uint16_t(bf16_rne(0.2 * (r % 3 + 1) * (i % 2 ? 1 : -1)));
    for (auto &v : x) v = bf16_f(bf16_rne(ux(rng)));
    T_CK(cudaMemset(table, 0x5A, lm_off));
    T_CK(cudaMemcpy(table + lm_off, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    T_CK(cudaMemset(table + lm_off + size_t(kRows) * PCI_CLUSTER_LM_ROW_BYTES, 0, size_t(kRows) * 4));
    for (unsigned i = 0; i < PCI_CLUSTER_LM_HIDDEN; i++) C()->x[i] = x[i];
    // host reference
    std::vector<unsigned> ref(kRows); double mx = -1e300;
    for (unsigned r = 0; r < kRows; r++) { double s = 0; for (unsigned i = 0; i < PCI_CLUSTER_LM_HIDDEN; i++) s += double(bf16_f(w[size_t(r) * PCI_CLUSTER_LM_HIDDEN + i])) * x[i];
        ref[r] = bf16_rne(s); mx = std::max(mx, double(bf16_f(ref[r]))); }
    double sum = 0; for (unsigned r = 0; r < kRows; r++) sum += std::exp(double(bf16_f(ref[r])) - mx);
    const double ref_lse = mx + std::log(sum);
    std::vector<unsigned> order(kRows); for (unsigned r = 0; r < kRows; r++) order[r] = r;
    std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return bf16_f(ref[a]) > bf16_f(ref[b]); });
    configure(table_bytes); launch();
    for (unsigned k : {1u, 1024u, PCI_CLUSTER_LM_MAX_K}) {
        set_request(k, lm_off, kRows);
        const bool acked = request();
        const auto *dw = reinterpret_cast<const unsigned *>(D()->result);
        const auto *sw = reinterpret_cast<const unsigned *>(slot->result);
        const unsigned n = dw[0];
        bool vals = true, slot_same = sw[0] == n && slot->status == 0 && slot->worker == 0 && slot->layer == 1;
        float ret_min = INFINITY; std::vector<char> got(kRows, 0); unsigned best_idx = 0; float best = -INFINITY;
        for (unsigned i = 1; i <= n && i <= PCI_CLUSTER_LM_MAX_K; i++) {
            const unsigned e = dw[i], idx = e & 0xffffu, b = e >> 16;
            if (sw[i] != e) slot_same = false;
            if (idx >= kRows || got[idx]) { vals = false; continue; }
            got[idx] = 1; const float v = bf16_f(b), rv = bf16_f(ref[idx]);
            if (std::fabs(v - rv) > std::ldexp(std::fabs(rv), -7) + 1e-30f) vals = false;      // <= 1 BF16 ulp
            ret_min = std::min(ret_min, v); if (v > best || (v == best && idx < best_idx)) { best = v; best_idx = idx; }
        }
        unsigned missing = 0;   // host values clearly (> 1 ulp) above the returned minimum but not returned
        for (unsigned r = 0; r < kRows; r++) { const float rv = bf16_f(ref[r]); if (!got[r] && rv > ret_min + std::ldexp(std::fabs(ret_min), -7)) missing++; }
        float lse; std::memcpy(&lse, (const void *)&slot->gate, 4);
        check(acked && D()->status == 0 && n == k && vals && missing == 0, "topk_" + std::to_string(k),
              "count," + std::to_string(n) + ",values_ok," + std::to_string(vals) + ",missing," + std::to_string(missing));
        check(bf16_f(ref[best_idx]) == bf16_f(ref[order[0]]), "argmax_" + std::to_string(k), "got," + std::to_string(best_idx) + ",ref," + std::to_string(order[0]));
        check(std::fabs(lse - ref_lse) < 1e-3 * std::max(1.0, std::fabs(ref_lse)), "lse_" + std::to_string(k), "got," + std::to_string(lse) + ",ref," + std::to_string(ref_lse));
        check(slot_same, "rtx_slot_" + std::to_string(k));
    }
    // rejection cases (each ends the grid; relaunch)
    struct Bad { const char *name; unsigned k; unsigned long long off, rows, tb; unsigned me; unsigned bit; };
    const Bad bad[] = {
        {"err_k0", 0, lm_off, kRows, table_bytes, kMe, PCI_CLUSTER_REASON_LM_CONFIG},
        {"err_k2560", 2560, lm_off, kRows, table_bytes, kMe, PCI_CLUSTER_REASON_LM_CONFIG},
        {"err_slice_not_at_end", 16, lm_off - 4096, kRows, table_bytes, kMe, PCI_CLUSTER_REASON_LM_SLICE},
        {"err_misaligned", 16, lm_off + 8, kRows, table_bytes, kMe, PCI_CLUSTER_REASON_LM_SLICE},
        {"err_table_size_unset", 16, lm_off, kRows, 0, kMe, PCI_CLUSTER_REASON_LM_CONFIG},
    };
    for (const auto &b : bad) {
        configure(b.tb); set_request(b.k, b.off, b.rows, b.me);
        slot->status = 0; const bool acked = request();
        const unsigned st = D()->status, ex = D()->exit_reason;
        T_CK(cudaDeviceSynchronize());
        check(acked && st == 1 && (ex & b.bit) && slot->status == 1 && (slot->worker & b.bit), b.name,
              "status," + std::to_string(st) + ",exit," + std::to_string(ex));
        configure(table_bytes); launch();
    }
    set_request(16, lm_off, kRows); const bool again = request();
    check(again && D()->status == 0 && reinterpret_cast<const unsigned *>(D()->result)[0] == 16, "topk_after_relaunch");
    C()->mode = 2; std::atomic_thread_fence(std::memory_order_seq_cst); C()->seq = ++seq;
    const bool stopped = await_u32(&D()->seq, seq, 10000); T_CK(cudaDeviceSynchronize());
    check(stopped, "clean_stop");
    // PLE region untouched by any scratch write (checked after the grid stopped: a blocking copy would
    // otherwise wait behind the persistent kernel until its idle deadline).
    std::vector<uint8_t> ple(lm_off); T_CK(cudaMemcpy(ple.data(), table, lm_off, cudaMemcpyDeviceToHost));
    check(std::all_of(ple.begin(), ple.end(), [](uint8_t b) { return b == 0x5A; }), "ple_region_untouched");
    std::printf("PCI_CLUSTER_LM_TOPK_TEST,%s,kernel,%s,failures,%d\n", g_fail ? "FAIL" : "PASS", kKernel, g_fail);
    return g_fail ? 1 : 0;
}
