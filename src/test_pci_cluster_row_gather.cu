// Offline test of the PCI-cluster row gather (worker mode 6, pci_cluster_exchange.cuh) in the real
// resident worker kernels: build with -DTEST_SHARED for resident_dynamic_shared (v2), without for
// resident_dynamic_routed (v3). One persistent worker grid on one GPU, a synthetic 1-MiB table,
// control block + direct result in mapped host memory laid out like the real control page.
// Prints one CHECK line per case and PCI_CLUSTER_ROW_GATHER_TEST,PASS|FAIL. Exit 0 only on PASS.
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
#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define T_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA,%s,%s\n", #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static constexpr size_t kControlBytes = 0x40000;           // worker BAR exposure size
static constexpr size_t kTableBytes = 1u << 20;
static constexpr unsigned kMe = 7;                          // this worker's peer index
static int g_fail = 0;

struct Rig {
    uint8_t *host_ctrl = nullptr, *dev_ctrl = nullptr;      // mapped host control page
    Return *hr = nullptr, *dr = nullptr;
    Grid *grid = nullptr; float *partial = nullptr;
    uint8_t *table = nullptr; std::vector<uint8_t> host_table;
    Q38CpuMailboxSlot *slot = nullptr, *dev_slot = nullptr;   // stands in for the remapped RTX return slot
    unsigned seq = 0;
    Control *c() { return reinterpret_cast<Control *>(host_ctrl); }
    ResidentDirectResult *direct() { return reinterpret_cast<ResidentDirectResult *>(host_ctrl + Q38_DIRECT_RESULT_OFF); }
};

static bool await_u32(const volatile unsigned *p, unsigned v, int ms) {
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    while (std::chrono::steady_clock::now() < end) { if (*p == v) return true; std::this_thread::sleep_for(std::chrono::microseconds(50)); }
    return false;
}
static void check(bool ok, const std::string &name, const std::string &detail = "") {
    std::printf("CHECK,%s,%s%s%s\n", name.c_str(), ok ? "OK" : "FAIL", detail.empty() ? "" : ",", detail.c_str());
    if (!ok) g_fail++;
}
static void launch(Rig &r) {
    T_CK(cudaMemset(r.grid, 0, sizeof(Grid)));
    r.hr->ready = 0; r.direct()->ready = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
#ifdef TEST_SHARED
    resident_dynamic_shared<<<10, T>>>(nullptr, reinterpret_cast<Control *>(r.dev_ctrl), r.grid, r.partial, r.dr, r.table, 20u, 30000000000ULL);
#else
    resident_dynamic_routed<<<10, THREADS>>>(nullptr, reinterpret_cast<Control *>(r.dev_ctrl), r.grid, nullptr, nullptr, r.dr, r.table, 30000000000ULL);
#endif
    T_CK(cudaGetLastError());
    if (!await_u32(&r.hr->ready, 1, 10000)) { std::fprintf(stderr, "worker startup timeout\n"); std::exit(3); }
}
// Publish one request: payload already written; then the sequence word (as the RTX/CPU publishers do).
static bool request(Rig &r, unsigned mode) {
    r.c()->mode = mode;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    r.c()->seq = ++r.seq;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return await_u32(&r.direct()->seq, r.seq, 10000);
}
static void configure(Rig &r, unsigned long long table_bytes, bool rtx_return) {
    r.c()->cpu_mailbox_device_ptr = reinterpret_cast<unsigned long long>(r.dev_slot);
    r.c()->cpu_mailbox_worker = kMe;
    r.c()->cpu_mailbox_reserved = rtx_return ? Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN : 0u;
    r.c()->ple_reserved[1] = unsigned(table_bytes); r.c()->ple_reserved[2] = unsigned(table_bytes >> 32);
}
static void set_rows(Rig &r, const std::vector<unsigned long long> &desc) {
    r.c()->ple_count = unsigned(desc.size());
    for (size_t i = 0; i < desc.size(); i++) r.c()->ple_offsets[i] = desc[i];
}
// Expected result bytes: this worker's rows, compacted in request order.
static std::vector<uint8_t> expected(const Rig &r, const std::vector<unsigned long long> &desc, unsigned *owned) {
    std::vector<uint8_t> out; unsigned k = 0;
    for (auto d : desc) if (unsigned(d >> PCI_CLUSTER_OWNER_SHIFT) == kMe) {
        const size_t off = size_t(d & PCI_CLUSTER_OFFSET_MASK);
        out.insert(out.end(), r.host_table.begin() + off, r.host_table.begin() + off + PCI_CLUSTER_ROW_BYTES); k++;
    }
    *owned = k; return out;
}
static void run_ok_case(Rig &r, const std::string &name, const std::vector<unsigned long long> &desc) {
    unsigned want_owned = 0; auto want = expected(r, desc, &want_owned);
    std::memset(r.direct()->result, 0xEE, sizeof(r.direct()->result));
    set_rows(r, desc);
    const bool acked = request(r, PCI_CLUSTER_MODE_ROW_GATHER);
    const auto *d = r.direct();
    unsigned owned; float g = d->gate; std::memcpy(&owned, &g, 4);
    const bool bytes_ok = acked && std::memcmp(d->result, want.data(), want.size()) == 0;
    // The RTX reads the return slot, not `direct`: header (MoE convention), count and rows must be there.
    // The slot is acknowledged after `direct`; wait on the slot's own seq, as the RTX does.
    const bool slot_acked = await_u32(&r.slot->seq, r.seq, 10000);
    const auto *sl = r.slot;
    unsigned slot_owned; float sg = sl->gate; std::memcpy(&slot_owned, &sg, 4);
    const bool slot_ok = slot_acked && sl->seq == r.seq && sl->status == 0 && sl->worker == 0 && sl->layer == 1 &&
                         slot_owned == want_owned && std::memcmp(sl->result, want.data(), want.size()) == 0;
    check(slot_ok, name + "_rtx_slot", "seq," + std::to_string(sl->seq) + ",status," + std::to_string(sl->status) +
          ",worker," + std::to_string(sl->worker) + ",layer," + std::to_string(sl->layer) + ",owned," + std::to_string(slot_owned));
    check(acked && d->status == 0 && d->exit_reason == 0 && owned == want_owned && bytes_ok, name,
          "rows," + std::to_string(desc.size()) + ",owned," + std::to_string(owned) + ",want," + std::to_string(want_owned) +
          ",status," + std::to_string(d->status) + ",exit," + std::to_string(d->exit_reason) + ",bytes_match," + std::to_string(bytes_ok));
}
// Error cases: the kernel must take its error exit with `bit` set and read nothing; relaunch after.
static void run_error_case(Rig &r, const std::string &name, unsigned bit) {
    const bool rtx_flag = (r.c()->cpu_mailbox_reserved & Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN) != 0;
    r.slot->status = 0; r.slot->worker = 0;
    const bool acked = request(r, PCI_CLUSTER_MODE_ROW_GATHER);
    const auto *d = r.direct();
    const unsigned status = d->status, exit_reason = d->exit_reason;
    if (rtx_flag && !await_u32(&r.slot->seq, r.seq, 10000)) std::fprintf(stderr, "slot error ack timeout\n");
    if (rtx_flag)   // during a peer session the RTX must see the error at once (reason 4), not a timeout
        check(r.slot->seq == r.seq && r.slot->status == 1 && (r.slot->worker & bit), name + "_rtx_slot_error",
              "seq," + std::to_string(r.slot->seq) + ",status," + std::to_string(r.slot->status) + ",bits," + std::to_string(r.slot->worker));
    T_CK(cudaDeviceSynchronize());                           // worker grid must have exited
    check(acked && status == 1 && (exit_reason & bit), name,
          "status," + std::to_string(status) + ",exit_reason," + std::to_string(exit_reason) + ",want_bit," + std::to_string(bit));
    configure(r, kTableBytes, true);
    launch(r);
    // The relaunched kernel must clear the previous error from the direct record (the services'
    // relaunch only clears the host return struct) - otherwise the RTX wait sees a stale exit_reason.
    check(r.direct()->status == 0 && r.direct()->exit_reason == 0, name + "_relaunch_clears_error",
          "status," + std::to_string(r.direct()->status) + ",exit_reason," + std::to_string(r.direct()->exit_reason));
}

int main() {
    Rig r;
    cudaDeviceProp prop{}; T_CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("PCI_CLUSTER_ROW_GATHER_TEST,kernel,%s,device,%s,sm,%d%d\n", kKernel, prop.name, prop.major, prop.minor);
    T_CK(cudaHostAlloc(&r.host_ctrl, kControlBytes, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(r.host_ctrl, 0, kControlBytes);
    T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&r.dev_ctrl), r.host_ctrl, 0));
    T_CK(cudaHostAlloc(&r.hr, sizeof(Return), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(r.hr, 0, sizeof(Return));
    T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&r.dr), r.hr, 0));
    T_CK(cudaHostAlloc(&r.slot, sizeof(Q38CpuMailboxSlot), cudaHostAllocMapped | cudaHostAllocPortable));
    std::memset(r.slot, 0, sizeof(Q38CpuMailboxSlot));
    T_CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&r.dev_slot), r.slot, 0));
    T_CK(cudaMalloc(&r.grid, sizeof(Grid)));
    T_CK(cudaMalloc(&r.partial, 10 * H * sizeof(float)));
    r.host_table.resize(kTableBytes);
    for (size_t i = 0; i < kTableBytes; i++) r.host_table[i] = uint8_t((i * 2654435761u) >> 13);
    T_CK(cudaMalloc(&r.table, kTableBytes));
    T_CK(cudaMemcpy(r.table, r.host_table.data(), kTableBytes, cudaMemcpyHostToDevice));
    configure(r, kTableBytes, true);
    launch(r);

    std::mt19937_64 rng(38);
    const unsigned rows_in_table = kTableBytes / PCI_CLUSTER_ROW_BYTES;
    auto row = [&](unsigned owner) { return pci_cluster_row_desc(owner, (rng() % rows_in_table) * PCI_CLUSTER_ROW_BYTES); };

    // 1. Decode-sized request: 16 rows over three owners (this worker owns 6).
    std::vector<unsigned long long> decode;
    for (int i = 0; i < 16; i++) decode.push_back(row(i % 3 == 0 ? kMe : (i % 3 == 1 ? 3u : 20u)));
    // Mode-4 reference over the same owned offsets (existing CPU path, per-worker offsets).
    {
        std::vector<unsigned long long> mine;
        for (auto d : decode) if (unsigned(d >> PCI_CLUSTER_OWNER_SHIFT) == kMe) mine.push_back(d & PCI_CLUSTER_OFFSET_MASK);
        set_rows(r, mine);
        const bool acked = request(r, 4);
        std::vector<uint8_t> mode4(reinterpret_cast<uint8_t *>(r.direct()->result),
                                   reinterpret_cast<uint8_t *>(r.direct()->result) + mine.size() * PCI_CLUSTER_ROW_BYTES);
        run_ok_case(r, "decode_16_rows_3_owners", decode);
        const bool same = acked && std::memcmp(mode4.data(), r.direct()->result, mode4.size()) == 0;
        check(same, "mode6_bytes_equal_mode4", "rows," + std::to_string(mine.size()));
    }
    // 2. Nothing owned: ack with 0 rows.
    { std::vector<unsigned long long> v; for (int i = 0; i < 16; i++) v.push_back(row(5)); run_ok_case(r, "zero_owned", v); }
    // 3. Maximum request: 508 rows, exactly 64 owned (the per-worker capacity).
    {
        std::vector<unsigned long long> v;
        for (unsigned i = 0; i < PCI_CLUSTER_ROW_GATHER_MAX_ROWS; i++) v.push_back(row(i % 8 == 0 && i / 8 < 64 ? kMe : 1u + i % 5));
        run_ok_case(r, "max_508_rows_64_owned", v);
    }
    // 4. A row owned by another worker with an offset far outside THIS table is ignored (not read).
    {
        std::vector<unsigned long long> v = {row(kMe), pci_cluster_row_desc(9u, 0xFFFFFFF000ull), row(kMe)};
        run_ok_case(r, "foreign_row_out_of_my_bounds_ignored", v);
    }
    // 5. Repeat the decode request (sequence continuity across mode switches).
    run_ok_case(r, "decode_repeat", decode);

    // Error cases (each ends the worker grid; relaunched in run_error_case).
    set_rows(r, {row(kMe), pci_cluster_row_desc(kMe, kTableBytes - 156)});
    run_error_case(r, "err_owned_row_past_end", PCI_CLUSTER_REASON_ROW_GATHER_ROW);
    set_rows(r, {pci_cluster_row_desc(kMe, 2)});
    run_error_case(r, "err_owned_row_misaligned", PCI_CLUSTER_REASON_ROW_GATHER_ROW);
    { std::vector<unsigned long long> v; for (int i = 0; i < 65; i++) v.push_back(row(kMe)); set_rows(r, v); }
    run_error_case(r, "err_65_owned_rows", PCI_CLUSTER_REASON_ROW_GATHER_ROW);
    set_rows(r, {row(kMe)}); configure(r, kTableBytes, false);
    run_error_case(r, "err_no_rtx_return_flag", PCI_CLUSTER_REASON_ROW_GATHER_CONFIG);
    set_rows(r, {row(kMe)}); configure(r, 0, true);
    run_error_case(r, "err_table_size_unset", PCI_CLUSTER_REASON_ROW_GATHER_CONFIG);
    { std::vector<unsigned long long> v; for (unsigned i = 0; i < PCI_CLUSTER_ROW_GATHER_MAX_ROWS + 1; i++) v.push_back(row(2)); set_rows(r, v); }
    run_error_case(r, "err_509_rows", PCI_CLUSTER_REASON_ROW_GATHER_CONFIG);

    // Worker still serves requests after the error relaunches; then a clean stop (mode 2).
    run_ok_case(r, "decode_after_error_relaunch", decode);
    const bool stopped = request(r, 2);
    T_CK(cudaDeviceSynchronize());
    check(stopped && r.direct()->exit_reason == 2, "clean_stop_mode2");
    std::printf("PCI_CLUSTER_ROW_GATHER_TEST,%s,kernel,%s,failures,%d\n", g_fail ? "FAIL" : "PASS", kKernel, g_fail);
    return g_fail ? 1 : 0;
}
