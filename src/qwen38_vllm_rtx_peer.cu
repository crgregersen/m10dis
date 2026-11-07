// vLLM-side GPU-originated request path for the already-running Qwen M10
// resident workers. The daemon owns each RM mailbox and model allocation;
// this library installs temporary RTX SYS aliases for the retained 64-KiB
// mailbox windows, then publishes and reduces on the RTX.
#define Q38_RTX_TRANSPORT_LIBRARY_ONLY
#define FENCE_HANDOFF_EVENT_ONLY 1
#include <cctype>
#include "bench_qwen38_rtx_m10_retained_roundtrip.cu"
#include "resident_direct_mailbox.cuh"
#include "pci_cluster_exchange.cuh"
#include "resident_batch_layout.cuh"
#include "resident_cpu_mailbox_abi.cuh"
#include "resident_multirow_abi.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <sstream>
#include <algorithm>

namespace {
constexpr NvU32 kMailboxBytes = 64u << 10;
constexpr NvU32 kMailboxLeaves = kMailboxBytes / PAGE;
// Q38_RTX_PEER_SMALL_LEAVES: remap only 4-KiB leaves on both sides.
// 64-KiB (RTX) and 128-KiB (GM107) remaps were observed to read an original
// PTE of 0x0 while the same VA was live in a 4-KiB table, and no RTX store
// through them ever arrived (planning brief 2026-11-02). The 4-KiB path is the
// one proven by the 4-KiB marker, receipt-gate and layer-0 reference runs.
#ifdef Q38_RTX_PEER_SMALL_LEAVES
constexpr NvU32 kReturnLeafBytes = PAGE;
constexpr NvU32 kReturnPages = 3;            // 16 + 2560*4 + 4 = 10260 bytes
#else
constexpr NvU32 kReturnLeafBytes = 128u << 10;
constexpr NvU32 kReturnPages = 1;
#endif
constexpr NvU32 kReturnSpanBytes = kReturnLeafBytes * kReturnPages;
// ACQUIRE_EXTERNAL_OWNER rejects RTX BAR1 windows below 64 KiB
// (uvm_tools.c: length < UVM_PAGE_SIZE_64K -> NV_ERR_INVALID_LIMIT); the
// remapped leaves (kReturnSpanBytes) sit at the start of the retained window.
constexpr NvU32 kReturnWindowBytes = kReturnSpanBytes < (64u << 10) ? (64u << 10) : kReturnSpanBytes;
constexpr NvU32 kReturnSlotBytes = 256u << 10;
constexpr NvU32 kAsyncStatusSlots = 64;
constexpr NvU32 kPexGroups = 6;
__device__ NvU32 g_q38_ack_profile = 0;   // Q38_RTX_PEER_ACK_PROFILE (see pci_cluster_ack_profile)
constexpr NvU32 kPexPagesPerGroup = 7;   // + 0xC000..0xEFFF (multi-row rows 2-3, 2026-11-04)
constexpr NvU32 kPexAliases = kPexGroups * kPexPagesPerGroup;
constexpr NvU32 kPexWindowBytes = 64u << 10;
constexpr NvU32 kPexLeafBytes = 64u << 10;
constexpr NvU32 kPexPageBytes = PAGE;
constexpr NvU32 kPexPageOffsets[kPexPagesPerGroup] = {0, 0x8000, 0x9000, 0xA000, 0xC000, 0xD000, 0xE000};
#ifdef Q38_RTX_PEER_SMALL_LEAVES
constexpr NvU32 kPexRemapBytes = PAGE;                    // one 4-KiB leaf per used page
constexpr NvU32 kPexRemapsPerGroup = kPexPagesPerGroup;
#else
constexpr NvU32 kPexRemapBytes = kPexLeafBytes;           // one 64-KiB leaf per group
constexpr NvU32 kPexRemapsPerGroup = 1;
#endif
struct PeerControlLayout {
    volatile NvU32 seq, layer, mode, reserved;
    NvU32 ple_count, ple_reserved[3];
    NvU64 ple_offsets[4096];
    NvS32 ids[10];
    float routes[10];
    float x[QWEN_H];
    NvU32 cpu_mailbox_worker, cpu_mailbox_reserved;
    NvU64 cpu_mailbox_device_ptr;
    NvU64 batch_scratch;                 // mode 8 (worker structs, 2026-11-03)
    NvU32 batch_scratch_bytes, batch_reserved;
    NvU64 batch_area;
};
constexpr NvU32 kControlIdsOff = offsetof(PeerControlLayout, ids);
constexpr NvU32 kControlRoutesOff = offsetof(PeerControlLayout, routes);
constexpr NvU32 kControlHOff = offsetof(PeerControlLayout, x);
constexpr NvU32 kDirectResultOff = Q38_DIRECT_RESULT_OFF;
#ifdef Q38_RTX_PEER_LEADER_AGG
// Leader aggregation slots/flags live at owner +0x28000..0x32803 (row-1 slots of the multi-row mode at
// 0x33000..0x3BFFF, worker timing ring at 0x3C000), inside the daemon's
// 0x40000-byte worker BAR1 exposure (resident_direct_mailbox.cuh).
constexpr size_t kOwnerDiagnosticMapBytes = Q38_WORKER_BAR_MAP_BYTES;
#else
constexpr size_t kOwnerDiagnosticMapBytes =
    (kDirectResultOff + sizeof(ResidentDirectResult) + PAGE - 1) & ~(size_t(PAGE) - 1);
#endif
constexpr NvU32 kResultDataOff = kDirectResultOff + offsetof(ResidentDirectResult, result);
constexpr NvU32 kResultGateOff = kDirectResultOff + offsetof(ResidentDirectResult, gate);
constexpr NvU32 kCpuMailboxWorkerOff = offsetof(PeerControlLayout, cpu_mailbox_worker);
constexpr NvU32 kCpuMailboxPtrOff = offsetof(PeerControlLayout, cpu_mailbox_device_ptr);
static_assert(kControlIdsOff == 32800 && kControlRoutesOff == 32840 && kControlHOff == 32880,
              "RTX publisher layout no longer matches the resident worker ABI");
static_assert(kControlIdsOff >= 0x8000 && kControlRoutesOff >= 0x8000 &&
              kControlHOff >= 0x8000 && kControlHOff + QWEN_H * sizeof(NvU32) <= 0xB000,
              "RTX multicast fields are outside the compact PEX page aliases");
static_assert(offsetof(PeerControlLayout, batch_scratch) == 0xA880 && offsetof(PeerControlLayout, batch_area) == 0xA890,
              "batch fields no longer match the resident worker control structs");
static_assert(sizeof(PeerControlLayout) <= Q38_DIRECT_RESULT_OFF,
              "RTX control writes overlap the resident result record");
static_assert(sizeof(ResidentDirectResult) == 10260 && kResultDataOff == 0x20010 &&
              kResultGateOff == 0x22810,
              "RTX reducer layout no longer matches the resident result ABI");
static_assert(offsetof(ResidentDirectResult, result) == offsetof(Q38CpuMailboxSlot, result) &&
              offsetof(ResidentDirectResult, gate) == offsetof(Q38CpuMailboxSlot, gate),
              "direct result payload no longer matches RTX reduction layout");
static_assert(kCpuMailboxWorkerOff == 0xA870 && kCpuMailboxPtrOff == 0xA878,
              "RTX publisher CPU-mailbox fields no longer match resident control ABI");
constexpr NvU64 kWaitCycles = 20000000000ull;
static std::string peer_hex(NvU64 value) {
    char text[32]; std::snprintf(text, sizeof(text), "0x%llx", (unsigned long long)value);
    return text;
}

// PCI-cluster row gather (pci_cluster_exchange.cuh): one entry per PLE shard, global rows
// [global_start, global_start+rows) live at byte `offset` of peer worker `owner`'s table.
struct PciClusterShard { NvU64 global_start, rows, offset; NvU32 owner, pad; };
constexpr NvU32 kPciClusterRowGatherMaxRows = 64;   // per exchange: even if one worker owns all rows, <= 64
static_assert(kPciClusterRowGatherMaxRows <= PCI_CLUSTER_ROWS_PER_WORKER &&
              kPciClusterRowGatherMaxRows <= PCI_CLUSTER_ROW_GATHER_MAX_ROWS, "row-gather bound");
static_assert(offsetof(PeerControlLayout, ple_count) == 16 && offsetof(PeerControlLayout, ple_reserved) == 20 &&
              offsetof(PeerControlLayout, ple_offsets) == 32, "row-gather request layout no longer matches the worker ABI");

struct PeerRecord {
    std::string bdf;
    int group = -1, worker = -1, ordinal = -1, minor = -1;
    NvU64 owner_va = 0, bar_bus = 0, return_alias_va = 0;
    NvProcessorUuid uuid{};
    std::vector<UVM_GLM53_VA_SPACE_PDB_ENTRY> requester_pdb_candidates;
};
struct PeerState {
    int rtx_ordinal = -1, uvm_fd = -1;
    int pex_dma_fd = -1;
    CUdevice rtx_device{};
    CUcontext rtx_context{};
    NvProcessorUuid rtx_uuid{};
    void *host_alias = nullptr;          // aligned view into host_alias_alloc
    void *host_alias_alloc = nullptr;    // raw cudaHostAlloc pointer (freed)
    NvU32 *device_alias = nullptr;
    void *pex_host_alias = nullptr;
    NvU64 *device_pex_aliases = nullptr;
    // Outer PEX8748 multicast (Q38_RTX_PEER_PEX_OUTER_BASE, 2026-11-04): one alias group whose writes the outer switch
    // copies to the six cards' multicast windows; then every request is published once (pub_groups = 1).
    NvU64 *device_pex_outer_aliases = nullptr;
    NvU64 pex_outer_base = 0;
    const NvU64 *pub_aliases = nullptr;
    NvU32 pub_groups = 6;
    // Q38_RTX_PEER_PUBLISH_ORDER: 4 (default since 2026-11-05) begin + publish + go in one kernel (outer group only;
    // without it as 0); 0 begin, payload, go as separate kernels; 1 / 2 / 3: publish-order experiments (TRANSPORT.md)
    NvU32 publish_order = 4;
    NvU32 publish_seq = 0x543210u;
    NvU32 go_gap_ns = 0;
    NvU32 go_lowfence = 0;               // Q38_RTX_PEER_GO_LOWFENCE=1: publish order 3, sequence words only, one fence                 // Q38_RTX_PEER_GO_GAP_NS: publish order 3, spin between cards' go words       // Q38_RTX_PEER_PUBLISH_SEQ="0,1,2,3,4,5": serial group order (a permutation)             // Q38_RTX_PEER_PUBLISH_ORDER: 0 payloads then go, 1 per group parallel, 2 per group serial
    bool pex_multicast = false;
    bool direct_unicast_diagnostic = false;
    bool pex_quiesced = true;
    struct PexLeaf {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS query{};
        UVM_GLM53_MULTI_PTE_REMAP_PARAMS active{};
        NvU64 destination = 0;
        bool acquired = false;
    };
    std::vector<PexLeaf> pex_routes;
    std::vector<NvU64> pex_ingresses;
    NvU64 *device_aliases = nullptr;
    NvU32 *device_status = nullptr;
    NvU32 *device_headers = nullptr;
    NvU64 *device_return_slots = nullptr;
    NvU64 *device_m10_return_aliases = nullptr;
    // Multi-row decode MoE (worker mode 10, Q38_RTX_PEER_MOE_MULTIROW=1 at open, resident_multirow_abi.cuh):
    // row-1 return records of the 9 reducers (card leaders, shared workers), 0 for the card peers.
    bool multirow = false;
    NvU32 mr_max_rows = 0;               // Q38_RTX_PEER_MOE_MULTIROW_MAX (2..4, default 2): rows the routes cover
    std::vector<NvU64> return_slots1;    // [(r-1)*24 + w], r = 1 .. mr_max_rows-1
    NvU64 *device_return_slots1 = nullptr;
    std::vector<PeerRecord> workers;
    std::vector<MultiRoute> routes;
    std::vector<MultiRoute> return_routes;
#ifdef Q38_RTX_PEER_LEADER_AGG
    // On-card leader push aggregation (2026-11-02): M10 peer -> M10 leader routes and
    // the saved owner-BAR words they overwrite (aggregation block and leader flags).
    std::vector<MultiRoute> agg_routes;
    struct AggSave { size_t worker; NvU64 offset; NvU32 words[8]; };
    std::vector<AggSave> agg_saved;
    NvU32 agg_skip_mask = 0;
#endif
    std::vector<int> owner_fds;
    std::vector<void *> owner_cpus;
    std::vector<NvU64> return_slots;
    std::vector<NvU32> sequence;
    struct SavedMailboxFields { NvU32 worker, flags; NvU64 ptr; };
    std::vector<SavedMailboxFields> saved_mailbox_fields;
    bool ingress_diagnostic = false;
    std::vector<NvU32> pending_sequences;
    cudaStream_t pending_stream = nullptr;
    QMem return_owner{};
    bool return_owner_active = false;
    bool m10_quiesced = true;
    bool in_flight = false;
    // Phase 2 GPU-driven exchange (q38_rtx_peer_forward_gpu): sequence counter and per-call
    // state in device memory; sticky error word and abort flag in mapped pinned host memory.
    // pipe_*: pipelined batched MoE (q38_rtx_peer_moe_batch_pipe_gpu): first sequence, first timing record, skip
    struct GpuDev { NvU32 seq, cur_seq, cur_skip, skips, calls, pad[3]; NvU32 pipe_base, pipe_rec, pipe_skip, pipe_pad; };
    struct GpuHost { NvU32 abort, err, err_layer, err_seq, err_reason, err_worker, err_status, err_exit; };
    GpuDev *gpu_dev = nullptr;
    GpuHost *gpu_host = nullptr, *gpu_host_dev = nullptr;
    bool gpu_used = false;
    void *timing_buf = nullptr;          // q38_rtx_peer_timing_enable (per-exchange %globaltimer records)
    void *batch_host = nullptr; char *batch_dev = nullptr;   // shared CPU mailbox (mode 8), RTX mapping
    NvU64 batch_calls = 0, batch_rows = 0, pipe_calls = 0;
    std::vector<NvU64> batch_area_saved;  // per worker, restored at close
    NvU32 timing_cap = 0;
    // Row gather (configured by q38_rtx_peer_configure_row_gather; ple_reserved words restored at close).
    bool rg_configured = false;
    PciClusterShard *rg_shards = nullptr; NvU32 rg_nshards = 0;
    NvU64 *rg_desc = nullptr; NvU32 *rg_rank = nullptr; NvU32 *rg_counts = nullptr;
    struct SavedTableWords { NvU32 w[3]; };
    std::vector<SavedTableWords> rg_saved;
    NvU64 rg_calls = 0, rg_rows = 0;
    // LM-head top-k (LMHEAD manifest rows, worker mode 7)
    bool lm_configured = false; NvU32 lm_vocab = 0;
    NvU64 *lm_off = nullptr, *lm_rows = nullptr; NvU32 *lm_row_start = nullptr;
    NvU64 lm_calls = 0;
};
std::unique_ptr<PeerState> g_peer;
std::mutex g_peer_mutex;
std::string g_peer_error;
bool g_last_open_rollback_complete = false;

static NvProcessorUuid parse_uuid(const std::string &s) {
    std::string hex;
    for (char c : s) if (std::isxdigit(static_cast<unsigned char>(c))) hex.push_back(c);
    if (hex.size() >= 2 && (hex.size() - 2) == 32) hex.erase(0, 2); // GPU- prefix has no hex digits
    if (hex.size() != 32) throw std::runtime_error("invalid GPU UUID in peer mapping manifest: " + s);
    NvProcessorUuid out{};
    auto nibble=[](char c)->unsigned { if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; return c-'A'+10; };
    for (size_t i=0;i<16;i++) reinterpret_cast<NvU8*>(&out)[i]=NvU8((nibble(hex[2*i])<<4)|nibble(hex[2*i+1]));
    return out;
}
static std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out; size_t start=0;
    while (true) { auto at=line.find('\t',start); out.push_back(line.substr(start,at-start)); if(at==std::string::npos)break; start=at+1; }
    return out;
}
struct PexGroupRecord {
    std::vector<std::string> gpus;
    NvU64 bridge_lo = 0, bridge_hi = 0, ingress = 0;
};
static std::vector<PexGroupRecord> load_pex_groups(const char *path,
                                                    const std::vector<PeerRecord> &workers) {
    if (!path || !*path) throw std::runtime_error("PEX multicast mode requires Q38_M10_GROUPS_TSV");
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot open active PEX group TSV: ") + path);
    std::vector<PexGroupRecord> groups;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_tabs(line);
        if (f.size() != 7) throw std::runtime_error("malformed active PEX group TSV row");
        PexGroupRecord g;
        auto split_csv = [](const std::string &s) {
            std::vector<std::string> v; size_t at = 0;
            while (true) { auto end = s.find(',', at); v.push_back(s.substr(at, end-at));
                if (end == std::string::npos) break; at = end + 1; }
            return v;
        };
        g.gpus = split_csv(f[3]);
        g.bridge_lo = std::stoull(f[4], nullptr, 0);
        g.bridge_hi = std::stoull(f[5], nullptr, 0);
        g.ingress = std::stoull(f[6], nullptr, 0);
        if (g.gpus.size() != 4 || g.ingress % kPexWindowBytes ||
            g.ingress < g.bridge_lo || g.bridge_hi < g.ingress ||
            kPexWindowBytes - 1 > g.bridge_hi - g.ingress)
            throw std::runtime_error("invalid PEX group GPU set, ingress alignment, or bridge coverage");
        groups.push_back(std::move(g));
    }
    if (groups.size() != kPexGroups || workers.size() != kPexGroups * 4)
        throw std::runtime_error("RTX PEX publisher requires six current groups and 24 resident workers");
    for (size_t group = 0; group < groups.size(); ++group)
        for (size_t worker = 0; worker < 4; ++worker) {
            const auto it = std::find_if(workers.begin(), workers.end(), [&](const PeerRecord &r) {
                return r.group == int(group) && r.worker == int(worker);
            });
            if (it == workers.end() || it->bdf != groups[group].gpus[worker])
                throw std::runtime_error("PEX topology GPU identity differs from resident mapping manifest");
        }
    return groups;
}
static std::pair<NvU64,NvU64> owner_bar1_range(const std::string &bdf) {
    std::ifstream resources("/sys/bus/pci/devices/"+bdf+"/resource");
    if (!resources) throw std::runtime_error("open PCI resource table for "+bdf);
    std::string line;
    if (!std::getline(resources,line) || !std::getline(resources,line))
        throw std::runtime_error("read BAR1 resource for "+bdf);
    std::istringstream fields(line);
    std::string first,last,flags;
    fields >> first >> last >> flags;
    if (!fields) throw std::runtime_error("parse BAR1 resource for "+bdf);
    return {std::stoull(first,nullptr,0),std::stoull(last,nullptr,0)};
}
static std::vector<PeerRecord> load_peer_records(const char *maps_path,const char *uuid_path) {
    std::ifstream mf(maps_path); if(!mf)throw std::runtime_error("open GPU mapping manifest failed");
    std::string line; if(!std::getline(mf,line)||line!="QWEN38_GPU_MAPPINGS_MANIFEST\t1")throw std::runtime_error("unsupported GPU mapping manifest");
    std::getline(mf,line); std::getline(mf,line); std::getline(mf,line);
    std::vector<PeerRecord> records;
    while(std::getline(mf,line)) {
        if(line.empty())continue; auto f=split_tabs(line); if(f.size()!=7)throw std::runtime_error("malformed GPU map record");
        PeerRecord r; r.group=std::stoi(f[0]); r.worker=std::stoi(f[1]); r.bdf=f[2]; r.ordinal=std::stoi(f[3]); r.minor=std::stoi(f[4]);
        r.owner_va=std::stoull(f[5],nullptr,0); r.bar_bus=std::stoull(f[6],nullptr,0); records.push_back(r);
    }
    if(records.size()!=24)throw std::runtime_error("RTX peer mode requires all 24 resident GPU mappings");
    std::ifstream uf(uuid_path); if(!uf)throw std::runtime_error("open GPU UUID manifest failed");
    std::unordered_map<std::string,NvProcessorUuid> uuids;
    while(std::getline(uf,line)) { if(line.empty()||line[0]=='#')continue; auto f=split_tabs(line); if(f.size()!=2)throw std::runtime_error("malformed GPU UUID record"); uuids[f[0]]=parse_uuid(f[1]); }
    for(auto &r:records) { auto it=uuids.find(r.bdf); if(it==uuids.end())throw std::runtime_error("UUID missing for resident GPU "+r.bdf); r.uuid=it->second; }
    const auto pdb_path=std::filesystem::path(maps_path).parent_path()/"gpu_pdb.manifest";
    std::ifstream pf(pdb_path); if(!pf)throw std::runtime_error("open worker-owned UVM PDB manifest failed: "+pdb_path.string());
    if(!std::getline(pf,line)||line!="QWEN38_GPU_PDB_MANIFEST\t2")throw std::runtime_error("unsupported worker PDB manifest");
    if(!std::getline(pf,line)||line.rfind("run_id\t",0)!=0)throw std::runtime_error("worker PDB manifest has no run id");
    if(!std::getline(pf,line)||line!="workers\t24")throw std::runtime_error("worker PDB manifest worker count mismatch");
    if(!std::getline(pf,line)||line.rfind("candidates\t",0)!=0)throw std::runtime_error("worker PDB manifest candidate count missing");
    const size_t candidate_count=std::stoull(line.substr(11));
    if(!std::getline(pf,line)||line!="group\tworker\tbdf\tcuda_va\tpid\tva_space\tgpu_va_space\troot_phys\troot_aperture")
        throw std::runtime_error("worker PDB manifest schema mismatch");
    std::unordered_map<std::string,size_t> by_bdf;for(size_t i=0;i<records.size();i++)by_bdf.emplace(records[i].bdf,i);
    std::vector<bool> pdb_seen(records.size(),false);
    size_t parsed_candidates=0;
    while(std::getline(pf,line)) {
        if(line.empty())continue;auto f=split_tabs(line);if(f.size()!=9)throw std::runtime_error("malformed worker PDB row");
        const unsigned group=std::stoul(f[0]),worker=std::stoul(f[1]);
        if(group>=6||worker>=4)throw std::runtime_error("worker PDB index out of range");
        auto it=by_bdf.find(f[2]);if(it==by_bdf.end())throw std::runtime_error("worker PDB BDF absent from GPU mapping manifest");
        auto &r=records[it->second];
        if(group!=unsigned(r.group)||worker!=unsigned(r.worker))throw std::runtime_error("worker PDB index differs from GPU mapping manifest");
        if(std::stoull(f[3],nullptr,0)!=r.owner_va)throw std::runtime_error("worker PDB VA differs from RM/VMM mapping");
        UVM_GLM53_VA_SPACE_PDB_ENTRY p{};p.gpu_uuid=r.uuid;p.current_pid=std::stoul(f[4]);
        p.va_space_ptr=std::stoull(f[5],nullptr,0);p.gpu_va_space_ptr=std::stoull(f[6],nullptr,0);
        p.root_phys_address=std::stoull(f[7],nullptr,0);p.root_phys_aperture=std::stoul(f[8]);p.did_set_page_directory=1;
        if(!p.current_pid||!p.va_space_ptr||!p.gpu_va_space_ptr||!p.root_phys_address)
            throw std::runtime_error("worker PDB row contains an invalid VA-space identity");
        for(const auto &old:r.requester_pdb_candidates)
            if(old.va_space_ptr==p.va_space_ptr&&old.gpu_va_space_ptr==p.gpu_va_space_ptr&&old.root_phys_address==p.root_phys_address)
                throw std::runtime_error("duplicate worker PDB candidate");
        r.requester_pdb_candidates.push_back(p);pdb_seen[it->second]=true;parsed_candidates++;
    }
    if(parsed_candidates!=candidate_count)throw std::runtime_error("worker PDB manifest candidate count mismatch");
    if(std::find(pdb_seen.begin(),pdb_seen.end(),false)!=pdb_seen.end())throw std::runtime_error("worker PDB manifest is incomplete");
    return records;
}

static void load_return_aliases(std::vector<PeerRecord> &records) {
    const char *path = std::getenv("Q38_M10_RTX_RETURN_ALIASES_MANIFEST");
    if (!path || !*path)
        throw std::runtime_error("mapped-host M10 return-alias manifest is required for the 128-KiB result route");
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("open return-alias manifest: ") + path);
    std::string line;
    if (!std::getline(in, line) || line != "QWEN38_RTX_RETURN_ALIASES\t1" ||
        !std::getline(in, line) || line.rfind("run_id\t", 0) != 0 || line.size() <= 7 ||
        !std::getline(in, line) || line != "records\t24" ||
        !std::getline(in, line) || line != "group\tworker\tbdf\tdevice_va")
        throw std::runtime_error("invalid return-alias manifest header");

    using Key = std::pair<unsigned, unsigned>;
    std::map<Key, NvU64> aliases;
    std::map<Key, std::string> bdfs;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto f = split_tabs(line);
        if (f.size() != 4) throw std::runtime_error("malformed return-alias row");
        const unsigned group = std::stoul(f[0]);
        const unsigned worker = std::stoul(f[1]);
        const NvU64 va = std::stoull(f[3], nullptr, 0);
        const Key key{group, worker};
        if (group >= 6 || worker >= 4 || !va || (va & (PAGE - 1)) ||
            !aliases.emplace(key, va).second)
            throw std::runtime_error("invalid, unaligned, or duplicate return-alias record");
        bdfs.emplace(key, f[2]);
    }
    if (aliases.size() != records.size())
        throw std::runtime_error("return-alias manifest does not contain all 24 workers");
    for (auto &r : records) {
        const Key key{unsigned(r.group), unsigned(r.worker)};
        const auto it = aliases.find(key);
        if (it == aliases.end() || bdfs.at(key) != r.bdf)
            throw std::runtime_error("return-alias identity differs from worker mapping: " + r.bdf);
        r.return_alias_va = it->second;
        std::fprintf(stderr,
            "RTX_PEER_RETURN_ALIAS,bdf,%s,device_va,0x%llx,record_bytes,%u,pages,%u,leaf_bytes,%u\n",
            r.bdf.c_str(), (unsigned long long)r.return_alias_va,
            (unsigned)sizeof(ResidentDirectResult), (unsigned)kReturnPages,
            (unsigned)kReturnLeafBytes);
    }
}

__global__ static void rtx_peer_publish(const void *h, int h_type,
        const int32_t *ids, const float *routes, const NvU64 *aliases,
        NvU32 workers, NvU32 row, NvU32 layer, NvU32 sequence) {
    if (blockIdx.x) return;
    const __half *h16=reinterpret_cast<const __half*>(h);
    const __nv_bfloat16 *hb=reinterpret_cast<const __nv_bfloat16*>(h);
    for (NvU32 i=threadIdx.x;i<QWEN_H;i+=blockDim.x) {
        float value = h_type==0 ? reinterpret_cast<const float*>(h)[size_t(row)*QWEN_H+i]
                    : h_type==1 ? __half2float(h16[size_t(row)*QWEN_H+i])
                                : __bfloat162float(hb[size_t(row)*QWEN_H+i]);
        NvU32 bits=__float_as_uint(value);
        #pragma unroll 1
        for(NvU32 w=0;w<workers;w++) {
            auto *dst=reinterpret_cast<volatile NvU32*>(aliases[w]);
            dst[kControlHOff/4+i]=bits;
        }
    }
    for (NvU32 k=threadIdx.x;k<10;k+=blockDim.x) {
        const NvU32 id=static_cast<NvU32>(ids[size_t(row)*10+k]);
        const NvU32 weight=__float_as_uint(routes[size_t(row)*10+k]);
        #pragma unroll 1
        for(NvU32 w=0;w<workers;w++) {
            auto *dst=reinterpret_cast<volatile NvU32*>(aliases[w]);
            dst[kControlIdsOff/4+k]=id;
            dst[kControlRoutesOff/4+k]=weight;
        }
    }
    __threadfence_system();
    __syncthreads();
    if(threadIdx.x==0) {
        #pragma unroll 1
        for(NvU32 w=0;w<workers;w++) {
            auto *dst=reinterpret_cast<volatile NvU32*>(aliases[w]);
            dst[1]=layer; dst[2]=1; // ResidentDynamicControl: layer, mode=MoE
            __threadfence_system();
            dst[0]=sequence;
        }
        __threadfence_system();
    }
}

__device__ __forceinline__ static volatile NvU32 *rtx_pex_word(
        const NvU64 *aliases, NvU32 group, NvU32 byte_offset) {
    // The mailbox has a large PLE area between its control words and expert
    // request fields. The PEX alias table is compact, not indexed by the
    // mailbox's absolute 4-KiB page number:
    //   alias 0 -> mailbox [0x0000,0x1000)
    //   alias 1 -> mailbox [0x8000,0x9000)
    //   alias 2 -> mailbox [0x9000,0xA000)
    //   alias 3 -> mailbox [0xA000,0xB000)
    NvU32 page, logical_page_base;
    if (byte_offset < kPexPageBytes) {
        page = 0;
        logical_page_base = 0;
    } else if (byte_offset >= 0x8000 && byte_offset < 0xB000) {
        page = 1 + (byte_offset - 0x8000) / kPexPageBytes;
        logical_page_base = 0x8000 + (page - 1) * kPexPageBytes;
    } else if (byte_offset >= 0xC000 && byte_offset < 0xF000) {      // alias 4..6 -> mailbox [0xC000,0xF000)
        page = 4 + (byte_offset - 0xC000) / kPexPageBytes;
        logical_page_base = 0xC000 + (page - 4) * kPexPageBytes;
    } else {
        asm volatile("trap;");
        return nullptr;
    }
    const NvU32 word = (byte_offset - logical_page_base) / sizeof(NvU32);
    return reinterpret_cast<volatile NvU32 *>(aliases[group*kPexPagesPerGroup+page]) + word;
}

// PEX ingress aliases are arranged as four independent 4-KiB requester leaves
// per card: control page, then the three pages covering IDs/routes/H. The PEX
// hardware fans each card-ingress write to that card's four retained mailboxes.
// Return-pointer metadata remains per worker and is initialized separately.
__global__ static void rtx_peer_pex_publish_payload(const void *h, int h_type,
        const int32_t *ids, const float *routes, const NvU64 *aliases,
        NvU32 layer) {
    const NvU32 group = blockIdx.x;
    if (group >= kPexGroups) return;
    const __half *h16 = reinterpret_cast<const __half *>(h);
    const __nv_bfloat16 *hb = reinterpret_cast<const __nv_bfloat16 *>(h);
    if (threadIdx.x == 0) {
        *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, layer)) = layer;
        *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, mode)) = 1;
        *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, reserved)) = 0;   // mode-1 x format: FP32
    }
    for (NvU32 i = threadIdx.x; i < QWEN_H; i += blockDim.x) {
        const float value = h_type == 0 ? reinterpret_cast<const float *>(h)[i]
                         : h_type == 1 ? __half2float(h16[i]) : __bfloat162float(hb[i]);
        *rtx_pex_word(aliases, group, kControlHOff + i*sizeof(NvU32)) = __float_as_uint(value);
    }
    for (NvU32 k = threadIdx.x; k < 10; k += blockDim.x) {
        *rtx_pex_word(aliases, group, kControlIdsOff + k*sizeof(NvS32)) = static_cast<NvU32>(ids[k]);
        *rtx_pex_word(aliases, group, kControlRoutesOff + k*sizeof(float)) = __float_as_uint(routes[k]);
    }
    __threadfence_system();
}

__global__ static void rtx_peer_pex_publish_sequence(const NvU64 *aliases, NvU32 sequence) {
    const NvU32 group = blockIdx.x;
    if (group < kPexGroups && threadIdx.x == 0) {
        auto *control = reinterpret_cast<volatile NvU32 *>(aliases[group*kPexPagesPerGroup]);
        // Repeat the operation selector at the final publication boundary.
        // The payload kernel's separate mode store was not visible in the BAR
        // readback even though H/IDs/routes were; placing it immediately
        // before seq makes the command state explicit before workers wake.
        control[2] = 1; // ResidentDynamicControl: MoE mode
        __threadfence_system();
        control[0] = sequence;
        __threadfence_system();
    }
}

// One-store diagnostic for separating GPU-originated PEX delivery from the
// larger request/polling protocol. Offset 0xfffc is the last word in the
// retained 64-KiB worker control allocation and is outside PeerControlLayout.
__global__ static void rtx_peer_pex_word_store(const NvU64 *aliases,
                                               NvU32 byte_offset,
                                               NvU32 value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        auto *dst = reinterpret_cast<volatile NvU32 *>(aliases[0]);
        dst[byte_offset / sizeof(NvU32)] = value;
        __threadfence_system();
    }
}

__global__ static void rtx_peer_set_return_metadata(const NvU64 *worker_aliases,
        const NvU64 *m10_return_aliases, NvU32 workers) {
    const NvU32 w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= workers) return;
    auto *control = reinterpret_cast<volatile PeerControlLayout *>(worker_aliases[w]);
    control->cpu_mailbox_worker = w;
    control->cpu_mailbox_reserved = Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN;
    control->cpu_mailbox_device_ptr = m10_return_aliases[w];
    __threadfence_system();
}

static NvU64 rtx_sys_pte_for_bus(NvU64 destination, NvU64 alignment = PAGE) {
    constexpr NvU64 ADDRESS_MASK = 0x003fffffffffff00ULL;
    constexpr NvU64 VALID = 1ULL, SYS_COHERENT = 2ULL << 1, VOL = 1ULL << 3;
    constexpr NvU64 ATOMIC_DISABLE = 1ULL << 7, GENERIC_KIND = 0x06ULL << 56;
    if (!alignment || (alignment & (alignment - 1)) || (destination & (alignment - 1)))
        throw std::runtime_error("RTX SYS PTE bus destination is not aligned to the requested leaf size");
    const NvU64 encoded = (destination >> 12) << 8;
    if ((encoded & ~ADDRESS_MASK) || (((encoded >> 8) << 12) != destination))
        throw std::runtime_error("RTX PEX bus destination exceeds requester SYS PTE address field");
    return VALID | SYS_COHERENT | VOL | ATOMIC_DISABLE | GENERIC_KIND | encoded;
}

static void restore_pex_routes(PeerState &s) {
    if (!s.pex_quiesced) throw std::runtime_error("RTX PEX stores may be outstanding; retaining aliases");
    for (auto it = s.pex_routes.rbegin(); it != s.pex_routes.rend(); ++it) {
        if (!it->acquired) continue;
        UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
        p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_RESTORE;
        p.mapping_id = it->active.mapping_id;
        p.target_va = it->active.target_va;
        p.target_pte_phys = it->active.target_pte_phys;
        errno = 0;
        const int rc = ioctl(s.uvm_fd, UVM_GLM53_MULTI_PTE_REMAP, &p);
        const int saved_errno = errno;
        if (rc || p.rmStatus != NV_OK || !p.pte_restored ||
            p.pte_after != it->query.pte_after) {
            std::fprintf(stderr, "PRESERVE,RTX_PEX_PTE_RESTORE_FAILED,va,0x%llx,rc,%d,errno,%d,rm,0x%08x,expected,0x%llx,actual,0x%llx\n",
                (unsigned long long)it->query.target_va, rc, saved_errno,
                (unsigned)p.rmStatus, (unsigned long long)it->query.pte_after,
                (unsigned long long)p.pte_after);
            throw std::runtime_error("could not restore RTX PEX alias exactly; retaining resources");
        }
        q38_journal_line("PTE_RESTORED\t%llu", (unsigned long long)it->active.mapping_id);
        std::fprintf(stderr,
            "RTX_PEX_PTE_RESTORE,va,0x%llx,expected,0x%llx,actual,0x%llx,restored,1\n",
            (unsigned long long)it->query.target_va,
            (unsigned long long)it->query.pte_after,
            (unsigned long long)p.pte_after);
        it->acquired = false;
    }
    s.pex_routes.clear();
}

static void verify_pex_dma_helper(const std::vector<PexGroupRecord> &groups, PeerState &s) {
    s.pex_dma_fd = open("/dev/rtx_mc_dma_map", O_RDWR | O_CLOEXEC);
    if (s.pex_dma_fd < 0)
        throw std::runtime_error(std::string("RTX PEX path requires the already-validated rtx_mc_dma_map resource mappings: ") + strerror(errno));
    auto read_param = [](const std::string &name) {
        std::ifstream f("/sys/module/rtx_mc_dma_map/parameters/" + name);
        std::string value;
        if (!f || !std::getline(f, value)) throw std::runtime_error("cannot read active rtx_mc_dma_map parameter " + name);
        return value;
    };
    if (std::stoi(read_param("groups")) != int(groups.size()))
        throw std::runtime_error("active rtx_mc_dma_map group count differs from topology");
    std::string ingress_text = read_param("ingress");
    std::stringstream ingress_stream(ingress_text);
    std::string token;
    for (size_t g=0; g<groups.size(); ++g) {
        if (!std::getline(ingress_stream, token, ','))
            throw std::runtime_error("active rtx_mc_dma_map ingress list is shorter than the topology");
        const NvU64 observed = std::stoull(token, nullptr, 0);
        if (observed != groups[g].ingress)
            throw std::runtime_error("active rtx_mc_dma_map ingress address differs from current topology");
        const NvU64 iova = std::stoull(read_param("iova" + std::to_string(g)), nullptr, 0);
        if (!iova || (iova & (kPexWindowBytes-1)))
            throw std::runtime_error("active rtx_mc_dma_map does not retain an aligned DMA resource mapping");
        std::fprintf(stderr,"RTX_PEX_DMA_RESOURCE,group,%zu,resource_bus,0x%llx,retained_dma_iova,0x%llx,direction,DMA_FROM_DEVICE,encoded_pte_uses,PCI_BUS_ADDRESS\n",
            g,(unsigned long long)observed,(unsigned long long)iova);
    }
    // Compare with the RTX this process opened (never a hardcoded address).
    char opened[32] = {};
    CK(cudaDeviceGetPCIBusId(opened, sizeof(opened), s.rtx_ordinal));
    std::string opened_bdf(opened);
    for (auto &c : opened_bdf) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (opened_bdf.rfind("00000000:", 0) == 0) opened_bdf = "0000:" + opened_bdf.substr(9);
    std::string requester = read_param("requester_bdf");
    for (auto &c : requester) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (requester != opened_bdf)
        throw std::runtime_error("rtx_mc_dma_map requester " + requester + " is not the opened RTX " + opened_bdf);
}

static void setup_pex_routes(PeerState &s, const char *groups_path,
        const UVM_GLM53_VA_SPACE_PDB_ENTRY &pdb) {
    const auto groups = load_pex_groups(groups_path, s.workers);
    const char *direct_env = std::getenv("Q38_RTX_PEER_DIRECT_UNICAST_DIAGNOSTIC");
    s.direct_unicast_diagnostic = direct_env && std::strcmp(direct_env, "1") == 0;
    if (!s.direct_unicast_diagnostic)
        verify_pex_dma_helper(groups, s);
    // Four logical mailbox windows fit inside one 64-KiB PEX ingress window.
    // Reserve alignment slop and map exactly one 64-KiB requester leaf per
    // switch group, reducing ingress remap records from 24 to 6.
    {const char *ob=std::getenv("Q38_RTX_PEER_PEX_OUTER_BASE");
     s.pex_outer_base=(ob&&*ob&&!s.direct_unicast_diagnostic)?std::stoull(ob,nullptr,0):0;
     if(s.pex_outer_base&(kPexLeafBytes-1))throw std::runtime_error("Q38_RTX_PEER_PEX_OUTER_BASE must be 64 KiB aligned");
     for(const auto &g:groups)if(s.pex_outer_base&&s.pex_outer_base+kPexLeafBytes-1>=g.bridge_lo&&s.pex_outer_base<=g.bridge_hi)
         throw std::runtime_error("outer multicast window overlaps a card bridge window");}
    const NvU32 outer_routes = s.pex_outer_base ? 1u : 0u;
    const size_t host_bytes = size_t(kPexGroups + 1 + outer_routes) * kPexLeafBytes;
    CK(cudaHostAlloc(&s.pex_host_alias, host_bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    void *dev_raw_ptr = nullptr;
    CK(cudaHostGetDevicePointer(&dev_raw_ptr, s.pex_host_alias, 0));
    const NvU64 dev_raw = reinterpret_cast<NvU64>(dev_raw_ptr);
    const NvU64 dev_base = (dev_raw + kPexLeafBytes - 1) & ~(NvU64(kPexLeafBytes) - 1);
    const size_t host_offset = size_t(dev_base - dev_raw);
    if (host_offset + size_t(kPexGroups + outer_routes) * kPexLeafBytes > host_bytes)
        throw std::runtime_error("aligned RTX PEX aliases exceed mapped-host allocation");
    auto *host_base = static_cast<unsigned char *>(s.pex_host_alias) + host_offset;
    auto *dev_base_ptr = reinterpret_cast<void *>(dev_base);
    const NvU32 route_count = s.direct_unicast_diagnostic ? 1 : kPexGroups + outer_routes;   // group 6 = outer
    const NvU32 prime_pages = route_count * (kPexLeafBytes / PAGE);
    prime_host_alias<<<1,1>>>(reinterpret_cast<volatile NvU32 *>(dev_base_ptr), prime_pages, 0x38e70000u);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    for (NvU32 i=0; i<prime_pages; ++i)
        if (reinterpret_cast<volatile NvU32 *>(host_base)[size_t(i)*PAGE/4] != 0x38e70000u+i)
            throw std::runtime_error("RTX PEX alias priming check failed at 4-KiB subpage " + std::to_string(i));

    std::vector<NvU64> aliases(kPexAliases), outer_aliases(kPexPagesPerGroup);
    s.pex_routes.reserve(route_count);
    for (NvU32 g=0; g<route_count; ++g) {
        const NvU64 va = dev_base + NvU64(g)*kPexLeafBytes;
        const bool outer = g == kPexGroups;
        const NvU64 destination = s.direct_unicast_diagnostic ? s.workers[0].bar_bus : outer ? s.pex_outer_base : groups[g].ingress;
        const NvU64 window_lo = outer ? s.pex_outer_base : groups[s.direct_unicast_diagnostic ? 0 : g].bridge_lo;
        const NvU64 window_hi = outer ? s.pex_outer_base + kPexLeafBytes - 1 : groups[s.direct_unicast_diagnostic ? 0 : g].bridge_hi;
        if (destination < window_lo || destination > window_hi ||
            kPexLeafBytes-1 > window_hi-destination)
            throw std::runtime_error(s.direct_unicast_diagnostic ?
                "RTX direct owner BAR1 leaf is outside the recorded bridge window" :
                "RTX PEX ingress leaf is outside the recorded bridge window");
        // Default build: one 64-KiB leaf per group (remap 0 at offset 0).
        // Q38_RTX_PEER_SMALL_LEAVES: one 4-KiB leaf per used control page.
        for (NvU32 remap=0; remap<kPexRemapsPerGroup; ++remap) {
        const NvU64 page_off = kPexRemapsPerGroup == 1 ? 0 : kPexPageOffsets[remap];
        const NvU64 leaf_va = va + page_off, leaf_dest = destination + page_off;
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q{};
        q.requester_uuid=s.rtx_uuid; q.op=UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
        q.va_space_ptr=pdb.va_space_ptr; q.gpu_va_space_ptr=pdb.gpu_va_space_ptr;
        q.root_phys_address=pdb.root_phys_address; q.root_phys_aperture=pdb.root_phys_aperture;
        q.target_va=leaf_va; q.page_size=kPexRemapBytes;
        errno=0; const int qrc=ioctl(s.uvm_fd,UVM_GLM53_SERIAL_BAR1_PEER_READ,&q); const int qe=errno;
        if (qrc || q.rmStatus!=NV_OK || !q.target_pte_phys)
            throw std::runtime_error("RTX PEX alias PTE query failed, rc="+std::to_string(qrc)+
                ", errno="+std::to_string(qe)+", RM="+std::to_string((unsigned)q.rmStatus));
        if (kPexRemapBytes == PAGE && !(q.pte_after & 1))
            throw std::runtime_error("RTX PEX 4-KiB alias leaf is not present before remap");
        UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
        p.gpu_uuid=s.rtx_uuid; p.op=UVM_GLM53_MULTI_PTE_REMAP_OP_APPLY;
        p.root_phys_aperture=q.root_phys_aperture; p.va_space_ptr=q.va_space_ptr;
        p.gpu_va_space_ptr=q.gpu_va_space_ptr; p.root_phys_address=q.root_phys_address;
        p.target_va=leaf_va; p.page_size=kPexRemapBytes; p.target_pte_phys=q.target_pte_phys;
        p.expected_current_pte=q.pte_after; p.requested_temporary_pte=rtx_sys_pte_for_bus(leaf_dest,kPexRemapBytes);
        p.requested_destination_gpu_sys_address=leaf_dest; p.requested_allow_writable=1;
        errno=0; const int arc=ioctl(s.uvm_fd,UVM_GLM53_MULTI_PTE_REMAP,&p); const int ae=errno;
        const bool acquired=arc==0 && p.rmStatus==NV_OK;
        if (acquired) {
            q38_journal_line("PTE\t%llu\t%s\t%llu\t%llu\t%llu\t%llu\t0",(unsigned long long)p.mapping_id,
                q38_uuid_hex(s.rtx_uuid).c_str(),(unsigned long long)p.target_va,(unsigned long long)p.target_pte_phys,
                (unsigned long long)p.temporary_pte,(unsigned long long)p.pte_before);
            PeerState::PexLeaf leaf{}; leaf.query=q; leaf.active=p; leaf.destination=leaf_dest; leaf.acquired=true;
            s.pex_routes.push_back(leaf);
        }
        std::fprintf(stderr,"RTX_PEX_PTE,group,%u,leaf_bytes,0x%x,va,0x%llx,destination_bus,0x%llx,original,0x%llx,candidate,0x%llx,after,0x%llx,rc,%d,errno,%d,rm,0x%08x,acquired,%u,page_offset,0x%llx,pte_phys,0x%llx,l2_invalidate,0x%08x,phys_tlb_invalidate,0x%08x\n",
            g,kPexRemapBytes,(unsigned long long)leaf_va,(unsigned long long)leaf_dest,
            (unsigned long long)q.pte_after,(unsigned long long)p.requested_temporary_pte,
            (unsigned long long)p.pte_after,arc,ae,(unsigned)p.rmStatus,acquired,
            (unsigned long long)page_off,(unsigned long long)q.target_pte_phys,
            (unsigned)p.requester_l2_invalidate_status,(unsigned)p.requester_phys_tlb_invalidate_status);
        if (!acquired || p.pte_after!=p.requested_temporary_pte) {
            if (acquired) restore_pex_routes(s);
            throw std::runtime_error("guarded RTX PEX PTE install did not match the requested bus address");
        }
        }
        if (s.direct_unicast_diagnostic) {
            aliases[0] = va;
        }
        else if (outer) {
            for (NvU32 pidx=0; pidx<kPexPagesPerGroup; ++pidx) outer_aliases[pidx]=va+kPexPageOffsets[pidx];
        }
        else {
            for (NvU32 pidx=0; pidx<kPexPagesPerGroup; ++pidx)
                aliases[g*kPexPagesPerGroup+pidx]=va+kPexPageOffsets[pidx];
        }
    }
    CK(cudaMalloc(&s.device_pex_aliases,aliases.size()*sizeof(NvU64)));
    CK(cudaMemcpy(s.device_pex_aliases,aliases.data(),aliases.size()*sizeof(NvU64),cudaMemcpyHostToDevice));
    s.pub_aliases=s.device_pex_aliases; s.pub_groups=kPexGroups;
    {const char *po=std::getenv("Q38_RTX_PEER_PUBLISH_ORDER"); s.publish_order=po&&*po?NvU32(std::atoi(po)):4u;   // default 4 (T293, T294-T309)
     if(s.publish_order>4)throw std::runtime_error("Q38_RTX_PEER_PUBLISH_ORDER must be 0..4");
     {const char *gg=std::getenv("Q38_RTX_PEER_GO_GAP_NS"); s.go_gap_ns=gg&&*gg?NvU32(std::atoi(gg)):0u; if(s.go_gap_ns>50000)throw std::runtime_error("Q38_RTX_PEER_GO_GAP_NS > 50000");}
     if(const char *q=std::getenv("Q38_RTX_PEER_PUBLISH_SEQ"); q&&*q){
        NvU32 v=0,seen=0,n=0; for(const char *c=q;*c;c++){ if(*c==',')continue; const NvU32 g=NvU32(*c-'0');
            if(g>=kPexGroups||(seen>>g&1u)||n>=kPexGroups)throw std::runtime_error("Q38_RTX_PEER_PUBLISH_SEQ must be a permutation of 0..5");
            seen|=1u<<g; v|=g<<(4*n); n++; }
        if(n!=kPexGroups)throw std::runtime_error("Q38_RTX_PEER_PUBLISH_SEQ must list all 6 groups"); s.publish_seq=v; }
     {const char *ap=std::getenv("Q38_RTX_PEER_ACK_PROFILE"); const NvU32 on=ap&&*ap=='1'?1u:0u;
      CK(cudaMemcpyToSymbol(g_q38_ack_profile,&on,sizeof(on))); std::fprintf(stderr,"RTX_ACK_PROFILE,%u\n",on);}
     {const char *lf=std::getenv("Q38_RTX_PEER_GO_LOWFENCE"); s.go_lowfence=lf&&*lf=='1'?1u:0u;}
     std::fprintf(stderr,"RTX_PEX_PUBLISH_ORDER,%u,seq,0x%06x,go_gap_ns,%u,lowfence,%u\n",s.publish_order,s.publish_seq,s.go_gap_ns,s.go_lowfence);}
    if (outer_routes) {
        CK(cudaMalloc(&s.device_pex_outer_aliases,outer_aliases.size()*sizeof(NvU64)));
        CK(cudaMemcpy(s.device_pex_outer_aliases,outer_aliases.data(),outer_aliases.size()*sizeof(NvU64),cudaMemcpyHostToDevice));
        s.pub_aliases=s.device_pex_outer_aliases; s.pub_groups=1;
        std::fprintf(stderr,"RTX_PEX_OUTER_MULTICAST,base,0x%llx,leaves,%u,publish_groups,1\n",(unsigned long long)s.pex_outer_base,kPexRemapsPerGroup);
    }
    if (s.direct_unicast_diagnostic) {
        s.pex_ingresses.push_back(s.workers[0].bar_bus);
    }
    else {
        s.pex_ingresses.reserve(groups.size());
        for (const auto &g : groups) s.pex_ingresses.push_back(g.ingress);
    }
    s.pex_multicast=true;
    if (s.direct_unicast_diagnostic)
        std::fprintf(stderr,"RTX_DIRECT_UNICAST_READY,target_bdf,%s,owner_bar1_bus,0x%llx,remap_leaves,%zu,leaf_bytes,%u\n",
            s.workers[0].bdf.c_str(),(unsigned long long)s.workers[0].bar_bus,s.pex_routes.size(),kPexLeafBytes);
    else
        std::fprintf(stderr,"RTX_PEX_MULTICAST_READY,groups,%u,worker_gpus,%zu,logical_aliases,%u,remap_leaves,%zu,leaf_bytes,%u,bytes_per_group,%u,source,active_topology_tsv\n",
            kPexGroups,s.workers.size(),kPexAliases,s.pex_routes.size(),kPexRemapBytes,kPexWindowBytes);
}

#ifdef Q38_RTX_PEER_LEADER_AGG
// Workers whose vector is already folded into their card leader's sum. Their RTX
// slot header is still validated; only the vector is skipped.
__device__ NvU32 g_q38_agg_skip_mask = 0;
__device__ NvU32 g_q38_batch_ok = 0;   // batched MoE: verdict of the part-flag check (PciClusterMoeBatchOp::combine)
#endif
__global__ static void rtx_peer_reduce(const NvU64 *return_slots,NvU32 workers,
        NvU32 sequence,NvU32 layer,float *output,NvU32 *status,NvU32 *observed_headers,
        NvU64 timeout_cycles) {
    __shared__ NvU32 headers_ok;
    if(threadIdx.x==0) {
        headers_ok=workers==24;
        const NvU64 wait_start=clock64();
        for(NvU32 w=0;w<workers;w++) {
            auto *r=reinterpret_cast<const volatile NvU32*>(return_slots[w]);
            while(r[0]!=sequence && clock64()-wait_start<timeout_cycles) { }
            observed_headers[w*4+0]=r[0];
            observed_headers[w*4+1]=r[1];
            observed_headers[w*4+2]=r[2];
            observed_headers[w*4+3]=r[3];
            // ResidentDirectResult header: seq, status, exit_reason, ready.
            // The worker identity is bound by the per-BDF requester PTE/slot map.
            if(r[0]!=sequence || r[1]!=0 || r[2]!=0 || r[3]!=1) { headers_ok=0; break; }
        }
        // Extra two words are reserved after the 24 four-word diagnostic
        // headers. This lets profiling split the reducer's peer wait from its
        // subsequent result gather/reduction without another kernel launch.
        const NvU64 wait_ticks=clock64()-wait_start;
        observed_headers[workers*4+0]=NvU32(wait_ticks);
        observed_headers[workers*4+1]=NvU32(wait_ticks>>32);
        *status=headers_ok?0u:1u;
    }
    __syncthreads();
    if(!headers_ok) return;
    const auto *gate_ptr=reinterpret_cast<const volatile float*>(return_slots[20]+offsetof(ResidentDirectResult, gate));
    const float gate=*gate_ptr;
    for(NvU32 j=threadIdx.x;j<QWEN_H;j+=blockDim.x) {
        float routed=0.0f,shared=0.0f;
#ifdef Q38_RTX_PEER_LEADER_AGG
        const NvU32 skip=g_q38_agg_skip_mask;
        for(NvU32 w=0;w<20;w++) if(!((skip>>w)&1u)) routed+=reinterpret_cast<const volatile float*>(return_slots[w]+offsetof(ResidentDirectResult, result))[j];
#else
        for(NvU32 w=0;w<20;w++) routed+=reinterpret_cast<const volatile float*>(return_slots[w]+offsetof(ResidentDirectResult, result))[j];
#endif
        for(NvU32 w=20;w<24;w++) shared+=reinterpret_cast<const volatile float*>(return_slots[w]+offsetof(ResidentDirectResult, result))[j];
        output[j]=routed+gate*shared;
    }
}

// ---------------------------------------------------------------------------------------------
// Phase 2: GPU-driven exchange. Every step is stream work, so the whole MoE call can be captured
// into a CUDA graph and replayed without host involvement. Error reasons (err_reason):
//   1 cycle timeout, 2 wall-clock deadline (%globaltimer), 3 host abort flag, 4 worker error status,
//   5 header mismatch (seq/ready), 6 sequence space exhausted; per-op reasons 7..11 (row gather, LM head, batch)
//   and 12 (multi-row: a reducer's row-1 record did not acknowledge, PciClusterMoe2Op).
__device__ __forceinline__ unsigned long long q38_globaltimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ static void q38_gpu_fail(PeerState::GpuHost *h, NvU32 reason, NvU32 layer, NvU32 seq,
                                    NvU32 worker, NvU32 status, NvU32 exit_reason) {
    volatile PeerState::GpuHost *v = h;
    if (v->err) return;                       // first error wins; single writer thread per call
    v->err_layer = layer; v->err_seq = seq; v->err_reason = reason;
    v->err_worker = worker; v->err_status = status; v->err_exit = exit_reason;
    __threadfence_system();
    v->err = 1;
    __threadfence_system();
}
// ---------------------------------------------------------------------------------------------
// PCI-cluster exchange, RTX side (2026-11-03). One skeleton for every RTX -> M10 request:
//   pci_cluster_begin<Op>          validate the request (Op::skip), take the next DEVICE sequence number
//   pci_cluster_publish<Op>        multicast header {layer, mode} + Op payload to all kPexGroups PEX groups
//   pci_cluster_publish_sequence   multicast the mode word, then the sequence word
//   pci_cluster_wait_combine<Op>   wait for all worker acknowledgements (common, pci_cluster_wait), then
//                                  Op::combine the 24 return slots into the output (or Op::zero on skip/failure)
// The worker side is pci_cluster_exchange.cuh plus the mode bodies in the resident worker kernels. All
// steps are stream work, so an exchange can be captured into a CUDA graph. One sequence counter serves
// every op. An Op provides:
//   Args                       kernel argument block (by value)
//   kMode                      worker mode word
//   skip(a)        -> NvU32    0 = publish, 1 = dummy request (no M10 traffic, output zeros)
//   publish(a, aliases, group) payload words of one PEX group (all threads of the group's block)
//   combine(a, slots)          output from the acknowledged return slots (all threads of one block)
//   zero(a)                    output for skipped or failed calls (all threads of one block)
// Optional per-exchange timing (q38_rtx_peer_timing_enable): %globaltimer stamps in a device ring.
// GpuDev::pad[1] counts records, pad[0] is the current record slot. Null pointer = off (default), so
// captured graphs are unchanged; enabling only makes the same kernels write their stamps.
struct PciClusterTimingRec { NvU32 layer, seq, skip, pad; unsigned long long t_begin, t_pub, t_seq, t_ack, t_done; };
static_assert(sizeof(PciClusterTimingRec) == 56, "timing record layout (read by the adapter)");
__device__ PciClusterTimingRec *g_pc_timing = nullptr;
__device__ NvU32 g_pc_timing_cap = 0;
__device__ __forceinline__ PciClusterTimingRec *pc_rec(const PeerState::GpuDev *d) {
    PciClusterTimingRec *t = g_pc_timing;
    return t ? t + (static_cast<const volatile PeerState::GpuDev *>(d)->pad[0] % g_pc_timing_cap) : nullptr;
}

__device__ __forceinline__ PciClusterTimingRec *pc_rec_pipe(const PeerState::GpuDev *d, NvU32 chunk) {
    PciClusterTimingRec *t = g_pc_timing;
    return t ? t + ((static_cast<const volatile PeerState::GpuDev *>(d)->pipe_rec + chunk) % g_pc_timing_cap) : nullptr;
}

template <class Op>
__global__ static void pci_cluster_begin(PeerState::GpuDev *d, PeerState::GpuHost *h, typename Op::Args a, NvU32 layer) {
    if (blockIdx.x || threadIdx.x) return;
    NvU32 skip = 0;
    if (static_cast<volatile PeerState::GpuHost *>(h)->err) skip = 2;      // after an error: no M10 traffic
    else skip = Op::skip(a);
    if (!skip) {
        const NvU32 next = d->seq + 1;
        if (!next || next == 0xffffffffu) { skip = 2; q38_gpu_fail(h, 6, layer, d->seq, 0, 0, 0); }
        else { d->seq = next; d->cur_seq = next; d->calls++; }
    } else if (skip == 1) d->skips++;
    d->cur_skip = skip;
    if (g_pc_timing) {
        const NvU32 slot = d->pad[1]++;
        d->pad[0] = slot;
        PciClusterTimingRec *r = g_pc_timing + slot % g_pc_timing_cap;
        r->layer = layer; r->seq = d->cur_seq; r->skip = skip;
        r->t_pub = r->t_seq = r->t_ack = r->t_done = 0;
        r->t_begin = q38_globaltimer();
    }
    __threadfence();
}
template <class Op>
__global__ static void pci_cluster_publish(const PeerState::GpuDev *d, typename Op::Args a, const NvU64 *aliases, NvU32 layer) {
    if (static_cast<const volatile PeerState::GpuDev *>(d)->cur_skip) return;
    const NvU32 group = blockIdx.x;
    if (group >= kPexGroups) return;
    if (threadIdx.x == 0) {
        *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, layer)) = layer;
        *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, mode)) = Op::kMode;
    }
    Op::publish(a, aliases, group);
    __threadfence_system();
    if (group == 0 && threadIdx.x == 0) { PciClusterTimingRec *r = pc_rec(d); if (r) r->t_pub = q38_globaltimer(); }
}
__global__ static void pci_cluster_publish_sequence(const PeerState::GpuDev *d, const NvU64 *aliases, NvU32 mode) {
    const volatile PeerState::GpuDev *v = d;
    if (v->cur_skip) return;
    const NvU32 group = blockIdx.x;
    if (group < kPexGroups && threadIdx.x == 0) {
        auto *control = reinterpret_cast<volatile NvU32 *>(aliases[group*kPexPagesPerGroup]);
        control[2] = mode;
        __threadfence_system();
        control[0] = v->cur_seq;
        __threadfence_system();
        if (group == 0) { PciClusterTimingRec *r = pc_rec(d); if (r) r->t_seq = q38_globaltimer(); }
    }
}
// Publish order experiment (2026-11-04, Q38_RTX_PEER_PUBLISH_ORDER): payload AND go (mode word, sequence) per group
// in one kernel. serial = 1: one block does group 0 (payload, fence, go), then group 1, ... so card 0 starts first and
// the returns spread out; serial = 0: block g does group g (no group waits for the other groups' payloads).
template <class Op>
__global__ static void pci_cluster_publish_go(const PeerState::GpuDev *d, typename Op::Args a, const NvU64 *aliases, NvU32 layer, NvU32 serial, NvU32 order) {
    const volatile PeerState::GpuDev *v = d;
    if (v->cur_skip) return;
    // serial: the i-th group to get payload + go is nibble i of order (default 0x543210 = card 0 first, shared card 5 last)
    const NvU32 i0 = serial ? 0 : blockIdx.x, i1 = serial ? kPexGroups : blockIdx.x + 1;
    for (NvU32 i = i0; i < i1 && i < kPexGroups; i++) {
        const NvU32 group = serial ? (order >> (4 * i)) & 0xFu : i;
        if (group >= kPexGroups) continue;
        if (threadIdx.x == 0) {
            *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, layer)) = layer;
            *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, mode)) = Op::kMode;
        }
        Op::publish(a, aliases, group);
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            auto *control = reinterpret_cast<volatile NvU32 *>(aliases[group*kPexPagesPerGroup]);
            control[2] = Op::kMode;
            __threadfence_system();
            control[0] = v->cur_seq;
            __threadfence_system();
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 && (serial || blockIdx.x == 0)) {
        PciClusterTimingRec *r = pc_rec(d); if (r) { r->t_pub = q38_globaltimer(); r->t_seq = r->t_pub; }
    }
}
// Publish order 4 (2026-11-05, default since T294-T309): pci_cluster_begin + pci_cluster_publish + pci_cluster_publish_sequence in ONE
// kernel (outer group only, 1 block): thread 0 does the begin logic, block barrier; all threads write the request;
// each thread fences (system), block barrier; thread 0 writes the sequence word. The mode word (control[2]) is the
// request header's mode field, written above before the fence, so the separate mode write + fence of
// pci_cluster_publish_sequence are not repeated. Two kernel boundaries and one to two fences fewer per exchange.
template <class Op>
__global__ static void pci_cluster_begin_publish_go(PeerState::GpuDev *d, PeerState::GpuHost *h, typename Op::Args a,
        const NvU64 *aliases, NvU32 layer) {
    if (blockIdx.x) return;
    __shared__ NvU32 skip_s;
    if (threadIdx.x == 0) {                                           // = pci_cluster_begin
        NvU32 skip = 0;
        if (static_cast<volatile PeerState::GpuHost *>(h)->err) skip = 2;
        else skip = Op::skip(a);
        if (!skip) {
            const NvU32 next = d->seq + 1;
            if (!next || next == 0xffffffffu) { skip = 2; q38_gpu_fail(h, 6, layer, d->seq, 0, 0, 0); }
            else { d->seq = next; d->cur_seq = next; d->calls++; }
        } else if (skip == 1) d->skips++;
        d->cur_skip = skip;
        if (g_pc_timing) {
            const NvU32 slot = d->pad[1]++;
            d->pad[0] = slot;
            PciClusterTimingRec *r = g_pc_timing + slot % g_pc_timing_cap;
            r->layer = layer; r->seq = d->cur_seq; r->skip = skip;
            r->t_pub = r->t_seq = r->t_ack = r->t_done = 0;
            r->t_begin = q38_globaltimer();
        }
        __threadfence();
        skip_s = skip;
    }
    __syncthreads();
    if (skip_s) return;                                               // dummy / failed rows: no M10 traffic (as before)
    if (threadIdx.x == 0) {                                           // = pci_cluster_publish, group 0 (outer)
        *rtx_pex_word(aliases, 0, offsetof(PeerControlLayout, layer)) = layer;
        *rtx_pex_word(aliases, 0, offsetof(PeerControlLayout, mode)) = Op::kMode;
    }
    Op::publish(a, aliases, 0);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        PciClusterTimingRec *r = pc_rec(d); if (r) r->t_pub = q38_globaltimer();
        auto *control = reinterpret_cast<volatile NvU32 *>(aliases[0]);
        control[0] = static_cast<const volatile PeerState::GpuDev *>(d)->cur_seq;   // the go
        if (r) r->t_seq = q38_globaltimer();
    }
}
// Publish order 3 (2026-11-04): payload once through the outer multicast group (pci_cluster_publish, 1 group), then
// the go words (mode, sequence) card by card through the per-card group mappings, which stay installed when the
// outer group is on. order = nibble list as for publish_go; gap_ns = spin between two cards' go words.
// lowfence = 1: only the sequence word per card, no fences between cards. The mode word (control[2]) was already
// written to every card by pci_cluster_publish through the outer group, which ends with __threadfence_system; that
// fence plus the kernel boundary orders it before every sequence word (as in the default doorbell).
__global__ static void pci_cluster_go_staggered(const PeerState::GpuDev *d, const NvU64 *card_aliases, NvU32 mode, NvU32 order, NvU32 gap_ns, NvU32 lowfence) {
    const volatile PeerState::GpuDev *v = d;
    if (v->cur_skip || blockIdx.x || threadIdx.x) return;
    const NvU32 seq = v->cur_seq;
    for (NvU32 i = 0; i < kPexGroups; i++) {
        const NvU32 group = (order >> (4 * i)) & 0xFu;
        if (group >= kPexGroups) continue;
        if (i && gap_ns) { const unsigned long long t0 = q38_globaltimer(); while (q38_globaltimer() - t0 < gap_ns) {} }
        auto *control = reinterpret_cast<volatile NvU32 *>(card_aliases[group*kPexPagesPerGroup]);
        if (lowfence) { control[0] = seq; continue; }
        control[2] = mode;
        __threadfence_system();
        control[0] = seq;
        __threadfence_system();
    }
    if (lowfence) __threadfence_system();
    PciClusterTimingRec *r = pc_rec(d); if (r) r->t_seq = q38_globaltimer();
}
// Thread 0 only: wait until every worker acknowledged the current sequence, with four exits
// (cycle timeout, %globaltimer deadline, host abort flag, worker error status) and a header check.
// Returns true on success; on failure records the sticky error (q38_gpu_fail).
// at_least (pipelined batch): a slot already acknowledging a later sequence also counts (wrap-safe compare);
// its header words then belong to that later request.
__device__ static bool pci_cluster_wait_seq(NvU32 sequence, bool at_least, PeerState::GpuHost *hst,
        const NvU64 *return_slots, NvU32 workers, NvU32 layer, NvU32 *observed_headers,
        NvU64 timeout_cycles, NvU64 deadline_ns) {
    const volatile PeerState::GpuHost *vh = hst;
    const NvU64 c0 = clock64(); const unsigned long long t0 = q38_globaltimer();
    NvU32 reason = 0, bad = 0, st = 0, ex = 0, polls = 0;
    bool ok = workers == 24;
    for (NvU32 w = 0; w < workers && !reason; w++) {
        auto *r = reinterpret_cast<const volatile NvU32 *>(return_slots[w]);
        for (;;) {
            const NvU32 got = r[0];
            if (got == sequence || (at_least && int(got - sequence) > 0)) break;
            if (clock64() - c0 > timeout_cycles) { reason = 1; break; }
            if ((++polls & 1023u) == 0) {
                if (q38_globaltimer() - t0 > deadline_ns) { reason = 2; break; }
                if (vh->abort) { reason = 3; break; }
            }
        }
        if (!reason) {
            observed_headers[w*4+0] = r[0]; observed_headers[w*4+1] = r[1];
            observed_headers[w*4+2] = r[2]; observed_headers[w*4+3] = r[3];
            if (r[1] != 0) { reason = 4; bad = w; st = r[1]; ex = r[2]; }
            else if (r[2] != 0 || r[3] != 1) { reason = 5; bad = w; st = r[1]; ex = r[2]; }
        } else bad = w;
    }
    if (reason) { ok = false; q38_gpu_fail(hst, reason, layer, sequence, bad, st, ex); }
    return ok;
}
__device__ static bool pci_cluster_wait(const PeerState::GpuDev *d, PeerState::GpuHost *hst,
        const NvU64 *return_slots, NvU32 workers, NvU32 layer, NvU32 *observed_headers,
        NvU64 timeout_cycles, NvU64 deadline_ns) {
    return pci_cluster_wait_seq(static_cast<const volatile PeerState::GpuDev *>(d)->cur_seq, false, hst, return_slots, workers,
                                layer, observed_headers, timeout_cycles, deadline_ns);
}
// Ack profile (2026-11-04, Q38_RTX_PEER_ACK_PROFILE=1): before the normal in-order wait, thread 0 polls all 24
// headers round-robin and records when the data headers (card leaders = workers not in g_q38_agg_skip_mask, and the
// shared workers 20..23) were complete and when all 24 were. rec->pad = min(t_all - t_data, 0xFFFFFF ns) |
// (index of the last header seen) << 24. Measurement only: the normal wait then runs unchanged and validates.
__device__ static void pci_cluster_ack_profile(const PeerState::GpuDev *d, const NvU64 *return_slots, NvU32 workers, NvU64 deadline_ns) {
    if (workers != 24) return;
    const NvU32 sequence = static_cast<const volatile PeerState::GpuDev *>(d)->cur_seq;
#ifdef Q38_RTX_PEER_LEADER_AGG
    const NvU32 data_mask = (~g_q38_agg_skip_mask & 0xFFFFFu) | 0xF00000u;
#else
    const NvU32 data_mask = 0xFFFFFFu;
#endif
    NvU32 seen = 0, last = 0;
    unsigned long long t_data = 0;
    const unsigned long long t0 = q38_globaltimer();
    while (seen != 0xFFFFFFu) {
        for (NvU32 w = 0; w < 24; w++) {
            if ((seen >> w) & 1u) continue;
            if (reinterpret_cast<const volatile NvU32 *>(return_slots[w])[0] == sequence) {
                seen |= 1u << w; last = w;
                if (!t_data && (seen & data_mask) == data_mask) t_data = q38_globaltimer();
            }
        }
        if (q38_globaltimer() - t0 > deadline_ns) return;      // the normal wait reports the failure
    }
    const unsigned long long t_all = q38_globaltimer();
    if (!t_data) t_data = t_all;
    PciClusterTimingRec *r = pc_rec(d);
    if (r) { const unsigned long long lag = t_all - t_data; r->pad = NvU32(lag > 0xFFFFFFull ? 0xFFFFFFull : lag) | (last << 24); }
}
template <class Op>
__global__ static void pci_cluster_wait_combine(const PeerState::GpuDev *d, PeerState::GpuHost *hst, typename Op::Args a,
        const NvU64 *return_slots, NvU32 workers, NvU32 layer, NvU32 *observed_headers,
        NvU64 timeout_cycles, NvU64 deadline_ns) {
    __shared__ NvU32 ok, skip;
    if (threadIdx.x == 0) {
        skip = static_cast<const volatile PeerState::GpuDev *>(d)->cur_skip; ok = 0;
        if (!skip && g_q38_ack_profile) pci_cluster_ack_profile(d, return_slots, workers, deadline_ns);
        if (!skip) ok = pci_cluster_wait(d, hst, return_slots, workers, layer, observed_headers, timeout_cycles, deadline_ns);
        PciClusterTimingRec *r = pc_rec(d); if (r) r->t_ack = q38_globaltimer();
    }
    __syncthreads();
    if (skip || !ok) Op::zero(a);                        // dummy rows and failed calls produce zeros
    else Op::combine(a, return_slots);
    __syncthreads();
    if (threadIdx.x == 0) { PciClusterTimingRec *r = pc_rec(d); if (r) r->t_done = q38_globaltimer(); }
}

// MoE wait + combine block size (2026-11-04: 256 -> 1024). Each thread sums whole output elements in the
// fixed worker order, so the block size does not change any result; more threads overlap the slot reads
// (mode 1: 9.2 us, mode 10 two rows: 28.7 us with 256 threads).
constexpr unsigned kMoeCombineThreads = 1024;
// Op: MoE (worker mode 1). Every worker computes its own experts for the broadcast hidden state;
// combine = sum of the 20 routed slots + shared gate * sum of the 4 shared slots.
struct PciClusterMoeOp {
    struct Args { const void *h; int h_type; const int32_t *ids; const float *routes; float *out; };
    static constexpr NvU32 kMode = 1;
    __device__ static NvU32 skip(const Args &a) {
        for (int k = 0; k < 10; k++) { const int e = a.ids[k]; if (e < 0 || e >= 512) return 1; }
        return 0;
    }
    __device__ static void publish(const Args &a, const NvU64 *aliases, NvU32 group) {
        const __half *h16 = reinterpret_cast<const __half *>(a.h);
        if (a.h_type == 2) {
            // BF16 hidden states (2026-11-03): x[] carries H/2 words of raw BF16 pairs, half the multicast bytes
            // (6 groups x 5 KB instead of 6 x 10 KB over the RTX link); the workers widen exactly
            // (control->reserved = 1, Q38_MODE1_X_BF16)
            const NvU32 *hw = reinterpret_cast<const NvU32 *>(a.h);
            for (NvU32 i = threadIdx.x; i < QWEN_H / 2; i += blockDim.x)
                *rtx_pex_word(aliases, group, kControlHOff + i*sizeof(NvU32)) = hw[i];
        } else {
            for (NvU32 i = threadIdx.x; i < QWEN_H; i += blockDim.x) {
                const float value = a.h_type == 0 ? reinterpret_cast<const float *>(a.h)[i] : __half2float(h16[i]);
                *rtx_pex_word(aliases, group, kControlHOff + i*sizeof(NvU32)) = __float_as_uint(value);
            }
        }
        if (threadIdx.x == 0) *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, reserved)) = a.h_type == 2 ? 1u : 0u;
        for (NvU32 k = threadIdx.x; k < 10; k += blockDim.x) {
            *rtx_pex_word(aliases, group, kControlIdsOff + k*sizeof(NvS32)) = static_cast<NvU32>(a.ids[k]);
            *rtx_pex_word(aliases, group, kControlRoutesOff + k*sizeof(float)) = __float_as_uint(a.routes[k]);
        }
    }
    __device__ static void zero(const Args &a) {
        for (NvU32 j = threadIdx.x; j < QWEN_H; j += blockDim.x) a.out[j] = 0.0f;
    }
    __device__ static void combine(const Args &a, const NvU64 *return_slots) {
        const auto *gate_ptr = reinterpret_cast<const volatile float *>(return_slots[20] + offsetof(ResidentDirectResult, gate));
        const float gate = *gate_ptr;
        for (NvU32 j = threadIdx.x; j < QWEN_H; j += blockDim.x) {
            float routed = 0.0f, shared = 0.0f;
#ifdef Q38_RTX_PEER_LEADER_AGG
            const NvU32 skipmask = g_q38_agg_skip_mask;
            for (NvU32 w = 0; w < 20; w++) if (!((skipmask >> w) & 1u)) routed += reinterpret_cast<const volatile float *>(return_slots[w] + offsetof(ResidentDirectResult, result))[j];
#else
            for (NvU32 w = 0; w < 20; w++) routed += reinterpret_cast<const volatile float *>(return_slots[w] + offsetof(ResidentDirectResult, result))[j];
#endif
            for (NvU32 w = 20; w < 24; w++) shared += reinterpret_cast<const volatile float *>(return_slots[w] + offsetof(ResidentDirectResult, result))[j];
            a.out[j] = routed + gate * shared;
        }
    }
};

#ifdef Q38_RTX_PEER_LEADER_AGG
// Op: multi-row decode MoE (worker mode 10, resident_multirow_abi.cuh; 2026-11-04). R = 1..2 BF16 rows in ONE
// multicast request; per row PciClusterMoeOp's combine (same order), row 1 from the reducers' second return
// records. A dummy row (an id outside [0, 512)) is sent as ids 0..9 with routes 0 and gives a zero output row.
// Error reason 12: a reducer's row-1 record does not acknowledge the request (header seq / status / ready).
struct PciClusterMoe2Op {
    // rows 1..4 (rows <= the max rows configured at open); slots1[(r-1)*24 + w] = row-r record of worker w (reducers)
    struct Args { const void *h; const int32_t *ids; const float *routes; float *out; NvU32 rows;
                  const NvU64 *slots1; const PeerState::GpuDev *d; PeerState::GpuHost *hst; NvU32 layer; };
    static constexpr NvU32 kMode = Q38_MR_MODE;
    __device__ static bool dummy(const Args &a, NvU32 r) {
        for (int k = 0; k < 10; k++) { const int e = a.ids[r * 10 + k]; if (e < 0 || e >= 512) return true; }
        return false;
    }
    __device__ static NvU32 skip(const Args &a) {
        for (NvU32 r = 0; r < a.rows; r++) if (!dummy(a, r)) return 0;
        return 1;
    }
    __device__ static void publish(const Args &a, const NvU64 *aliases, NvU32 group) {
        const NvU32 *hw = reinterpret_cast<const NvU32 *>(a.h);
        for (NvU32 r = 0; r < a.rows; r++)
            for (NvU32 i = threadIdx.x; i < QWEN_H / 2; i += blockDim.x)
                *rtx_pex_word(aliases, group, q38_mr_x_off(r) + i * sizeof(NvU32)) = hw[r * (QWEN_H / 2) + i];
        if (threadIdx.x == 0) {
            *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, reserved)) = 1u;      // BF16 x
            *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, ple_count)) = a.rows;
        }
        for (NvU32 t = threadIdx.x; t < a.rows * 10; t += blockDim.x) {
            const NvU32 r = t / 10, k = t % 10;
            const bool dm = dummy(a, r);
            *rtx_pex_word(aliases, group, q38_mr_ids_off(r) + k * sizeof(NvS32)) = dm ? k : NvU32(a.ids[t]);
            *rtx_pex_word(aliases, group, q38_mr_routes_off(r) + k * sizeof(float)) = dm ? 0u : __float_as_uint(a.routes[t]);
        }
    }
    __device__ static void zero(const Args &a) {
        for (NvU32 j = threadIdx.x; j < a.rows * QWEN_H; j += blockDim.x) a.out[j] = 0.0f;
    }
    __device__ static void combine(const Args &a, const NvU64 *return_slots) {
        __shared__ NvU32 ok;
        const NvU32 skipmask = g_q38_agg_skip_mask;
        if (threadIdx.x == 0) {
            ok = 1;
            const NvU32 seq = static_cast<const volatile PeerState::GpuDev *>(a.d)->cur_seq;
            for (NvU32 r = 1; r < a.rows && ok; r++)
                for (NvU32 w = 0; w < 24 && ok; w++) {
                    if (w < 20 && ((skipmask >> w) & 1u)) continue;
                    const NvU64 s1 = a.slots1[(r - 1) * 24 + w];
                    const volatile NvU32 *h = reinterpret_cast<const volatile NvU32 *>(s1);
                    if (!s1) { ok = 0; q38_gpu_fail(a.hst, 12, a.layer, seq, w, r, 0); }
                    else if (h[0] != seq || h[1] != 0 || h[2] != 0 || h[3] != 1) { ok = 0; q38_gpu_fail(a.hst, 12, a.layer, seq, w + 32 * r, h[1], h[0]); }
                }
        }
        __syncthreads();
        if (!ok) { zero(a); return; }
        for (NvU32 row = 0; row < a.rows; row++) {
            const NvU64 *slots = row ? a.slots1 + (row - 1) * 24 : return_slots;
            const float gate = *reinterpret_cast<const volatile float *>(slots[20] + offsetof(ResidentDirectResult, gate));
            const bool dm = dummy(a, row);
            for (NvU32 j = threadIdx.x; j < QWEN_H; j += blockDim.x) {
                float routed = 0.0f, shared = 0.0f;
                for (NvU32 w = 0; w < 20; w++) if (!((skipmask >> w) & 1u)) routed += reinterpret_cast<const volatile float *>(slots[w] + offsetof(ResidentDirectResult, result))[j];
                for (NvU32 w = 20; w < 24; w++) shared += reinterpret_cast<const volatile float *>(slots[w] + offsetof(ResidentDirectResult, result))[j];
                a.out[row * QWEN_H + j] = dm ? 0.0f : routed + gate * shared;
            }
        }
    }
};
#endif

// Op: row gather (worker mode 6). The RTX resolves every global row to (owner worker, byte offset)
// in skip() (one thread, <= 64 rows, binary search over the shard table), broadcasts the descriptors,
// each worker copies only its own rows (compacted, request order), and combine() picks row i from
// slot[owner] at that worker's rank. Each worker reports its owned count in `gate`; combine checks it.
// Error reasons: 7 row outside every shard, 8 more rows for one worker than its result record holds,
// 9 a worker's owned count differs from the request.
struct PciClusterRowGatherOp {
    struct Args { const long long *rows; NvU32 n; NvU8 *out; const PciClusterShard *shards; NvU32 nshards;
                  NvU64 *desc; NvU32 *rank; NvU32 *counts; PeerState::GpuHost *hst; NvU32 layer; };
    static constexpr NvU32 kMode = PCI_CLUSTER_MODE_ROW_GATHER;
    __device__ static NvU32 skip(const Args &a) {
        if (!a.n) return 1;
        for (NvU32 w = 0; w < 24; w++) a.counts[w] = 0;
        for (NvU32 i = 0; i < a.n; i++) {
            const long long row = a.rows[i];
            NvU32 lo = 0, hi = a.nshards;                       // last shard with global_start <= row
            while (lo < hi) { const NvU32 mid = (lo + hi) / 2; if (a.shards[mid].global_start <= NvU64(row)) lo = mid + 1; else hi = mid; }
            if (row < 0 || !lo || NvU64(row) - a.shards[lo-1].global_start >= a.shards[lo-1].rows) {
                q38_gpu_fail(a.hst, 7, a.layer, 0, i, NvU32(row), NvU32(NvU64(row) >> 32)); return 2;
            }
            const PciClusterShard &sh = a.shards[lo-1];
            const NvU32 w = sh.owner;
            if (w >= 24 || a.counts[w] >= PCI_CLUSTER_ROWS_PER_WORKER) { q38_gpu_fail(a.hst, 8, a.layer, 0, w, a.counts[w], 0); return 2; }
            a.desc[i] = pci_cluster_row_desc(w, sh.offset + (NvU64(row) - sh.global_start) * PCI_CLUSTER_ROW_BYTES);
            a.rank[i] = (w << 8) | a.counts[w]++;
        }
        return 0;
    }
    __device__ static void publish(const Args &a, const NvU64 *aliases, NvU32 group) {
        if (threadIdx.x == 0) *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, ple_count)) = a.n;
        for (NvU32 i = threadIdx.x; i < a.n; i += blockDim.x) {
            const NvU64 d = static_cast<const volatile NvU64 *>(a.desc)[i];
            const NvU32 off = offsetof(PeerControlLayout, ple_offsets) + i * 8;
            *rtx_pex_word(aliases, group, off) = NvU32(d);
            *rtx_pex_word(aliases, group, off + 4) = NvU32(d >> 32);
        }
    }
    __device__ static void zero(const Args &a) {
        for (NvU32 j = threadIdx.x; j < a.n * PCI_CLUSTER_ROW_WORDS; j += blockDim.x) reinterpret_cast<NvU32 *>(a.out)[j] = 0;
    }
    __device__ static void combine(const Args &a, const NvU64 *return_slots) {
        __shared__ NvU32 counts_ok;
        if (threadIdx.x == 0) {
            counts_ok = 1;
            for (NvU32 w = 0; w < 24 && counts_ok; w++) {
                const NvU32 got = __float_as_uint(*reinterpret_cast<const volatile float *>(return_slots[w] + offsetof(ResidentDirectResult, gate)));
                if (got != a.counts[w]) { counts_ok = 0; q38_gpu_fail(a.hst, 9, a.layer, 0, w, got, a.counts[w]); }
            }
        }
        __syncthreads();
        if (!counts_ok) { zero(a); return; }
        for (NvU32 j = threadIdx.x; j < a.n * PCI_CLUSTER_ROW_WORDS; j += blockDim.x) {
            const NvU32 i = j / PCI_CLUSTER_ROW_WORDS, word = j % PCI_CLUSTER_ROW_WORDS, r = a.rank[i];
            const auto *src = reinterpret_cast<const volatile NvU32 *>(return_slots[r >> 8] + offsetof(ResidentDirectResult, result));
            reinterpret_cast<NvU32 *>(a.out)[j] = src[(r & 255u) * PCI_CLUSTER_ROW_WORDS + word];
        }
    }
};

// Op: LM-head top-k (worker mode 7). Broadcast x (as the MoE hidden state), K and every worker's slice
// (offset, rows); each worker returns its exact local top-K (BF16 value, local index) and its slice
// log-sum-exp. combine: check every count (= min(K, rows)), fill the logits row with -inf, scatter the
// candidates to row_start + index, and combine the 24 log-sum-exps (written to lse_out).
// Error reason 10: a worker's candidate count differs from the request.
// nrows 2 (2026-11-04, MTP verify): two BF16 rows in one request (PCI_CLUSTER_LM_FMT_BF16X2); out / lse_out hold
// nrows rows (vocab BF16 logits, one log-sum-exp each).
static_assert(PCI_CLUSTER_LM_X1_OFF == Q38_MR_X1_OFF, "LM-head row 1 uses the multi-row x offset");
struct PciClusterLmHeadOp {
    struct Args { const void *x; int x_type; NvU16 *out; NvU32 vocab, k; const NvU64 *off, *rows; const NvU32 *row_start;
                  float *lse_out; PeerState::GpuHost *hst; NvU32 layer; NvU32 nrows = 1; };
    static constexpr NvU32 kMode = PCI_CLUSTER_MODE_LM_TOPK;
    __device__ static NvU32 skip(const Args &) { return 0; }
    __device__ static void publish(const Args &a, const NvU64 *aliases, NvU32 group) {
        PciClusterMoeOp::Args h{a.x, a.x_type, nullptr, nullptr, nullptr};
        const __half *h16 = reinterpret_cast<const __half *>(h.h);
        const __nv_bfloat16 *hb = reinterpret_cast<const __nv_bfloat16 *>(h.h);
        if (a.nrows >= 2) {                                   // N BF16 rows as raw words (exact)
            const NvU32 *hw = reinterpret_cast<const NvU32 *>(a.x);
            for (NvU32 r = 0; r < a.nrows; r++)
                for (NvU32 i = threadIdx.x; i < QWEN_H / 2; i += blockDim.x)
                    *rtx_pex_word(aliases, group, pci_cluster_lm_x_off(r) + i * sizeof(NvU32)) = hw[r * (QWEN_H / 2) + i];
        } else
        for (NvU32 i = threadIdx.x; i < QWEN_H; i += blockDim.x) {
            const float value = h.h_type == 0 ? reinterpret_cast<const float *>(h.h)[i]
                             : h.h_type == 1 ? __half2float(h16[i]) : __bfloat162float(hb[i]);
            *rtx_pex_word(aliases, group, kControlHOff + i*sizeof(NvU32)) = __float_as_uint(value);
        }
        if (threadIdx.x == 0) {
            *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, ple_count)) = a.k;
            *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, reserved)) = a.nrows >= 2 ? a.nrows : 0u;
        }
        for (NvU32 w = threadIdx.x; w < 24; w += blockDim.x) {
            const NvU32 o = offsetof(PeerControlLayout, ple_offsets) + w * 16;
            *rtx_pex_word(aliases, group, o) = NvU32(a.off[w]);       *rtx_pex_word(aliases, group, o + 4) = NvU32(a.off[w] >> 32);
            *rtx_pex_word(aliases, group, o + 8) = NvU32(a.rows[w]);  *rtx_pex_word(aliases, group, o + 12) = NvU32(a.rows[w] >> 32);
        }
    }
    __device__ static void zero(const Args &a) {          // failed call: all-zero logits (error is sticky)
        for (NvU32 j = threadIdx.x; j < a.nrows * a.vocab / 2; j += blockDim.x) reinterpret_cast<NvU32 *>(a.out)[j] = 0;
    }
    // Row r of worker w: count at result word r*(k+1), candidates after it; LSE in gate (row 0) or word LSE1 (row 1).
    __device__ static float lse_of(const NvU64 slot, NvU32 row) {
        return row ? reinterpret_cast<const volatile float *>(slot + offsetof(ResidentDirectResult, result))[PCI_CLUSTER_LM_LSE1_WORD - (row - 1)]
                   : *reinterpret_cast<const volatile float *>(slot + offsetof(ResidentDirectResult, gate));
    }
    __device__ static void combine(const Args &a, const NvU64 *return_slots) {
        __shared__ NvU32 ok;
        if (threadIdx.x == 0) {
            ok = 1;
            for (NvU32 row = 0; row < a.nrows && ok; row++) {
                float m = -INFINITY;
                for (NvU32 w = 0; w < 24 && ok; w++) {
                    const auto *r = reinterpret_cast<const volatile NvU32 *>(return_slots[w] + offsetof(ResidentDirectResult, result));
                    const NvU32 want = a.k < a.rows[w] ? a.k : NvU32(a.rows[w]);
                    if (r[row * (a.k + 1)] != want) { ok = 0; q38_gpu_fail(a.hst, 10, a.layer, row, w, r[row * (a.k + 1)], want); }
                    m = fmaxf(m, lse_of(return_slots[w], row));
                }
                if (ok) {
                    float s = 0.0f;
                    for (NvU32 w = 0; w < 24; w++) s += expf(lse_of(return_slots[w], row) - m);
                    if (a.lse_out) a.lse_out[row] = m + logf(s);
                }
            }
        }
        __syncthreads();
        if (!ok) { zero(a); return; }
        for (NvU32 j = threadIdx.x; j < a.nrows * a.vocab / 2; j += blockDim.x) reinterpret_cast<NvU32 *>(a.out)[j] = 0xff80ff80u;   // BF16 -inf
        __syncthreads();
        for (NvU32 row = 0; row < a.nrows; row++)
        for (NvU32 w = 0; w < 24; w++) {
            const auto *r = reinterpret_cast<const volatile NvU32 *>(return_slots[w] + offsetof(ResidentDirectResult, result)) + row * (a.k + 1);
            const NvU32 n = r[0], base = row * a.vocab + a.row_start[w];
            for (NvU32 i = threadIdx.x; i < n; i += blockDim.x) { const NvU32 e = r[1 + i]; a.out[base + (e & 0xffffu)] = NvU16(e >> 16); }
        }
    }
};

// Op: batched MoE (worker mode 8; docs/BATCHED_MOE_EXCHANGE_RESEARCH.md). The rows go once into the
// host batch area of the shared CPU mailbox (FP32 x, ids, routes); the multicast carries only the
// doorbell {layer, mode 8, R}. Workers compute expert-major, reduce on the card (leader = own + peers)
// and leave reducer rows in their part areas; combine = sum of the routed reducers in worker order +
// gate[row] * shared reducer (worker 20): the exact per-row (mode 1) order.
// Error reason 11: a reducer's flag does not carry this request's sequence.
struct PciClusterMoeBatchOp {
    // chunk / pipe: pipelined calls (mode 9) use sequence pipe_base + chunk and host slot sequence & 1 (the workers
    // derive the slot from the sequence; pipe_base is only known on the device). Mode 8 uses slot 0.
    // t20b (2026-11-03): the slot was chunk & 1 on the RTX; with an odd pipe_base every worker read the other slot's
    // header and failed the request with Q38_BATCH_REASON_CONFIG.
    struct Args { const void *h; int h_type; const int32_t *ids; const float *routes; NvU32 rows; float *out;
                  char *batch; const PeerState::GpuDev *d; PeerState::GpuHost *hst; NvU32 layer;
                  NvU32 chunk = 0, pipe = 0; };
    __device__ static NvU32 seq_of(const Args &a) {
        const volatile PeerState::GpuDev *v = a.d;
        return a.pipe ? v->pipe_base + a.chunk : v->cur_seq;
    }
    __device__ static NvU32 slot_of(const Args &a) { return a.pipe ? (seq_of(a) % Q38_BATCH_PIPE_SLOTS) : 0u; }
    static constexpr NvU32 kMode = Q38_BATCH_MODE;
    // Dummy rows (any id outside [0, 512), as in mode 1) are sent as ids 0..9 with routes 0 and x 0, and
    // produce zeros in combine; the whole call is skipped only when every row is a dummy.
    __device__ static bool dummy_row(const Args &a, NvU32 r) {
        for (NvU32 k = 0; k < Q38_BATCH_K; k++) { const int e = a.ids[r * Q38_BATCH_K + k]; if (e < 0 || e >= 512) return true; }
        return false;
    }
    __device__ static NvU32 skip(const Args &a) {
        for (NvU32 r = 0; r < a.rows; r++) if (!dummy_row(a, r)) return 0;
        return 1;
    }
    // rows -> host request area of the request's slot (all blocks of the grid). Pipelined: block 0 also writes the slot header
    // {fmt, rows, layer, seq}; the doorbell kernel that follows orders it before the workers look.
    __device__ static void write_request(const Args &a) {
        float *hx = reinterpret_cast<float *>(a.batch + q38_batch_req_x_off(slot_of(a)));
        int32_t *hid = reinterpret_cast<int32_t *>(a.batch + q38_batch_req_ids_off(slot_of(a)));
        float *hr = reinterpret_cast<float *>(a.batch + q38_batch_req_routes_off(slot_of(a)));
        const __half *h16 = reinterpret_cast<const __half *>(a.h);
        const NvU32 n = a.rows * QWEN_H, stride = gridDim.x * blockDim.x;
        if (a.h_type == 2) {
            // BF16 hidden states: send the raw BF16 bits (half the bytes every worker reads from host memory);
            // the workers widen exactly (Q38_BATCH_FMT_BF16)
            const NvU32 *hw = reinterpret_cast<const NvU32 *>(a.h);
            NvU32 *hx2 = reinterpret_cast<NvU32 *>(hx);
            for (NvU32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += stride)
                hx2[i] = dummy_row(a, (2 * i) / QWEN_H) ? 0u : hw[i];
        } else {
            for (NvU32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
                hx[i] = dummy_row(a, i / QWEN_H) ? 0.0f : a.h_type == 0 ? reinterpret_cast<const float *>(a.h)[i] : __half2float(h16[i]);
        }
        for (NvU32 i = blockIdx.x * blockDim.x + threadIdx.x; i < a.rows * Q38_BATCH_K; i += stride) {
            const bool dummy = dummy_row(a, i / Q38_BATCH_K);
            hid[i] = dummy ? int32_t(i % Q38_BATCH_K) : a.ids[i]; hr[i] = dummy ? 0.0f : a.routes[i];
        }
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            NvU32 *hd = reinterpret_cast<NvU32 *>(a.batch + q38_batch_req_hdr_off(slot_of(a)));
            hd[0] = a.h_type == 2 ? Q38_BATCH_FMT_BF16 : Q38_BATCH_FMT_FP32;
            if (a.pipe) { hd[1] = a.rows; hd[2] = a.layer; __threadfence_system(); hd[3] = seq_of(a); }
        }
    }
    __device__ static void publish(const Args &a, const NvU64 *aliases, NvU32 group) {
        // rows -> host request area, shared by the kPexGroups blocks; then this group's doorbell count
        write_request(a);
        if (threadIdx.x == 0) *rtx_pex_word(aliases, group, offsetof(PeerControlLayout, ple_count)) = a.rows;
    }
    // v2 step 3/4 (2026-11-03): wait_combine runs zero()/combine() on one block only to decide; the output is
    // written by q38_batch_combine (rows x column blocks). combine() checks the reducers' part flags (routed
    // workers not in g_q38_agg_skip_mask: card leaders hold the on-card sum, plus the shared reducer 20).
    __device__ static void zero(const Args &) {
        if (threadIdx.x == 0) { g_q38_batch_ok = 0; __threadfence(); }
    }
    __device__ static void combine(const Args &a, const NvU64 *) {
        if (threadIdx.x == 0) {
            NvU32 ok = 1; const NvU32 seq = seq_of(a);
#ifdef Q38_RTX_PEER_LEADER_AGG
            const NvU32 skipmask = g_q38_agg_skip_mask;
#else
            const NvU32 skipmask = 0;
#endif
            for (NvU32 w = 0; w < 21 && ok; w++) {
                if (w < 20 && ((skipmask >> w) & 1u)) continue;     // routed peers: summed by their leader
                const NvU32 got = *reinterpret_cast<const volatile NvU32 *>(a.batch + q38_batch_flag_off(w, slot_of(a)));
                if (got != seq) { ok = 0; q38_gpu_fail(a.hst, 11, a.layer, seq, w, got, 0); }
            }
            g_q38_batch_ok = ok; __threadfence();
        }
    }
};

// Batched MoE output (grid: x = column blocks of 1024 floats, y = rows; 256 threads, one float4 each).
// Same sum order as mode 1: routed = 0 + the reducers' areas in worker order (a card leader's area holds
// ((own + peer 1) + peer 2) + peer 3 from the on-card reduce; workers outside leader aggregation hold
// their own rows), out = routed + gate * shared. Host areas are read with 16-byte cache-volatile loads
// (they are rewritten every exchange). Dummy rows and failed calls give 0. The last block stamps t_done.
__device__ NvU32 g_q38_batch_done = 0;
__global__ static void q38_batch_combine(const PeerState::GpuDev *d, PciClusterMoeBatchOp::Args a) {
    __shared__ NvU32 ok;
    static_assert(QWEN_H % 4 == 0, "float4 combine");
    const NvU32 r = blockIdx.y, h4 = blockIdx.x * blockDim.x + threadIdx.x, slot = PciClusterMoeBatchOp::slot_of(a);
    if (threadIdx.x == 0) ok = *static_cast<volatile NvU32 *>(&g_q38_batch_ok) && !PciClusterMoeBatchOp::dummy_row(a, r);
    __syncthreads();
    if (h4 < QWEN_H / 4) {
        const size_t i4 = size_t(r) * (QWEN_H / 4) + h4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (ok) {
#ifdef Q38_RTX_PEER_LEADER_AGG
            const NvU32 skipmask = g_q38_agg_skip_mask;
#else
            const NvU32 skipmask = 0;
#endif
            auto part = [&](NvU32 w) { return __ldcv(reinterpret_cast<const float4 *>(a.batch + q38_batch_part_off(w, slot)) + i4); };
            float4 routed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (NvU32 w = 0; w < 20; w++) {
                if ((skipmask >> w) & 1u) continue;
                const float4 p = part(w);
                routed.x += p.x; routed.y += p.y; routed.z += p.z; routed.w += p.w;
            }
            const float gate = __ldcv(reinterpret_cast<const float *>(a.batch + q38_batch_gate_off(slot)) + r);
            const float4 sh = part(20);
            v.x = routed.x + gate * sh.x; v.y = routed.y + gate * sh.y; v.z = routed.z + gate * sh.z; v.w = routed.w + gate * sh.w;
        }
        reinterpret_cast<float4 *>(a.out)[i4] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&g_q38_batch_done, 1u) == gridDim.x * gridDim.y - 1) {
            g_q38_batch_done = 0;
            PciClusterTimingRec *rec = a.pipe ? pc_rec_pipe(d, a.chunk) : pc_rec(d); if (rec) rec->t_done = q38_globaltimer();
        }
    }
}

// Restore the per-worker row-gather words (table size) saved by q38_rtx_peer_configure_row_gather.
static void restore_row_gather_words(PeerState &s) {
    for (size_t i = 0; i < s.rg_saved.size() && i < s.owner_cpus.size(); i++) {
        if (s.owner_cpus[i] == MAP_FAILED) continue;
        auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
        for (int k = 0; k < 3; k++) c->ple_reserved[k] = s.rg_saved[i].w[k];
    }
    asm volatile("sfence" ::: "memory");
    bool ok = true;
    for (size_t i = 0; i < s.rg_saved.size() && i < s.owner_cpus.size(); i++) {
        if (s.owner_cpus[i] == MAP_FAILED) continue;
        auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
        for (int k = 0; k < 3; k++) ok &= c->ple_reserved[k] == s.rg_saved[i].w[k];
    }
    std::fprintf(stderr, "PCI_CLUSTER_ROW_GATHER_RESTORE,workers,%zu,match,%u\n", s.rg_saved.size(), ok ? 1u : 0u);
    if (!ok) throw std::runtime_error("row-gather table-size words did not restore");
    s.rg_saved.clear();
}

static void cleanup_peer_state(PeerState &s) {
    if(s.in_flight)throw std::runtime_error("RTX peer work may still be outstanding; retaining mappings and allocations");
    if(!s.m10_quiesced)throw std::runtime_error("M10 resident workers must acknowledge PARK before return PTE teardown");
    restore_pex_routes(s);
#ifdef Q38_RTX_PEER_LEADER_AGG
    for(auto it=s.agg_saved.rbegin();it!=s.agg_saved.rend();++it){
        auto *w=reinterpret_cast<volatile NvU32*>(reinterpret_cast<volatile NvU8*>(s.owner_cpus.at(it->worker))+it->offset);
        for(int k=0;k<8;k++)w[k]=it->words[k];
        asm volatile("sfence" ::: "memory");
        bool ok=true;for(int k=0;k<8;k++)ok&=w[k]==it->words[k];
        std::fprintf(stderr,"RTX_PEER_AGG_RESTORE,bdf,%s,offset,0x%llx,match,%u\n",s.workers[it->worker].bdf.c_str(),(unsigned long long)it->offset,ok?1u:0u);
        if(!ok)throw std::runtime_error("aggregation owner word restore mismatch");
    }
    s.agg_saved.clear();
    for(size_t i=s.agg_routes.size();i>0;i--)multi_route_cleanup(s.agg_routes[i-1],s.uvm_fd);
    s.agg_routes.clear();
    if(s.agg_skip_mask){NvU32 zero=0;cudaMemcpyToSymbol(g_q38_agg_skip_mask,&zero,sizeof(zero));s.agg_skip_mask=0;}
#endif
    if(!s.rg_saved.empty())restore_row_gather_words(s);
    if(!s.batch_area_saved.empty()){
        bool ok=true;
        for(size_t i=0;i<s.batch_area_saved.size()&&i<s.owner_cpus.size();i++){if(s.owner_cpus[i]==MAP_FAILED){ok=false;continue;}
            auto*c=reinterpret_cast<volatile PeerControlLayout*>(s.owner_cpus[i]);c->batch_area=s.batch_area_saved[i];ok&=c->batch_area==s.batch_area_saved[i];}
        std::fprintf(stderr,"PCI_CLUSTER_MOE_BATCH_AREA_RESTORE,workers,%zu,match,%u\n",s.batch_area_saved.size(),ok?1u:0u);
        s.batch_area_saved.clear();
    }
    for(size_t i=0;i<s.owner_cpus.size();i++)if(s.owner_cpus[i]!=MAP_FAILED){
        void *p=s.owner_cpus[i];
        auto *control=reinterpret_cast<volatile PeerControlLayout*>(p);
        if (i < s.saved_mailbox_fields.size()) {
            control->cpu_mailbox_worker=s.saved_mailbox_fields[i].worker;
            control->cpu_mailbox_reserved=s.saved_mailbox_fields[i].flags;
            control->cpu_mailbox_device_ptr=s.saved_mailbox_fields[i].ptr;
        }
    }
    asm volatile("sfence" ::: "memory");
    for(size_t i=s.routes.size();i>0;i--)multi_route_cleanup(s.routes[i-1],s.uvm_fd);
    for(size_t i=s.return_routes.size();i>0;i--)multi_route_cleanup(s.return_routes[i-1],s.uvm_fd);
    for(size_t i=0;i<s.owner_cpus.size();i++){if(s.owner_cpus[i]!=MAP_FAILED)munmap(s.owner_cpus[i],kOwnerDiagnosticMapBytes);if(i<s.owner_fds.size()&&s.owner_fds[i]>=0)close(s.owner_fds[i]);}
    if(s.device_aliases)cudaFree(s.device_aliases);
    if(s.device_status)cudaFree(s.device_status);
    if(s.device_headers)cudaFree(s.device_headers);
    if(s.batch_host){cudaHostUnregister(s.batch_host);s.batch_host=nullptr;s.batch_dev=nullptr;}
    if(s.timing_buf){PciClusterTimingRec*z=nullptr;cudaMemcpyToSymbol(g_pc_timing,&z,sizeof(z));cudaFree(s.timing_buf);s.timing_buf=nullptr;}
    if(s.gpu_dev)cudaFree(s.gpu_dev);
    if(s.gpu_host)cudaFreeHost(s.gpu_host);
    if(s.rg_shards)cudaFree(s.rg_shards);
    if(s.rg_desc)cudaFree(s.rg_desc);
    if(s.rg_rank)cudaFree(s.rg_rank);
    if(s.rg_counts)cudaFree(s.rg_counts);
    s.rg_shards=nullptr;s.rg_desc=nullptr;s.rg_rank=nullptr;s.rg_counts=nullptr;s.rg_configured=false;
    if(s.lm_off)cudaFree(s.lm_off);
    if(s.lm_rows)cudaFree(s.lm_rows);
    if(s.lm_row_start)cudaFree(s.lm_row_start);
    s.lm_off=nullptr;s.lm_rows=nullptr;s.lm_row_start=nullptr;s.lm_configured=false;
    s.gpu_dev=nullptr;s.gpu_host=s.gpu_host_dev=nullptr;
    if(s.device_return_slots)cudaFree(s.device_return_slots);
    if(s.device_return_slots1)cudaFree(s.device_return_slots1);
    s.device_return_slots1=nullptr; s.multirow=false; s.mr_max_rows=0;
    if(s.device_m10_return_aliases)cudaFree(s.device_m10_return_aliases);
    if(s.device_pex_aliases)cudaFree(s.device_pex_aliases);
    if(s.device_pex_outer_aliases)cudaFree(s.device_pex_outer_aliases);
    s.device_pex_outer_aliases=nullptr; s.pub_aliases=nullptr; s.pub_groups=kPexGroups; s.pex_outer_base=0;
    if(s.host_alias_alloc)cudaFreeHost(s.host_alias_alloc);
    if(s.pex_host_alias)cudaFreeHost(s.pex_host_alias);
    if(s.pex_dma_fd>=0)close(s.pex_dma_fd);
    if(s.uvm_fd>=0)close(s.uvm_fd);
    if(s.return_owner_active)qdestroy_mem(s.return_owner);
    if(s.rtx_context)cuDevicePrimaryCtxRelease(s.rtx_device);
    std::fprintf(stderr,"RTX_PEER_STATE_CLEANUP,PASS\n");
}

// UVM lists global VA-space entries and stamps current_pid with the ioctl
// caller's PID. For RM-backed and mapped-host allocations, range_count may be
// zero even when the process has a live page directory. Select only this
// process's valid root for the RTX, then validate the actual mapped alias PTE.
static UVM_GLM53_VA_SPACE_PDB_ENTRY find_pdb_for_mapped_va(
        int fd, const NvProcessorUuid &uuid, NvU64 va) {
    std::vector<UVM_GLM53_VA_SPACE_PDB_ENTRY> entries(128);
    UVM_GLM53_LIST_VA_SPACE_PDBS_PARAMS list{};
    list.entry_capacity=entries.size(); list.entries=(NvU64)entries.data();
    if (ioctl(fd,UVM_GLM53_LIST_VA_SPACE_PDBS,&list) || list.rmStatus!=NV_OK)
        throw std::runtime_error("UVM PDB listing failed while selecting mapped RTX VA space");
    std::fprintf(stderr,
        "RTX_ALIAS_PDB_SNAPSHOT,pid,%u,count,%u,total,%u,matched,%u,overflow,%u,va,0x%llx\n",
        static_cast<unsigned>(getpid()), static_cast<unsigned>(list.entry_count),
        static_cast<unsigned>(list.total_va_spaces),
        static_cast<unsigned>(list.matched_va_spaces), static_cast<unsigned>(list.overflow),
        static_cast<unsigned long long>(va));
    UVM_GLM53_VA_SPACE_PDB_ENTRY selected{};
    NvU32 matches=0;
    for (NvU32 i=0;i<list.entry_count;i++) {
        const auto &e=entries[i];
        if (e.current_pid!=(NvU32)getpid() || !e.did_set_page_directory ||
            !e.va_space_ptr || !e.gpu_va_space_ptr || !e.root_phys_address ||
            std::memcmp(&e.gpu_uuid,&uuid,sizeof(uuid))) continue;
        std::fprintf(stderr,
            "RTX_ALIAS_PDB_CANDIDATE,pid,%u,index,%u,va_space,0x%llx,gpu_va_space,0x%llx,mm,0x%llx,root,0x%llx,range_count,%llu,state,%u,did_set,%u\n",
            static_cast<unsigned>(getpid()), static_cast<unsigned>(i),
            static_cast<unsigned long long>(e.va_space_ptr),
            static_cast<unsigned long long>(e.gpu_va_space_ptr),
            static_cast<unsigned long long>(e.mm_ptr),
            static_cast<unsigned long long>(e.root_phys_address),
            static_cast<unsigned long long>(e.range_count), e.state, e.did_set_page_directory);
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q{};
        q.requester_uuid=uuid; q.op=UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
        q.va_space_ptr=e.va_space_ptr; q.gpu_va_space_ptr=e.gpu_va_space_ptr;
        q.root_phys_address=e.root_phys_address; q.root_phys_aperture=e.root_phys_aperture;
        q.target_va=va; q.page_size=PAGE;
        const bool query_ok = ioctl_log(fd,"RTX_ALIAS_PDB_PTE_QUERY",q);
        if (query_ok && (q.pte_after&1) && q.target_pte_phys) {
            selected=e; matches++;
        }
    }
    if (matches!=1)
        throw std::runtime_error("mapped RTX alias resolves in "+std::to_string(matches)+
                                 " candidate GPU page-table roots; expected exactly one");
    std::fprintf(stderr,"RTX_ALIAS_PDB_SELECTED,va,0x%llx,va_space,0x%llx,gpu_va_space,0x%llx,root,0x%llx\n",
        (unsigned long long)va,(unsigned long long)selected.va_space_ptr,
        (unsigned long long)selected.gpu_va_space_ptr,(unsigned long long)selected.root_phys_address);
    return selected;
}

static UVM_GLM53_VA_SPACE_PDB_ENTRY find_worker_pdb_for_mapped_va(
        int fd, const NvProcessorUuid &uuid,
        const std::vector<UVM_GLM53_VA_SPACE_PDB_ENTRY> &candidates, NvU64 va,
        const std::string &bdf) {
    UVM_GLM53_VA_SPACE_PDB_ENTRY selected{};
    unsigned matches=0;
    for(size_t i=0;i<candidates.size();i++) {
        auto q=UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS{};
        q.requester_uuid=uuid;
        q.op=UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
        q.va_space_ptr=candidates[i].va_space_ptr;
        q.gpu_va_space_ptr=candidates[i].gpu_va_space_ptr;
        q.root_phys_address=candidates[i].root_phys_address;
        q.root_phys_aperture=candidates[i].root_phys_aperture;
        q.target_va=va;q.page_size=PAGE;
        const bool ok=ioctl_log(fd,"WORKER_RETURN_ALIAS_PTE_QUERY",q);
        const bool mapped=ok&&(q.pte_after&1)&&q.target_pte_phys;
        std::fprintf(stderr,
            "RTX_WORKER_PDB_CANDIDATE,bdf,%s,alias_va,0x%llx,index,%zu,pid,%u,va_space,0x%llx,gpu_va_space,0x%llx,root,0x%llx,pte_phys,0x%llx,pte,0x%llx,mapped,%u\n",
            bdf.c_str(),(unsigned long long)va,i,candidates[i].current_pid,
            (unsigned long long)candidates[i].va_space_ptr,
            (unsigned long long)candidates[i].gpu_va_space_ptr,
            (unsigned long long)candidates[i].root_phys_address,
            (unsigned long long)q.target_pte_phys,(unsigned long long)q.pte_after,mapped?1u:0u);
        std::fflush(stderr);
        if(mapped){selected=candidates[i];matches++;}
    }
    if(matches!=1)
        throw std::runtime_error("worker return alias "+peer_hex(va)+" on "+bdf+
            " resolves in "+std::to_string(matches)+" exported PDB candidates; expected exactly one");
    std::fprintf(stderr,"RTX_WORKER_PDB_SELECTED,bdf,%s,alias_va,0x%llx,va_space,0x%llx,gpu_va_space,0x%llx,root,0x%llx\n",
        bdf.c_str(),(unsigned long long)va,(unsigned long long)selected.va_space_ptr,
        (unsigned long long)selected.gpu_va_space_ptr,(unsigned long long)selected.root_phys_address);
    std::fflush(stderr);
    return selected;
}

static void peer_open(const char *rtx_bdf,const char *maps_path,const char *uuid_path) {
    if(g_peer)throw std::runtime_error("RTX peer mailbox already open");
    auto s=std::make_unique<PeerState>(); s->workers=load_peer_records(maps_path,uuid_path);
    const char *diag_env = std::getenv("Q38_RTX_PEER_INGRESS_DIAGNOSTIC");
    s->ingress_diagnostic = diag_env && std::strcmp(diag_env, "1") == 0;
    if (!s->ingress_diagnostic)
        load_return_aliases(s->workers);
    const char *bus_route_env = std::getenv("Q38_RTX_PEER_USE_OWNER_BUS_ADDRESS");
    const bool use_owner_bus_address = bus_route_env && std::strcmp(bus_route_env, "1") == 0;
    std::fprintf(stderr, "RTX_PEER_ADDRESS_MODE,%s\n",
                 use_owner_bus_address ? "owner_pci_bus" : "requester_dma");
    for (auto &r : s->workers) {
        if (!r.owner_va || (!s->ingress_diagnostic &&
            (!r.return_alias_va || (r.return_alias_va & (PAGE - 1)))))
            throw std::runtime_error("worker mapped-host result VA is invalid or not 4-KiB aligned: " + r.bdf);
    }
    try {
    CK(cudaDeviceGetByPCIBusId(&s->rtx_ordinal,rtx_bdf));
    CUKQ(cuDeviceGet(&s->rtx_device,s->rtx_ordinal)); CUKQ(cuDevicePrimaryCtxRetain(&s->rtx_context,s->rtx_device));
    CUKQ(cuCtxSetCurrent(s->rtx_context)); CK(cudaSetDevice(s->rtx_ordinal)); CK(cudaFree(nullptr));
    CUuuid rtx_uuid{}; CUKQ(cuDeviceGetUuid(&rtx_uuid,s->rtx_device));
    static_assert(sizeof(rtx_uuid)==sizeof(s->rtx_uuid));
    std::memcpy(&s->rtx_uuid,&rtx_uuid,sizeof(rtx_uuid));
    q38_journal_open_from_env();   // route journal for daemon RECOVER_PEER_CLIENT (no-op if unset)
    s->uvm_fd=open("/dev/nvidia-uvm",O_RDWR|O_CLOEXEC); if(s->uvm_fd<0)throw std::runtime_error("open /dev/nvidia-uvm failed");
    UVM_INITIALIZE_PARAMS init{}; if(ioctl(s->uvm_fd,UVM_INITIALIZE,&init)||init.rmStatus!=NV_OK)throw std::runtime_error("UVM initialize failed for RTX peer client");
    auto register_gpu=[&](const NvProcessorUuid &uuid){
        UVM_REGISTER_GPU_PARAMS p{};
        p.gpu_uuid=uuid;
        errno=0;
        const int ioctl_rc=ioctl(s->uvm_fd,UVM_REGISTER_GPU,&p);
        const int ioctl_errno=errno;
        const auto *uuid_bytes=reinterpret_cast<const unsigned char *>(&uuid);
        std::fprintf(stderr,
            "RTX_UVM_REGISTER_GPU,uuid,%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x,ioctl_rc,%d,errno,%d,rm_status,0x%08x\n",
            uuid_bytes[0],uuid_bytes[1],uuid_bytes[2],uuid_bytes[3],
            uuid_bytes[4],uuid_bytes[5],uuid_bytes[6],uuid_bytes[7],
            uuid_bytes[8],uuid_bytes[9],uuid_bytes[10],uuid_bytes[11],
            uuid_bytes[12],uuid_bytes[13],uuid_bytes[14],uuid_bytes[15],
            ioctl_rc,ioctl_errno,static_cast<unsigned>(p.rmStatus));
        std::fflush(stderr);
        if(ioctl_rc||p.rmStatus!=NV_OK)
            throw std::runtime_error("UVM GPU registration failed in RTX peer client");
    };
    register_gpu(s->rtx_uuid);
    if (!s->ingress_diagnostic) {
        for(const auto&r:s->workers)register_gpu(r.uuid);
    }
    else {
        // The bounded PEX-ingress diagnostic only changes requester RTX PTEs
        // and reads owner BAR1 mappings from the CPU. It installs no M10
        // requester aliases, so registering each owner GPU in this new UVM
        // client is unnecessary and can be rejected when this client lacks
        // the daemon's RM client handles.
        std::fprintf(stderr,
            "RTX_UVM_REGISTER_GPU_SKIPPED,owner_gpus,%zu,reason,PEX_ingress_only_no_owner_UVM_aliases\n",
            s->workers.size());
    }
    if (!s->ingress_diagnostic) {
        auto get_export=reinterpret_cast<ExportGetterQ>(dlsym(RTLD_DEFAULT,"rm_trace_get_last_export"));
        if(!get_export)throw std::runtime_error("RM export capture preload is required for RTX-owned return BAR1 allocation");
        constexpr size_t return_map_bytes=24u*kReturnSlotBytes;
        s->return_owner_active=true; // qdestroy_mem also handles a partially created QMem.
        qcreate_mapping(s->return_owner,rtx_bdf,-1,-1,get_export,return_map_bytes);
        set_device_context(s->return_owner);
        CK(cudaMemset(reinterpret_cast<void*>(s->return_owner.va),0,return_map_bytes));
    }
    // CUDA does not guarantee page alignment for mapped host memory (2026-11-03: a no-offload run got
    // an unaligned pointer). Over-allocate one mailbox and use a kMailboxBytes-aligned view, as the
    // PEX alias does; the 4-KiB leaf PTE remaps need page-aligned device addresses.
    {
        const size_t alias_bytes=size_t(24)*kMailboxBytes, alloc_bytes=alias_bytes+kMailboxBytes;
        CK(cudaHostAlloc(&s->host_alias_alloc,alloc_bytes,cudaHostAllocMapped|cudaHostAllocPortable));
        void *dev_raw_ptr=nullptr; CK(cudaHostGetDevicePointer(&dev_raw_ptr,s->host_alias_alloc,0));
        const NvU64 dev_raw=reinterpret_cast<NvU64>(dev_raw_ptr);
        const NvU64 dev_base=(dev_raw+kMailboxBytes-1)&~NvU64(kMailboxBytes-1);
        const size_t off=size_t(dev_base-dev_raw);
        if(off+alias_bytes>alloc_bytes)throw std::runtime_error("aligned request alias exceeds mapped-host allocation");
        s->host_alias=static_cast<unsigned char*>(s->host_alias_alloc)+off;
        s->device_alias=reinterpret_cast<NvU32*>(dev_base);
        std::fprintf(stderr,"RTX_REQUEST_ALIAS,dev_raw,0x%llx,dev_aligned,0x%llx,offset,0x%zx\n",
                     (unsigned long long)dev_raw,(unsigned long long)dev_base,off);
    }
    if(reinterpret_cast<NvU64>(s->device_alias)&(PAGE-1))throw std::runtime_error("request alias is not 4-KiB aligned");
    constexpr NvU32 alias_pages=24*kMailboxLeaves;
    prime_host_alias<<<1,1>>>(reinterpret_cast<volatile NvU32*>(s->device_alias),alias_pages,0x51a70000u);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    for(NvU32 page=0;page<alias_pages;page++)
        if(reinterpret_cast<volatile NvU32*>(s->host_alias)[page*PAGE_WORDS]!=0x51a70000u+page)
            throw std::runtime_error("RTX request/return alias priming readback mismatch at page "+std::to_string(page));
    const auto request_alias_pdb=find_pdb_for_mapped_va(s->uvm_fd,s->rtx_uuid,(NvU64)s->device_alias);
    const char *pex_env = std::getenv("Q38_RTX_PEER_PEX_MULTICAST");
    const bool use_pex_multicast = pex_env && std::strcmp(pex_env, "1") == 0;
    const char *direct_env = std::getenv("Q38_RTX_PEER_DIRECT_UNICAST_DIAGNOSTIC");
    const bool use_direct_unicast = direct_env && std::strcmp(direct_env, "1") == 0;
    if (use_pex_multicast || use_direct_unicast)
        setup_pex_routes(*s, std::getenv("Q38_M10_GROUPS_TSV"), request_alias_pdb);
    s->routes.resize(24);s->return_routes.resize(24);s->sequence.resize(24);s->owner_fds.assign(24,-1);s->owner_cpus.assign(24,MAP_FAILED);
    {const char *mr=std::getenv("Q38_RTX_PEER_MOE_MULTIROW"); s->multirow=mr&&std::strcmp(mr,"1")==0;
     const char *mx=std::getenv("Q38_RTX_PEER_MOE_MULTIROW_MAX"); s->mr_max_rows=s->multirow?(mx&&*mx?NvU32(std::atoi(mx)):2u):0u;
     if(s->multirow&&(s->mr_max_rows<2||s->mr_max_rows>Q38_MR_MAX_ROWS))throw std::runtime_error("Q38_RTX_PEER_MOE_MULTIROW_MAX must be 2..4");}
    s->return_slots1.assign(size_t(24)*(Q38_MR_MAX_ROWS-1),0);
#if !defined(Q38_RTX_PEER_SMALL_LEAVES) || !defined(Q38_RTX_PEER_LEADER_AGG)
    if(s->multirow)throw std::runtime_error("Q38_RTX_PEER_MOE_MULTIROW needs the small-leaf leader-aggregation build");
#else
    static_assert(kReturnPages*kReturnLeafBytes==Q38_MR_RECORD_SHIFT,"row-1 record shift must equal one return record");
#endif
    if(s->multirow&&(!use_pex_multicast||s->ingress_diagnostic))
        throw std::runtime_error("Q38_RTX_PEER_MOE_MULTIROW needs the PEX multicast path");
    s->return_slots.resize(24);
    s->saved_mailbox_fields.resize(24);
    for(size_t i=0;i<24;i++) {
        auto&r=s->workers[i];
        QMem owner{};owner.map_bytes=kMailboxBytes;owner.rmmap.addr=r.bar_bus;
        s->routes[i].requester_quiesced=true;
        if (!use_pex_multicast && !use_direct_unicast) {
            // PEX mode publishes the request directly through the PEX ingress
            // aliases above. Do not also install the legacy 16-leaf RTX→M10
            // peer alias for every worker: it is unused in this mode and
            // needlessly consumes guarded UVM remap slots.
            multi_route_setup(s->routes[i],s->uvm_fd,s->rtx_uuid,r.uuid,owner,
                NvU64(s->device_alias)+i*kMailboxBytes,0,nullptr,0,kMailboxBytes,kMailboxLeaves,
                &request_alias_pdb,PAGE,true,use_owner_bus_address);
        } else {
            std::fprintf(stderr,"RTX_PEER_REQUEST_ROUTE_SKIPPED,bdf,%s,reason,PEX_ingress_is_request_transport\n",
                         r.bdf.c_str());
        }
        const auto bar=owner_bar1_range(r.bdf);
        if (r.bar_bus<bar.first || r.bar_bus>bar.second ||
            kOwnerDiagnosticMapBytes-1>bar.second-r.bar_bus || ((r.bar_bus-bar.first)&(PAGE-1)))
            throw std::runtime_error("retained owner BAR1 range is outside or unaligned in resource: "+r.bdf);
        const std::string resource="/sys/bus/pci/devices/"+r.bdf+"/resource1";
        s->owner_fds[i]=open(resource.c_str(),O_RDWR|O_SYNC|O_CLOEXEC);
        if(s->owner_fds[i]<0)throw std::runtime_error("open owner BAR1 resource "+resource+": "+strerror(errno));
        const off_t bar_offset=static_cast<off_t>(r.bar_bus-bar.first);
        s->owner_cpus[i]=mmap(nullptr,kOwnerDiagnosticMapBytes,PROT_READ|PROT_WRITE,MAP_SHARED,s->owner_fds[i],bar_offset);
        if(s->owner_cpus[i]==MAP_FAILED)throw std::runtime_error("map owner BAR1 resource "+r.bdf+": "+strerror(errno));
        s->sequence[i]=*reinterpret_cast<volatile NvU32*>(s->owner_cpus[i]);
        if(i&&s->sequence[i]!=s->sequence[0])throw std::runtime_error("resident worker sequences are not synchronized");
        std::fprintf(stderr,"RTX_PEER_OWNER,bdf,%s,owner_va,0x%llx,bar1_bus,0x%llx,cpu_bar1_start,0x%llx,bar1_offset,0x%llx,cpu_mapping,%p,sequence,%u\n",
            r.bdf.c_str(),(unsigned long long)r.owner_va,(unsigned long long)r.bar_bus,
            (unsigned long long)bar.first,(unsigned long long)(r.bar_bus-bar.first),s->owner_cpus[i],s->sequence[i]);

        auto *control = reinterpret_cast<volatile PeerControlLayout *>(s->owner_cpus[i]);
        s->saved_mailbox_fields[i] = {control->cpu_mailbox_worker,
                                      control->cpu_mailbox_reserved,
                                      control->cpu_mailbox_device_ptr};

        // Reuse the daemon's existing cudaHostRegister mailbox. Align its
        // per-worker pointer down to the M10's 128-KiB big-page leaf; the
        // pointer's in-leaf offset is applied to the matching RTX BAR1 slot.
        const NvU64 slot_offset=NvU64(i)*kReturnSlotBytes;
        s->return_routes[i].requester_quiesced=true;
        if (!s->ingress_diagnostic) {
        if(slot_offset+kReturnSlotBytes>s->return_owner.map_bytes)
            throw std::runtime_error("RTX return slot exceeds retained BAR1 window");
        // The worker process exports its own VA-space root in gpu_pdb.manifest.
        // Do not rediscover it by scanning this RTX process's global UVM list:
        // the numeric CUDA VA can also be mapped in unrelated VA spaces, and
        // a present leaf there says nothing about the resident worker's PDB.
        // load_peer_records() already validated the worker identity and VA.
        const auto worker_return_pdb=find_worker_pdb_for_mapped_va(
            s->uvm_fd,r.uuid,r.requester_pdb_candidates,r.return_alias_va,r.bdf);
        const NvU64 return_leaf_offset = r.return_alias_va & (kReturnLeafBytes - 1);
        const NvU64 return_leaf_va = r.return_alias_va - return_leaf_offset;
        // kReturnSpanBytes == kReturnLeafBytes in the default 128-KiB build.
        if (return_leaf_offset + sizeof(Q38CpuMailboxSlot) > kReturnSpanBytes)
            throw std::runtime_error("worker result mailbox crosses its return leaf span: " + r.bdf);
        const NvU64 slot_bus = s->return_owner.rmmap.addr + slot_offset;
        const NvU64 owner_leaf_delta = (kReturnLeafBytes -
            (slot_bus & (kReturnLeafBytes - 1))) & (kReturnLeafBytes - 1);
        const NvU64 owner_window_offset = slot_offset + owner_leaf_delta;
        if (owner_window_offset + kReturnWindowBytes > slot_offset + kReturnSlotBytes ||
            owner_window_offset + kReturnWindowBytes > s->return_owner.map_bytes)
            throw std::runtime_error("aligned RTX return window exceeds its reserved 256-KiB slot");
        // Mode 10 (Q38_RTX_PEER_MOE_MULTIROW, N = Q38_RTX_PEER_MOE_MULTIROW_MAX rows): the 9 reducers (card leaders
        // 0,4,..,16 and shared workers 20..23) get N return records in the same route (3N pages): record r at row 0 +
        // r * 3 pages; backwards (row 0 - r * 3 pages) where the next VA pages are the mode-8 request area (worker 23,
        // and 22 for N >= 3). Card peers keep 3 pages (header only).
        const NvU32 mr_n = s->mr_max_rows;
        const bool mr_reducer = s->multirow && ((i < 20 && i % 4 == 0) || i >= 20);
        const bool mr_back = mr_reducer && (i == 23 || (i == 22 && mr_n >= 3));
        const NvU32 return_pages = mr_reducer ? mr_n * kReturnPages : kReturnPages;
        const NvU64 return_span = NvU64(return_pages) * kReturnLeafBytes;
        const NvU64 back_shift = mr_back ? NvU64(mr_n - 1) * Q38_MR_RECORD_SHIFT : 0;
        const NvU64 route_va = return_leaf_va - back_shift;
        if (return_span > kReturnWindowBytes)
            throw std::runtime_error("multi-row return records exceed the retained RTX return window");
        multi_route_setup(s->return_routes[i],s->uvm_fd,r.uuid,s->rtx_uuid,
            s->return_owner,route_va,0,nullptr,owner_window_offset,kReturnWindowBytes,return_pages,
            &worker_return_pdb,kReturnLeafBytes,true,use_owner_bus_address);
        if(s->return_routes[i].owner_offset < slot_offset ||
           s->return_routes[i].owner_offset-slot_offset > kReturnSlotBytes-return_span ||
           s->return_routes[i].owner_offset+return_span > s->return_owner.map_bytes)
            throw std::runtime_error("aligned M10→RTX result slot exceeds RTX BAR1 allocation");
        s->return_slots[i]=s->return_owner.va+s->return_routes[i].owner_offset+return_leaf_offset+back_shift;
        if (mr_reducer)
            for (NvU32 rr = 1; rr < mr_n; rr++)
                s->return_slots1[size_t(rr - 1) * 24 + i] = mr_back ? s->return_slots[i] - NvU64(rr) * Q38_MR_RECORD_SHIFT
                                                                    : s->return_slots[i] + NvU64(rr) * Q38_MR_RECORD_SHIFT;
        std::fprintf(stderr,"RTX_PEER_PUSH,bdf,%s,requester_alias_va,0x%llx,requester_leaf_va,0x%llx,leaf_offset,0x%llx,owner_bar_bus_leaf,0x%llx,owner_offset,0x%llx,rtx_local_va,0x%llx,pages,%u,leaf_bytes,0x%x,writable,1\n",
            r.bdf.c_str(),(unsigned long long)r.return_alias_va,
            (unsigned long long)return_leaf_va,(unsigned long long)return_leaf_offset,
            (unsigned long long)(s->return_owner.rmmap.addr+s->return_routes[i].owner_offset),
            (unsigned long long)s->return_routes[i].owner_offset,
            (unsigned long long)s->return_slots[i],(unsigned)return_pages,(unsigned)kReturnLeafBytes);

        // The worker's mailbox pointer is its original mapped-host CUDA VA,
        // and the guarded return PTE above now makes that VA target this
        // worker's slot in the RTX BAR1 allocation. In PEX mode the RTX
        // device_aliases are intentionally not mapped to the workers, so a
        // CUDA store through those aliases only updates RTX-local host memory.
        // Install this small, static routing metadata through the retained
        // owner BAR mapping instead of pretending that alias is remote.
        control->cpu_mailbox_worker = static_cast<NvU32>(i);
        control->cpu_mailbox_reserved = Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN;
        control->cpu_mailbox_device_ptr = r.return_alias_va;
        asm volatile("sfence" ::: "memory");
        if (control->cpu_mailbox_worker != i ||
            control->cpu_mailbox_reserved != Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN ||
            control->cpu_mailbox_device_ptr != r.return_alias_va)
            throw std::runtime_error("owner BAR readback disagrees with installed RTX return-mailbox metadata: " + r.bdf);
        std::fprintf(stderr,"RTX_PEER_RETURN_METADATA,bdf,%s,worker,%zu,flag,0x%x,worker_mailbox_va,0x%llx,verified,1,transport,owner_cpu_BAR1_setup\n",
            r.bdf.c_str(),i,Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN,
            (unsigned long long)r.return_alias_va);
        } else {
            // Diagnostic mode deliberately omits the 24 GPU return PTEs. Keep
            // worker output in its direct BAR1 result record and disable any
            // stale RTX-return pointer left by an earlier interrupted client.
            control->cpu_mailbox_worker = static_cast<NvU32>(i);
            control->cpu_mailbox_reserved = 0;
            control->cpu_mailbox_device_ptr = 0;
            asm volatile("sfence" ::: "memory");
            if (control->cpu_mailbox_reserved != 0 || control->cpu_mailbox_device_ptr != 0)
                throw std::runtime_error("failed to disable stale return-mailbox metadata for diagnostic");
        }
    }
    CUKQ(cuCtxSetCurrent(s->rtx_context));CK(cudaSetDevice(s->rtx_ordinal));
    std::vector<NvU64> aliases(24);for(size_t i=0;i<24;i++)aliases[i]=NvU64(s->device_alias)+i*kMailboxBytes;
    CK(cudaMalloc(&s->device_aliases,aliases.size()*sizeof(NvU64)));
    CK(cudaMemcpy(s->device_aliases,aliases.data(),aliases.size()*sizeof(NvU64),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&s->device_status,kAsyncStatusSlots*sizeof(NvU32)));
    CK(cudaMalloc(&s->device_headers,(size_t(24)*4+2)*sizeof(NvU32)));
    CK(cudaMalloc(&s->gpu_dev,sizeof(PeerState::GpuDev)));CK(cudaMemset(s->gpu_dev,0,sizeof(PeerState::GpuDev)));
    CK(cudaHostAlloc(reinterpret_cast<void**>(&s->gpu_host),sizeof(PeerState::GpuHost),cudaHostAllocMapped|cudaHostAllocPortable));
    std::memset(static_cast<void*>(s->gpu_host),0,sizeof(PeerState::GpuHost));
    CK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&s->gpu_host_dev),s->gpu_host,0));
    CK(cudaMalloc(&s->device_return_slots,24*sizeof(NvU64)));
    CK(cudaMemcpy(s->device_return_slots,s->return_slots.data(),24*sizeof(NvU64),cudaMemcpyHostToDevice));
    if (!s->ingress_diagnostic) {
        std::vector<NvU64> worker_return_aliases(24);
        for(size_t i=0;i<24;i++)worker_return_aliases[i]=s->workers[i].return_alias_va;
        CK(cudaMalloc(&s->device_m10_return_aliases,worker_return_aliases.size()*sizeof(NvU64)));
        CK(cudaMemcpy(s->device_m10_return_aliases,worker_return_aliases.data(),
                      worker_return_aliases.size()*sizeof(NvU64),cudaMemcpyHostToDevice));
    }
    if (!s->pex_multicast) {
        rtx_peer_set_return_metadata<<<1,32>>>(s->device_aliases,s->device_m10_return_aliases,24);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::fprintf(stderr,"RTX_PEER_RETURN_METADATA_INITIALIZED,workers,24,once_per_open,1,transport,RTX_peer_alias\n");
    }
#ifdef Q38_RTX_PEER_LEADER_AGG
    if (!s->ingress_diagnostic) {
        // Q38_RTX_PEER_AGG_CARDS: comma list of routed card groups (0..4). Card c has
        // leader worker 4c and peers 4c+1..4c+3. Each peer's result goes through 3 x
        // 4 KiB GM107 leaves to slot (peer-1) of a 64 KiB window at leader owner
        // +0x28000 (slot stride 0x4000, flag at +10240). The peer VA is the
        // neighbouring CPU-mailbox slot (return alias + 16 KiB), which is mapped in
        // every routed device but unused by that device.
        constexpr NvU64 kAggBlockOff=0xB000, kAggSlotOff=0x28000, kAggSlotStride=0x4000, kAggWindow=0x10000;
        constexpr NvU64 kAggLeaderMap=Q38_WORKER_BAR_MAP_BYTES;
        constexpr NvU32 kAggMagic=0x51384147u;
        static_assert(kAggSlotOff+kAggWindow<=kOwnerDiagnosticMapBytes,"aggregation window outside owner CPU map");
        const char *cards_env=std::getenv("Q38_RTX_PEER_AGG_CARDS");
        std::vector<int> cards;
        if(cards_env&&*cards_env){std::string v(cards_env);size_t a=0;while(a<=v.size()){size_t b=v.find(',',a);if(b==std::string::npos)b=v.size();if(b>a)cards.push_back(std::stoi(v.substr(a,b-a)));a=b+1;}}
        auto save=[&](size_t w,NvU64 off){PeerState::AggSave sv{};sv.worker=w;sv.offset=off;
            auto *p=reinterpret_cast<volatile NvU32*>(reinterpret_cast<volatile NvU8*>(s->owner_cpus.at(w))+off);
            for(int k=0;k<8;k++)sv.words[k]=p[k];s->agg_saved.push_back(sv);return p;};
        auto put=[&](volatile NvU32*p,const NvU32*v,const char*what,size_t w){for(int k=0;k<8;k++)p[k]=v[k];asm volatile("sfence" ::: "memory");
            for(int k=0;k<8;k++)if(p[k]!=v[k])throw std::runtime_error(std::string("aggregation ")+what+" readback mismatch on "+s->workers[w].bdf);};
        NvU32 mask=0;
        for(int c:cards){
            if(c<0||c>4)throw std::runtime_error("leader aggregation is defined for routed cards 0..4 only");
            const size_t L=size_t(c)*4; const auto &leader=s->workers[L];
            QMem lw{}; lw.map_bytes=kAggLeaderMap; lw.rmmap.addr=leader.bar_bus;
            for(int pi=1;pi<=3;pi++){
                const size_t P=L+pi; const auto &peer=s->workers[P];
                const NvU64 push_va=peer.return_alias_va+Q38_CPU_MAILBOX_SLOT_STRIDE;
                const auto pdb=find_worker_pdb_for_mapped_va(s->uvm_fd,peer.uuid,peer.requester_pdb_candidates,push_va,peer.bdf);
                s->agg_routes.emplace_back(); auto &rt=s->agg_routes.back(); rt.requester_quiesced=true;
                multi_route_setup(rt,s->uvm_fd,peer.uuid,leader.uuid,lw,push_va,NvU64(pi-1)*kAggSlotStride,nullptr,
                    kAggSlotOff,kAggWindow,3,&pdb,PAGE,true,use_owner_bus_address);
                if(rt.owner_offset!=kAggSlotOff+NvU64(pi-1)*kAggSlotStride)
                    throw std::runtime_error("aggregation slot landed at an unexpected leader offset for "+peer.bdf);
                const NvU32 blk[8]={kAggMagic,2u,NvU32(pi),3u,NvU32(push_va),NvU32(push_va>>32),NvU32(kAggSlotOff),NvU32(kAggSlotStride)};
                put(save(P,kAggBlockOff),blk,"peer block",P);
                mask|=1u<<P;
                std::fprintf(stderr,"RTX_PEER_AGG_ROUTE,card,%d,peer,%s,leader,%s,push_va,0x%llx,leader_bus,0x%llx,pages,3,leaf_bytes,4096\n",
                    c,peer.bdf.c_str(),leader.bdf.c_str(),(unsigned long long)push_va,
                    (unsigned long long)(leader.bar_bus+rt.owner_offset));
            }
            for(int k=0;k<3;k++){const NvU32 z[8]={0,0,0,0,0,0,0,0};put(save(L,kAggSlotOff+NvU64(k)*kAggSlotStride+Q38_CPU_MAILBOX_HIDDEN*4),z,"leader flag",L);}
            const NvU32 blk[8]={kAggMagic,1u,0u,3u,0u,0u,NvU32(kAggSlotOff),NvU32(kAggSlotStride)};
            put(save(L,kAggBlockOff),blk,"leader block",L);
        }
        if(mask){CK(cudaMemcpyToSymbol(g_q38_agg_skip_mask,&mask,sizeof(mask)));s->agg_skip_mask=mask;}
        std::fprintf(stderr,"RTX_PEER_AGG_READY,cards,%zu,peer_routes,%zu,skip_mask,0x%x\n",cards.size(),s->agg_routes.size(),mask);
        if(s->multirow){
            // Mode 10 (resident_multirow_abi.cuh), N = mr_max_rows: every routed card aggregated; each peer pushes rows
            // 1..N-1 packed (10,244 B each) through ONE route of ceil((N-1)*10244/4 KiB) leaves (its VA: slot P+2) into
            // the leader's block Q38_MR_PEER_BASE(pi-1); every worker gets its multi-row block (role | N << 8; saved /
            // restored with the aggregation words).
            std::vector<int> sorted_cards=cards; std::sort(sorted_cards.begin(),sorted_cards.end());
            if(sorted_cards!=std::vector<int>{0,1,2,3,4})
                throw std::runtime_error("Q38_RTX_PEER_MOE_MULTIROW needs Q38_RTX_PEER_AGG_CARDS=0,1,2,3,4");
            const NvU32 N=s->mr_max_rows;
            const NvU32 push_pages=NvU32(((N-1)*Q38_MR_PUSH_ROW_BYTES+PAGE-1)/PAGE);
            static_assert(kAggSlotOff+2*kAggSlotStride+Q38_CPU_MAILBOX_HIDDEN*4+8<=Q38_MR_PEER_BASE(2),"peer block 2 overlaps the row-0 slots");
            static_assert(Q38_MR_PEER_BASE(2)+Q38_MR_PEER_BLOCK_BYTES<=PCI_CLUSTER_WTIMING_OFF,"peer block 2 overlaps the worker timing ring");
            static_assert(Q38_MR_PEER_BASE(1)+Q38_MR_PEER_BLOCK_BYTES<=Q38_DIRECT_RESULT_OFF,"peer block 1 overlaps the direct record");
            static_assert(Q38_MR_BLOCK_OFF>=kAggBlockOff+32&&Q38_MR_BLOCK_OFF+sizeof(Q38MrBlock)<=kAggBlockOff+PAGE,"multi-row block overlaps the aggregation block");
            const NvU32 step=NvU32(Q38_MR_RECORD_SHIFT), back_step=NvU32(0u-Q38_MR_RECORD_SHIFT);
            // Q38_RTX_PEER_MR_STREAM (default on; 0 = off, 2026-11-04): streamed finish with per-block flags (resident_multirow_abi.cuh)
            const char *st_env=std::getenv("Q38_RTX_PEER_MR_STREAM");
            // default on since the PoC (T256/T257: 141.3 -> 133.1 us per layer for 1 row, 400.4 -> 379.9 for 4, bit-identical)
            const NvU32 stream_bit=(st_env&&std::strcmp(st_env,"0")==0)?0u:Q38_MR_FLAG_STREAM;
            // Q38_RTX_PEER_MR_DIAG_NORTX=1 (diagnostic only, results wrong): reducers skip their result writes to the RTX
            const char *dg_env=std::getenv("Q38_RTX_PEER_MR_DIAG_NORTX");
            const NvU32 diag_bit=(dg_env&&std::strcmp(dg_env,"1")==0)?Q38_MR_FLAG_DIAG_NORTX:0u;
            if(diag_bit)std::fprintf(stderr,"RTX_PEER_MR_DIAG_NORTX,1,results_invalid\n");
            // Q38_RTX_PEER_MR_PROFILE=1: card leaders record their finish phases (resident_multirow_abi.cuh)
            const char *pf_env=std::getenv("Q38_RTX_PEER_MR_PROFILE");
            const NvU32 prof_bit=(pf_env&&std::strcmp(pf_env,"1")==0)?Q38_MR_FLAG_PROFILE:0u;
            if(prof_bit)std::fprintf(stderr,"RTX_PEER_MR_PROFILE,1\n");
            // Q38_RTX_PEER_MR_LL=1 (PoC, kernels built with -DQ38_MR_LL_BUILD): LL peer push for 1-row exchanges; the LL row
            // lives in the rows->=1 peer blocks, so their push routes must map at least Q38_MR_LL_BYTES
            const char *ll_env=std::getenv("Q38_RTX_PEER_MR_LL");
            const NvU32 ll_bit=(ll_env&&std::strcmp(ll_env,"1")==0)?Q38_MR_FLAG_LL:0u;
            if(ll_bit&&(!stream_bit||NvU64(push_pages)*PAGE<Q38_MR_LL_BYTES))
                throw std::runtime_error("Q38_RTX_PEER_MR_LL needs the streamed finish and push routes of >= 20480 B (Q38_RTX_PEER_MOE_MULTIROW_MAX=4)");
            if(ll_bit)std::fprintf(stderr,"RTX_PEER_MR_LL,1,push_bytes,%llu\n",(unsigned long long)push_pages*PAGE);
            static_assert(Q38_MR_BLOCK_FLAG_OFF>=Q38_CPU_MAILBOX_HIDDEN*4+8&&Q38_MR_BLOCK_FLAG_OFF+4*Q38_MR_MAX_BLOCKS<=3*PAGE,"block flags outside the row-0 window");
            auto back=[&](size_t w){return w==23||(w==22&&N>=3);};
            auto ret1_va=[&](size_t w){return back(w)?s->workers[w].return_alias_va-Q38_MR_RECORD_SHIFT:s->workers[w].return_alias_va+Q38_MR_RECORD_SHIFT;};
            auto put_mr=[&](size_t w,const Q38MrBlock &b,const char*what){NvU32 v[8];std::memcpy(v,&b,sizeof(v));put(save(w,Q38_MR_BLOCK_OFF),v,what,w);};
            size_t mr_routes=0;
            for(int c=0;c<5;c++){
                const size_t L=size_t(c)*4; const auto &leader=s->workers[L];
                QMem lw{}; lw.map_bytes=kAggLeaderMap; lw.rmmap.addr=leader.bar_bus;
                for(int pi=1;pi<=3;pi++){
                    const size_t P=L+pi; const auto &peer=s->workers[P];
                    const NvU64 push1_va=peer.return_alias_va+2*Q38_CPU_MAILBOX_SLOT_STRIDE;
                    const NvU64 base=Q38_MR_PEER_BASE(pi-1), win=base>=0x30000?0x30000:0x0, winlen=base>=0x30000?0x10000:0x20000;
                    const auto pdb=find_worker_pdb_for_mapped_va(s->uvm_fd,peer.uuid,peer.requester_pdb_candidates,push1_va,peer.bdf);
                    s->agg_routes.emplace_back(); auto &rt=s->agg_routes.back(); rt.requester_quiesced=true;
                    multi_route_setup(rt,s->uvm_fd,peer.uuid,leader.uuid,lw,push1_va,base-win,nullptr,
                        win,winlen,push_pages,&pdb,PAGE,true,use_owner_bus_address);
                    if(rt.owner_offset!=base)
                        throw std::runtime_error("multi-row peer block landed at an unexpected leader offset for "+peer.bdf);
                    put_mr(P,Q38MrBlock{Q38_MR_MAGIC,Q38_MR_ROLE_PEER|(N<<8),stream_bit|ll_bit,0u,0ull,push1_va},"multi-row peer block");
                    for(NvU64 o=0;o<4*Q38_MR_MAX_BLOCKS;o+=32){const NvU32 z[8]={0,0,0,0,0,0,0,0};
                        put(save(L,kAggSlotOff+NvU64(pi-1)*kAggSlotStride+Q38_MR_BLOCK_FLAG_OFF+o),z,"multi-row block flags",L);}
                    mr_routes++;
                    std::fprintf(stderr,"RTX_PEER_MR_ROUTE,card,%d,peer,%s,leader,%s,push1_va,0x%llx,leader_bus,0x%llx,pages,%u,leaf_bytes,4096\n",
                        c,peer.bdf.c_str(),leader.bdf.c_str(),(unsigned long long)push1_va,(unsigned long long)(leader.bar_bus+rt.owner_offset),push_pages);
                    for(NvU32 r=1;r<N;r++){const NvU32 z[8]={0,0,0,0,0,0,0,0};
                        put(save(L,base+NvU64(r-1)*Q38_MR_PUSH_ROW_BYTES+Q38_CPU_MAILBOX_HIDDEN*4),z,"multi-row leader flag",L);}
                }
                put_mr(L,Q38MrBlock{Q38_MR_MAGIC,Q38_MR_ROLE_LEADER|(N<<8),1u|stream_bit|diag_bit|prof_bit|ll_bit,back(L)?back_step:step,ret1_va(L),0ull},"multi-row leader block");
            }
            for(size_t w=20;w<24;w++)put_mr(w,Q38MrBlock{Q38_MR_MAGIC,Q38_MR_ROLE_SHARED|(N<<8),1u|diag_bit,back(w)?back_step:step,ret1_va(w),0ull},"multi-row shared block");
            for(size_t w=0;w<24;w++){const bool red=(w<20&&w%4==0)||w>=20;
                for(NvU32 r=1;r<N;r++)if(red!=(s->return_slots1[size_t(r-1)*24+w]!=0))throw std::runtime_error("multi-row return records do not match the reducer set");}
            CK(cudaMalloc(&s->device_return_slots1,s->return_slots1.size()*sizeof(NvU64)));
            CK(cudaMemcpy(s->device_return_slots1,s->return_slots1.data(),s->return_slots1.size()*sizeof(NvU64),cudaMemcpyHostToDevice));
            std::fprintf(stderr,"RTX_PEER_MR_READY,max_rows,%u,row_peer_routes,%zu,push_pages,%u,reducer_records,%u,routes_total,%zu,stream,%u\n",
                         N,mr_routes,push_pages,9*N,s->agg_routes.size()+24,stream_bit?1u:0u);
        }
    }
#endif
    g_peer=std::move(s);
    } catch (...) {
        g_last_open_rollback_complete = false;
        if(s) {
            try {
                cleanup_peer_state(*s);
                g_last_open_rollback_complete = true;
                std::fprintf(stderr,"RTX_PEER_OPEN_ROLLBACK,PASS\n");
            } catch (...) {
                std::fprintf(stderr,"RTX_PEER_OPEN_ROLLBACK_INCOMPLETE; retaining process state\n");
            }
        }
        throw;
    }
}

static void enqueue_peer_request(PeerState &s, cudaStream_t stream,
        const void *h, int h_type, const int32_t *ids, const float *routes,
        NvU32 layer, NvU32 sequence) {
    if (s.pex_multicast) {
        rtx_peer_pex_publish_payload<<<kPexGroups,256,0,stream>>>(h,h_type,ids,routes,
            s.device_pex_aliases,layer);
        CK(cudaGetLastError());
        const char *readback = std::getenv("Q38_RTX_PEER_DEBUG_PEX_READBACK");
        if (readback && std::strcmp(readback, "1") == 0) {
            // Diagnostic boundary: inspect the target BAR1 mailboxes after
            // payload publication but before sequence publication can wake a
            // resident worker. This deliberately adds a stream sync and CPU
            // MMIO reads, and is never part of the normal/timed path.
            CK(cudaStreamSynchronize(stream));
            for (size_t i=0; i<s.workers.size(); ++i) {
                auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
                std::fprintf(stderr,
                    "RTX_PEX_PAYLOAD_READBACK,bdf,%s,seq,%u,layer,%u,mode,%u,id0,%d,route0,%.9g,h0,%.9g,mailbox_flag,%u,mailbox_va,0x%llx\n",
                    s.workers[i].bdf.c_str(), c->seq, c->layer, c->mode,
                    c->ids[0], double(c->routes[0]), double(c->x[0]),
                    c->cpu_mailbox_reserved,
                    (unsigned long long)c->cpu_mailbox_device_ptr);
            }
            std::fflush(stderr);
        }
        // A separate same-stream kernel publishes command sequence only after
        // every group CTA's payload stores and system fences have completed.
        rtx_peer_pex_publish_sequence<<<kPexGroups,1,0,stream>>>(s.device_pex_aliases,sequence);
        CK(cudaGetLastError());
        if (readback && std::strcmp(readback, "1") == 0) {
            // Diagnostic-only observation of the command word after its final
            // publisher has completed. Workers may now be consuming it.
            CK(cudaStreamSynchronize(stream));
            for (size_t i=0; i<s.workers.size(); ++i) {
                auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
                std::fprintf(stderr,
                    "RTX_PEX_COMMAND_READBACK,bdf,%s,seq,%u,layer,%u,mode,%u,id0,%d,mailbox_flag,%u\n",
                    s.workers[i].bdf.c_str(), c->seq, c->layer, c->mode,
                    c->ids[0], c->cpu_mailbox_reserved);
            }
            std::fflush(stderr);
        }
    } else {
        rtx_peer_publish<<<1,256,0,stream>>>(h,h_type,ids,routes,s.device_aliases,
            24,0,layer,sequence);
        CK(cudaGetLastError());
    }
}

// Diagnostic-only bridge for isolating RTX PEX ingress and resident worker
// execution without consuming the 24 guarded M10->RTX return PTE records.
// This intentionally reads worker records via CPU BAR1 and performs a CPU
// reduction before copying the vector to `out`; it is not the production G2G
// result path and must never be used for transport timing.
static void ingress_diagnostic_join(PeerState &s, NvU32 sequence, float *out,
                                    cudaStream_t stream) {
    CK(cudaStreamSynchronize(stream));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    std::vector<volatile ResidentDirectResult *> direct(s.workers.size());
    for (size_t i = 0; i < s.workers.size(); ++i) {
        direct[i] = reinterpret_cast<volatile ResidentDirectResult *>(
            reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]) + kDirectResultOff);
        while (direct[i]->seq != sequence && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (direct[i]->seq != sequence) {
            std::fprintf(stderr,
                "RTX_INGRESS_DIAG_TIMEOUT,index,%zu,bdf,%s,expected_seq,%u,observed_seq,%u,status,%u,exit_reason,%u\n",
                i, s.workers[i].bdf.c_str(), sequence, direct[i]->seq,
                direct[i]->status, direct[i]->exit_reason);
            std::fflush(stderr);
            throw std::runtime_error("resident worker did not publish the requested PEX sequence");
        }
        if (direct[i]->ready != 1 || direct[i]->status != 0 || direct[i]->exit_reason != 0)
            throw std::runtime_error("resident worker published a non-success direct result header");
    }

    std::vector<float> joined(QWEN_H, 0.0f);
    const float gate = direct[20]->gate;
    for (NvU32 j = 0; j < QWEN_H; ++j) {
        float routed = 0.0f, shared = 0.0f;
        for (size_t i = 0; i < 20; ++i) routed += direct[i]->result[j];
        for (size_t i = 20; i < 24; ++i) shared += direct[i]->result[j];
        joined[j] = routed + gate * shared;
    }
    for (size_t i = 0; i < direct.size(); ++i)
        std::fprintf(stderr,
            "RTX_INGRESS_DIAG_WORKER,index,%zu,bdf,%s,seq,%u,status,%u,exit_reason,%u,ready,%u,gate,%.9g,result0,%.9g,result1,%.9g\n",
            i, s.workers[i].bdf.c_str(), direct[i]->seq, direct[i]->status,
            direct[i]->exit_reason, direct[i]->ready, double(direct[i]->gate),
            double(direct[i]->result[0]), double(direct[i]->result[1]));
    CK(cudaMemcpyAsync(out, joined.data(), joined.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream));
    CK(cudaStreamSynchronize(stream));
    s.in_flight = false;
    s.pex_quiesced = true;
    std::fprintf(stderr,
        "RTX_INGRESS_DIAGNOSTIC_CPU_JOIN,PASS,sequence,%u,workers,%zu,hidden,%u,cpu_payload_relay,YES,return_PTEs,0\n",
        sequence, s.workers.size(), QWEN_H);
    std::fflush(stderr);
}

static void debug_stage_sync(cudaStream_t stream, const char *stage) {
    const char *enabled = std::getenv("Q38_RTX_PEER_DEBUG_STAGE_SYNC");
    if (!enabled || std::strcmp(enabled, "1") != 0)
        return;

    const cudaError_t status = cudaStreamSynchronize(stream);
    std::fprintf(stderr, "RTX_PEER_DEBUG_STAGE,%s,cuda_status,%d,name,%s\n",
                 stage, static_cast<int>(status), cudaGetErrorName(status));
    std::fflush(stderr);
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("debug stage ") + stage +
                                 " failed: " + cudaGetErrorString(status));
}

static void peer_close() {
    if(!g_peer)return;
    if(g_peer->gpu_used && g_peer->in_flight) {
        // q38_rtx_peer_forward_gpu enqueues without waiting: prove every queued exchange has finished
        // (reducers end by completion, timeout, deadline or abort) before routes are restored.
        cudaError_t sync=cudaDeviceSynchronize();
        if(sync!=cudaSuccess) throw std::runtime_error(std::string("GPU-driven exchanges not quiescent: ")+cudaGetErrorString(sync)+"; retaining mappings");
        for(auto &route:g_peer->routes)route.requester_quiesced=true;
        for(auto &route:g_peer->return_routes)route.requester_quiesced=true;
        g_peer->in_flight=false; g_peer->pex_quiesced=true;
    }
    cleanup_peer_state(*g_peer);g_peer.reset();
}
}

extern "C" int q38_rtx_peer_open(const char *rtx_bdf,const char *maps,const char *uuids,
        char *err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    g_last_open_rollback_complete = false;
    try{peer_open(rtx_bdf,maps,uuids);g_peer_error.clear();return 0;}
    catch(const std::exception&e){g_peer_error=e.what();if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());try{peer_close();}catch(...){}return 1;}
}
extern "C" int q38_rtx_peer_open_rollback_complete(void) {
    return g_last_open_rollback_complete ? 1 : 0;
}
extern "C" int q38_rtx_peer_forward(uint32_t layer,const void*h,int h_type,const int32_t*ids,
        const float*routes,float*out,void*stream,char*err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    const auto profile_path = std::getenv("Q38_RTX_PEER_PHASE_CSV");
    const char *diag_env = std::getenv("Q38_RTX_PEER_INGRESS_DIAGNOSTIC");
    const bool ingress_diagnostic = diag_env && std::strcmp(diag_env, "1") == 0;
    const bool profile = profile_path && *profile_path && !ingress_diagnostic;
    using PeerClock = std::chrono::steady_clock;
    const auto t_entry = PeerClock::now();
    double context_us=0, sequence_us=0, publish_enqueue_us=0, clear_enqueue_us=0;
    double reduce_enqueue_us=0, status_copy_enqueue_us=0, stream_sync_us=0, cleanup_us=0;
    float publish_gpu_ms=0, clear_gpu_ms=0, reduce_gpu_ms=0, status_gpu_ms=0;
    NvU64 peer_wait_cycles=0;
    cudaEvent_t ev_publish_begin{}, ev_publish_end{}, ev_clear_end{}, ev_reduce_end{}, ev_status_end{};
    auto elapsed_us=[](auto a,auto b){return std::chrono::duration<double,std::micro>(b-a).count();};
    try {
        if(!g_peer)throw std::runtime_error("RTX peer client is not open");
        if(layer>=Q38_LAYERS_WITH_MTP||!h||!ids||!routes||!out||h_type<0||h_type>2)throw std::runtime_error("invalid RTX peer forward arguments");
        auto&s=*g_peer;
        if (s.ingress_diagnostic != ingress_diagnostic)
            throw std::runtime_error("ingress diagnostic mode changed after peer-open");
        if(!s.pending_sequences.empty())
            throw std::runtime_error("synchronous peer forward cannot overtake an asynchronous window; flush it first");
        auto phase=PeerClock::now();
        CUKQ(cuCtxSetCurrent(s.rtx_context));CK(cudaSetDevice(s.rtx_ordinal));
        auto now=PeerClock::now(); context_us=elapsed_us(phase,now); phase=now;
        // PLE uses the same resident control sequence through the established
        // CPU/PEX path. Refresh it from every retained owner BAR mapping so
        // MoE and PLE can safely share one mailbox without stale local counters.
        NvU32 previous=*reinterpret_cast<volatile NvU32*>(s.owner_cpus[0]);
        for(size_t i=1;i<s.owner_cpus.size();i++) {
            NvU32 current=*reinterpret_cast<volatile NvU32*>(s.owner_cpus[i]);
            if(current!=previous)throw std::runtime_error("resident worker sequences diverged before RTX publication");
        }
        now=PeerClock::now(); sequence_us=elapsed_us(phase,now); phase=now;
        NvU32 seq=previous+1;if(previous>=UINT32_MAX-1||!seq)throw std::runtime_error("resident sequence space exhausted");
        for(size_t i=0;i<24;i++){s.routes[i].requester_quiesced=false;s.return_routes[i].requester_quiesced=false;}s.in_flight=true;
        s.pex_quiesced=false;
        s.m10_quiesced=false;
        auto st=reinterpret_cast<cudaStream_t>(stream);
        if(profile) {
            CK(cudaEventCreateWithFlags(&ev_publish_begin,cudaEventDefault));
            CK(cudaEventCreateWithFlags(&ev_publish_end,cudaEventDefault));
            CK(cudaEventCreateWithFlags(&ev_clear_end,cudaEventDefault));
            CK(cudaEventCreateWithFlags(&ev_reduce_end,cudaEventDefault));
            CK(cudaEventCreateWithFlags(&ev_status_end,cudaEventDefault));
            CK(cudaEventRecord(ev_publish_begin,st));
        }
        phase=PeerClock::now();
        enqueue_peer_request(s,st,h,h_type,ids,routes,layer,seq);
        // Diagnostic-only boundary: distinguish a fault in the RTX→PEX
        // publication kernels from a fault in the later RTX result reader.
        // The stream sync is disabled in normal and timed runs.
        debug_stage_sync(st, "request_publication");
        now=PeerClock::now(); publish_enqueue_us=elapsed_us(phase,now); phase=now;
        if (s.ingress_diagnostic) {
            ingress_diagnostic_join(s, seq, out, st);
            for (size_t i=0; i<s.sequence.size(); ++i) s.sequence[i]=seq;
            return 0;
        }
        if(profile) CK(cudaEventRecord(ev_publish_end,st));
        CK(cudaMemsetAsync(s.device_headers,0xff,(size_t(24)*4+2)*sizeof(NvU32),st));
        now=PeerClock::now(); clear_enqueue_us=elapsed_us(phase,now); phase=now;
        if(profile) CK(cudaEventRecord(ev_clear_end,st));
        rtx_peer_reduce<<<1,256,0,st>>>(s.device_return_slots,24,seq,layer,out,s.device_status,
                                        s.device_headers,kWaitCycles);
        CK(cudaGetLastError());
        debug_stage_sync(st, "peer_reduce");
        now=PeerClock::now(); reduce_enqueue_us=elapsed_us(phase,now); phase=now;
        if(profile) CK(cudaEventRecord(ev_reduce_end,st));
        NvU32 status=~0u;CK(cudaMemcpyAsync(&status,s.device_status,sizeof(status),cudaMemcpyDeviceToHost,st));
        now=PeerClock::now(); status_copy_enqueue_us=elapsed_us(phase,now); phase=now;
        if(profile) CK(cudaEventRecord(ev_status_end,st));
        const auto sync_started=PeerClock::now();
        CK(cudaStreamSynchronize(st));
        now=PeerClock::now(); stream_sync_us=elapsed_us(sync_started,now); phase=now;
        if(profile) {
            CK(cudaEventSynchronize(ev_status_end));
            CK(cudaEventElapsedTime(&publish_gpu_ms,ev_publish_begin,ev_publish_end));
            CK(cudaEventElapsedTime(&clear_gpu_ms,ev_publish_end,ev_clear_end));
            CK(cudaEventElapsedTime(&reduce_gpu_ms,ev_clear_end,ev_reduce_end));
            CK(cudaEventElapsedTime(&status_gpu_ms,ev_reduce_end,ev_status_end));
            NvU32 wait_words[2]{};
            CK(cudaMemcpy(wait_words,s.device_headers+24*4,sizeof(wait_words),cudaMemcpyDeviceToHost));
            peer_wait_cycles=NvU64(wait_words[0])|(NvU64(wait_words[1])<<32);
        }
        for(size_t i=0;i<24;i++){s.routes[i].requester_quiesced=true;s.return_routes[i].requester_quiesced=true;s.sequence[i]=seq;}s.in_flight=false;
        s.pex_quiesced=true;
        if(s.gpu_dev)CK(cudaMemcpy(&s.gpu_dev->seq,&seq,sizeof(seq),cudaMemcpyHostToDevice));
        now=PeerClock::now(); cleanup_us=elapsed_us(phase,now);
        if(profile) {
            std::ofstream log(profile_path,std::ios::app);
            if(!log) throw std::runtime_error("cannot append RTX peer phase profile");
            log << layer << ',' << seq << ',' << context_us << ',' << sequence_us << ','
                << publish_enqueue_us << ',' << clear_enqueue_us << ',' << reduce_enqueue_us << ','
                << status_copy_enqueue_us << ',' << stream_sync_us << ',' << cleanup_us << ',' << publish_gpu_ms*1000.0f << ','
                << clear_gpu_ms*1000.0f << ',' << reduce_gpu_ms*1000.0f << ','
                << status_gpu_ms*1000.0f << ',' << peer_wait_cycles << ','
                << elapsed_us(t_entry,now) << '\n';
        }
        if(profile) for(cudaEvent_t ev : {ev_publish_begin,ev_publish_end,ev_clear_end,ev_reduce_end,ev_status_end})
            if(ev) CK(cudaEventDestroy(ev));
        if(status) {
            std::vector<NvU32> observed(size_t(24)*4);
            CK(cudaMemcpy(observed.data(),s.device_headers,observed.size()*sizeof(NvU32),cudaMemcpyDeviceToHost));
            std::fprintf(stderr,
                         "RTX_PEER_FAILURE_DIAG,sequence,%u,workers,%zu,owner_cmd_seq_layer_mode,direct_header,return_bar_header,return_result0_4\n",
                         seq,s.workers.size());
            for(size_t i=0;i<s.workers.size();i++) {
                auto *owner_words=reinterpret_cast<volatile NvU32*>(s.owner_cpus[i]);
                auto *direct=reinterpret_cast<volatile NvU32*>(
                    reinterpret_cast<volatile NvU8*>(s.owner_cpus[i])+kDirectResultOff);
                const NvU64 leaf_offset = s.workers[i].return_alias_va & (kReturnLeafBytes - 1);
                auto *ret=reinterpret_cast<volatile NvU32 *>(
                    reinterpret_cast<volatile NvU8 *>(s.return_owner.cpu) +
                    s.return_routes[i].owner_offset + leaf_offset);
                auto *ret_result=reinterpret_cast<volatile float*>(ret+4);
                std::fprintf(stderr,
                    "RTX_PEER_FAILURE_WORKER,index,%zu,bdf,%s,command,%u,%u,%u,direct,%u,%u,%u,%u,return,%u,%u,%u,%u,result0_4,%.9g,%.9g,%.9g,%.9g\n",
                    i,s.workers[i].bdf.c_str(),owner_words[0],owner_words[1],owner_words[2],
                    direct[0],direct[1],direct[2],direct[3],ret[0],ret[1],ret[2],ret[3],
                    double(ret_result[0]),double(ret_result[1]),double(ret_result[2]),double(ret_result[3]));
            }
            std::fflush(stderr);
            std::ostringstream detail;
            detail << "RTX peer reducer rejected headers; observed workers [";
            for(size_t i=0;i<24;i++) {
                if(i)detail << ';';
                detail << i << ':' << std::hex << observed[i*4] << ',' << observed[i*4+1]
                       << ',' << observed[i*4+2] << ',' << observed[i*4+3] << std::dec;
            }
            detail << ']';
            throw std::runtime_error(detail.str());
        }
        return 0;
    } catch(const std::exception&e){
        if(profile) for(cudaEvent_t ev : {ev_publish_begin,ev_publish_end,ev_clear_end,ev_reduce_end,ev_status_end})
            if(ev) cudaEventDestroy(ev);
        g_peer_error=e.what();
        if(g_peer&&g_peer->in_flight) {
            // A host-side launch/status error does not prove that an already
            // submitted requester operation has stopped. Establish device
            // quiescence before allowing adapter teardown to restore aliases.
            cudaError_t sync=cudaDeviceSynchronize();
            if(sync==cudaSuccess) {
                for(auto &route:g_peer->routes)route.requester_quiesced=true;
                for(auto &route:g_peer->return_routes)route.requester_quiesced=true;
                g_peer->in_flight=false;
                g_peer->pex_quiesced=true;
            } else {
                g_peer_error += std::string("; CUDA quiescence not established: ")+
                    cudaGetErrorString(sync)+"; retain peer mappings and owner allocations";
            }
        }
        if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());return 1;
    }
}
// ---- Phase 2 host API (GPU-driven exchange) -------------------------------------------------
// Enqueue one row's exchange on `stream`: begin (skip/sequence) -> payload -> sequence publish ->
// spin-wait reducer. No host reads, no synchronization: capturable into a CUDA graph. Errors are
// reported through q38_rtx_peer_check_error(); dummy rows (ids outside [0,512)) produce zeros.
extern "C" int q38_rtx_peer_forward_gpu(uint32_t layer,const void*h,int h_type,const int32_t*ids,
        const float*routes,float*out,void*stream,char*err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if(!g_peer)throw std::runtime_error("RTX peer client is not open");
        auto &s=*g_peer;
        if(!s.pex_multicast||!s.gpu_dev)throw std::runtime_error("GPU-driven exchange needs the PEX multicast path");
        if(layer>=Q38_LAYERS_WITH_MTP||!h||!ids||!routes||!out||h_type<0||h_type>2)throw std::runtime_error("invalid RTX peer forward arguments");
        if(!s.pending_sequences.empty())throw std::runtime_error("GPU-driven exchange cannot follow a pending async window");
        auto st=reinterpret_cast<cudaStream_t>(stream);
        for(size_t i=0;i<24;i++){s.routes[i].requester_quiesced=false;s.return_routes[i].requester_quiesced=false;}
        s.in_flight=true; s.pex_quiesced=false; s.m10_quiesced=false; s.gpu_used=true;
        using Op=PciClusterMoeOp;
        const Op::Args a{h,h_type,ids,routes,out};
        pci_cluster_begin<Op><<<1,32,0,st>>>(s.gpu_dev,s.gpu_host_dev,a,layer);
        CK(cudaGetLastError());
        pci_cluster_publish<Op><<<s.pub_groups,256,0,st>>>(s.gpu_dev,a,s.pub_aliases,layer);
        CK(cudaGetLastError());
        pci_cluster_publish_sequence<<<s.pub_groups,1,0,st>>>(s.gpu_dev,s.pub_aliases,Op::kMode);
        CK(cudaGetLastError());
        pci_cluster_wait_combine<Op><<<1,kMoeCombineThreads,0,st>>>(s.gpu_dev,s.gpu_host_dev,a,s.device_return_slots,24,layer,
                                                     s.device_headers,kWaitCycles,2000000000ull);
        CK(cudaGetLastError());
        return 0;
    } catch(const std::exception&e) {
        g_peer_error=e.what(); if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str()); return 1;
    }
}
// Multi-row decode MoE (worker mode 10): rows 1..2 BF16 rows of h / ids / routes -> out [rows][H] FP32, one
// exchange (enqueue only, capturable). Needs Q38_RTX_PEER_MOE_MULTIROW=1 at open.
extern "C" int q38_rtx_peer_moe2_gpu(uint32_t layer,const void*h,int h_type,const int32_t*ids,const float*routes,
        uint32_t rows,float*out,void*stream,char*err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
#ifdef Q38_RTX_PEER_LEADER_AGG
        if(!g_peer)throw std::runtime_error("RTX peer client is not open");
        auto &s=*g_peer;
        if(!s.pex_multicast||!s.gpu_dev)throw std::runtime_error("GPU-driven exchange needs the PEX multicast path");
        if(!s.multirow||!s.device_return_slots1)throw std::runtime_error("multi-row exchange is not configured (Q38_RTX_PEER_MOE_MULTIROW=1 at open)");
        if(layer>=Q38_LAYERS_WITH_MTP||!h||!ids||!routes||!out||h_type!=2||rows<1||rows>s.mr_max_rows)
            throw std::runtime_error("invalid multi-row forward arguments (BF16 rows 1..max rows, layer < 49)");
        if(!s.pending_sequences.empty())throw std::runtime_error("GPU-driven exchange cannot follow a pending async window");
        auto st=reinterpret_cast<cudaStream_t>(stream);
        for(size_t i=0;i<24;i++){s.routes[i].requester_quiesced=false;s.return_routes[i].requester_quiesced=false;}
        s.in_flight=true; s.pex_quiesced=false; s.m10_quiesced=false; s.gpu_used=true;
        using Op=PciClusterMoe2Op;
        const Op::Args a{h,ids,routes,out,rows,s.device_return_slots1,s.gpu_dev,s.gpu_host_dev,layer};
        if (s.publish_order == 4 && s.pub_groups == 1) {      // default: begin + publish + go in one kernel (outer group)
            pci_cluster_begin_publish_go<Op><<<1,256,0,st>>>(s.gpu_dev,s.gpu_host_dev,a,s.pub_aliases,layer);
            CK(cudaGetLastError());
        } else {
        pci_cluster_begin<Op><<<1,32,0,st>>>(s.gpu_dev,s.gpu_host_dev,a,layer);
        CK(cudaGetLastError());
        if (s.publish_order == 3 && s.pub_groups == 1) {     // needs the outer group; else the default path below
            pci_cluster_publish<Op><<<s.pub_groups,256,0,st>>>(s.gpu_dev,a,s.pub_aliases,layer);
            CK(cudaGetLastError());
            pci_cluster_go_staggered<<<1,1,0,st>>>(s.gpu_dev,s.device_pex_aliases,Op::kMode,s.publish_seq,s.go_gap_ns,s.go_lowfence);
            CK(cudaGetLastError());
        } else if ((s.publish_order == 1 || s.publish_order == 2) && s.pub_groups == kPexGroups) {
            pci_cluster_publish_go<Op><<<s.publish_order == 2 ? 1 : kPexGroups,256,0,st>>>(s.gpu_dev,a,s.pub_aliases,layer,s.publish_order == 2 ? 1u : 0u,s.publish_seq);
            CK(cudaGetLastError());
        } else {
        pci_cluster_publish<Op><<<s.pub_groups,256,0,st>>>(s.gpu_dev,a,s.pub_aliases,layer);
        CK(cudaGetLastError());
        pci_cluster_publish_sequence<<<s.pub_groups,1,0,st>>>(s.gpu_dev,s.pub_aliases,Op::kMode);
        CK(cudaGetLastError());
        }
        }   // publish order 4 else
        pci_cluster_wait_combine<Op><<<1,kMoeCombineThreads,0,st>>>(s.gpu_dev,s.gpu_host_dev,a,s.device_return_slots,24,layer,
                                                     s.device_headers,kWaitCycles,2000000000ull);
        CK(cudaGetLastError());
        return 0;
#else
        (void)layer;(void)h;(void)h_type;(void)ids;(void)routes;(void)rows;(void)out;(void)stream;
        throw std::runtime_error("multi-row exchange needs the leader-aggregation build");
#endif
    } catch(const std::exception&e) {
        g_peer_error=e.what(); if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str()); return 1;
    }
}
// Re-read the workers' control sequence (all must agree) into the device counter. Call only while
// no GPU-driven exchange is queued (e.g. after the stream was synchronized for a PLE lookup).
extern "C" int q38_rtx_peer_gpu_resync(char*err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if(!g_peer||!g_peer->gpu_dev)throw std::runtime_error("RTX peer client is not open");
        auto &s=*g_peer;
        NvU32 seq=*reinterpret_cast<volatile NvU32*>(s.owner_cpus[0]);
        for(size_t i=1;i<s.owner_cpus.size();i++)
            if(*reinterpret_cast<volatile NvU32*>(s.owner_cpus[i])!=seq)throw std::runtime_error("resident worker sequences diverged");
        CK(cudaMemcpy(&s.gpu_dev->seq,&seq,sizeof(seq),cudaMemcpyHostToDevice));
        return 0;
    } catch(const std::exception&e) {
        g_peer_error=e.what(); if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str()); return 1;
    }
}
static std::string pci_cluster_canon_bdf(std::string b) {
    for (auto &ch : b) ch = char(std::tolower(static_cast<unsigned char>(ch)));
    const auto colon = b.find(':');
    if (colon == std::string::npos) throw std::runtime_error("malformed PCI address: " + b);
    if (b.find(':', colon + 1) == std::string::npos) return "0000:" + b;          // bus:dev.fn
    const unsigned long domain = std::stoul(b.substr(0, colon), nullptr, 16);
    char head[16]; std::snprintf(head, sizeof(head), "%04lx", domain);
    return std::string(head) + b.substr(colon);
}
// Configure the PCI-cluster row gather from the daemon's PLE IPC manifest (DEVICE and SHARD rows):
// uploads the shard table (owner = peer worker index, matched by PCI address) and writes each
// worker's table size into ple_reserved[1..2] through the owner BAR mapping (previous words saved,
// readback verified, restored at close). Call after open, while the workers are parked.
extern "C" int q38_rtx_peer_configure_row_gather(const char *manifest_path, char *err, size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.pex_multicast || !s.gpu_dev) throw std::runtime_error("row gather needs the GPU-driven PEX multicast path");
        if (s.rg_configured || !s.rg_saved.empty()) throw std::runtime_error("row gather is already configured");
        if (s.in_flight) throw std::runtime_error("row gather must be configured before any exchange");
        if (!manifest_path || !*manifest_path) throw std::runtime_error("empty PLE manifest path");
        std::map<std::string, NvU32> worker_for_bdf;
        for (size_t i = 0; i < s.workers.size(); i++) worker_for_bdf[pci_cluster_canon_bdf(s.workers[i].bdf)] = NvU32(i);
        std::ifstream f(manifest_path);
        if (!f) throw std::runtime_error(std::string("open PLE manifest: ") + manifest_path);
        std::map<NvU32, NvU64> table_bytes;                 // worker -> bytes
        std::map<NvU32, PciClusterShard> by_id;
        struct LmSlice { NvU64 off, rows, row_start; };
        std::map<NvU32, LmSlice> lm;                        // worker -> LM-head slice (LMHEAD rows)
        std::string line;
        while (std::getline(f, line)) {
            std::vector<std::string> v; std::stringstream ss(line); std::string tok;
            while (std::getline(ss, tok, '\t')) v.push_back(tok);
            if (v.size() < 2 || v[1] == "cuda_ordinal" || v[1] == "id") continue;
            if (v[0] == "DEVICE") {
                if (v.size() != 5) throw std::runtime_error("malformed DEVICE row in PLE manifest");
                auto it = worker_for_bdf.find(pci_cluster_canon_bdf(v[2]));
                if (it == worker_for_bdf.end()) throw std::runtime_error("PLE device is not a peer worker: " + v[2]);
                table_bytes[it->second] = std::stoull(v[3]);
            } else if (v[0] == "LMHEAD") {
                if (v.size() != 7 || v[6] != "5120") throw std::runtime_error("malformed LMHEAD row in PLE manifest");
                auto it = worker_for_bdf.find(pci_cluster_canon_bdf(v[2]));
                if (it == worker_for_bdf.end()) throw std::runtime_error("LM-head slice is not on a peer worker: " + v[2]);
                if (!lm.emplace(it->second, LmSlice{std::stoull(v[3]), std::stoull(v[4]), std::stoull(v[5])}).second)
                    throw std::runtime_error("two LM-head slices for one worker: " + v[2]);
            } else if (v[0] == "SHARD") {
                if (v.size() != 8 || v[7] != "160") throw std::runtime_error("malformed SHARD row in PLE manifest");
                auto it = worker_for_bdf.find(pci_cluster_canon_bdf(v[3]));
                if (it == worker_for_bdf.end()) throw std::runtime_error("PLE shard is not on a peer worker: " + v[3]);
                const NvU32 id = NvU32(std::stoul(v[1]));
                const NvU64 bytes = std::stoull(v[4]), offset = std::stoull(v[5]), rows = std::stoull(v[6]);
                if (bytes != rows * PCI_CLUSTER_ROW_BYTES || (offset & 3ull) || offset > PCI_CLUSTER_OFFSET_MASK)
                    throw std::runtime_error("invalid PLE shard geometry, id " + v[1]);
                // Global row numbering follows the fixed shard ids (as the CPU direct client).
                if (!by_id.emplace(id, PciClusterShard{NvU64(id) * rows, rows, offset, it->second, 0}).second)
                    throw std::runtime_error("duplicate PLE shard id " + v[1]);
            }
        }
        if (by_id.empty()) throw std::runtime_error("PLE manifest contains no shards");
        std::vector<PciClusterShard> shards;
        for (auto &[id, sh] : by_id) {
            auto tb = table_bytes.find(sh.owner);
            if (tb == table_bytes.end() || sh.offset + sh.rows * PCI_CLUSTER_ROW_BYTES > tb->second)
                throw std::runtime_error("PLE shard exceeds its worker's table, id " + std::to_string(id));
            shards.push_back(sh);
        }
        std::sort(shards.begin(), shards.end(), [](const PciClusterShard &a, const PciClusterShard &b) { return a.global_start < b.global_start; });
        for (size_t i = 1; i < shards.size(); i++)
            if (shards[i-1].global_start + shards[i-1].rows > shards[i].global_start) throw std::runtime_error("PLE shards overlap");
        CK(cudaMalloc(&s.rg_shards, shards.size() * sizeof(PciClusterShard)));
        CK(cudaMemcpy(s.rg_shards, shards.data(), shards.size() * sizeof(PciClusterShard), cudaMemcpyHostToDevice));
        s.rg_nshards = NvU32(shards.size());
        CK(cudaMalloc(&s.rg_desc, kPciClusterRowGatherMaxRows * sizeof(NvU64)));
        CK(cudaMalloc(&s.rg_rank, kPciClusterRowGatherMaxRows * sizeof(NvU32)));
        CK(cudaMalloc(&s.rg_counts, 24 * sizeof(NvU32)));
        // Per-worker table size: save, write, verify.
        s.rg_saved.resize(s.owner_cpus.size());
        for (size_t i = 0; i < s.owner_cpus.size(); i++) {
            auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
            for (int k = 0; k < 3; k++) s.rg_saved[i].w[k] = c->ple_reserved[k];
        }
        for (size_t i = 0; i < s.owner_cpus.size(); i++) {
            auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
            const NvU64 bytes = table_bytes.count(NvU32(i)) ? table_bytes[NvU32(i)] : 0;
            c->ple_reserved[1] = NvU32(bytes); c->ple_reserved[2] = NvU32(bytes >> 32);
        }
        asm volatile("sfence" ::: "memory");
        for (size_t i = 0; i < s.owner_cpus.size(); i++) {
            auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
            const NvU64 bytes = table_bytes.count(NvU32(i)) ? table_bytes[NvU32(i)] : 0;
            const NvU64 got = NvU64(c->ple_reserved[1]) | (NvU64(c->ple_reserved[2]) << 32);
            std::fprintf(stderr, "PCI_CLUSTER_ROW_GATHER_TABLE,worker,%zu,bdf,%s,bytes,%llu,readback,%llu,saved,0x%x/0x%x/0x%x\n",
                         i, s.workers[i].bdf.c_str(), (unsigned long long)bytes, (unsigned long long)got,
                         s.rg_saved[i].w[0], s.rg_saved[i].w[1], s.rg_saved[i].w[2]);
            if (got != bytes) throw std::runtime_error("row-gather table size readback mismatch at " + s.workers[i].bdf);
        }
        s.rg_configured = true;
        if (!lm.empty()) {
            // Every worker holds one slice; slices tile [0, vocab) contiguously; each ends exactly at its
            // worker's table end (the same rule worker mode 7 enforces before reading anything).
            if (lm.size() != 24) throw std::runtime_error("LM head: expected 24 slices, manifest has " + std::to_string(lm.size()));
            std::vector<std::pair<NvU64, NvU32>> by_start; for (auto &[w, sl] : lm) by_start.push_back({sl.row_start, w});
            std::sort(by_start.begin(), by_start.end());
            NvU64 next_row = 0; std::vector<NvU64> off(24), rows(24); std::vector<NvU32> start(24);
            for (auto &[st, w] : by_start) {
                const auto &sl = lm[w];
                const NvU64 tb = table_bytes.count(w) ? table_bytes[w] : 0;
                if (st != next_row || !sl.rows || sl.rows > PCI_CLUSTER_LM_MAX_ROWS || (sl.off & 15ull) ||
                    sl.off + sl.rows * (PCI_CLUSTER_LM_ROW_BYTES + 4ull) != tb)
                    throw std::runtime_error("LM-head slice geometry invalid for worker " + std::to_string(w));
                off[w] = sl.off; rows[w] = sl.rows; start[w] = NvU32(sl.row_start); next_row += sl.rows;
            }
            CK(cudaMalloc(&s.lm_off, 24 * sizeof(NvU64))); CK(cudaMalloc(&s.lm_rows, 24 * sizeof(NvU64)));
            CK(cudaMalloc(&s.lm_row_start, 24 * sizeof(NvU32)));
            CK(cudaMemcpy(s.lm_off, off.data(), 24 * sizeof(NvU64), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s.lm_rows, rows.data(), 24 * sizeof(NvU64), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s.lm_row_start, start.data(), 24 * sizeof(NvU32), cudaMemcpyHostToDevice));
            s.lm_vocab = NvU32(next_row); s.lm_configured = true;
            std::fprintf(stderr, "PCI_CLUSTER_LM_HEAD_CONFIGURED,vocab,%u,slices,24\n", s.lm_vocab);
        }
        std::fprintf(stderr, "PCI_CLUSTER_ROW_GATHER_CONFIGURED,shards,%u,workers_with_tables,%zu\n", s.rg_nshards, table_bytes.size());
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Enqueue one row gather (n <= 64 global rows, device int64) on `stream`: out receives n x 160 bytes.
// Capturable; failures produce zeros and the sticky error (q38_rtx_peer_check_error).
extern "C" int q38_rtx_peer_row_gather_gpu(const long long *rows, uint32_t n, uint8_t *out, void *stream, char *err, size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.rg_configured) throw std::runtime_error("row gather is not configured");
        if (!rows || !out || !n || n > kPciClusterRowGatherMaxRows) throw std::runtime_error("invalid row-gather arguments");
        if (!s.pending_sequences.empty()) throw std::runtime_error("GPU-driven exchange cannot follow a pending async window");
        auto st = reinterpret_cast<cudaStream_t>(stream);
        for (size_t i = 0; i < 24; i++) { s.routes[i].requester_quiesced = false; s.return_routes[i].requester_quiesced = false; }
        s.in_flight = true; s.pex_quiesced = false; s.m10_quiesced = false; s.gpu_used = true;
        using Op = PciClusterRowGatherOp;
        constexpr NvU32 kRowGatherLayer = 0x100;            // error records: "layer 256" = row gather
        const Op::Args a{rows, n, out, s.rg_shards, s.rg_nshards, s.rg_desc, s.rg_rank, s.rg_counts, s.gpu_host_dev, kRowGatherLayer};
        pci_cluster_begin<Op><<<1, 32, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, kRowGatherLayer);
        CK(cudaGetLastError());
        pci_cluster_publish<Op><<<s.pub_groups, 128, 0, st>>>(s.gpu_dev, a, s.pub_aliases, kRowGatherLayer);
        CK(cudaGetLastError());
        pci_cluster_publish_sequence<<<s.pub_groups, 1, 0, st>>>(s.gpu_dev, s.pub_aliases, Op::kMode);
        CK(cudaGetLastError());
        pci_cluster_wait_combine<Op><<<1, 256, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, s.device_return_slots, 24, kRowGatherLayer,
                                                        s.device_headers, kWaitCycles, 2000000000ull);
        CK(cudaGetLastError());
        s.rg_calls++; s.rg_rows += n;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str()); return 1;
    }
}
#include "qwen38_kv_tier.cuh"
// ---------------------------------------------------------------------------------------------
// Zero-copy row gather from a pinned host table (2026-11-03, Q38_EMBEDDING_ZERO_COPY=1): the RTX
// reads the needed rows straight from host RAM over PCIe (one 5-KiB row per decoded token), so the
// embedding lookup is stream work and is captured in the CUDA graph. Independent of the peer session.
// Ids outside [0, rows_total) produce zero rows and increment *bad (device counter).
__global__ static void q38_zero_copy_gather_kernel(const uint4 *table, NvU64 rows_total, NvU32 row_vec,
        const long long *ids, NvU32 n, uint4 *out, NvU32 *bad) {
    for (NvU32 t = blockIdx.x; t < n; t += gridDim.x) {
        const long long id = ids[t];
        const bool ok = id >= 0 && NvU64(id) < rows_total;
        if (!ok && threadIdx.x == 0 && bad) atomicAdd(bad, 1u);
        for (NvU32 v = threadIdx.x; v < row_vec; v += blockDim.x)
            out[NvU64(t) * row_vec + v] = ok ? table[NvU64(id) * row_vec + v] : make_uint4(0, 0, 0, 0);
    }
}
// Pin `bytes` of host memory at `host` (mapped, read-only when supported) and return the device
// pointer the gather kernel uses. Pair with q38_zero_copy_unregister.
extern "C" int q38_zero_copy_register(void *host, size_t bytes, void **device_ptr, int *read_only, char *err, size_t cap) {
    try {
        if (!host || !bytes || !device_ptr) throw std::runtime_error("invalid zero-copy registration arguments");
        cudaError_t rc = cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable | cudaHostRegisterReadOnly);
        int ro = 1;
        if (rc != cudaSuccess) {                   // read-only registration is optional
            (void)cudaGetLastError(); ro = 0;
            CK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        }
        CK(cudaHostGetDevicePointer(device_ptr, host, 0));
        if (read_only) *read_only = ro;
        return 0;
    } catch (const std::exception &e) {
        if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1;
    }
}
extern "C" int q38_zero_copy_unregister(void *host, char *err, size_t cap) {
    try { CK(cudaHostUnregister(host)); return 0; }
    catch (const std::exception &e) { if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1; }
}
// Enqueue the gather on `stream`: out[t] = table[ids[t]] (row_bytes, multiple of 16). Capturable.
extern "C" int q38_zero_copy_gather(const void *device_table, unsigned long long rows_total, uint32_t row_bytes,
        const long long *ids, uint32_t n, void *out, uint32_t *bad, void *stream, char *err, size_t cap) {
    try {
        if (!device_table || !ids || !out || !n || !row_bytes || (row_bytes % 16u) ||
            (reinterpret_cast<uintptr_t>(device_table) % 16u) || (reinterpret_cast<uintptr_t>(out) % 16u))
            throw std::runtime_error("invalid zero-copy gather arguments");
        const NvU32 blocks = n < 1024u ? n : 1024u;
        q38_zero_copy_gather_kernel<<<blocks, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<const uint4 *>(device_table), rows_total, row_bytes / 16u, ids, n,
            reinterpret_cast<uint4 *>(out), bad);
        CK(cudaGetLastError());
        return 0;
    } catch (const std::exception &e) {
        if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1;
    }
}
// Host-memory K/V tier (docs/research/KV_HOST_TIER_PLAN.md, Q38_KV_TIER=1): see qwen38_kv_tier.cuh.
extern "C" size_t q38_kv_tier_state_size() { return sizeof(Q38KvTierState); }
extern "C" int q38_kv_tier_register(void *host, size_t bytes, void **device_ptr, char *err, size_t cap) {
    try {
        CK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        CK(cudaHostGetDevicePointer(device_ptr, host, 0));
        return 0;
    } catch (const std::exception &e) { if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1; }
}
// Copy `pages` pool pages of one layer to host RAM, in token order. `table`: host copy of the block table.
extern "C" int q38_kv_tier_populate(const uint8_t *kv, uint64_t block_stride, uint32_t page_tokens, uint32_t token_stride,
        const int32_t *table, uint32_t pages, uint8_t *host, char *err, size_t cap) {
    try {
        for (uint32_t i = 0; i < pages; i++)
            CK(cudaMemcpy(host + uint64_t(i) * page_tokens * 2048u, kv + uint64_t(table[i]) * block_stride, uint64_t(page_tokens) * token_stride, cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception &e) { if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1; }
}
extern "C" int q38_kv_resolve(const int32_t *in, int32_t *out, uint32_t width, uint32_t rows, const int64_t *positions, uint8_t *kv,
        uint64_t block_stride, uint32_t token_stride, uint32_t page_tokens, const int32_t *block_table, uint8_t *host,
        int32_t *block_to_slot, int32_t *slot_to_block, uint32_t *slot_epoch, Q38KvTierState *st, const uint8_t *inbox, void *stream, char *err, size_t cap) {
    try {
        Q38KvTierArgs a{in, out, width, rows, positions, kv, block_stride, token_stride, page_tokens, block_table, host, block_to_slot, slot_to_block, slot_epoch, st, inbox};
        q38_kv_resolve_kernel<<<1, 1024, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a);
        CK(cudaGetLastError());
        return 0;
    } catch (const std::exception &e) { if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1; }
}
extern "C" int q38_kv_write_through(const uint8_t *inbox, uint8_t *host, const int64_t *positions, uint32_t rows, const int32_t *in, uint32_t width, void *stream, char *err, size_t cap) {
    try {
        q38_kv_write_through_kernel<<<rows < 64u ? 1u : 16u, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const uint4 *>(inbox), host, positions, rows, in, width);
        CK(cudaGetLastError());
        return 0;
    } catch (const std::exception &e) { if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1; }
}
extern "C" int q38_kv_verify(const int32_t *in, int32_t *out, uint32_t width, uint32_t rows, const int64_t *positions, uint8_t *kv,
        uint64_t block_stride, uint32_t token_stride, uint32_t page_tokens, const int32_t *block_table, uint8_t *host,
        Q38KvTierState *st, uint32_t cache_tokens, uint64_t *checked, uint64_t *bad, void *stream, char *err, size_t cap) {
    try {
        Q38KvTierArgs a{in, out, width, rows, positions, kv, block_stride, token_stride, page_tokens, block_table, host, nullptr, nullptr, nullptr, st, nullptr};
        q38_kv_verify_kernel<<<16, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(a, cache_tokens, reinterpret_cast<unsigned long long *>(checked), reinterpret_cast<unsigned long long *>(bad));
        CK(cudaGetLastError());
        return 0;
    } catch (const std::exception &e) { if (err && cap) std::snprintf(err, cap, "%s", e.what()); return 1; }
}
// Enqueue one LM-head top-k exchange for one row: x (QWEN_H values, h_type 0 f32 / 1 f16 / 2 bf16, device)
// -> out (vocab BF16 logits: top candidates, -inf elsewhere), lse_out (FP32 global log-sum-exp, may be
// null). 1 <= k <= 2559. Capturable; failures give zero logits and the sticky error.
extern "C" int q38_rtx_peer_lm_head_topk_rows(const void *x, int x_type, uint16_t *out, float *lse_out, uint32_t k,
        uint32_t nrows, uint32_t *vocab_out, void *stream, char *err, size_t cap);
extern "C" int q38_rtx_peer_lm_head_topk(const void *x, int x_type, uint16_t *out, float *lse_out, uint32_t k,
        uint32_t *vocab_out, void *stream, char *err, size_t cap) {
    return q38_rtx_peer_lm_head_topk_rows(x, x_type, out, lse_out, k, 1, vocab_out, stream, err, cap);
}
// nrows = 2 (2026-11-04): two BF16 rows (x [2][H], out [2][vocab], lse_out [2]) in one exchange; needs 2(k+1) <= 2559.
extern "C" int q38_rtx_peer_lm_head_topk_rows(const void *x, int x_type, uint16_t *out, float *lse_out, uint32_t k,
        uint32_t nrows, uint32_t *vocab_out, void *stream, char *err, size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.lm_configured) throw std::runtime_error("LM head is not configured (no LMHEAD rows in the PLE manifest)");
        if (vocab_out) *vocab_out = s.lm_vocab;
        if (!x || !out) return 0;                            // vocab query only
        if (x_type < 0 || x_type > 2 || k < 1 || k > PCI_CLUSTER_LM_MAX_K) throw std::runtime_error("invalid LM-head top-k arguments");
        if (nrows < 1 || nrows > PCI_CLUSTER_LM_MAX_NROWS || (nrows >= 2 && (x_type != 2 || nrows * (k + 1) > 2560 - nrows)))
            throw std::runtime_error("N-row LM head needs BF16 rows, N <= 4 and N(k+1) <= 2560 - N");
        if (!s.pending_sequences.empty()) throw std::runtime_error("GPU-driven exchange cannot follow a pending async window");
        auto st = reinterpret_cast<cudaStream_t>(stream);
        for (size_t i = 0; i < 24; i++) { s.routes[i].requester_quiesced = false; s.return_routes[i].requester_quiesced = false; }
        s.in_flight = true; s.pex_quiesced = false; s.m10_quiesced = false; s.gpu_used = true;
        using Op = PciClusterLmHeadOp;
        constexpr NvU32 kLmLayer = 0x200;                    // error records: "layer 512" = LM head
        const Op::Args a{x, x_type, reinterpret_cast<NvU16 *>(out), s.lm_vocab, k, s.lm_off, s.lm_rows, s.lm_row_start,
                         lse_out, s.gpu_host_dev, kLmLayer, nrows};
        pci_cluster_begin<Op><<<1, 32, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, kLmLayer);
        CK(cudaGetLastError());
        pci_cluster_publish<Op><<<s.pub_groups, 256, 0, st>>>(s.gpu_dev, a, s.pub_aliases, kLmLayer);
        CK(cudaGetLastError());
        pci_cluster_publish_sequence<<<s.pub_groups, 1, 0, st>>>(s.gpu_dev, s.pub_aliases, Op::kMode);
        CK(cudaGetLastError());
        pci_cluster_wait_combine<Op><<<1, 1024, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, s.device_return_slots, 24, kLmLayer,
                                                         s.device_headers, kWaitCycles, 2000000000ull);
        CK(cudaGetLastError());
        s.lm_calls++;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Batched MoE: pin and map the shared CPU mailbox (the client's shm, registered in every worker by the
// daemon) on the RTX. `host` is the client's mmap of the whole region (Q38_CPU_MAILBOX_BYTES).
extern "C" int q38_rtx_peer_configure_moe_batch(void *host, size_t bytes, char *err, size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (s.batch_host) throw std::runtime_error("batched MoE is already configured");
        if (!host || bytes != Q38_CPU_MAILBOX_BYTES)
            throw std::runtime_error("CPU mailbox size " + std::to_string(bytes) + " is not the batch ABI size " + std::to_string(Q38_CPU_MAILBOX_BYTES));
        if (!s.pex_multicast || !s.gpu_dev) throw std::runtime_error("batched MoE needs the GPU-driven PEX multicast path");
        if (s.workers.size() != Q38_CPU_MAILBOX_WORKERS || s.owner_cpus.size() != Q38_CPU_MAILBOX_WORKERS)
            throw std::runtime_error("batched MoE needs all 24 workers");
        // Worker VA of the batch area = own slot VA (return_alias_va, from the daemon's mailbox
        // registration) - worker * slot stride + Q38_BATCH_OFF. Written through the owner BAR mapping
        // while the workers are parked: previous value saved, readback verified, restored at close.
        std::vector<NvU64> area(Q38_CPU_MAILBOX_WORKERS), saved(Q38_CPU_MAILBOX_WORKERS);
        for (size_t i = 0; i < area.size(); i++) {
            const NvU64 slot = s.workers[i].return_alias_va;
            if (!slot || s.owner_cpus[i] == MAP_FAILED || slot < i * Q38_CPU_MAILBOX_SLOT_STRIDE)
                throw std::runtime_error("batched MoE: no mailbox slot VA / owner mapping for worker " + std::to_string(i));
            area[i] = slot - i * Q38_CPU_MAILBOX_SLOT_STRIDE + Q38_BATCH_OFF;
            saved[i] = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i])->batch_area;
        }
        CK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
        s.batch_area_saved = saved;
        bool ok = true;
        for (size_t i = 0; i < area.size(); i++) {
            auto *c = reinterpret_cast<volatile PeerControlLayout *>(s.owner_cpus[i]);
            c->batch_area = area[i];
            const NvU64 rb = c->batch_area; ok &= rb == area[i];
            std::fprintf(stderr, "PCI_CLUSTER_MOE_BATCH_AREA,worker,%zu,bdf,%s,slot_va,0x%llx,batch_area,0x%llx,saved,0x%llx,readback,0x%llx\n",
                i, s.workers[i].bdf.c_str(), (unsigned long long)s.workers[i].return_alias_va, (unsigned long long)area[i],
                (unsigned long long)saved[i], (unsigned long long)rb);
        }
        if (!ok) throw std::runtime_error("batched MoE: batch_area readback mismatch");
        void *dev = nullptr;
        cudaError_t e = cudaHostGetDevicePointer(&dev, host, 0);
        if (e != cudaSuccess) { cudaHostUnregister(host); throw std::runtime_error(std::string("cudaHostGetDevicePointer mailbox: ") + cudaGetErrorString(e)); }
        s.batch_host = host; s.batch_dev = static_cast<char *>(dev) + Q38_BATCH_OFF;
        std::fprintf(stderr, "PCI_CLUSTER_MOE_BATCH_CONFIGURED,bytes,%zu,max_rows,%u\n", bytes, Q38_BATCH_MAX_ROWS);
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Enqueue one batched MoE exchange for `rows` (1..Q38_BATCH_MAX_ROWS) token rows of one layer: h [rows][2560]
// (h_type 0 f32 / 1 f16 / 2 bf16), ids/routes [rows][10] (device) -> out [rows][2560] f32. Capturable.
extern "C" int q38_rtx_peer_moe_batch_gpu(uint32_t layer, const void *h, int h_type, const int32_t *ids, const float *routes,
        uint32_t rows, float *out, void *stream, char *err, size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.batch_dev) throw std::runtime_error("batched MoE is not configured");
        if (layer >= Q38_LAYERS_WITH_MTP || !h || !ids || !routes || !out || h_type < 0 || h_type > 2 || rows < 1 || rows > Q38_BATCH_MAX_ROWS)
            throw std::runtime_error("invalid batched MoE arguments");
        if (!s.pending_sequences.empty()) throw std::runtime_error("GPU-driven exchange cannot follow a pending async window");
        auto st = reinterpret_cast<cudaStream_t>(stream);
        for (size_t i = 0; i < 24; i++) { s.routes[i].requester_quiesced = false; s.return_routes[i].requester_quiesced = false; }
        s.in_flight = true; s.pex_quiesced = false; s.m10_quiesced = false; s.gpu_used = true;
        using Op = PciClusterMoeBatchOp;
        const Op::Args a{h, h_type, ids, routes, rows, out, s.batch_dev, s.gpu_dev, s.gpu_host_dev, layer};
        pci_cluster_begin<Op><<<1, 32, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, layer);
        CK(cudaGetLastError());
        pci_cluster_publish<Op><<<s.pub_groups, 256, 0, st>>>(s.gpu_dev, a, s.pub_aliases, layer);
        CK(cudaGetLastError());
        pci_cluster_publish_sequence<<<s.pub_groups, 1, 0, st>>>(s.gpu_dev, s.pub_aliases, Op::kMode);
        CK(cudaGetLastError());
        pci_cluster_wait_combine<Op><<<1, 32, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, s.device_return_slots, 24, layer,
                                                       s.device_headers, kWaitCycles, 2000000000ull);
        CK(cudaGetLastError());
        q38_batch_combine<<<dim3((QWEN_H / 4 + 255) / 256, rows), 256, 0, st>>>(s.gpu_dev, a);
        CK(cudaGetLastError());
        s.batch_calls++; s.batch_rows += rows;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str()); return 1;
    }
}
// ---------------------------------------------------------------------------------------------
// Pipelined batched MoE (worker mode 9, 2026-11-03). One call = n chunks of <= Q38_BATCH_PIPE_ROWS rows of one
// layer, sequences pipe_base .. pipe_base + n - 1. Stream order:
//   begin(n)  P0 D0  P1 D1  W0 C0  P2 D2  W1 C1  P3 D3 ...  W(n-1) C(n-1)
// Host slot = sequence & 1 (as on the workers), so chunks c and c+2 share a slot.
// (P publish rows to the slot, D doorbell = multicast mode 9 + sequence, W wait for the 24 acks and the reducer
// flags, C combine). A slot is republished only after its previous chunk is combined (stream order), and the
// workers process sequences strictly in order, so at most two chunks are outstanding. While the workers compute
// chunk c+1 the RTX combines chunk c and publishes chunk c+2. An error (sticky GpuHost::err) before begin skips
// every chunk (zeros); a failure inside the pipeline skips the remaining combines (zeros) like a failed call.
__global__ static void q38_pipe_begin(PeerState::GpuDev *d, PeerState::GpuHost *h, NvU32 layer, NvU32 n) {
    if (blockIdx.x || threadIdx.x) return;
    NvU32 skip = static_cast<volatile PeerState::GpuHost *>(h)->err ? 2u : 0u;
    if (!skip) {
        const NvU32 first = d->seq + 1, last = d->seq + n;
        if (!first || last < first || last == 0xffffffffu) { skip = 2; q38_gpu_fail(h, 6, layer, d->seq, 0, 0, 0); }
        else { d->pipe_base = first; d->seq = last; d->cur_seq = last; d->calls += n; }
    }
    d->pipe_skip = skip; d->cur_skip = skip;
    if (g_pc_timing) {
        const NvU32 first = d->pad[1];
        d->pad[1] += n; d->pad[0] = first + n - 1; d->pipe_rec = first;
        const unsigned long long now = q38_globaltimer();
        for (NvU32 c = 0; c < n; c++) {
            PciClusterTimingRec *r = g_pc_timing + (first + c) % g_pc_timing_cap;
            r->layer = layer; r->seq = skip ? 0 : d->pipe_base + c; r->skip = skip;
            r->t_pub = r->t_seq = r->t_ack = r->t_done = 0; r->t_begin = now;
        }
    }
    __threadfence();
}
__global__ static void q38_pipe_publish(const PeerState::GpuDev *d, PciClusterMoeBatchOp::Args a) {
    if (static_cast<const volatile PeerState::GpuDev *>(d)->pipe_skip) return;
    if (static_cast<volatile PeerState::GpuHost *>(a.hst)->err) return;      // pipeline already failed: no new requests
    PciClusterMoeBatchOp::write_request(a);
    __threadfence_system();
    if (blockIdx.x == 0 && threadIdx.x == 0) { PciClusterTimingRec *r = pc_rec_pipe(d, a.chunk); if (r) r->t_pub = q38_globaltimer(); }
}
__global__ static void q38_pipe_doorbell(const PeerState::GpuDev *d, PeerState::GpuHost *hst, const NvU64 *aliases, NvU32 chunk) {
    const volatile PeerState::GpuDev *v = d;
    if (v->pipe_skip || static_cast<volatile PeerState::GpuHost *>(hst)->err) return;
    const NvU32 group = blockIdx.x;
    if (group < kPexGroups && threadIdx.x == 0) {
        auto *control = reinterpret_cast<volatile NvU32 *>(aliases[group*kPexPagesPerGroup]);
        control[2] = Q38_BATCH_PIPE_MODE;
        __threadfence_system();
        control[0] = v->pipe_base + chunk;
        __threadfence_system();
        if (group == 0) { PciClusterTimingRec *r = pc_rec_pipe(d, chunk); if (r) r->t_seq = q38_globaltimer(); }
    }
}
// W: thread 0 waits for every return slot to acknowledge >= this chunk's sequence, then checks the reducer flags of
// the chunk's slot (PciClusterMoeBatchOp::combine with a.pipe) and sets the verdict for q38_batch_combine.
__global__ static void q38_pipe_wait(const PeerState::GpuDev *d, PeerState::GpuHost *hst, PciClusterMoeBatchOp::Args a,
        const NvU64 *return_slots, NvU32 *observed_headers, NvU64 timeout_cycles, NvU64 deadline_ns) {
    if (threadIdx.x) return;
    const bool skip = static_cast<const volatile PeerState::GpuDev *>(d)->pipe_skip || static_cast<volatile PeerState::GpuHost *>(hst)->err;
    bool ok = false;
    if (!skip) ok = pci_cluster_wait_seq(PciClusterMoeBatchOp::seq_of(a), true, hst, return_slots, 24, a.layer, observed_headers, timeout_cycles, deadline_ns);
    PciClusterTimingRec *r = pc_rec_pipe(d, a.chunk); if (r) r->t_ack = q38_globaltimer();
    if (ok) PciClusterMoeBatchOp::combine(a, return_slots);
    else PciClusterMoeBatchOp::zero(a);
}

// Enqueue one pipelined batched MoE call for `rows` (>= 1) token rows of one layer (same arguments as
// q38_rtx_peer_moe_batch_gpu, any row count): ceil(rows / Q38_BATCH_PIPE_ROWS) worker exchanges in mode 9.
// Capturable (no host sync), but meant for eager prefill steps.
extern "C" int q38_rtx_peer_moe_batch_pipe_gpu(uint32_t layer, const void *h, int h_type, const int32_t *ids, const float *routes,
        uint32_t rows, float *out, void *stream, char *err, size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.batch_dev) throw std::runtime_error("batched MoE is not configured");
        constexpr uint32_t kMaxChunks = 4096;
        const uint32_t n = (rows + Q38_BATCH_PIPE_ROWS - 1) / Q38_BATCH_PIPE_ROWS;
        if (layer >= Q38_LAYERS_WITH_MTP || !h || !ids || !routes || !out || h_type < 0 || h_type > 2 || rows < 1 || n > kMaxChunks)
            throw std::runtime_error("invalid pipelined batched MoE arguments");
        if (!s.pending_sequences.empty()) throw std::runtime_error("GPU-driven exchange cannot follow a pending async window");
        auto st = reinterpret_cast<cudaStream_t>(stream);
        for (size_t i = 0; i < 24; i++) { s.routes[i].requester_quiesced = false; s.return_routes[i].requester_quiesced = false; }
        s.in_flight = true; s.pex_quiesced = false; s.m10_quiesced = false; s.gpu_used = true;
        const size_t hsz = h_type == 0 ? 4 : 2;
        auto args = [&](uint32_t c) {
            const uint32_t r0 = c * Q38_BATCH_PIPE_ROWS, cnt = std::min<uint32_t>(Q38_BATCH_PIPE_ROWS, rows - r0);
            PciClusterMoeBatchOp::Args a{static_cast<const char *>(h) + size_t(r0) * QWEN_H * hsz, h_type, ids + size_t(r0) * Q38_BATCH_K,
                                         routes + size_t(r0) * Q38_BATCH_K, cnt, out + size_t(r0) * QWEN_H, s.batch_dev, s.gpu_dev,
                                         s.gpu_host_dev, layer};
            a.chunk = c; a.pipe = 1;
            return a;
        };
        auto publish = [&](uint32_t c) {
            q38_pipe_publish<<<kPexGroups, 256, 0, st>>>(s.gpu_dev, args(c));
            CK(cudaGetLastError());
            q38_pipe_doorbell<<<s.pub_groups, 1, 0, st>>>(s.gpu_dev, s.gpu_host_dev, s.pub_aliases, c);
            CK(cudaGetLastError());
        };
        q38_pipe_begin<<<1, 32, 0, st>>>(s.gpu_dev, s.gpu_host_dev, layer, n);
        CK(cudaGetLastError());
        for (uint32_t c = 0; c < std::min<uint32_t>(n, Q38_BATCH_PIPE_SLOTS); c++) publish(c);
        for (uint32_t c = 0; c < n; c++) {
            const auto a = args(c);
            q38_pipe_wait<<<1, 32, 0, st>>>(s.gpu_dev, s.gpu_host_dev, a, s.device_return_slots, s.device_headers, kWaitCycles, 2000000000ull);
            CK(cudaGetLastError());
            q38_batch_combine<<<dim3((QWEN_H / 4 + 255) / 256, a.rows), 256, 0, st>>>(s.gpu_dev, a);
            CK(cudaGetLastError());
            if (c + Q38_BATCH_PIPE_SLOTS < n) publish(c + Q38_BATCH_PIPE_SLOTS);
        }
        s.batch_calls += n; s.batch_rows += rows; s.pipe_calls++;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Per-exchange timing: enable with a ring of `cap` records (the device must be idle: call after a
// synchronize), run exchanges or graph replays, then read (synchronizes, copies, disables, frees).
extern "C" int q38_rtx_peer_timing_enable(uint32_t cap, char *err, size_t errcap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer || !g_peer->gpu_dev) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (s.timing_buf || !cap) throw std::runtime_error("timing already enabled or zero capacity");
        CK(cudaDeviceSynchronize());
        CK(cudaMalloc(&s.timing_buf, size_t(cap) * sizeof(PciClusterTimingRec)));
        CK(cudaMemset(s.timing_buf, 0, size_t(cap) * sizeof(PciClusterTimingRec)));
        NvU32 zero[2] = {0, 0};
        CK(cudaMemcpy(&s.gpu_dev->pad[0], zero, sizeof(zero), cudaMemcpyHostToDevice));
        auto *ptr = static_cast<PciClusterTimingRec *>(s.timing_buf);
        CK(cudaMemcpyToSymbol(g_pc_timing_cap, &cap, sizeof(cap)));
        CK(cudaMemcpyToSymbol(g_pc_timing, &ptr, sizeof(ptr)));
        s.timing_cap = cap;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && errcap) std::snprintf(err, errcap, "%s", g_peer_error.c_str()); return 1;
    }
}
extern "C" int q38_rtx_peer_timing_read(void *host, uint32_t max_records, uint32_t *written, char *err, size_t errcap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer || !g_peer->timing_buf) throw std::runtime_error("timing is not enabled");
        auto &s = *g_peer;
        CK(cudaDeviceSynchronize());
        NvU32 count = 0;
        CK(cudaMemcpy(&count, &s.gpu_dev->pad[1], sizeof(count), cudaMemcpyDeviceToHost));
        // Ring order: if it wrapped, the oldest record sits at count % cap. Return the newest n in order.
        std::vector<PciClusterTimingRec> ring(s.timing_cap);
        CK(cudaMemcpy(ring.data(), s.timing_buf, ring.size() * sizeof(PciClusterTimingRec), cudaMemcpyDeviceToHost));
        const NvU32 held = std::min(count, s.timing_cap);
        const NvU32 oldest = count > s.timing_cap ? count % s.timing_cap : 0;
        const NvU32 n = std::min(held, max_records);
        auto *out = static_cast<PciClusterTimingRec *>(host);
        for (NvU32 i = 0; i < n; i++) out[i] = ring[(oldest + (held - n) + i) % s.timing_cap];
        PciClusterTimingRec *z = nullptr;
        CK(cudaMemcpyToSymbol(g_pc_timing, &z, sizeof(z)));
        CK(cudaFree(s.timing_buf)); s.timing_buf = nullptr; s.timing_cap = 0;
        if (written) *written = n;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && errcap) std::snprintf(err, errcap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Worker-side MoE timing (pci_cluster_exchange.cuh): set ple_reserved[0] bit 0 on every worker and zero
// its 32-KiB ring through the owner BAR mapping. Needs the row-gather configuration (its saved
// ple_reserved words are restored at close). Call with the device idle (after a synchronize).
extern "C" int q38_rtx_peer_worker_timing_enable(char *err, size_t errcap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (s.rg_saved.empty()) throw std::runtime_error("worker timing needs the row-gather configuration (saved words)");
        static_assert(PCI_CLUSTER_WTIMING_OFF + PCI_CLUSTER_WTIMING_RECS * sizeof(PciClusterWorkerTiming) <= kOwnerDiagnosticMapBytes,
                      "worker timing ring outside the owner BAR mapping");
        CK(cudaDeviceSynchronize());
        for (size_t i = 0; i < s.owner_cpus.size(); i++) {
            auto *base = reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]);
            auto *ring = reinterpret_cast<volatile NvU32 *>(base + PCI_CLUSTER_WTIMING_OFF);
            for (size_t w = 0; w < PCI_CLUSTER_WTIMING_RECS * sizeof(PciClusterWorkerTiming) / 4; w++) ring[w] = 0;
            auto *c = reinterpret_cast<volatile PeerControlLayout *>(base);
            c->ple_reserved[0] = s.rg_saved[i].w[0] | 1u;
        }
        asm volatile("sfence" ::: "memory");
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && errcap) std::snprintf(err, errcap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Clear the flag and copy every worker's valid records (seq != 0): out[worker * 512 + slot].
extern "C" int q38_rtx_peer_worker_timing_read(void *out, uint32_t max_records, uint32_t *written, char *err, size_t errcap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        CK(cudaDeviceSynchronize());
        uint32_t n = 0;
        auto *dst = static_cast<PciClusterWorkerTiming *>(out);
        for (size_t i = 0; i < s.owner_cpus.size(); i++) {
            auto *base = reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]);
            reinterpret_cast<volatile PeerControlLayout *>(base)->ple_reserved[0] = s.rg_saved[i].w[0];
            asm volatile("sfence" ::: "memory");
            auto *ring = reinterpret_cast<const volatile PciClusterWorkerTiming *>(base + PCI_CLUSTER_WTIMING_OFF);
            for (NvU32 k = 0; k < PCI_CLUSTER_WTIMING_RECS && n < max_records; k++) {
                PciClusterWorkerTiming r;
                r.seq = ring[k].seq; if (!r.seq) continue;
                r.layer = ring[k].layer; r.role = ring[k].role; r.kind = ring[k].kind | (NvU32(i) << 8);   // worker index in bits 8+
                r.t_seen = ring[k].t_seen; r.t_checked = ring[k].t_checked; r.t_gateup = ring[k].t_gateup;
                r.t_down = ring[k].t_down; r.t_done = ring[k].t_done; r.t_extra = ring[k].t_extra;
                dst[n++] = r;
            }
        }
        if (written) *written = n;
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what(); if (err && errcap) std::snprintf(err, errcap, "%s", g_peer_error.c_str()); return 1;
    }
}
// Non-blocking: the sticky error word lives in mapped host memory. Returns 1 and a description if
// any GPU-driven exchange failed (reason codes at pci_cluster_wait / q38_gpu_fail).
extern "C" int q38_rtx_peer_check_error(char*err,size_t cap) {
    if(!g_peer||!g_peer->gpu_host)return 0;
    volatile PeerState::GpuHost *h=g_peer->gpu_host;
    if(!h->err)return 0;
    if(err&&cap)std::snprintf(err,cap,"GPU-driven exchange failed: reason %u, layer %u, seq %u, worker %u, status %u, exit 0x%x",
                              h->err_reason,h->err_layer,h->err_seq,h->err_worker,h->err_status,h->err_exit);
    return 1;
}
// Ask every spinning reducer to give up (it records reason 3). Safe from any thread.
extern "C" void q38_rtx_peer_abort(void) {
    if(g_peer&&g_peer->gpu_host){static_cast<volatile PeerState::GpuHost*>(g_peer->gpu_host)->abort=1;asm volatile("sfence" ::: "memory");}
}
// Counters for reporting: GPU-driven calls and dummy skips (device copy; synchronous).
extern "C" int q38_rtx_peer_gpu_counters(uint32_t *calls,uint32_t *skips) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    if(!g_peer||!g_peer->gpu_dev)return 1;
    PeerState::GpuDev d{}; if(cudaMemcpy(&d,g_peer->gpu_dev,sizeof(d),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
    *calls=d.calls;*skips=d.skips;return 0;
}
// Queue one layer's request and reducer on the caller's stream without
// copying status to the host or waiting for completion. The returned tensor is
// therefore stream-ordered work; callers must keep its inputs/outputs alive
// under the normal CUDA stream lifetime rules and flush before teardown.
extern "C" int q38_rtx_peer_forward_async(uint32_t layer,const void*h,int h_type,const int32_t*ids,
        const float*routes,float*out,void*stream,char*err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if(!g_peer)throw std::runtime_error("RTX peer client is not open");
        if(layer>=Q38_LAYERS_WITH_MTP||!h||!ids||!routes||!out||h_type<0||h_type>2)
            throw std::runtime_error("invalid asynchronous RTX peer forward arguments");
        auto &s=*g_peer;
        auto st=reinterpret_cast<cudaStream_t>(stream);
        if(s.pending_sequences.size()>=kAsyncStatusSlots)
            throw std::runtime_error("asynchronous peer status ring is full; flush before enqueue");
        if(!s.pending_sequences.empty() && s.pending_stream!=st)
            throw std::runtime_error("asynchronous peer window crossed CUDA streams");

        // PLE can advance the same resident-worker command sequence. Sample
        // BAR1 only at a window boundary; sequences inside a window are owned
        // by this queued RTX stream and are not polled by the CPU.
        NvU32 previous=s.sequence.empty()?0:s.sequence[0];
        if(s.pending_sequences.empty()) {
            previous=*reinterpret_cast<volatile NvU32*>(s.owner_cpus[0]);
            for(size_t i=1;i<s.owner_cpus.size();i++) {
                NvU32 current=*reinterpret_cast<volatile NvU32*>(s.owner_cpus[i]);
                if(current!=previous)throw std::runtime_error("resident worker sequences diverged at async window start");
            }
            std::fill(s.sequence.begin(),s.sequence.end(),previous);
            s.pending_stream=st;
        }
        if(previous>=UINT32_MAX-1)throw std::runtime_error("resident sequence space exhausted");
        const NvU32 seq=previous+1;
        const NvU32 status_slot=seq%kAsyncStatusSlots;
        auto *status=s.device_status+status_slot;
        for(size_t i=0;i<s.routes.size();i++) {
            s.routes[i].requester_quiesced=false;
            s.return_routes[i].requester_quiesced=false;
        }
        s.pex_quiesced=false;
        s.in_flight=true;
        // Record ownership before issuing asynchronous work so any partial
        // enqueue failure remains visible to flush/teardown safety checks.
        s.pending_sequences.push_back(seq);
        std::fill(s.sequence.begin(),s.sequence.end(),seq);
        CK(cudaMemsetAsync(status,0xff,sizeof(*status),st));
        CK(cudaMemsetAsync(s.device_headers,0xff,(size_t(24)*4+2)*sizeof(NvU32),st));
        enqueue_peer_request(s,st,h,h_type,ids,routes,layer,seq);
        rtx_peer_reduce<<<1,256,0,st>>>(s.device_return_slots,24,seq,layer,out,status,
            s.device_headers,kWaitCycles);
        CK(cudaGetLastError());
        if(err&&cap)err[0]='\0';
        return 0;
    } catch(const std::exception &e) {
        g_peer_error=e.what();
        if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());
        return 1;
    }
}

// Complete one queued window and check all per-request status words with one
// batched device-to-host read. A failed synchronization leaves the mappings
// marked in flight, so callers must retain resources and recover safely.
extern "C" int q38_rtx_peer_async_flush(void *stream,char *err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if(!g_peer)throw std::runtime_error("RTX peer client is not open");
        auto &s=*g_peer;
        if(s.pending_sequences.empty()) { if(err&&cap)err[0]='\0'; return 0; }
        auto st=reinterpret_cast<cudaStream_t>(stream);
        if(st!=s.pending_stream)throw std::runtime_error("async flush used a different CUDA stream than its window");
        CUKQ(cuCtxSetCurrent(s.rtx_context));
        CK(cudaSetDevice(s.rtx_ordinal));
        CK(cudaStreamSynchronize(st));
        // The requester work has now stopped, even if a device status reports
        // a protocol error; safe route restoration is permitted after checks.
        for(auto &route:s.routes)route.requester_quiesced=true;
        for(auto &route:s.return_routes)route.requester_quiesced=true;
        s.in_flight=false;
        s.pex_quiesced=true;
        std::vector<NvU32> all_status(kAsyncStatusSlots);
        CK(cudaMemcpy(all_status.data(),s.device_status,
            all_status.size()*sizeof(NvU32),cudaMemcpyDeviceToHost));
        NvU32 failed_seq=0,failed_status=0;
        for(NvU32 seq:s.pending_sequences) {
            NvU32 status=all_status[seq%kAsyncStatusSlots];
            if(status!=0 && !failed_seq) { failed_seq=seq; failed_status=status; }
        }
        const size_t completed=s.pending_sequences.size();
        s.pending_sequences.clear();
        s.pending_stream=nullptr;
        if(failed_seq) {
            std::vector<NvU32> observed(size_t(24)*4);
            CK(cudaMemcpy(observed.data(),s.device_headers,observed.size()*sizeof(NvU32),cudaMemcpyDeviceToHost));
            std::fprintf(stderr,"RTX_PEER_ASYNC_FAILURE,sequence,%u,status,%u,window_requests,%zu\n",
                failed_seq,failed_status,completed);
            for(size_t i=0;i<24;i++)
                std::fprintf(stderr,"RTX_PEER_ASYNC_WORKER,index,%zu,bdf,%s,header,%u,%u,%u,%u\n",
                    i,s.workers[i].bdf.c_str(),observed[i*4],observed[i*4+1],
                    observed[i*4+2],observed[i*4+3]);
            throw std::runtime_error("asynchronous RTX peer window reported a failed worker status");
        }
        if(err&&cap)err[0]='\0';
        return 0;
    } catch(const std::exception &e) {
        g_peer_error=e.what();
        if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());
        return 1;
    }
}
extern "C" int q38_rtx_peer_pex_word_probe(NvU32 marker, NvU32 *before,
        NvU32 *observed, NvU32 *restored, char *err, size_t cap) {
    // Use the known, allocated control word rather than an uninitialized
    // mailbox-tail location. The resident workers are parked for this probe.
    constexpr NvU32 kProbeOffset = offsetof(PeerControlLayout, reserved);
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.pex_multicast || !s.ingress_diagnostic)
            throw std::runtime_error("one-word probe requires the PEX ingress diagnostic mapping");
        if (s.in_flight || !s.pex_quiesced || !s.pending_sequences.empty())
            throw std::runtime_error("peer routes are not quiescent for the one-word probe");
        if (s.workers.size() < 4 || s.owner_cpus.size() < 4 || !before || !observed || !restored)
            throw std::runtime_error("one-word probe is missing the first M10's four owner mappings");

        for (NvU32 i = 0; i < 4; ++i) {
            auto *word = reinterpret_cast<volatile NvU32 *>(
                reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]) + kProbeOffset);
            before[i] = *word;
        }
        if (marker == before[0] || marker == before[1] || marker == before[2] || marker == before[3])
            throw std::runtime_error("probe marker collides with an existing mailbox-tail word");

        CUKQ(cuCtxSetCurrent(s.rtx_context));
        CK(cudaSetDevice(s.rtx_ordinal));
        s.in_flight = true;
        s.pex_quiesced = false;
        rtx_peer_pex_word_store<<<1, 1>>>(s.device_pex_aliases, kProbeOffset, marker);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        // Synchronization confirms the single RTX store has completed; only
        // now may CPU BAR reads or the regular teardown path proceed.
        s.in_flight = false;
        s.pex_quiesced = true;

        NvU32 matched = 0;
        for (NvU32 i = 0; i < 4; ++i) {
            auto *word = reinterpret_cast<volatile NvU32 *>(
                reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]) + kProbeOffset);
            observed[i] = *word;
            matched += observed[i] == marker;
            std::fprintf(stderr,
                "RTX_PEX_WORD_PROBE,bdf,%s,offset,0x%x,before,0x%08x,marker,0x%08x,observed,0x%08x,match,%u\n",
                s.workers[i].bdf.c_str(), kProbeOffset, before[i], marker,
                observed[i], observed[i] == marker ? 1u : 0u);
        }

        // Restore each original word through its owner's CPU BAR mapping.
        // This is outside the measured one-store action and is verified below.
        for (NvU32 i = 0; i < 4; ++i) {
            auto *word = reinterpret_cast<volatile NvU32 *>(
                reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]) + kProbeOffset);
            *word = before[i];
        }
        asm volatile("sfence" ::: "memory");
        NvU32 restored_count = 0;
        for (NvU32 i = 0; i < 4; ++i) {
            auto *word = reinterpret_cast<volatile NvU32 *>(
                reinterpret_cast<volatile NvU8 *>(s.owner_cpus[i]) + kProbeOffset);
            restored[i] = *word;
            restored_count += restored[i] == before[i];
            std::fprintf(stderr,
                "RTX_PEX_WORD_RESTORE,bdf,%s,offset,0x%x,expected,0x%08x,observed,0x%08x,match,%u\n",
                s.workers[i].bdf.c_str(), kProbeOffset, before[i], restored[i],
                restored[i] == before[i] ? 1u : 0u);
        }
        std::fflush(stderr);
        if (restored_count != 4)
            throw std::runtime_error("one-word probe could not restore every original BAR word");
        if (matched != 4)
            throw std::runtime_error("RTX PEX single store was not observed by all four M10-local GPU BAR mappings");
        if (err && cap) err[0] = '\0';
        return 0;
    } catch (const std::exception &e) {
        g_peer_error = e.what();
        if (err && cap) std::snprintf(err, cap, "%s", g_peer_error.c_str());
        return 1;
    }
}
extern "C" int q38_rtx_peer_direct_word_probe(NvU32 marker, NvU32 *before,
        NvU32 *observed, NvU32 *restored, char *err, size_t cap) {
    constexpr NvU32 kProbeOffset = offsetof(PeerControlLayout, reserved);
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if (!g_peer) throw std::runtime_error("RTX peer client is not open");
        auto &s = *g_peer;
        if (!s.direct_unicast_diagnostic || !s.ingress_diagnostic ||
            !s.pex_multicast || s.pex_routes.size() != 1)
            throw std::runtime_error("direct one-word probe requires one direct RTX→M10 BAR1 route");
        if (s.in_flight || !s.pex_quiesced || !s.pending_sequences.empty() ||
            s.owner_cpus.empty() || !before || !observed || !restored)
            throw std::runtime_error("direct RTX→M10 route is not quiescent or lacks its owner BAR mapping");
        auto *word = reinterpret_cast<volatile NvU32 *>(
            reinterpret_cast<volatile NvU8 *>(s.owner_cpus[0]) + kProbeOffset);
        before[0] = *word;
        if (marker == before[0]) throw std::runtime_error("direct probe marker collides with saved owner word");

        CUKQ(cuCtxSetCurrent(s.rtx_context));
        CK(cudaSetDevice(s.rtx_ordinal));
        s.in_flight = true;
        s.pex_quiesced = false;
        rtx_peer_pex_word_store<<<1, 1>>>(s.device_pex_aliases, kProbeOffset, marker);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        s.in_flight = false;
        s.pex_quiesced = true;

        observed[0] = *word;
        std::fprintf(stderr,
            "RTX_DIRECT_WORD_PROBE,bdf,%s,owner_bar1_bus,0x%llx,offset,0x%x,before,0x%08x,marker,0x%08x,observed,0x%08x,match,%u\n",
            s.workers[0].bdf.c_str(),(unsigned long long)s.workers[0].bar_bus,
            kProbeOffset,before[0],marker,observed[0],observed[0]==marker?1u:0u);
        *word = before[0];
        asm volatile("sfence" ::: "memory");
        restored[0] = *word;
        std::fprintf(stderr,
            "RTX_DIRECT_WORD_RESTORE,bdf,%s,expected,0x%08x,observed,0x%08x,match,%u\n",
            s.workers[0].bdf.c_str(),before[0],restored[0],restored[0]==before[0]?1u:0u);
        std::fflush(stderr);
        if (observed[0] != marker || restored[0] != before[0])
            throw std::runtime_error("direct RTX→M10 BAR1 word store or restoration failed");
        if (err && cap) err[0] = '\0';
        return 0;
    }
    catch(const std::exception &e) {
        g_peer_error=e.what();
        if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());
        return 1;
    }
}
extern "C" int q38_rtx_peer_mappings_quiesced(void) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    return !g_peer || (!g_peer->in_flight && g_peer->pex_quiesced);
}
extern "C" int q38_rtx_peer_close(char *err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try{peer_close();q38_journal_line("CLOSED");g_peer_error.clear();return 0;}
    catch(const std::exception&e){g_peer_error=e.what();if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());return 1;}
}
extern "C" int q38_rtx_peer_mark_m10_quiesced(char *err,size_t cap) {
    std::lock_guard<std::mutex> lock(g_peer_mutex);
    try {
        if(!g_peer)throw std::runtime_error("RTX peer client is not open");
        for(auto &route:g_peer->return_routes)route.requester_quiesced=true;
        g_peer->m10_quiesced=true;
        if(err&&cap)err[0]='\0';
        return 0;
    } catch(const std::exception&e) {
        g_peer_error=e.what();if(err&&cap)std::snprintf(err,cap,"%s",g_peer_error.c_str());return 1;
    }
}
