// Compare CPU and RTX publication into the already-active six card-local
// PEX8747 multicast windows. The destination is a 12-KiB scratch range beyond
// the daemon's documented mailbox fields in each retained worker allocation.
// The route wrapper may temporarily clear ACS redirects on the relevant outer
// switch branches; the multicast-engine configuration and model weights stay
// untouched, and the wrapper restores the saved ACS words after safe teardown.
#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <emmintrin.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <nvstatus.h>
#include <uvm_linux_ioctl.h>
}

namespace {
constexpr uint64_t kPage = 4096;
constexpr uint64_t kWindow = 64 * 1024;
constexpr uint32_t kScratchOffset = 0xc000;
constexpr uint32_t kLogicalBytes = 2560u * 4u + 10u * 4u + 10u * 4u;
constexpr uint32_t kTransferBytes = 3u * kPage;
constexpr uint32_t kWords = kTransferBytes / 4;
constexpr uint32_t kHElements = 2560;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kMaxGroups = 6;
// The RTX PCI address is never hardcoded: Q38_RTX_BDF (from qwen38_pci_discover.py) is required.
static const char *rtx_bdf_from_env() {
    const char *v = std::getenv("Q38_RTX_BDF");
    if (!v || !*v) throw std::runtime_error("Q38_RTX_BDF is not set; run qwen38_pci_discover.py --env first");
    return v;
}

struct Group {
    std::string serial, upstream;
    std::vector<std::string> ports, gpus;
    uint64_t bridge_lo = 0, bridge_hi = 0, ingress = 0;
};
struct WorkerMap {
    int group = -1, worker = -1;
    std::string bdf;
    int ordinal = -1;
    uint64_t cuda_va = 0, bar1_bus = 0;
};
struct Receiver {
    WorkerMap identity;
    int fd = -1;
    uint8_t *mapping = reinterpret_cast<uint8_t *>(MAP_FAILED);
    std::vector<uint32_t> saved;
};
struct PteLeaf {
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS query{};
    uint64_t destination = 0, candidate = 0;
    UVM_GLM53_MULTI_PTE_REMAP_PARAMS active{};
    bool acquired = false;
};

static void require(bool ok, const std::string &why) {
    if (!ok) throw std::runtime_error(why);
}
static void ck(cudaError_t e, const char *where) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
#define CK(x) ck((x), #x)
static void cuk(CUresult e, const char *where) {
    if (e == CUDA_SUCCESS) return;
    const char *name = nullptr, *desc = nullptr;
    cuGetErrorName(e, &name); cuGetErrorString(e, &desc);
    throw std::runtime_error(std::string(where) + ": " + (name ? name : "CUDA") +
                             ": " + (desc ? desc : "unknown"));
}
#define CUK(x) cuk((x), #x)
static std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out; std::stringstream ss(s); std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
}
static uint64_t parse_u64(const std::string &s) {
    return std::strtoull(s.c_str(), nullptr, 0);
}
static std::vector<Group> read_groups(const char *path, uint32_t limit) {
    std::ifstream f(path); require(bool(f), "cannot open active group TSV");
    std::vector<Group> out; std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto c = split(line, '\t'); require(c.size() == 7, "malformed group TSV row");
        Group g{c[0], c[1], split(c[2], ','), split(c[3], ','),
                parse_u64(c[4]), parse_u64(c[5]), parse_u64(c[6])};
        require(g.ports.size() == 4 && g.gpus.size() == 4, "group is not four-GPU M10");
        require(g.ingress % kWindow == 0 && g.ingress >= g.bridge_lo &&
                g.ingress + kWindow - 1 <= g.bridge_hi, "active PEX ingress not contained/aligned");
        out.push_back(std::move(g));
    }
    require(out.size() >= limit && limit >= 1 && limit <= kMaxGroups,
            "active group manifest does not contain requested 1..6 groups");
    out.resize(limit); return out;
}
static std::vector<WorkerMap> read_workers(const char *path, const std::vector<Group> &groups) {
    std::ifstream f(path); require(bool(f), "cannot open resident GPU mapping manifest");
    std::string line; std::vector<WorkerMap> out;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line, '\t');
        if (c.size() != 7 || c[0] == "group") continue;
        WorkerMap w; w.group = std::stoi(c[0]); w.worker = std::stoi(c[1]);
        w.bdf = c[2]; w.ordinal = std::stoi(c[3]); w.cuda_va = parse_u64(c[5]);
        w.bar1_bus = parse_u64(c[6]);
        if (w.group < (int)groups.size()) out.push_back(std::move(w));
    }
    require(out.size() == groups.size() * 4, "mapping manifest must identify all four workers per group");
    for (const auto &w : out) {
        require(w.group >= 0 && w.group < (int)groups.size() && w.worker >= 0 && w.worker < 4,
                "invalid worker group/index");
        require(w.bdf == groups[w.group].gpus[w.worker], "worker BDF differs from active PEX group");
        require(w.bar1_bus % kWindow == 0, "worker BAR1 alias is not 64-KiB aligned");
    }
    return out;
}
static std::vector<uint64_t> read_iovas(const char *dir, uint32_t groups) {
    std::vector<uint64_t> out;
    for (uint32_t i = 0; i < groups; ++i) {
        std::ifstream f(std::string(dir) + "/iova" + std::to_string(i));
        require(bool(f), "cannot read live DMA IOVA for group " + std::to_string(i));
        std::string s; f >> s; uint64_t a = parse_u64(s);
        require(a && !(a & (kWindow - 1)), "DMA IOVA is not valid/64-KiB aligned");
        out.push_back(a);
    }
    return out;
}
static uint64_t read_host_iova(const char *dir) {
    std::ifstream f(std::string(dir) + "/host_iova");
    require(bool(f), "cannot read disposable coherent-buffer DMA address");
    std::string text; f >> text;
    const uint64_t value = parse_u64(text);
    require(value && !(value & (kPage - 1)), "coherent-buffer DMA address is invalid/unaligned");
    return value;
}
static uint64_t read_peer_iova(const char *dir) {
    std::ifstream f(std::string(dir) + "/peer_iova");
    require(bool(f), "cannot read direct-M10 BAR DMA address");
    std::string text; f >> text;
    const uint64_t value = parse_u64(text);
    require(value && !(value & (kWindow - 1)), "direct-M10 DMA address is invalid/64-KiB unaligned");
    return value;
}
static uint64_t bar1_resource_base(const std::string &bdf) {
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/resource");
    require(bool(f), "cannot read PCI resources for " + bdf);
    std::string row; require(bool(std::getline(f, row)) && bool(std::getline(f, row)),
                             "GPU BAR1 resource row missing for " + bdf);
    std::stringstream ss(row); uint64_t start = 0, end = 0, flags = 0;
    ss >> std::hex >> start >> end >> flags;
    require(bool(ss) && start && end >= start && (flags & 0x200),
            "GPU BAR1 resource invalid for " + bdf);
    return start;
}
static std::vector<Receiver> map_receivers(const std::vector<WorkerMap> &workers) {
    std::vector<Receiver> out; out.reserve(workers.size());
    for (const auto &w : workers) {
        Receiver r; r.identity = w;
        const uint64_t base = bar1_resource_base(w.bdf);
        require(w.bar1_bus >= base, "RM BAR1 bus base precedes PCI resource for " + w.bdf);
        const uint64_t offset = w.bar1_bus - base + kScratchOffset;
        require(!(offset & (kPage - 1)) && offset + kTransferBytes <= (1ull << 28),
                "BAR1 scratch interval invalid for " + w.bdf);
        const std::string p = "/sys/bus/pci/devices/" + w.bdf + "/resource1";
        r.fd = open(p.c_str(), O_RDWR | O_SYNC | O_CLOEXEC);
        require(r.fd >= 0, "open " + p + ": " + strerror(errno));
        r.mapping = static_cast<uint8_t *>(mmap(nullptr, kTransferBytes, PROT_READ | PROT_WRITE,
                                                MAP_SHARED, r.fd, (off_t)offset));
        require(r.mapping != MAP_FAILED, "mmap BAR1 " + w.bdf + ": " + strerror(errno));
        r.saved.resize(kWords);
        auto *mmio = reinterpret_cast<volatile uint32_t *>(r.mapping);
        for (uint32_t j = 0; j < kWords; ++j) r.saved[j] = mmio[j];
        std::printf("RECEIVER_MAP,group,%d,worker,%d,bdf,%s,cuda_va,0x%llx,bar1_bus,0x%llx,bar1_cpu_base,0x%llx,bar1_offset,0x%llx,cpu_mmio_va,%p,scratch_offset,0x%x,length,%u\n",
                    w.group, w.worker, w.bdf.c_str(), (unsigned long long)w.cuda_va,
                    (unsigned long long)w.bar1_bus, (unsigned long long)base,
                    (unsigned long long)offset, r.mapping, kScratchOffset, kTransferBytes);
        out.push_back(std::move(r));
    }
    return out;
}
static void restore_and_close_receivers(std::vector<Receiver> &receivers) {
    for (auto &r : receivers) {
        if (r.mapping != MAP_FAILED && !r.saved.empty()) {
            auto *mmio = reinterpret_cast<volatile uint32_t *>(r.mapping);
            for (uint32_t j = 0; j < kWords; ++j) mmio[j] = r.saved[j];
            asm volatile("sfence" ::: "memory");
            bool ok = true;
            for (uint32_t j = 0; j < kWords; ++j) if (mmio[j] != r.saved[j]) { ok = false; break; }
            if (!ok) {
                std::fprintf(stderr, "PRESERVE,receiver scratch restoration failed for %s; retain mappings\n",
                             r.identity.bdf.c_str());
                for (;;) sleep(60);
            }
        }
    }
    for (auto &r : receivers) {
        if (r.mapping != MAP_FAILED) {
            munmap(r.mapping, kTransferBytes);
            r.mapping = reinterpret_cast<uint8_t *>(MAP_FAILED);
        }
        if (r.fd >= 0) { close(r.fd); r.fd = -1; }
    }
    std::puts("RECEIVER_SCRATCH_RESTORE,PASS,all_saved_words_match");
}
static std::vector<uint32_t> make_payload(uint32_t seq) {
    std::vector<uint32_t> w(kWords);
    auto bits = [](float x) { uint32_t b; std::memcpy(&b, &x, sizeof(b)); return b; };
    for (uint32_t i = 0; i < kHElements; ++i)
        w[i] = bits(float((i * 7u + seq * 3u) % 97u) * 0.125f);
    for (uint32_t k = 0; k < kTopK; ++k)
        w[kHElements + k] = 3u + ((k * 17u + seq) % 128u);
    for (uint32_t k = 0; k < kTopK; ++k)
        w[kHElements + kTopK + k] = bits(float((k + 1u) * (seq + 1u)) * 0.0625f);
    for (uint32_t i = kLogicalBytes / 4; i < kWords; ++i)
        w[i] = 0x6d100000u ^ (seq * 0x00010101u) ^ (i * 0x9e3779b9u);
    return w;
}
static void compare_receivers(const std::vector<Receiver> &receivers,
                             const std::vector<uint32_t> &expected,
                             const char *path, uint32_t seq) {
    uint64_t bad = 0;
    for (const auto &r : receivers) {
        auto *mmio = reinterpret_cast<volatile const uint32_t *>(r.mapping);
        uint32_t local = 0, first = UINT32_MAX;
        for (uint32_t j = 0; j < kWords; ++j) {
            const uint32_t got = mmio[j];
            if (got != expected[j]) { if (first == UINT32_MAX) first = j; ++local; }
        }
        bad += local;
        std::printf("VERIFY,mode,%s,seq,%u,group,%d,worker,%d,bdf,%s,mismatches,%u,first_mismatch,%s,word0_expected,0x%08x,word0_observed,0x%08x,word1_observed,0x%08x,last_observed,0x%08x\n",
                    path, seq, r.identity.group, r.identity.worker, r.identity.bdf.c_str(), local,
                    local ? std::to_string(first).c_str() : "none", expected[0], mmio[0],
                    mmio[1], mmio[kWords - 1]);
    }
    require(!bad, std::string(path) + " payload mismatch, total=" + std::to_string(bad));
}
static void cpu_publish(const std::vector<Group> &groups,
                        const std::vector<uint32_t> &payload, uint32_t seq) {
    int fd = open("/dev/m10_qwen_mc_wc6", O_RDWR | O_CLOEXEC);
    require(fd >= 0, std::string("open CPU multicast WC mapper: ") + strerror(errno));
    std::vector<void *> maps(groups.size(), MAP_FAILED);
    try {
        for (size_t g = 0; g < groups.size(); ++g) {
            maps[g] = mmap(nullptr, kWindow, PROT_READ | PROT_WRITE, MAP_SHARED, fd, g * kWindow);
            require(maps[g] != MAP_FAILED, "map CPU PEX multicast ingress group " + std::to_string(g));
        }
        auto t0 = std::chrono::steady_clock::now();
        for (size_t g = 0; g < groups.size(); ++g) {
            auto *dst = reinterpret_cast<volatile uint32_t *>(static_cast<uint8_t *>(maps[g]) + kScratchOffset);
            for (uint32_t i = 0; i < kWords; ++i) dst[i] = payload[i];
        }
        asm volatile("sfence" ::: "memory");
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        auto t1 = std::chrono::steady_clock::now();
        std::printf("TX_TIME,mode,CPU,seq,%u,groups,%zu,per_group_bytes,%u,total_injected_bytes,%zu,store_fence_us,%.3f\n",
                    seq, groups.size(), kTransferBytes, groups.size() * size_t(kTransferBytes),
                    std::chrono::duration<double, std::micro>(t1-t0).count());
    } catch (...) {
        for (void *p : maps) if (p != MAP_FAILED) munmap(p, kWindow);
        close(fd); throw;
    }
    for (void *p : maps) if (p != MAP_FAILED) munmap(p, kWindow);
    close(fd);
}

static UVM_GLM53_VA_SPACE_PDB_ENTRY find_pdb(int fd, const NvProcessorUuid &uuid,
                                             uint64_t probe_va) {
    // The matching diagnostic UVM ioctl caps snapshots at 128 entries.
    std::vector<UVM_GLM53_VA_SPACE_PDB_ENTRY> entries(128);
    UVM_GLM53_LIST_VA_SPACE_PDBS_PARAMS p{};
    p.entry_capacity = entries.size(); p.entries = (NvU64)entries.data();
    errno = 0;
    const int rc = ioctl(fd, UVM_GLM53_LIST_VA_SPACE_PDBS, &p);
    const int saved_errno = errno;
    std::fprintf(stderr,
        "RTX_PDB_LIST,rc,%d,errno,%d,rm,0x%08x,capacity,%u,count,%u,matched,%u,total,%u,overflow,%u,pid,%u\n",
        rc, saved_errno, (unsigned)p.rmStatus, p.entry_capacity, p.entry_count,
        p.matched_va_spaces, p.total_va_spaces, p.overflow, (unsigned)getpid());
    if (rc != 0 || p.rmStatus != NV_OK)
        throw std::runtime_error("UVM PDB list ioctl/RM failure; see RTX_PDB_LIST record");
    UVM_GLM53_VA_SPACE_PDB_ENTRY selected{};
    uint32_t matches = 0;
    for (NvU32 i = 0; i < p.entry_count; ++i) {
        const auto &e = entries[i];
        if (std::memcmp(&e.gpu_uuid, &uuid, sizeof(uuid)) || !e.did_set_page_directory ||
            !e.va_space_ptr || !e.gpu_va_space_ptr || !e.root_phys_address)
            continue;
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q{};
        q.requester_uuid = uuid;
        q.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
        q.va_space_ptr = e.va_space_ptr;
        q.gpu_va_space_ptr = e.gpu_va_space_ptr;
        q.root_phys_address = e.root_phys_address;
        q.root_phys_aperture = e.root_phys_aperture;
        q.target_va = probe_va;
        q.page_size = kPage;
        errno = 0;
        const int query_rc = ioctl(fd, UVM_GLM53_SERIAL_BAR1_PEER_READ, &q);
        const int query_errno = errno;
        const bool valid = query_rc == 0 && q.rmStatus == NV_OK && q.target_pte_phys &&
                           (q.pte_after & 1);
        std::fprintf(stderr,
            "RTX_PDB_CANDIDATE,index,%u,va_space,0x%llx,gpu_va_space,0x%llx,root,0x%llx,query_rc,%d,errno,%d,rm,0x%08x,pte_phys,0x%llx,pte,0x%llx,valid,%u\n",
            i, (unsigned long long)e.va_space_ptr, (unsigned long long)e.gpu_va_space_ptr,
            (unsigned long long)e.root_phys_address, query_rc, query_errno,
            (unsigned)q.rmStatus, (unsigned long long)q.target_pte_phys,
            (unsigned long long)q.pte_after, valid);
        if (valid) { selected = e; ++matches; }
    }
    require(matches == 1, "RTX alias PTE resolves in " + std::to_string(matches) +
                          " candidate page-directory roots; expected exactly one");
    std::printf("RTX_PDB_SELECTED,va,0x%llx,va_space,0x%llx,gpu_va_space,0x%llx,root,0x%llx\n",
                (unsigned long long)probe_va, (unsigned long long)selected.va_space_ptr,
                (unsigned long long)selected.gpu_va_space_ptr,
                (unsigned long long)selected.root_phys_address);
    return selected;
}
static UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS query_leaf(
        int fd, const NvProcessorUuid &uuid, const UVM_GLM53_VA_SPACE_PDB_ENTRY &pdb,
        uint64_t va) {
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q{};
    q.requester_uuid = uuid; q.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
    q.va_space_ptr = pdb.va_space_ptr; q.gpu_va_space_ptr = pdb.gpu_va_space_ptr;
    q.root_phys_address = pdb.root_phys_address; q.root_phys_aperture = pdb.root_phys_aperture;
    q.target_va = va; q.page_size = kPage;
    if (ioctl(fd, UVM_GLM53_SERIAL_BAR1_PEER_READ, &q) || q.rmStatus != NV_OK ||
        !q.target_pte_phys || !(q.pte_after & 1))
        throw std::runtime_error("requester PTE query failed at " + std::to_string(va) +
                                 ", errno=" + std::to_string(errno) + ", RM=" +
                                 std::to_string((unsigned)q.rmStatus));
    return q;
}
static uint64_t candidate_pte(uint64_t destination) {
    // Exact 580.178.04 RTX/Ampere UVM HAL encoding:
    //   Ampere make_pte is inherited from make_pte_turing() in
    //   nvidia-uvm/uvm_ampere_mmu.c -> uvm_turing_mmu.c. For SYS,
    //   READ_WRITE, FLAGS_NONE it sets VALID, coherent SYS aperture, VOL,
    //   ATOMIC_DISABLE, generic-memory KIND, and ADDRESS_SYS. The Turing
    //   dev_mmu.h fields are VALID[0], APERTURE[2:1], VOL[3],
    //   ATOMIC_DISABLE[7], ADDRESS_SYS[53:8] (address shift 12), KIND[63:56].
    // The live guarded UVM remap validates exact equality against this same
    // active RTX HAL before it can write any PTE.
    constexpr uint64_t PTE_ADDRESS_SYS_MASK = 0x003fffffffffff00ULL; // bits 53:8
    constexpr uint64_t PTE_VALID = 1ULL;
    constexpr uint64_t PTE_SYS_COHERENT = 2ULL << 1;
    constexpr uint64_t PTE_VOL = 1ULL << 3;             // FLAGS_NONE
    constexpr uint64_t PTE_ATOMIC_DISABLE = 1ULL << 7;  // READ_WRITE, not ATOMIC
    constexpr uint64_t PTE_KIND_GENERIC_MEMORY = 0x06ULL << 56;
    require((destination & (kPage - 1)) == 0, "SYS PTE destination is unaligned");
    const uint64_t address = (destination >> 12) << 8;
    require((address & ~PTE_ADDRESS_SYS_MASK) == 0 &&
            ((address >> 8) << 12) == destination,
            "SYS destination exceeds the RTX PTE address field");
    const uint64_t pte = PTE_VALID | PTE_SYS_COHERENT | PTE_VOL |
                         PTE_ATOMIC_DISABLE | PTE_KIND_GENERIC_MEMORY | address;
    return pte;
}
static PteLeaf install_one_pte(int fd, const NvProcessorUuid &uuid,
                               const UVM_GLM53_VA_SPACE_PDB_ENTRY &pdb,
                               uint64_t va, uint64_t destination,
                               const char *label) {
    PteLeaf l; l.destination = destination;
    require(!(va & (kPage - 1)) && !(destination & (kPage - 1)),
            "unaligned requester alias/destination");
    l.query = query_leaf(fd, uuid, pdb, va);
    l.candidate = candidate_pte(destination);
    std::printf("RTX_PTE_CANDIDATE,kind,%s,method,installed_580.178.04_GA104_ampere_hal_inherited_turing_make_pte,va,0x%llx,original,0x%llx,destination_address,0x%llx,candidate,0x%llx\n",
                label, (unsigned long long)va, (unsigned long long)l.query.pte_after,
                (unsigned long long)destination, (unsigned long long)l.candidate);
    UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
    p.gpu_uuid = uuid; p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_APPLY;
    p.root_phys_aperture = pdb.root_phys_aperture; p.va_space_ptr = pdb.va_space_ptr;
    p.gpu_va_space_ptr = pdb.gpu_va_space_ptr; p.root_phys_address = pdb.root_phys_address;
    p.target_va = va; p.page_size = kPage; p.target_pte_phys = l.query.target_pte_phys;
    p.expected_current_pte = l.query.pte_after; p.requested_temporary_pte = l.candidate;
    p.requested_destination_gpu_sys_address = destination; p.requested_allow_writable = 1;
    errno = 0;
    const int rc = ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &p); const int e = errno;
    if (rc == 0 && p.rmStatus == NV_OK) { l.active = p; l.acquired = true; }
    std::printf("RTX_PTE_INSTALL,kind,%s,rc,%d,errno,%d,rm,0x%08x,lookup,0x%08x,read,0x%08x,validate,0x%08x,l2,0x%08x,tlb,0x%08x,root_ap,%u,va,0x%llx,original,0x%llx,destination_address,0x%llx,candidate,0x%llx,pte_before,0x%llx,after,0x%llx,acquired,%u\n",
                label, rc, e, (unsigned)p.rmStatus, (unsigned)p.lookup_status,
                (unsigned)p.read_status, (unsigned)p.pte_validation_status,
                (unsigned)p.requester_l2_invalidate_status,
                (unsigned)p.requester_phys_tlb_invalidate_status,
                pdb.root_phys_aperture, (unsigned long long)va,
                (unsigned long long)p.pte_before, (unsigned long long)destination,
                (unsigned long long)l.candidate, (unsigned long long)p.pte_before,
                (unsigned long long)p.pte_after, l.acquired);
    if (l.acquired && p.pte_after != l.candidate) {
        UVM_GLM53_MULTI_PTE_REMAP_PARAMS restore{};
        restore.op = UVM_GLM53_MULTI_PTE_REMAP_OP_RESTORE;
        restore.mapping_id = l.active.mapping_id;
        restore.target_va = l.active.target_va;
        restore.target_pte_phys = l.active.target_pte_phys;
        const int restore_rc = ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &restore);
        if (restore_rc != 0 || restore.rmStatus != NV_OK || !restore.pte_restored ||
            restore.pte_after != l.query.pte_after) {
            std::fprintf(stderr, "PRESERVE,unexpected successful PTE acquisition failed metadata check and immediate exact restore\n");
            for (;;) sleep(60);
        }
        l.acquired = false;
    }
    require(l.acquired && p.pte_after == l.candidate,
            "guarded requester PTE install failed exact candidate check");
    return l;
}
static void install_pte_aliases(int fd, const NvProcessorUuid &uuid,
                                const UVM_GLM53_VA_SPACE_PDB_ENTRY &pdb,
                                uint64_t alias_base, const std::vector<uint64_t> &iovas,
                                const std::vector<Group> &group_info, bool use_bus_address,
                                uint32_t group_count, std::vector<PteLeaf> &leaves) {
    leaves.reserve(group_count * 3);
    for (uint32_t g = 0; g < group_count; ++g) for (uint32_t page = 0; page < 3; ++page) {
        const uint64_t va = alias_base + uint64_t(g * 3 + page) * kPage;
        const uint64_t offset = kScratchOffset + uint64_t(page) * kPage;
        const uint64_t dest = (use_bus_address ? group_info[g].ingress : iovas[g]) + offset;
        std::printf("RTX_PTE_DESTINATION,group,%u,page,%u,kind,%s,address,0x%llx,ingress_bus,0x%llx,dma_iova,0x%llx,offset,0x%llx\n",
                    g, page, use_bus_address ? "LOCAL_PCI_BUS" : "IOMMU_DMA_IOVA",
                    (unsigned long long)dest,
                    (unsigned long long)group_info[g].ingress,
                    (unsigned long long)iovas[g], (unsigned long long)offset);
        leaves.push_back(install_one_pte(fd, uuid, pdb, va, dest, "PEX_INGRESS"));
    }
}
static void restore_ptes(int fd, std::vector<PteLeaf> &leaves) {
    for (auto it = leaves.rbegin(); it != leaves.rend(); ++it) {
        if (!it->acquired) continue;
        UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{}; p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_RESTORE;
        p.mapping_id = it->active.mapping_id; p.target_va = it->active.target_va;
        p.target_pte_phys = it->active.target_pte_phys;
        errno = 0; const int rc = ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &p); const int e = errno;
        const bool ok = rc == 0 && p.rmStatus == NV_OK && p.pte_restored &&
                        p.pte_after == it->query.pte_after;
        std::printf("RTX_PTE_RESTORE,va,0x%llx,rc,%d,errno,%d,rm,0x%08x,expected,0x%llx,after,0x%llx,restored,%u\n",
                    (unsigned long long)it->query.target_va, rc, e, (unsigned)p.rmStatus,
                    (unsigned long long)it->query.pte_after, (unsigned long long)p.pte_after,
                    p.pte_restored);
        if (!ok) {
            std::fprintf(stderr, "PRESERVE,exact requester PTE restore failed; retain DMA mapping and backing\n");
            for (;;) sleep(60);
        }
        it->acquired = false;
    }
    leaves.clear();
    std::puts("RTX_PTE_RESTORE_SUMMARY,PASS");
}

__global__ static void tx_multicast_pages(const uint32_t *source,
                                         const uint64_t *destinations,
                                         uint32_t word_count) {
    const uint32_t group = blockIdx.x;
    auto *dst = reinterpret_cast<volatile uint32_t *>(destinations[group]);
    for (uint32_t i = threadIdx.x; i < word_count; i += blockDim.x) dst[i] = source[i];
    // Each thread fences the stores it issued before the host observes kernel
    // completion and validates the individual GPU BAR1 views.
    __threadfence_system();
}
__global__ static void tx_probe_word(uint64_t destination, uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *reinterpret_cast<volatile uint32_t *>(destination) = value;
        __threadfence_system();
    }
}
__global__ static void tx_probe_groups(const uint64_t *destinations, uint32_t value) {
    const uint32_t group = blockIdx.x;
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint32_t *>(destinations[group]) = value;
        __threadfence_system();
    }
}
__global__ static void sanity_word(uint32_t *word, uint32_t value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *word = value;
}
__global__ static void prime_mapped_host_alias(volatile uint32_t *ptr, uint32_t pages,
                                               uint32_t marker) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (uint32_t page = 0; page < pages; ++page)
            ptr[page * (4096 / sizeof(uint32_t))] = marker + page;
}
static void cuda_health_check() {
    int count = 0; CK(cudaGetDeviceCount(&count));
    require(count == 25, "expected 24 GM107s plus RTX, found " + std::to_string(count));
    for (int d = 0; d < count; ++d) {
        CK(cudaSetDevice(d));
        cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, d));
        uint32_t *ptr = nullptr; CK(cudaMalloc(&ptr, sizeof(*ptr)));
        const uint32_t expected = 0x6a510000u + (uint32_t)d;
        sanity_word<<<1, 1>>>(ptr, expected); CK(cudaGetLastError());
        uint32_t observed = 0; CK(cudaMemcpy(&observed, ptr, sizeof(observed), cudaMemcpyDeviceToHost));
        CK(cudaFree(ptr));
        require(observed == expected, "CUDA sanity mismatch on device " + std::to_string(d));
        char bdf[32]{}; CK(cudaDeviceGetPCIBusId(bdf, sizeof(bdf), d));
        std::printf("CUDA_SANITY,device,%d,bdf,%s,name,%s,expected,0x%08x,observed,0x%08x,result,PASS\n",
                    d, bdf, prop.name, expected, observed);
    }
}
static bool wait_event(cudaEvent_t ev, unsigned timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        cudaError_t e = cudaEventQuery(ev);
        if (e == cudaSuccess) return true;
        if (e != cudaErrorNotReady) {
            std::fprintf(stderr, "CUDA_EVENT_FAILURE,%s\n", cudaGetErrorString(e));
            return false;
        }
        if (std::chrono::steady_clock::now() >= deadline) return false;
        usleep(1000);
    }
}
static void save_payload(const std::string &path, const std::vector<uint32_t> &p) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    require(bool(f), "cannot write expected payload artifact");
    f.write(reinterpret_cast<const char *>(p.data()), p.size() * sizeof(uint32_t));
    require(bool(f), "failed writing expected payload artifact");
}
static uint64_t stamp() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: %s GROUPS.tsv GPU_MAPPINGS.tsv IOVA_SYSFS_DIR GROUP_COUNT cpu|rtx|rtx-bus|both ARTIFACT_DIR\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[5], artifact_dir = argv[6];
    const uint32_t group_count = std::strtoul(argv[4], nullptr, 10);
    require(mode == "cpu" || mode == "rtx" || mode == "both" ||
            mode == "rtx-direct" || mode == "rtx-bus" || mode == "rtx-bus-all",
            "invalid send mode");
    std::vector<Group> groups;
    std::vector<WorkerMap> workers;
    std::vector<Receiver> receivers;
    void *host_alias = nullptr, *device_alias = nullptr;
    uint32_t *device_source = nullptr;
    uint64_t *device_destinations = nullptr;
    std::vector<uint8_t> host_alias_expected;
    int uvm_fd = -1, host_test_fd = -1;
    void *host_test_map = MAP_FAILED;
    uint64_t host_test_iova = 0;
    bool rtx_context_retained = false, rtx_quiesced = true, safe_to_unmap = false;
    CUdevice rtx_device{}; CUcontext rtx_context{}; int rtx_ordinal = -1;
    std::vector<PteLeaf> active_ptes;
    bool passed = true;
    try {
        const char *kRtxBdf = rtx_bdf_from_env();
        require(geteuid() == 0, "run as root for PCI BAR MMIO and UVM diagnostic ioctls");
        require(group_count >= 1 && group_count <= kMaxGroups, "group count must be 1..6");
        groups = read_groups(argv[1], group_count);
        workers = read_workers(argv[2], groups);
        auto iovas = read_iovas(argv[3], group_count);
        host_test_iova = read_host_iova(argv[3]);
        receivers = map_receivers(workers);
        std::printf("RTX_PEX_MULTICAST_CONFIG,requester,%s,groups,%u,m10_gpus,%zu,worker_activity,PARKED,scratch_offset,0x%x,logical_bytes,%u,transfer_bytes,%u\n",
                    kRtxBdf, group_count, workers.size(), kScratchOffset, kLogicalBytes, kTransferBytes);
        for (uint32_t g = 0; g < group_count; ++g)
            std::printf("DMA_DESTINATION,group,%u,ingress_bus,0x%llx,requester_iova,0x%llx,length,%llu,direction,DMA_FROM_DEVICE\n",
                        g, (unsigned long long)groups[g].ingress, (unsigned long long)iovas[g],
                        (unsigned long long)kWindow);

        CUK(cuInit(0)); CK(cudaDeviceGetByPCIBusId(&rtx_ordinal, kRtxBdf));
        CUK(cuDeviceGet(&rtx_device, rtx_ordinal));
        CUK(cuDevicePrimaryCtxRetain(&rtx_context, rtx_device)); rtx_context_retained = true;
        CUK(cuCtxSetCurrent(rtx_context)); CK(cudaSetDevice(rtx_ordinal));
        CK(cudaFree(nullptr));
        CUuuid c_uuid{}; CUK(cuDeviceGetUuid(&c_uuid, rtx_device));
        NvProcessorUuid uuid{}; static_assert(sizeof(uuid) == sizeof(c_uuid.bytes));
        std::memcpy(&uuid, c_uuid.bytes, sizeof(uuid));

        // The entire payload occupies three 4-KiB aliases per M10-local group.
        CK(cudaHostAlloc(&host_alias, size_t(group_count) * 3 * kPage,
                         cudaHostAllocMapped | cudaHostAllocPortable));
        CK(cudaHostGetDevicePointer(&device_alias, host_alias, 0));
        const uint64_t alias_base = reinterpret_cast<uint64_t>(device_alias);
        require(!(alias_base & (kPage - 1)), "RTX mapped-host alias is not 4-KiB aligned");
        const size_t host_alias_size = size_t(group_count) * 3 * kPage;
        std::memset(host_alias, 0x5a, host_alias_size);
        host_alias_expected.assign(host_alias_size, 0x5a);
        const uint32_t alias_pages = group_count * 3;
        prime_mapped_host_alias<<<1, 1>>>(reinterpret_cast<volatile uint32_t *>(device_alias),
                                          alias_pages, 0x7a510000u);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        for (uint32_t page = 0; page < alias_pages; ++page)
            require(reinterpret_cast<volatile uint32_t *>(host_alias)[page * (kPage / 4)] ==
                        0x7a510000u + page,
                    "RTX mapped-host alias priming readback mismatch at page " +
                        std::to_string(page));
        for (uint32_t page = 0; page < alias_pages; ++page) {
            const uint32_t marker = 0x7a510000u + page;
            std::memcpy(host_alias_expected.data() + size_t(page) * kPage,
                        &marker, sizeof(marker));
        }
        std::printf("RTX_ALIAS_PRIME,PASS,pages,%u,device_va,0x%llx,marker,0x7a510000\n",
                    alias_pages, (unsigned long long)alias_base);
        CK(cudaMalloc(&device_source, kTransferBytes));
        std::vector<uint64_t> dest_ptrs(group_count);
        for (uint32_t g = 0; g < group_count; ++g)
            dest_ptrs[g] = alias_base + uint64_t(g) * 3 * kPage;
        CK(cudaMalloc(&device_destinations, dest_ptrs.size() * sizeof(uint64_t)));
        CK(cudaMemcpy(device_destinations, dest_ptrs.data(), dest_ptrs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

        uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        require(uvm_fd >= 0, std::string("open /dev/nvidia-uvm: ") + strerror(errno));
        UVM_INITIALIZE_PARAMS init{};
        require(ioctl(uvm_fd, UVM_INITIALIZE, &init) == 0 && init.rmStatus == NV_OK,
                "UVM initialize failed");
        UVM_REGISTER_GPU_PARAMS reg{}; reg.gpu_uuid = uuid;
        require(ioctl(uvm_fd, UVM_REGISTER_GPU, &reg) == 0 && reg.rmStatus == NV_OK,
                "RTX UVM registration failed");
        const auto pdb = find_pdb(uvm_fd, uuid, alias_base);
        host_test_fd = open("/dev/rtx_mc_dma_map", O_RDWR | O_CLOEXEC);
        require(host_test_fd >= 0, std::string("open coherent DMA control: ") + strerror(errno));
        host_test_map = mmap(nullptr, 64 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED,
                             host_test_fd, 0);
        require(host_test_map != MAP_FAILED,
                std::string("mmap coherent DMA control: ") + strerror(errno));
        std::memset(host_test_map, 0, 64 * 1024);
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        std::printf("RTX_DMA_CONTROL,kind,COHERENT_HOST_RAM,iova,0x%llx,length,%u,cpu_map,%p,device,/dev/rtx_mc_dma_map\n",
                    (unsigned long long)host_test_iova, 64 * 1024, host_test_map);
        active_ptes.reserve(group_count * 3 + 1);
        active_ptes.push_back(install_one_pte(uvm_fd, uuid, pdb, alias_base,
                                               host_test_iova, "COHERENT_HOST_CONTROL"));
        constexpr uint32_t host_probe_value = 0xc01dcafeu;
        cudaStream_t host_probe_stream{}; cudaEvent_t host_probe_done{};
        CK(cudaStreamCreateWithFlags(&host_probe_stream, cudaStreamNonBlocking));
        CK(cudaEventCreate(&host_probe_done));
        rtx_quiesced = false;
        tx_probe_word<<<1, 1, 0, host_probe_stream>>>(alias_base, host_probe_value);
        CK(cudaGetLastError()); CK(cudaEventRecord(host_probe_done, host_probe_stream));
        if (!wait_event(host_probe_done, 10000)) {
            std::fprintf(stderr, "PRESERVE,RTX coherent-RAM control store did not complete; retain PTE and DMA resources\n");
            for (;;) sleep(60);
        }
        rtx_quiesced = true;
        CK(cudaEventDestroy(host_probe_done)); CK(cudaStreamDestroy(host_probe_stream));
        auto *host_test_words = reinterpret_cast<volatile uint32_t *>(host_test_map);
        std::printf("RTX_DMA_CONTROL_READBACK,expected,0x%08x,observed,0x%08x,match,%u\n",
                    host_probe_value, host_test_words[0], host_test_words[0] == host_probe_value);
        require(host_test_words[0] == host_probe_value,
                "RTX SYS-PTE store did not reach its coherent host-RAM DMA control buffer");
        require(host_test_words[1] == 0,
                "RTX one-word coherent-RAM control changed an adjacent word");
        restore_ptes(uvm_fd, active_ptes);
        if (mode == "rtx-direct") {
            require(group_count == 1, "direct M10 control is limited to one card group");
            const std::string &owner_bdf = groups[0].gpus[0];
            const uint64_t owner_bar1 = bar1_resource_base(owner_bdf);
            require(workers[0].bar1_bus == owner_bar1,
                    "resident mapping BAR address does not match the live owner BAR1 resource base");
            const uint64_t peer_iova = read_peer_iova(argv[3]);
            const uint64_t peer_destination = peer_iova + kScratchOffset;
            std::printf("RTX_DIRECT_M10_TARGET,owner_bdf,%s,bar1_resource_base,0x%llx,bar1_offset,0x%x,resource_address,0x%llx,dma_iova,0x%llx,length,%u\n",
                        owner_bdf.c_str(), (unsigned long long)owner_bar1, kScratchOffset,
                        (unsigned long long)(owner_bar1 + kScratchOffset),
                        (unsigned long long)peer_iova, (unsigned)kWindow);
            active_ptes.push_back(install_one_pte(uvm_fd, uuid, pdb, alias_base,
                                                   peer_destination, "DIRECT_M10_BAR1"));
            constexpr uint32_t direct_probe_value = 0xd1ec7ba1u;
            cudaStream_t direct_stream{}; cudaEvent_t direct_done{};
            CK(cudaStreamCreateWithFlags(&direct_stream, cudaStreamNonBlocking));
            CK(cudaEventCreate(&direct_done));
            rtx_quiesced = false;
            tx_probe_word<<<1, 1, 0, direct_stream>>>(alias_base, direct_probe_value);
            CK(cudaGetLastError()); CK(cudaEventRecord(direct_done, direct_stream));
            if (!wait_event(direct_done, 10000)) {
                std::fprintf(stderr, "PRESERVE,direct M10 BAR store did not complete; retain all PTE/DMA/receiver resources\n");
                for (;;) sleep(60);
            }
            rtx_quiesced = true;
            CK(cudaEventDestroy(direct_done)); CK(cudaStreamDestroy(direct_stream));
            bool direct_ok = true;
            for (const auto &r : receivers) {
                auto *mmio = reinterpret_cast<volatile const uint32_t *>(r.mapping);
                const uint32_t observed = mmio[0];
                const uint32_t expected = r.identity.worker == 0 ? direct_probe_value : r.saved[0];
                std::printf("RTX_DIRECT_M10_READBACK,owner_bdf,%s,read_bdf,%s,expected,0x%08x,observed,0x%08x,match,%u\n",
                            owner_bdf.c_str(), r.identity.bdf.c_str(), expected, observed,
                            observed == expected);
                direct_ok &= observed == expected;
            }
            restore_ptes(uvm_fd, active_ptes);
            if (!direct_ok)
                throw std::runtime_error("direct RTX-to-M10 BAR control had unexpected receiver value(s); see all readback records");
            std::puts("RTX_DIRECT_M10_CONTROL,PASS,one_owner_BAR1_word_verified_and_other_group_members_unchanged");
        }
        else {
            install_pte_aliases(uvm_fd, uuid, pdb, alias_base, iovas, groups,
                                mode == "rtx-bus" || mode == "rtx-bus-all",
                                group_count, active_ptes);
        }

        if (mode == "cpu" || mode == "both") {
            for (uint32_t seq = 1; seq <= 2; ++seq) {
                auto payload = make_payload(seq);
                save_payload(artifact_dir + "/cpu_expected_" + std::to_string(seq) + ".bin", payload);
                const auto t0 = stamp(); cpu_publish(groups, payload, seq);
                asm volatile("sfence" ::: "memory");
                compare_receivers(receivers, payload, "CPU", seq);
                std::printf("EPOCH_RESULT,mode,CPU,seq,%u,mismatches,0,elapsed_ns,%llu\n",
                            seq, (unsigned long long)(stamp() - t0));
            }
        }

        if (mode == "rtx" || mode == "both" || mode == "rtx-bus" ||
            mode == "rtx-bus-all") {
            cudaStream_t stream{}; cudaEvent_t start{}, done{};
            CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CK(cudaEventCreate(&start)); CK(cudaEventCreate(&done));
            constexpr uint32_t probe_value = 0xd15ea5e0u;
            CK(cudaEventRecord(start, stream));
            rtx_quiesced = false;
            tx_probe_groups<<<group_count, 1, 0, stream>>>(device_destinations, probe_value);
            CK(cudaGetLastError());
            CK(cudaEventRecord(done, stream));
            if (!wait_event(done, 10000)) {
                std::fprintf(stderr, "PRESERVE,RTX single-word probe did not reach confirmed completion; keep all PTE/DMA/receiver resources\n");
                for (;;) sleep(60);
            }
            rtx_quiesced = true;
            for (uint32_t g = 0; g < group_count; ++g)
                std::printf("RTX_WORD_PROBE,group,%u,device_va,0x%llx,pte_destination_kind,%s,pte_destination,0x%llx,dma_iova,0x%llx,resource_address,0x%llx,value,0x%08x,completed,1\n",
                            g, (unsigned long long)dest_ptrs[g],
                            (mode == "rtx-bus" || mode == "rtx-bus-all") ? "LOCAL_PCI_BUS" : "IOMMU_DMA_IOVA",
                            (unsigned long long)(((mode == "rtx-bus" || mode == "rtx-bus-all") ? groups[g].ingress : iovas[g]) + kScratchOffset),
                            (unsigned long long)(iovas[g] + kScratchOffset),
                            (unsigned long long)(groups[g].ingress + kScratchOffset), probe_value);
            bool word_probe_ok = true;
            for (const auto &r : receivers) {
                auto *mmio = reinterpret_cast<volatile const uint32_t *>(r.mapping);
                const uint32_t observed = mmio[0];
                std::printf("RTX_WORD_PROBE_READBACK,group,%d,worker,%d,bdf,%s,expected,0x%08x,observed,0x%08x,match,%u\n",
                            r.identity.group, r.identity.worker, r.identity.bdf.c_str(),
                            probe_value, observed, observed == probe_value);
                word_probe_ok &= observed == probe_value;
            }
            require(word_probe_ok,
                    "RTX single-word store did not appear at every CPU BAR1 receiver; see all readback records");
            for (uint32_t seq = 1; seq <= 2; ++seq) {
                auto payload = make_payload(seq + 10);
                save_payload(artifact_dir + "/rtx_expected_" + std::to_string(seq) + ".bin", payload);
                CK(cudaMemcpy(device_source, payload.data(), kTransferBytes, cudaMemcpyHostToDevice));
                const uint64_t t0 = stamp();
                CK(cudaEventRecord(start, stream));
                rtx_quiesced = false;
                tx_multicast_pages<<<group_count, 256, 0, stream>>>(device_source, device_destinations, kWords);
                CK(cudaGetLastError());
                CK(cudaEventRecord(done, stream));
                if (!wait_event(done, 10000)) {
                    std::fprintf(stderr, "PRESERVE,RTX transmit did not reach confirmed completion; keep all PTE/DMA/receiver resources\n");
                    for (;;) sleep(60);
                }
                rtx_quiesced = true;
                const uint64_t t1 = stamp();
                float device_ms = 0.0f; CK(cudaEventElapsedTime(&device_ms, start, done));
                compare_receivers(receivers, payload, "RTX", seq);
                std::printf("EPOCH_RESULT,mode,RTX,seq,%u,mismatches,0,gpu_event_ms,%.6f,submit_to_complete_ns,%llu,complete_and_verify_ns,%llu,groups,%u,bytes_per_group,%u,total_injected_bytes,%u\n",
                            seq, device_ms, (unsigned long long)(t1-t0),
                            (unsigned long long)(stamp()-t0), group_count,
                            kTransferBytes, group_count*kTransferBytes);
            }
            CK(cudaEventDestroy(done)); CK(cudaEventDestroy(start)); CK(cudaStreamDestroy(stream));
        }

        if (!rtx_quiesced) {
            std::fprintf(stderr, "PRESERVE,requester may still access mapped ingress\n");
            for (;;) sleep(60);
        }
        restore_ptes(uvm_fd, active_ptes);
        CK(cudaDeviceSynchronize());
        const auto *pinned = reinterpret_cast<const uint8_t *>(host_alias);
        const auto mismatch = std::mismatch(host_alias_expected.begin(), host_alias_expected.end(), pinned);
        if (mismatch.first != host_alias_expected.end()) {
            const size_t offset = size_t(mismatch.first - host_alias_expected.begin());
            throw std::runtime_error("RTX pinned-host alias guard changed at byte " +
                                     std::to_string(offset));
        }
        std::printf("RTX_HOST_ALIAS_GUARD,PASS,bytes,%zu,baseline,post_prime_markers_plus_0x5a_fill\n",
                    host_alias_size);
        safe_to_unmap = true;
        std::puts("SAFE_TO_UNMAP,DMA mappings can be released after this process exits");
    } catch (const std::exception &e) {
        passed = false;
        std::fprintf(stderr, "FAIL,%s\n", e.what());
        if (!rtx_quiesced) {
            std::fprintf(stderr, "PRESERVE,uncertain GPU completion; keep mappings and resources\n");
            for (;;) sleep(60);
        }
        if (!active_ptes.empty() && uvm_fd >= 0) restore_ptes(uvm_fd, active_ptes);
        safe_to_unmap = active_ptes.empty() || uvm_fd >= 0;
    }
    if (!safe_to_unmap) {
        std::fprintf(stderr, "PRESERVE,cleanup state ambiguous\n");
        for (;;) sleep(60);
    }
    if (host_test_map != MAP_FAILED) { munmap(host_test_map, 64 * 1024); host_test_map = MAP_FAILED; }
    if (host_test_fd >= 0) { close(host_test_fd); host_test_fd = -1; }
    if (!receivers.empty()) restore_and_close_receivers(receivers);
    if (device_destinations) cudaFree(device_destinations);
    if (device_source) cudaFree(device_source);
    if (host_alias) cudaFreeHost(host_alias);
    if (uvm_fd >= 0) close(uvm_fd);
    if (rtx_context_retained) {
        CUK(cuCtxSetCurrent(rtx_context));
        CUK(cuDevicePrimaryCtxRelease(rtx_device));
    }
    if (passed) {
        try { cuda_health_check(); }
        catch (const std::exception &e) {
            passed = false; std::fprintf(stderr, "CUDA_SANITY,FAIL,%s\n", e.what());
        }
    }
    if (safe_to_unmap) std::puts("SAFE_TO_UNMAP,all acquired GPU and PTE objects released");
    std::printf("RUN_RESULT,%s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
