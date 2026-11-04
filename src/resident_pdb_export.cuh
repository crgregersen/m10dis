#pragma once

// Emit the worker process's own UVM VA-space identity for a retained RM/VMM
// mailbox. The RTX client can then ask the diagnostic UVM route transaction
// to operate on this live worker VA, without pretending the VA belongs to the
// client process.
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

extern "C" {
#include <nvstatus.h>
#include <uvm_linux_ioctl.h>
}

struct ResidentPdbRecord {
    uint64_t va_space = 0;
    uint64_t gpu_va_space = 0;
    uint64_t root_phys = 0;
    uint32_t root_aperture = 0;
    uint32_t pid = 0;
};

static int resident_pdb_uvm_fd = -1;

static std::vector<ResidentPdbRecord> resident_query_pdb(CUdevice device) {
    if (resident_pdb_uvm_fd < 0) {
        resident_pdb_uvm_fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        if (resident_pdb_uvm_fd < 0)
            throw std::runtime_error(std::string("open UVM for worker PDB export: ") + strerror(errno));
        UVM_INITIALIZE_PARAMS init{};
        if (ioctl(resident_pdb_uvm_fd, UVM_INITIALIZE, &init) || init.rmStatus != NV_OK)
            throw std::runtime_error("UVM_INITIALIZE failed while exporting worker PDB");
    }

    CUuuid cu_uuid{};
    CUKQ(cuDeviceGetUuid(&cu_uuid, device));
    NvProcessorUuid uuid{};
    static_assert(sizeof(uuid) == sizeof(cu_uuid.bytes));
    std::memcpy(&uuid, cu_uuid.bytes, sizeof(uuid));
    UVM_REGISTER_GPU_PARAMS reg{};
    reg.gpu_uuid = uuid;
    if (ioctl(resident_pdb_uvm_fd, UVM_REGISTER_GPU, &reg) || reg.rmStatus != NV_OK)
        throw std::runtime_error("UVM_REGISTER_GPU failed while exporting worker PDB");

    std::vector<UVM_GLM53_VA_SPACE_PDB_ENTRY> entries(128);
    UVM_GLM53_LIST_VA_SPACE_PDBS_PARAMS list{};
    list.entry_capacity = static_cast<NvU32>(entries.size());
    list.entries = reinterpret_cast<NvU64>(entries.data());
    errno = 0;
    const int list_rc = ioctl(resident_pdb_uvm_fd, UVM_GLM53_LIST_VA_SPACE_PDBS, &list);
    const int list_errno = errno;
    if (list_rc || list.rmStatus != NV_OK) {
        std::fprintf(stderr,
            "GPU_PDB_LIST_FAILURE,pid,%u,device,%d,uvm_fd,%d,rc,%d,errno,%d,status,0x%08x,capacity,%u,count,%u,matched,%u,total,%u,overflow,%u\n",
            static_cast<unsigned>(getpid()), static_cast<int>(device), resident_pdb_uvm_fd,
            list_rc, list_errno, static_cast<unsigned>(list.rmStatus),
            static_cast<unsigned>(list.entry_capacity), static_cast<unsigned>(list.entry_count),
            static_cast<unsigned>(list.matched_va_spaces), static_cast<unsigned>(list.total_va_spaces),
            static_cast<unsigned>(list.overflow));
        std::fflush(stderr);
        throw std::runtime_error("UVM PDB enumeration failed in resident worker process");
    }

    std::fprintf(stderr,
        "GPU_PDB_SNAPSHOT,pid,%u,device,%d,count,%u,total,%u,matched,%u,overflow,%u\n",
        static_cast<unsigned>(getpid()), static_cast<int>(device),
        static_cast<unsigned>(list.entry_count), static_cast<unsigned>(list.total_va_spaces),
        static_cast<unsigned>(list.matched_va_spaces), static_cast<unsigned>(list.overflow));
    for (NvU32 i = 0; i < list.entry_count; ++i) {
        const auto &entry = entries[i];
        if (std::memcmp(&entry.gpu_uuid, &uuid, sizeof(uuid)) == 0)
            std::fprintf(stderr,
                "GPU_PDB_ENTRY,pid,%u,index,%u,va_space,0x%llx,gpu_va_space,0x%llx,mm,0x%llx,root,0x%llx,current_pid,%u,target_pid,%u,range_count,%llu,state,%u,did_set,%u\n",
                static_cast<unsigned>(getpid()), static_cast<unsigned>(i),
                static_cast<unsigned long long>(entry.va_space_ptr),
                static_cast<unsigned long long>(entry.gpu_va_space_ptr),
                static_cast<unsigned long long>(entry.mm_ptr),
                static_cast<unsigned long long>(entry.root_phys_address),
                entry.current_pid, entry.target_pid,
                static_cast<unsigned long long>(entry.range_count),
                entry.state, entry.did_set_page_directory);
    }

    std::vector<ResidentPdbRecord> candidates;
    for (NvU32 i = 0; i < list.entry_count; ++i) {
        const auto &entry = entries[i];
        if (std::memcmp(&entry.gpu_uuid, &uuid, sizeof(uuid)) != 0 ||
            entry.current_pid != static_cast<NvU32>(getpid()) ||
            !entry.did_set_page_directory || !entry.va_space_ptr ||
            !entry.gpu_va_space_ptr || !entry.root_phys_address)
            continue;

        candidates.push_back({entry.va_space_ptr, entry.gpu_va_space_ptr,
                              entry.root_phys_address, entry.root_phys_aperture,
                              entry.current_pid});
    }
    if (candidates.empty())
        throw std::runtime_error("worker PDB snapshot has no valid current-process root for this GPU");
    return candidates;
}

static void resident_emit_pdb_record(int group, int worker, const char *bdf,
                                     CUdevice device, CUdeviceptr cuda_va) {
    std::fprintf(stderr,
        "GPU_PDB_QUERY_BEGIN,group,%d,worker,%d,bdf,%s,device,%d,cuda_va,0x%llx,pid,%u\n",
        group, worker, bdf, static_cast<int>(device),
        static_cast<unsigned long long>(cuda_va), static_cast<unsigned>(getpid()));
    std::fflush(stderr);
    std::vector<ResidentPdbRecord> candidates;
    try {
        candidates = resident_query_pdb(device);
    }
    catch (const std::exception &e) {
        std::fprintf(stderr,
            "GPU_PDB_QUERY_ERROR,group,%d,worker,%d,bdf,%s,device,%d,cuda_va,0x%llx,error,%s\n",
            group, worker, bdf, static_cast<int>(device),
            static_cast<unsigned long long>(cuda_va), e.what());
        std::fflush(stderr);
        throw;
    }
    for (const auto &p : candidates) {
        std::fprintf(stderr,
            "GPU_PDB_CANDIDATE,group,%d,worker,%d,bdf,%s,cuda_va,0x%llx,pid,%u,va_space,0x%llx,gpu_va_space,0x%llx,root_phys,0x%llx,root_aperture,%u\n",
            group, worker, bdf, static_cast<unsigned long long>(cuda_va), p.pid,
            static_cast<unsigned long long>(p.va_space),
            static_cast<unsigned long long>(p.gpu_va_space),
            static_cast<unsigned long long>(p.root_phys), p.root_aperture);
    }
    std::fflush(stderr);
    std::fprintf(stderr, "GPU_PDB_QUERY_END,group,%d,worker,%d,bdf,%s\n",
                 group, worker, bdf);
    std::fflush(stderr);
}

static void resident_close_pdb_export() {
    if (resident_pdb_uvm_fd >= 0) {
        close(resident_pdb_uvm_fd);
        resident_pdb_uvm_fd = -1;
    }
}
