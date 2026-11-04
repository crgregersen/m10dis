#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "resident_cpu_mailbox_abi.cuh"

struct Q38CpuMailboxManifest {
    std::string run_id;
    std::string shm_name;
};

struct Q38CpuMailboxRegion {
    int fd = -1;
    void *host = MAP_FAILED;
    bool registered = false;
    std::string run_id;
    std::vector<void *> device_slots;
};

inline Q38CpuMailboxManifest q38_read_cpu_mailbox_manifest(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("open CPU mailbox manifest: " + path);
    std::string line;
    if (!std::getline(in, line) || line != "QWEN38_CPU_MAILBOX_MANIFEST\t1")
        throw std::runtime_error("unsupported CPU mailbox manifest version");
    std::map<std::string, std::string> fields;
    while (std::getline(in, line)) {
        const auto tab = line.find('\t');
        if (tab == std::string::npos || fields.count(line.substr(0, tab)))
            throw std::runtime_error("malformed CPU mailbox manifest field");
        fields.emplace(line.substr(0, tab), line.substr(tab + 1));
    }
    if (fields.size() != 6 || !fields.count("run_id") || !fields.count("shm_name") ||
        !fields.count("bytes") || !fields.count("workers") ||
        !fields.count("slot_stride") || !fields.count("slot_bytes"))
        throw std::runtime_error("CPU mailbox manifest schema mismatch");
    if (fields.at("run_id").empty() || fields.at("shm_name").empty() ||
        fields.at("shm_name")[0] != '/')
        throw std::runtime_error("CPU mailbox manifest has invalid run_id or shm_name");
    if (std::stoull(fields.at("bytes")) != Q38_CPU_MAILBOX_BYTES ||
        std::stoul(fields.at("workers")) != Q38_CPU_MAILBOX_WORKERS ||
        std::stoull(fields.at("slot_stride")) != Q38_CPU_MAILBOX_SLOT_STRIDE ||
        std::stoull(fields.at("slot_bytes")) != Q38_CPU_MAILBOX_SLOT_BYTES)
        throw std::runtime_error("CPU mailbox manifest layout is incompatible");
    return {fields.at("run_id"), fields.at("shm_name")};
}

inline void q38_unregister_cpu_mailbox(Q38CpuMailboxRegion &region) {
    if (region.registered && region.host != MAP_FAILED) {
        const cudaError_t e = cudaHostUnregister(region.host);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("cudaHostUnregister CPU mailbox: ") + cudaGetErrorString(e));
        region.registered = false;
    }
    if (region.host != MAP_FAILED) { munmap(region.host, Q38_CPU_MAILBOX_BYTES); region.host = MAP_FAILED; }
    if (region.fd >= 0) { close(region.fd); region.fd = -1; }
    region.device_slots.clear();
    region.run_id.clear();
}

inline Q38CpuMailboxRegion q38_register_cpu_mailbox(const Q38CpuMailboxManifest &manifest,
                                                     const std::vector<unsigned> &global_workers) {
    if (global_workers.size() != 20 && global_workers.size() != 4)
        throw std::runtime_error("CPU mailbox registration requires 20 routed or 4 shared devices");
    Q38CpuMailboxRegion region;
    try {
        region.fd = shm_open(manifest.shm_name.c_str(), O_RDWR | O_CLOEXEC, 0);
        if (region.fd < 0) throw std::runtime_error(std::string("shm_open CPU mailbox: ") + strerror(errno));
        struct stat st{};
        if (fstat(region.fd, &st) || size_t(st.st_size) != Q38_CPU_MAILBOX_BYTES)
            throw std::runtime_error("CPU mailbox shared-memory size mismatch");
        region.host = mmap(nullptr, Q38_CPU_MAILBOX_BYTES, PROT_READ | PROT_WRITE,
                           MAP_SHARED, region.fd, 0);
        if (region.host == MAP_FAILED)
            throw std::runtime_error(std::string("mmap CPU mailbox: ") + strerror(errno));
        cudaError_t e = cudaHostRegister(region.host, Q38_CPU_MAILBOX_BYTES,
                                         cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("cudaHostRegister CPU mailbox: ") + cudaGetErrorString(e));
        region.registered = true;
        region.run_id = manifest.run_id;
        region.device_slots.resize(global_workers.size());
        for (size_t d = 0; d < global_workers.size(); ++d) {
            if (global_workers[d] >= Q38_CPU_MAILBOX_WORKERS)
                throw std::runtime_error("CPU mailbox worker index outside 0..23");
            e = cudaSetDevice(static_cast<int>(d));
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("cudaSetDevice for CPU mailbox: ") + cudaGetErrorString(e));
            void *base = nullptr;
            e = cudaHostGetDevicePointer(&base, region.host, 0);
            if (e != cudaSuccess)
                throw std::runtime_error(std::string("cudaHostGetDevicePointer CPU mailbox: ") + cudaGetErrorString(e));
            region.device_slots[d] = static_cast<char *>(base) +
                size_t(global_workers[d]) * Q38_CPU_MAILBOX_SLOT_STRIDE;
        }
        return region;
    } catch (...) {
        try { q38_unregister_cpu_mailbox(region); } catch (...) {}
        throw;
    }
}
