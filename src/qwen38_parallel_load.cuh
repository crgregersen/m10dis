// Parallel, verified host->GPU loading for the Qwen3.8 M10 loaders.
//
// One loader thread per M10 card (GPUs grouped by their PEX8747 upstream port, found from
// sysfs, never hardcoded). Each thread streams files in kChunk pieces:
//   pread (O_DIRECT into pinned slot A)  ->  cudaMemcpyAsync H2D  ->  cudaMemcpyAsync D2H into
//   pinned readback slot  ->  memcmp after the event, before slot A is reused.
// Two slots per thread let the next disk read overlap the current chunk's PCIe transfers, and
// the six card links run concurrently. Every byte that reaches the GPU is read back and
// compared, so this keeps the PLE loader's full-readback guarantee.
//
// O_DIRECT needs 4 KiB aligned offsets; any read that is not aligned falls back to buffered
// pread for that file (reported in the stats). Buffers come from cudaHostAlloc (page aligned).
#pragma once
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace q38load {

constexpr size_t kChunk = size_t(32) << 20;
constexpr size_t kAlign = 4096;

struct Stats {
  std::atomic<unsigned long long> bytes{0}, direct_files{0}, buffered_files{0};
  std::atomic<unsigned long long> read_ns{0}, wait_ns{0};
};

inline unsigned long long ns_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();
}

inline void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Pinned double-buffer set owned by one loader thread.
struct Slots {
  uint8_t *in[2]{}, *back[2]{};
  cudaEvent_t done[2]{};
  bool pending[2]{};
  size_t pending_bytes[2]{};
  int pending_device[2]{-1, -1};
  int event_device = -1;                 // CUDA events belong to one device context
  explicit Slots() {
    for (int i = 0; i < 2; i++) {
      cuda_check(cudaHostAlloc(reinterpret_cast<void **>(&in[i]), kChunk, cudaHostAllocPortable), "cudaHostAlloc in");
      cuda_check(cudaHostAlloc(reinterpret_cast<void **>(&back[i]), kChunk, cudaHostAllocPortable), "cudaHostAlloc back");
    }
  }
  ~Slots() {
    for (int i = 0; i < 2; i++) {
      if (done[i]) { cudaSetDevice(event_device); cudaEventDestroy(done[i]); }
      cudaFreeHost(in[i]); cudaFreeHost(back[i]);
    }
  }
  Slots(const Slots &) = delete;
  Slots &operator=(const Slots &) = delete;
};

// Wait for the transfer in slot i and compare its readback. Throws on mismatch.
inline void settle(Slots &s, int i, Stats &st, const std::string &what) {
  if (!s.pending[i]) return;
  auto t = std::chrono::steady_clock::now();
  cuda_check(cudaSetDevice(s.pending_device[i]), "cudaSetDevice settle");
  cuda_check(cudaEventSynchronize(s.done[i]), "cudaEventSynchronize");
  st.wait_ns += ns_since(t);
  if (std::memcmp(s.in[i], s.back[i], s.pending_bytes[i]) != 0) {
    size_t first = 0;
    while (first < s.pending_bytes[i] && s.in[i][first] == s.back[i][first]) ++first;
    throw std::runtime_error("readback mismatch in " + what + " at chunk byte " + std::to_string(first));
  }
  s.pending[i] = false;
}

// Stream bytes [offset, offset+bytes) of path into device pointer dst on `device`, verified.
inline void upload_file(Slots &s, Stats &st, int device, cudaStream_t stream, const std::string &path,
                        unsigned long long offset, unsigned long long bytes, void *dst) {
  bool direct = offset % kAlign == 0;
  int fd = direct ? ::open(path.c_str(), O_RDONLY | O_DIRECT) : -1;
  if (fd < 0) { direct = false; fd = ::open(path.c_str(), O_RDONLY); }
  if (fd < 0) throw std::runtime_error("open failed: " + path);
  struct stat sb{};
  if (fstat(fd, &sb) != 0 || static_cast<unsigned long long>(sb.st_size) < offset + bytes) {
    ::close(fd); throw std::runtime_error("file shorter than requested range: " + path);
  }
  (direct ? st.direct_files : st.buffered_files) += 1;
  cuda_check(cudaSetDevice(device), "cudaSetDevice upload");
  if (s.event_device != device) {         // nothing is pending here: every upload_file settles both slots
    for (int i = 0; i < 2; i++) {
      if (s.done[i]) { cudaSetDevice(s.event_device); cudaEventDestroy(s.done[i]); s.done[i] = nullptr; }
    }
    cuda_check(cudaSetDevice(device), "cudaSetDevice events");
    for (int i = 0; i < 2; i++) cuda_check(cudaEventCreateWithFlags(&s.done[i], cudaEventDisableTiming), "cudaEventCreate");
    s.event_device = device;
  }
  unsigned long long pos = 0; int slot = 0;
  try {
    while (pos < bytes) {
      const size_t n = static_cast<size_t>(std::min<unsigned long long>(kChunk, bytes - pos));
      settle(s, slot, st, path);                      // slot reuse: previous transfer verified first
      cuda_check(cudaSetDevice(device), "cudaSetDevice chunk");
      auto t = std::chrono::steady_clock::now();
      // O_DIRECT reads a 4 KiB multiple; at EOF the kernel returns the short remainder.
      const size_t want = direct ? ((n + kAlign - 1) / kAlign) * kAlign : n;
      size_t got = 0;
      while (got < n) {
        ssize_t r = ::pread(fd, s.in[slot] + got, want - got, static_cast<off_t>(offset + pos + got));
        if (r < 0 && direct && errno == EINVAL && got == 0) {    // filesystem refused O_DIRECT
          ::close(fd); fd = ::open(path.c_str(), O_RDONLY); direct = false;
          if (fd < 0) throw std::runtime_error("reopen failed: " + path);
          st.direct_files -= 1; st.buffered_files += 1; continue;
        }
        if (r <= 0) throw std::runtime_error("short read in " + path);
        got += static_cast<size_t>(r);
        if (direct && got < n && got % kAlign != 0)       // unaligned short read: cannot continue O_DIRECT
          throw std::runtime_error("unaligned short O_DIRECT read in " + path);
      }
      st.read_ns += ns_since(t);
      auto *d = static_cast<uint8_t *>(dst) + pos;
      cuda_check(cudaMemcpyAsync(d, s.in[slot], n, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
      cuda_check(cudaMemcpyAsync(s.back[slot], d, n, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync D2H");
      cuda_check(cudaEventRecord(s.done[slot], stream), "cudaEventRecord");
      s.pending[slot] = true; s.pending_bytes[slot] = n; s.pending_device[slot] = device;
      pos += n; st.bytes += n; slot ^= 1;
    }
    settle(s, 0, st, path); settle(s, 1, st, path);
  } catch (...) {
    ::close(fd);
    for (int i = 0; i < 2; i++) if (s.pending[i]) { cudaSetDevice(s.pending_device[i]); cudaEventSynchronize(s.done[i]); s.pending[i] = false; }
    throw;
  }
  ::close(fd);
}

// PEX8747 upstream port of a CUDA device: GPU -> inner downstream port -> upstream port.
inline std::string card_of_device(int device) {
  char bdf[32]{};
  cuda_check(cudaDeviceGetPCIBusId(bdf, sizeof(bdf), device), "cudaDeviceGetPCIBusId");
  std::string b(bdf);
  for (auto &c : b) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (b.size() > 12 && b.rfind("0000", 0) == 0 && b[8] == ':') b = b.substr(4);  // 00000000:05:00.0 -> 0000:05:00.0
  auto p = std::filesystem::canonical("/sys/bus/pci/devices/" + b);
  return p.parent_path().parent_path().filename().string();
}

// Group devices by card; returns card -> device list (insertion order by first device).
inline std::vector<std::vector<int>> group_by_card(const std::vector<int> &devices) {
  std::map<std::string, std::vector<int>> m; std::vector<std::string> order;
  for (int d : devices) { auto c = card_of_device(d); if (!m.count(c)) order.push_back(c); m[c].push_back(d); }
  std::vector<std::vector<int>> out;
  for (auto &c : order) out.push_back(m[c]);
  return out;
}

// Run fn(group_index, devices) on one thread per card; rethrows the first failure.
template <class F> void per_card(const std::vector<std::vector<int>> &groups, F fn) {
  std::vector<std::thread> th; std::mutex em; std::string first_error;
  for (size_t gi = 0; gi < groups.size(); gi++)
    th.emplace_back([&, gi] {
      try { fn(gi, groups[gi]); }
      catch (const std::exception &e) { std::lock_guard<std::mutex> l(em); if (first_error.empty()) first_error = e.what(); }
    });
  for (auto &t : th) t.join();
  if (!first_error.empty()) throw std::runtime_error(first_error);
}

}  // namespace q38load
