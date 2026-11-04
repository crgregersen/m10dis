// Recovery/inspection tool for patched-UVM route records left by an RTX peer client.
// Run as root (the glm53 ioctls require CAP_SYS_RAWIO). Used by the resident daemon's
// RECOVER_PEER_CLIENT command; see docs/RESIDENT_DAEMON_INTERFACE.md.
//
//   q38_peer_route_tool scan [MAX_ID]      read-only: QUERY multi-PTE records 1..MAX_ID (default 4096)
//                                          and print every active one. Writes nothing.
//   q38_peer_route_tool restore JOURNAL    replay a route journal written by the peer library:
//     1. every PTE record not marked PTE_RESTORED is restored, newest first, through the kernel's
//        guarded RESTORE (it re-finds the live GPU VA space and writes the saved original only if
//        the current PTE still equals the temporary value). Outcomes per record:
//          RESTORED      original written back and read back
//          NOT_ACTIVE    no such record (already restored)
//          VA_SPACE_GONE the owning VA space no longer exists; nothing written, record stays
//                        in the table (released by UVM's lifecycle hook or REAP_DEAD if present)
//          PTE_CHANGED   PTE no longer holds our temporary value; nothing written
//          FAILED        anything else (blocking)
//     2. only if no PTE ended FAILED: DMA maps not marked DMA_UNMAPPED are unmapped, then owner
//        retains not marked OWNER_RELEASED are released, newest first.
//   Exit 0 when nothing ended FAILED; 3 otherwise (DMA/owner records are then left untouched).
//   A RECOVERED line is appended to the journal on success.
#include <cinttypes>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
extern "C" {
#include <nvstatus.h>
#include <uvm_linux_ioctl.h>
}
#include "qwen38_route_journal.h"

static bool uuid_from_hex(const std::string &h, NvProcessorUuid &u) {
  if (h.size() != 2 * sizeof(u)) return false;
  auto *b = reinterpret_cast<unsigned char *>(&u);
  for (size_t i = 0; i < sizeof(u); i++) b[i] = (unsigned char)std::stoul(h.substr(2 * i, 2), nullptr, 16);
  return true;
}

static int open_uvm() {
  int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
  if (fd < 0) { perror("open /dev/nvidia-uvm"); return -1; }
  UVM_INITIALIZE_PARAMS init{};
  if (ioctl(fd, UVM_INITIALIZE, &init) || init.rmStatus != NV_OK) { fprintf(stderr, "UVM_INITIALIZE failed\n"); close(fd); return -1; }
  return fd;
}

static int scan(int fd, unsigned long long max_id) {
  unsigned long long active = 0; NvU32 count = 0;
  for (unsigned long long id = 1; id <= max_id; id++) {
    UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
    p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_QUERY; p.mapping_id = id;
    ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &p);
    count = p.active_count;
    if (p.rmStatus != NV_OK) continue;
    active++;
    printf("ACTIVE_RECORD,id,%llu,slot,%u,gpu,%s,va,0x%llx,pte_phys,0x%llx,original,0x%llx,temporary,0x%llx,"
           "destination,0x%llx,va_space_state,%u\n", id, p.slot, q38_uuid_hex(p.gpu_uuid).c_str(),
           (unsigned long long)p.active_target_va, (unsigned long long)p.active_target_pte_phys,
           (unsigned long long)p.saved_original_pte, (unsigned long long)p.temporary_pte,
           (unsigned long long)p.active_destination_gpu_sys_address, p.retained_gpu_va_space_state);
  }
  printf("SCAN_DONE,ids_checked,%llu,active_found,%llu,kernel_active_count,%u\n", max_id, active, count);
  return 0;
}

struct Pte { unsigned long long id, va, phys, temp, orig, map_id; std::string gpu; };
struct Dma { unsigned long long map_id, route_id; std::string req, own; };
struct Own { unsigned long long map_id; std::string req, own; };

static int restore(int fd, const char *journal) {
  std::ifstream in(journal);
  if (!in) { fprintf(stderr, "cannot read journal %s\n", journal); return 2; }
  std::vector<Pte> ptes; std::vector<Dma> dmas; std::vector<Own> owners;
  std::set<unsigned long long> pte_done, own_done; std::set<std::pair<unsigned long long, unsigned long long>> dma_done;
  std::string line; bool closed = false;
  while (std::getline(in, line)) {
    std::istringstream ss(line); std::string tag; std::getline(ss, tag, '\t');
    std::vector<std::string> f; std::string x; while (std::getline(ss, x, '\t')) f.push_back(x);
    auto u = [&](size_t i) { return std::stoull(f.at(i)); };
    if (tag == "PTE") ptes.push_back({u(0), u(2), u(3), u(4), u(5), u(6), f.at(1)});
    else if (tag == "PTE_RESTORED") pte_done.insert(u(0));
    else if (tag == "DMA") dmas.push_back({u(0), u(1), f.at(2), f.at(3)});
    else if (tag == "DMA_UNMAPPED") dma_done.insert({u(0), u(1)});
    else if (tag == "OWNER") owners.push_back({u(0), f.at(1), f.at(2)});
    else if (tag == "OWNER_RELEASED") own_done.insert(u(0));
    else if (tag == "CLOSED") closed = true;
  }
  printf("JOURNAL,%s,pte,%zu,pte_restored,%zu,dma,%zu,dma_unmapped,%zu,owner,%zu,owner_released,%zu,closed,%d\n",
         journal, ptes.size(), pte_done.size(), dmas.size(), dma_done.size(), owners.size(), own_done.size(), closed);
  std::map<std::string, int> outcome; int failed = 0;
  for (auto it = ptes.rbegin(); it != ptes.rend(); ++it) {
    if (pte_done.count(it->id)) continue;
    UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
    p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_RESTORE; p.mapping_id = it->id; p.target_va = it->va; p.target_pte_phys = it->phys;
    errno = 0; int rc = ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &p); int e = errno;
    const char *kind;
    if (rc == 0 && p.rmStatus == NV_OK && p.pte_restored && p.pte_after == it->orig) kind = "RESTORED";
    else if (p.rmStatus == NV_ERR_OBJECT_NOT_FOUND && p.lookup_status == NV_OK && !p.pte_update_attempted && p.va_space_ptr == 0) kind = "NOT_ACTIVE";
    else if (p.lookup_status != NV_OK) kind = "VA_SPACE_GONE";
    else if (p.rmStatus == NV_ERR_OBJECT_TYPE_MISMATCH && !p.pte_update_attempted && p.read_status == NV_OK && p.pte_before == 0)
      kind = "VA_SPACE_GONE";   // pointer now names a different (new) GPU VA space; nothing read or written
    else if (p.duplicate_or_stale && p.pte_before != it->temp && !p.pte_update_attempted) kind = "PTE_CHANGED";
    else kind = "FAILED";
    if (!strcmp(kind, "FAILED")) failed++;
    outcome[kind]++;
    printf("PTE_RECOVER,id,%llu,gpu,%s,va,0x%llx,outcome,%s,rc,%d,errno,%d,rm,0x%08x,lookup,0x%08x,before,0x%llx,after,0x%llx,expected,0x%llx\n",
           it->id, it->gpu.c_str(), it->va, kind, rc, e, (unsigned)p.rmStatus, (unsigned)p.lookup_status,
           (unsigned long long)p.pte_before, (unsigned long long)p.pte_after, it->orig);
    if (!strcmp(kind, "RESTORED")) q38_journal_line("PTE_RESTORED\t%llu", it->id);
  }
  if (failed) {
    printf("RECOVER_RESULT,FAILED,pte_failed,%d,dma_and_owner_records_left_untouched\n", failed);
    return 3;
  }
  for (auto it = dmas.rbegin(); it != dmas.rend(); ++it) {
    if (dma_done.count({it->map_id, it->route_id})) continue;
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS p{};
    if (!uuid_from_hex(it->req, p.requester_uuid) || !uuid_from_hex(it->own, p.owner_uuid)) { printf("DMA_RECOVER,bad_uuid\n"); return 2; }
    p.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_UNMAP_REQUESTER; p.mapping_id = it->map_id; p.route_id = it->route_id;
    int rc = ioctl(fd, UVM_GLM53_SERIAL_BAR1_PEER_READ, &p);
    const bool ok = rc == 0 && p.rmStatus == NV_OK && !p.dma_active;
    outcome[ok ? "DMA_UNMAPPED" : "DMA_UNMAP_NOT_DONE"]++;
    printf("DMA_RECOVER,map_id,%llu,route_id,%llu,rc,%d,rm,0x%08x,dma_active,%u,ok,%d\n", it->map_id, it->route_id, rc, (unsigned)p.rmStatus, p.dma_active, ok);
    if (ok) q38_journal_line("DMA_UNMAPPED\t%llu\t%llu", it->map_id, it->route_id);
  }
  for (auto it = owners.rbegin(); it != owners.rend(); ++it) {
    if (own_done.count(it->map_id)) continue;
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS p{};
    if (!uuid_from_hex(it->req, p.requester_uuid) || !uuid_from_hex(it->own, p.owner_uuid)) { printf("OWNER_RECOVER,bad_uuid\n"); return 2; }
    p.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_RELEASE_OWNER; p.mapping_id = it->map_id;
    int rc = ioctl(fd, UVM_GLM53_SERIAL_BAR1_PEER_READ, &p);
    const bool ok = rc == 0 && p.rmStatus == NV_OK && !p.owner_active;
    outcome[ok ? "OWNER_RELEASED" : "OWNER_RELEASE_NOT_DONE"]++;
    printf("OWNER_RECOVER,map_id,%llu,rc,%d,rm,0x%08x,owner_active,%u,ok,%d\n", it->map_id, rc, (unsigned)p.rmStatus, p.owner_active, ok);
    if (ok) q38_journal_line("OWNER_RELEASED\t%llu", it->map_id);
  }
  printf("RECOVER_RESULT,OK");
  for (auto &kv : outcome) printf(",%s,%d", kv.first.c_str(), kv.second);
  printf("\n");
  q38_journal_line("RECOVERED");
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s scan [MAX_ID] | restore JOURNAL\n", argv[0]); return 2; }
  int fd = open_uvm(); if (fd < 0) return 2;
  std::string mode = argv[1];
  if (mode == "scan") return scan(fd, argc > 2 ? std::stoull(argv[2]) : 4096);
  if (mode == "restore" && argc == 3) {
    setenv("Q38_RTX_PEER_ROUTE_JOURNAL", argv[2], 1);    // append outcomes to the same journal
    q38_journal_open_from_env();
    return restore(fd, argv[2]);
  }
  fprintf(stderr, "usage: %s scan [MAX_ID] | restore JOURNAL\n", argv[0]); return 2;
}
