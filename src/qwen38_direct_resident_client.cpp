// Direct Qwen3.8 request publisher for already-resident routed/shared workers.
// No request/response payload crosses a process socket, pipe, or file in this
// program: requests go through the PEX WC windows. Results use owner BAR1 in
// the baseline, or registered mapped-host mailboxes in CPU-return mode. Files
// are setup/reference inputs only.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <immintrin.h>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <xmmintrin.h>
#include "resident_cpu_mailbox_abi.cuh"
#include "resident_direct_mailbox.cuh"

namespace {
// The PEX multicast aperture remains 64 KiB. Worker BAR1 mappings also expose
// the direct result record at the newer, separately aligned 128-KiB leaf.
constexpr size_t kWindow = 64u << 10;
constexpr size_t kWorkerBarBytes = Q38_WORKER_BAR_MAP_BYTES;
constexpr size_t kResultOff = Q38_DIRECT_RESULT_OFF;
constexpr int kH = 2560, kK = 10, kRouted = 20, kShared = 4, kWorkers = 24;
constexpr int kPollTimeoutMs = 10000;
constexpr uint32_t kPleMode = 4;
constexpr uint32_t kPleRowBytes = 160;
constexpr uint32_t kPleRowsPerMailbox = 2560 * sizeof(float) / kPleRowBytes;

struct WireControl {
    uint32_t seq, layer, mode, reserved;
    uint32_t ple_count, ple_reserved[3];
    uint64_t ple_offsets[4096];
    int32_t ids[kK];
    float routes[kK];
    float x[kH];
    uint32_t cpu_mailbox_worker, cpu_mailbox_reserved;
    uint64_t cpu_mailbox_device_ptr;
};
struct DirectResult {
    volatile uint32_t seq, status, exit_reason, ready;
    volatile float result[kH];
    volatile float gate;
};
struct DirectResultSnapshot {
    uint32_t seq, status, exit_reason, ready;
    float result[kH];
    float gate;
};
static_assert(sizeof(WireControl) == 43136);
static_assert(offsetof(WireControl, ids) == 32800);
static_assert(sizeof(DirectResult) == 10260);
static_assert(sizeof(DirectResultSnapshot) == sizeof(DirectResult));
static_assert(sizeof(Q38CpuMailboxSlot) == sizeof(DirectResultSnapshot));
static_assert(kResultOff + sizeof(DirectResult) <= kWorkerBarBytes);

struct Mapping {
    int group = -1, worker = -1;
    std::string bdf;
    uint64_t cuda_va = 0, bar_bus = 0, bar_start = 0, bar_end = 0;
    int fd = -1;
    volatile uint8_t *base = nullptr;
    volatile DirectResult *result = nullptr;
};
struct PleShard {
    uint64_t global_start = 0, rows = 0, allocation_offset = 0, bytes = 0;
    int worker = -1;
};
struct PleEntry { uint32_t output_index; uint64_t offset; };
struct PleLookupBreakdown {
    double resolve_us=0, sequence_check_us=0, command_store_us=0;
    double command_fence_us=0, sequence_publish_us=0, ack_wait_us=0;
    double result_status_us=0, result_copy_us=0, total_us=0;
    uint64_t batches=0, poll_passes=0, poll_reads=0;
};
struct Request { uint32_t layer; std::array<float,kH> x; std::array<int32_t,kK> ids; std::array<float,kK> routes; };
struct Mode1PublishBreakdown {
    double request_copy_us = 0.0;
    double request_copy_cpu_us = 0.0;
    double input_validation_us = 0.0;
    double input_validation_cpu_us = 0.0;
    double input_prepare_us = 0.0;
    double sequence_check_us = 0.0;
    double payload_store_us = 0.0;
    double wc_drain_us = 0.0;
    double request_readback_us = 0.0;
    double sequence_publish_us = 0.0;
};

double thread_cpu_us() {
    timespec ts{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return double(ts.tv_sec) * 1.0e6 + double(ts.tv_nsec) / 1.0e3;
}

uint64_t parse_hex(const std::string &s) {
    return std::stoull(s, nullptr, 0);
}
std::string normalize_bdf(const std::string &bdf) {
    const auto c1=bdf.find(':');
    const auto c2=c1==std::string::npos?c1:bdf.find(':',c1+1);
    const auto dot=c2==std::string::npos?c2:bdf.find('.',c2+1);
    if(c1==std::string::npos || c2==std::string::npos || dot==std::string::npos)
        throw std::runtime_error("malformed PCI BDF: "+bdf);
    const auto domain=std::stoul(bdf.substr(0,c1),nullptr,16);
    const auto bus=std::stoul(bdf.substr(c1+1,c2-c1-1),nullptr,16);
    const auto device=std::stoul(bdf.substr(c2+1,dot-c2-1),nullptr,16);
    const auto function=std::stoul(bdf.substr(dot+1),nullptr,16);
    char normalized[16];
    std::snprintf(normalized,sizeof(normalized),"%04x:%02x:%02x.%x",
                  unsigned(domain),unsigned(bus),unsigned(device),unsigned(function));
    return normalized;
}
std::vector<std::string> split(const std::string &s, char c) {
    std::stringstream ss(s); std::string t; std::vector<std::string> out;
    while (std::getline(ss,t,c)) out.push_back(t);
    return out;
}
void require(bool ok, const std::string &msg) { if (!ok) throw std::runtime_error(msg); }
void require(bool ok, const char *msg) { if (!ok) throw std::runtime_error(msg); }

std::vector<std::array<std::string,7>> read_groups(const char *path) {
    std::ifstream f(path); require(bool(f), std::string("open groups TSV: ")+path);
    std::vector<std::array<std::string,7>> rows; std::string line;
    while (std::getline(f,line)) {
        if (line.empty()) continue;
        auto v=split(line,'\t'); require(v.size()==7,"bad groups TSV row");
        std::array<std::string,7> a; std::copy(v.begin(),v.end(),a.begin()); rows.push_back(a);
    }
    require(rows.size()==6,"expected six M10 groups"); return rows;
}
std::map<std::string,uint64_t> resource1(const std::string &bdf) {
    std::ifstream f("/sys/bus/pci/devices/"+bdf+"/resource");
    require(bool(f),"open PCI resource table for "+bdf);
    std::string line; require(bool(std::getline(f,line)),"read BAR0 for "+bdf);
    require(bool(std::getline(f,line)),"read BAR1 for "+bdf);
    std::istringstream s(line); std::string a,b,flags; s>>a>>b>>flags;
    require(bool(s),"parse BAR1 for "+bdf);
    return {{"start",parse_hex(a)},{"end",parse_hex(b)}};
}
#ifndef Q38_DIRECT_LIBRARY
Request read_request(const char *path) {
    std::ifstream f(path,std::ios::binary); require(bool(f),std::string("open request: ")+path);
    Request r{}; f.read(reinterpret_cast<char*>(&r.layer),4);
    f.read(reinterpret_cast<char*>(r.x.data()),sizeof(r.x));
    f.read(reinterpret_cast<char*>(r.ids.data()),sizeof(r.ids));
    f.read(reinterpret_cast<char*>(r.routes.data()),sizeof(r.routes));
    require(bool(f) && f.peek()==EOF,"request must be exactly layer+H+IDs+routes");
    require(r.layer<48,"request layer outside [0,47]");
    for(int i=0;i<kK;i++) require(r.ids[i]>=0&&r.ids[i]<512&&std::isfinite(r.routes[i]),"invalid expert route");
    for(float x:r.x) require(std::isfinite(x),"non-finite H element");
    return r;
}
std::vector<float> read_floats(const char *path,size_t n) {
    std::ifstream f(path,std::ios::binary); require(bool(f),std::string("open expected file: ")+path);
    std::vector<float> v(n); f.read(reinterpret_cast<char*>(v.data()),n*sizeof(float));
    require(bool(f)&&f.peek()==EOF,std::string("bad expected file size: ")+path); return v;
}
#endif
std::vector<Mapping> read_mappings(const char *path) {
    std::ifstream f(path); require(bool(f),std::string("open GPU mapping manifest: ")+path);
    std::string line;
    require(bool(std::getline(f,line))&&line=="QWEN38_GPU_MAPPINGS_MANIFEST\t1",
            "unsupported GPU mapping manifest magic/version");
    require(bool(std::getline(f,line)),"mapping manifest is missing run_id");
    auto run=split(line,'\t'); require(run.size()==2&&run[0]=="run_id"&&!run[1].empty(),
            "malformed mapping manifest run_id");
    require(bool(std::getline(f,line))&&line=="records\t24",
            "mapping manifest must declare exactly 24 records");
    require(bool(std::getline(f,line))&&line=="group\tworker\tbdf\tcuda_ordinal\tnvidia_minor\tcuda_va\tbar1_bus",
            "mapping manifest field schema mismatch");
    std::vector<Mapping> m; m.reserve(kWorkers); std::set<std::string> seen_bdfs;
    std::map<int,bool> seen;
    while(std::getline(f,line)) {
        if(line.empty()) continue;
        auto v=split(line,'\t'); require(v.size()==7,"malformed GPU mapping manifest record");
        Mapping x; x.group=std::stoi(v[0]); x.worker=std::stoi(v[1]);
        x.bdf=v[2]; x.cuda_va=parse_hex(v[5]); x.bar_bus=parse_hex(v[6]);
        const int logical=x.group*4+x.worker;
        require(x.group>=0&&x.group<6&&x.worker>=0&&x.worker<4,
                "GPU mapping manifest worker index is outside the six-M10 layout");
        require(!seen[logical],"duplicate GPU mapping manifest worker index");
        seen[logical]=true;
        require(seen_bdfs.insert(x.bdf).second,"duplicate BDF in GPU mapping manifest: "+x.bdf);
        require(std::stoi(v[3])>=0,"invalid CUDA ordinal in mapping manifest");
        require(std::stoi(v[4])>=0,"invalid NVIDIA device minor in mapping manifest");
        auto bar=resource1(x.bdf); x.bar_start=bar.at("start");x.bar_end=bar.at("end");
        require(x.bar_bus>=x.bar_start && x.bar_bus+kWorkerBarBytes-1<=x.bar_end,"RM mapping not contained in BAR1 for "+x.bdf);
        uint64_t off=x.bar_bus-x.bar_start; require((off&4095)==0,"BAR1 offset is not page aligned");
        const char *wc=getenv("Q38_BAR1_WC");
        std::string rp="/sys/bus/pci/devices/"+x.bdf+
                       ((wc&&std::strcmp(wc,"0")!=0)?"/resource1_wc":"/resource1");
        x.fd=open(rp.c_str(),O_RDWR|O_SYNC|O_CLOEXEC); require(x.fd>=0,"open "+rp+": "+strerror(errno));
        void *p=mmap(nullptr,kWorkerBarBytes,PROT_READ|PROT_WRITE,MAP_SHARED,x.fd,off);
        require(p!=MAP_FAILED,"mmap BAR1 resource for "+x.bdf+": "+strerror(errno));
        x.base=static_cast<volatile uint8_t*>(p);
        x.result=reinterpret_cast<volatile DirectResult*>(x.base+kResultOff);
        m.push_back(x);
    }
    require(m.size()==kWorkers,"expected 24 manifest records, got "+std::to_string(m.size()));
    std::sort(m.begin(),m.end(),[](const Mapping&a,const Mapping&b){return a.group*4+a.worker<b.group*4+b.worker;});
    for(int i=0;i<kWorkers;i++) require(seen[i]&&m[i].group*4+m[i].worker==i,"missing or duplicate mapping index");
    return m;
}
bool wait_all(const std::vector<Mapping>&m,uint32_t seq,
              uint64_t *poll_passes=nullptr,uint64_t *poll_reads=nullptr) {
    auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(kPollTimeoutMs);
    while(std::chrono::steady_clock::now()<end) {
        if(poll_passes)++*poll_passes;
        bool all=true;
        for(const auto&x:m) {if(poll_reads)++*poll_reads;if(x.result->seq!=seq) { all=false; break; }}
        if(all) return true;
        std::this_thread::yield();
    }
    return false;
}
void write_request(void *const mc[6],const std::vector<Mapping>&m,const Request&r,uint32_t seq,
                   Mode1PublishBreakdown *breakdown=nullptr, uint32_t mode=1) {
    require(mode==1||mode==3||mode==5,"direct request mode must be compute (1), ack (3), or echo (5)");
    WireControl routed{}; routed.layer=r.layer; routed.mode=mode;
    std::copy(r.ids.begin(),r.ids.end(),routed.ids);
    std::copy(r.routes.begin(),r.routes.end(),routed.routes);
    std::copy(r.x.begin(),r.x.end(),routed.x);
    auto phase=breakdown?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    for(int g=0;g<6;g++) {
        auto *dst=static_cast<volatile uint32_t*>(mc[g]);
        // Mode 1 does not consume the PLE count/offset array. Leave those
        // resident control fields untouched instead of broadcasting 32 KiB
        // of zero offsets on every MoE request.
        dst[offsetof(WireControl,layer)/4]=r.layer;
        dst[offsetof(WireControl,mode)/4]=mode;
        if(g<5) {
            const auto *ids=reinterpret_cast<const uint32_t*>(routed.ids);
            const auto *routes=reinterpret_cast<const uint32_t*>(routed.routes);
            for(size_t i=0;i<kK;i++) {
                dst[offsetof(WireControl,ids)/4+i]=ids[i];
                dst[offsetof(WireControl,routes)/4+i]=routes[i];
            }
        }
        const auto *x=reinterpret_cast<const uint32_t*>(routed.x);
        for(size_t i=0;i<kH;i++) dst[offsetof(WireControl,x)/4+i]=x[i];
    }
    auto mark=[&](double &dst) {
        const auto now=std::chrono::steady_clock::now();
        dst=std::chrono::duration<double,std::micro>(now-phase).count();
        phase=now;
    };
    if(breakdown) mark(breakdown->payload_store_us);
    _mm_sfence(); std::atomic_thread_fence(std::memory_order_seq_cst);
    if(breakdown) mark(breakdown->wc_drain_us);
    // Optional diagnostic readback of the multicast destinations. This is not
    // the completion protocol: the sequence publication and per-worker ack
    // below provide the request/response boundary. Reading all 24 endpoint
    // BAR mappings here adds 24 serialized PCIe reads to every request.
    const char *readback_env=std::getenv("Q38_REQUEST_READBACK_MODE");
    const char *readback_mode=readback_env?readback_env:"all";
    require(std::strcmp(readback_mode,"all")==0||std::strcmp(readback_mode,"group")==0||
            std::strcmp(readback_mode,"none")==0,"Q38_REQUEST_READBACK_MODE must be all, group, or none");
    constexpr size_t kIdsWord = offsetof(WireControl,ids)/4;
    constexpr size_t kRoutesWord = offsetof(WireControl,routes)/4;
    constexpr size_t kXWord = offsetof(WireControl,x)/4;
    for(int i=0;i<kWorkers;i++) {
        if(std::strcmp(readback_mode,"none")==0) break;
        if(std::strcmp(readback_mode,"group")==0 && (i%4)!=0) continue;
        const auto *p=reinterpret_cast<const volatile uint32_t*>(m[i].base);
        const bool routed_worker=i<kRouted;
        const auto *ids=reinterpret_cast<const uint32_t*>(routed.ids);
        const auto *routes=reinterpret_cast<const uint32_t*>(routed.routes);
        const auto *x=reinterpret_cast<const uint32_t*>(routed.x);
        const bool wrong_common=p[1]!=r.layer||p[2]!=mode||p[kXWord]!=x[0]||
            p[kXWord+kH/2]!=x[kH/2]||p[kXWord+kH-1]!=x[kH-1];
        const bool wrong_route=routed_worker&&
            (p[kIdsWord]!=ids[0]||p[kRoutesWord]!=routes[0]);
        if(wrong_common||wrong_route) {
            auto *wc=static_cast<volatile uint32_t*>(mc[i/4]);
            char msg[512];
            std::snprintf(msg,sizeof(msg),"GPU BAR1 request readback mismatch bdf=%s pex=[%08x,%08x,%08x,%08x] bar=[%08x,%08x,%08x,%08x] expected=[layer=%u,mode=%u,id0=%08x,route0=%08x,x0=%08x,xmid=%08x,xlast=%08x]",
                m[i].bdf.c_str(),wc[1],wc[2],wc[kIdsWord],wc[kXWord],p[1],p[2],p[kIdsWord],p[kXWord],r.layer,mode,ids[0],routes[0],x[0],x[kH/2],x[kH-1]);
            throw std::runtime_error(msg);
        }
    }
    if(breakdown) mark(breakdown->request_readback_us);
    for(int g=0;g<6;g++) static_cast<volatile uint32_t*>(mc[g])[0]=seq;
    _mm_sfence(); std::atomic_thread_fence(std::memory_order_seq_cst);
    if(breakdown) mark(breakdown->sequence_publish_us);
}

struct DirectState {
    std::vector<Mapping> mappings;
    std::vector<PleShard> ple_shards;
    unsigned gather_threads = 1;
    Q38CpuMailboxSlot *cpu_mailbox = nullptr;
    int mc_fd=-1;
    void *mc[6]={MAP_FAILED,MAP_FAILED,MAP_FAILED,MAP_FAILED,MAP_FAILED,MAP_FAILED};
    ~DirectState() {
        for(void *p:mc) if(p!=MAP_FAILED) munmap(p,kWindow);
        if(mc_fd>=0) close(mc_fd);
        for(const auto &m:mappings) {
            if(m.base) munmap(const_cast<uint8_t*>(m.base),kWorkerBarBytes);
            if(m.fd>=0) close(m.fd);
        }
    }
};
Q38CpuMailboxSlot &cpu_mailbox_slot(DirectState &state, int worker) {
    require(state.cpu_mailbox != nullptr, "CPU mailbox is not configured");
    require(worker >= 0 && worker < kWorkers, "CPU mailbox worker index is invalid");
    auto *base = reinterpret_cast<uint8_t *>(state.cpu_mailbox);
    return *reinterpret_cast<Q38CpuMailboxSlot *>(
        base + size_t(worker) * Q38_CPU_MAILBOX_SLOT_STRIDE);
}
std::unique_ptr<DirectState> g_api;
std::mutex g_api_mutex;
std::string g_api_error;

void api_open(const char *groups_path,const char *maps_path) {
    require(!g_api,"direct mailbox API already open");
    auto s=std::make_unique<DirectState>();
    if(const char *v=getenv("Q38_GATHER_THREADS")) {
        const long n=std::strtol(v,nullptr,10);
        require(n>=1&&n<=kWorkers,"Q38_GATHER_THREADS must be in [1,24]");
        s->gather_threads=static_cast<unsigned>(n);
    }
    (void)read_groups(groups_path);
    s->mappings=read_mappings(maps_path);
    s->mc_fd=open("/dev/m10_qwen_mc_wc6",O_RDWR|O_CLOEXEC);
    require(s->mc_fd>=0,std::string("open PEX mapper: ")+strerror(errno));
    for(int g=0;g<6;g++) {
        s->mc[g]=mmap(nullptr,kWindow,PROT_READ|PROT_WRITE,MAP_SHARED,s->mc_fd,g*kWindow);
        require(s->mc[g]!=MAP_FAILED,std::string("mmap PEX window ")+std::to_string(g)+": "+strerror(errno));
    }
    uint32_t seq=reinterpret_cast<const volatile uint32_t*>(s->mappings[0].base)[0];
    require(seq!=UINT32_MAX,"invalid sequence readback from BAR1");
    for(const auto &m:s->mappings) {
        require(m.result->ready==1,"resident direct mailbox not ready at "+m.bdf);
        require(reinterpret_cast<const volatile uint32_t*>(m.base)[0]==seq,
                "worker command sequences are not aligned at API open");
    }
    g_api=std::move(s);
}

void api_configure_ple(const char *groups_path,const char *manifest_path) {
    require(bool(g_api),"direct mailbox API must be opened before PLE configuration");
    require(manifest_path && *manifest_path,"empty PLE IPC manifest path");
    std::map<std::string,int> worker_for_bdf;
    const auto groups=read_groups(groups_path);
    for(size_t g=0;g<groups.size();g++) {
        const auto bdfs=split(groups[g][3],',');
        require(bdfs.size()==4,"PLE group has an invalid GPU list");
        for(size_t w=0;w<4;w++) worker_for_bdf[normalize_bdf(bdfs[w])]=int(g*4+w);
    }
    std::ifstream f(manifest_path);
    require(bool(f),std::string("open PLE IPC manifest: ")+manifest_path);
    struct RawShard { uint64_t rows,offset,bytes; int worker; };
    std::map<uint32_t,RawShard> by_id;
    std::string line;
    while(std::getline(f,line)) {
        if(line.rfind("SHARD\t",0)!=0) continue;
        auto v=split(line,'\t');
        require(v.size()==8,"malformed SHARD row in PLE IPC manifest");
        if(v[1]=="id") continue;
        const uint32_t id=uint32_t(std::stoul(v[1]));
        const std::string bdf=v[3];
        auto it=worker_for_bdf.find(normalize_bdf(bdf));
        require(it!=worker_for_bdf.end(),"PLE shard BDF is outside six-card worker topology: "+bdf);
        const uint64_t bytes=std::stoull(v[4]), offset=std::stoull(v[5]), rows=std::stoull(v[6]);
        require(v[7]=="160" && bytes==rows*kPleRowBytes,"invalid PLE shard row size");
        require(by_id.emplace(id,RawShard{rows,offset,bytes,it->second}).second,
                "duplicate PLE shard id");
    }
    require(!by_id.empty(),"PLE IPC manifest contains no shards");
    g_api->ple_shards.clear();
    for(const auto &[id,r]:by_id) {
        // PLE global row numbering follows fixed shard IDs even when only a
        // bounded subset is resident. Missing shards remain rejected below.
        g_api->ple_shards.push_back({uint64_t(id)*r.rows,r.rows,r.offset,r.bytes,r.worker});
    }
    std::sort(g_api->ple_shards.begin(),g_api->ple_shards.end(),
        [](const PleShard&a,const PleShard&b){return a.global_start<b.global_start;});
}

int api_ple_lookup(const uint64_t *global_rows,uint32_t count,uint8_t *output,
                   double *elapsed_us,PleLookupBreakdown *breakdown=nullptr) {
    require(bool(g_api),"direct mailbox API is not open");
    require(!g_api->ple_shards.empty(),"direct PLE manifest is not configured");
    require(count==0 || (global_rows && output),"null direct PLE input/output");
    if(count==0) { if(elapsed_us)*elapsed_us=0; return 0; }
    const auto total_started=std::chrono::steady_clock::now();
    std::vector<std::vector<PleEntry>> per_worker(kWorkers);
    for(uint32_t i=0;i<count;i++) {
        const uint64_t row=global_rows[i];
        auto it=std::upper_bound(g_api->ple_shards.begin(),g_api->ple_shards.end(),row,
            [](uint64_t value,const PleShard &s){return value<s.global_start;});
        require(it!=g_api->ple_shards.begin(),"PLE global row precedes first shard");
        --it;
        const uint64_t local=row-it->global_start;
        require(local<it->rows,"PLE global row lies beyond manifest");
        per_worker[it->worker].push_back({i,it->allocation_offset+local*kPleRowBytes});
    }
    size_t batches=0;
    for(const auto &v:per_worker) batches=std::max(batches,(v.size()+kPleRowsPerMailbox-1)/kPleRowsPerMailbox);
    const auto started=std::chrono::steady_clock::now();
    auto mark=[&](double &slot,const auto &a,const auto &b){slot+=std::chrono::duration<double,std::micro>(b-a).count();};
    if(breakdown){breakdown->batches=batches;breakdown->resolve_us=std::chrono::duration<double,std::micro>(started-total_started).count();}
    for(size_t batch=0;batch<batches;batch++) {
        auto phase=std::chrono::steady_clock::now();
        const uint32_t previous=reinterpret_cast<const volatile uint32_t*>(g_api->mappings[0].base)[0];
        require(previous<UINT32_MAX-1,"direct PLE sequence space exhausted");
        for(const auto &m:g_api->mappings)
            require(reinterpret_cast<const volatile uint32_t*>(m.base)[0]==previous,
                    "worker PLE command sequences diverged");
        const uint32_t seq=previous+1;
        auto now=std::chrono::steady_clock::now();if(breakdown)mark(breakdown->sequence_check_us,phase,now);
        phase=now;
        for(int worker=0;worker<kWorkers;worker++) {
            const size_t begin=batch*kPleRowsPerMailbox;
            const size_t n=begin<per_worker[worker].size()?
                std::min<size_t>(kPleRowsPerMailbox,per_worker[worker].size()-begin):0;
            auto *control=reinterpret_cast<volatile uint32_t*>(g_api->mappings[worker].base);
            control[2]=kPleMode;
            control[4]=uint32_t(n);
            for(size_t j=0;j<n;j++) {
                const auto &entry=per_worker[worker][begin+j];
                // ple_offsets is a naturally aligned uint64_t array at byte
                // offset 32. Adjacent entries occupy two 32-bit words each.
                auto *offsets=reinterpret_cast<volatile uint64_t*>(
                    reinterpret_cast<volatile uint8_t*>(control)+32);
                offsets[j]=entry.offset;
            }
        }
        now=std::chrono::steady_clock::now();if(breakdown)mark(breakdown->command_store_us,phase,now);
        phase=now;_mm_sfence(); std::atomic_thread_fence(std::memory_order_seq_cst);
        now=std::chrono::steady_clock::now();if(breakdown)mark(breakdown->command_fence_us,phase,now);
        phase=now;
        for(const auto &m:g_api->mappings)
            reinterpret_cast<volatile uint32_t*>(m.base)[0]=seq;
        _mm_sfence(); std::atomic_thread_fence(std::memory_order_seq_cst);
        now=std::chrono::steady_clock::now();if(breakdown)mark(breakdown->sequence_publish_us,phase,now);
        phase=now;
        require(wait_all(g_api->mappings,seq,breakdown?&breakdown->poll_passes:nullptr,
                         breakdown?&breakdown->poll_reads:nullptr),"resident direct PLE gather timed out; retain worker mappings");
        now=std::chrono::steady_clock::now();if(breakdown)mark(breakdown->ack_wait_us,phase,now);
        phase=now;
        for(int worker=0;worker<kWorkers;worker++) {
            const auto &m=g_api->mappings[worker];
            if(m.result->seq!=seq || m.result->status!=0) {
                throw std::runtime_error("resident direct PLE worker failed at "+m.bdf+
                    " seq="+std::to_string(m.result->seq)+" expected="+std::to_string(seq)+
                    " status="+std::to_string(m.result->status)+
                    " exit_reason="+std::to_string(m.result->exit_reason));
            }
        }
        now=std::chrono::steady_clock::now();
        if(breakdown)mark(breakdown->result_status_us,phase,now);
        phase=now;
        const char *width_env=std::getenv("Q38_PLE_READ_WIDTH");
        const unsigned width=width_env?static_cast<unsigned>(std::strtoul(width_env,nullptr,10)):1u;
        require(width==1||width==4||width==8||width==16||width==32,
                "Q38_PLE_READ_WIDTH must be 1, 4, 8, 16, or 32");
        for(int worker=0;worker<kWorkers;worker++) {
            const auto &m=g_api->mappings[worker];
            const size_t begin=batch*kPleRowsPerMailbox;
            const size_t n=begin<per_worker[worker].size()?
                std::min<size_t>(kPleRowsPerMailbox,per_worker[worker].size()-begin):0;
            const volatile uint8_t *src=reinterpret_cast<const volatile uint8_t*>(m.result->result);
            for(size_t j=0;j<n;j++) {
                auto *dst=output+size_t(per_worker[worker][begin+j].output_index)*kPleRowBytes;
                const size_t source_offset=j*kPleRowBytes;
                const unsigned alignment=width==32?16:width;
                require((reinterpret_cast<uintptr_t>(src+source_offset)&(alignment-1))==0,
                        "PLE result row is not aligned for selected BAR read width");
                require(kPleRowBytes%width==0,"PLE row size is not divisible by selected BAR read width");
                if(width==1) {
                    for(size_t b=0;b<kPleRowBytes;b++) dst[b]=src[source_offset+b];
                } else if(width==4) {
                    const auto *src32=reinterpret_cast<const volatile uint32_t*>(src+source_offset);
                    for(size_t b=0;b<kPleRowBytes;b+=4) {
                        const uint32_t value=src32[b/4];
                        std::memcpy(dst+b,&value,sizeof(value));
                    }
                } else if(width==8) {
                    const auto *src64=reinterpret_cast<const volatile uint64_t*>(src+source_offset);
                    for(size_t b=0;b<kPleRowBytes;b+=8) {
                        const uint64_t value=src64[b/8];
                        std::memcpy(dst+b,&value,sizeof(value));
                    }
                } else if(width==16) {
                    for(size_t b=0;b<kPleRowBytes;b+=16) {
                        __m128i value;
                        const volatile uint8_t *address=src+source_offset+b;
                        asm volatile("movdqa (%1), %0" : "=x"(value) : "r"(address) : "memory");
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst+b),value);
                    }
                } else {
                    for(size_t b=0;b<kPleRowBytes;b+=32) {
                        __m256i value;
                        const volatile uint8_t *address=src+source_offset+b;
                        asm volatile("vmovdqu (%1), %0" : "=x"(value) : "r"(address) : "memory");
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+b),value);
                    }
                }
            }
        }
        now=std::chrono::steady_clock::now();if(breakdown)mark(breakdown->result_copy_us,phase,now);
    }
    const auto finished=std::chrono::steady_clock::now();
    if(elapsed_us)*elapsed_us=std::chrono::duration<double,std::micro>(finished-started).count();
    if(breakdown)breakdown->total_us=std::chrono::duration<double,std::micro>(finished-total_started).count();
    return 0;
}

int api_run(uint32_t layer,const float*h,const int32_t*ids,const float*routes,
            float*out,double*publish_us,double*ack_us,double*gather_us,double*join_us,
            Mode1PublishBreakdown *publish_breakdown=nullptr, uint32_t mode=1) {
    const auto api_started=std::chrono::steady_clock::now();
    require(bool(g_api),"direct mailbox API is not open");
    require(layer<48,"layer id outside [0,47]");
    require(h&&ids&&routes&&out,"null direct request/output pointer");
    const auto copy_started=std::chrono::steady_clock::now();
    const double copy_cpu_started=thread_cpu_us();
    Request r{};r.layer=layer;
    std::copy(h,h+kH,r.x.begin());std::copy(ids,ids+kK,r.ids.begin());std::copy(routes,routes+kK,r.routes.begin());
    const auto copy_done=std::chrono::steady_clock::now();
    const double copy_cpu_done=thread_cpu_us();
    const double validation_cpu_started=copy_cpu_done;
    float wsum=0.0f;
    for(int k=0;k<kK;k++) {
        require(r.ids[k]>=0&&r.ids[k]<512,"expert id outside [0,511]");
        require(std::isfinite(r.routes[k])&&r.routes[k]>=0,"invalid route weight");
        require(std::isfinite(r.x[k]),"non-finite initial H elements");
        for(int j=0;j<k;j++) require(r.ids[j]!=r.ids[k],"duplicate expert id");
        wsum+=r.routes[k];
    }
    for(int j=kK;j<kH;j++) require(std::isfinite(r.x[j]),"non-finite H element");
    require(std::abs(wsum-1.0f)<=1e-5f,"route weights must sum to one");
    const auto input_prepared=std::chrono::steady_clock::now();
    const double validation_cpu_done=thread_cpu_us();
    if(publish_breakdown) {
        publish_breakdown->request_copy_us=std::chrono::duration<double,std::micro>(copy_done-copy_started).count();
        publish_breakdown->request_copy_cpu_us=copy_cpu_done-copy_cpu_started;
        publish_breakdown->input_validation_us=std::chrono::duration<double,std::micro>(input_prepared-copy_done).count();
        publish_breakdown->input_validation_cpu_us=validation_cpu_done-validation_cpu_started;
    }
    uint32_t previous=reinterpret_cast<const volatile uint32_t*>(g_api->mappings[0].base)[0];
    require(previous<UINT32_MAX-1,"sequence space exhausted");
    for(const auto&m:g_api->mappings)
        require(reinterpret_cast<const volatile uint32_t*>(m.base)[0]==previous,
                "worker command sequences diverged before publication");
    const uint32_t seq=previous+1;
    auto t0=std::chrono::steady_clock::now();
    if(publish_breakdown) {
        publish_breakdown->input_prepare_us=std::chrono::duration<double,std::micro>(input_prepared-api_started).count();
        publish_breakdown->sequence_check_us=std::chrono::duration<double,std::micro>(t0-input_prepared).count();
    }
    write_request(g_api->mc,g_api->mappings,r,seq,publish_breakdown,mode);
    auto tpub=std::chrono::steady_clock::now();
    if(g_api->cpu_mailbox) {
        auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(kPollTimeoutMs);
        bool all=false;
        while(std::chrono::steady_clock::now()<deadline) {
            all=true;
            for(int i=0;i<kWorkers;i++)
                if(cpu_mailbox_slot(*g_api,i).seq!=seq){all=false;break;}
            if(all)break;
            std::this_thread::yield();
        }
        require(all,"CPU mailbox sequence timeout; retain shared memory and resident workers");
    } else {
        require(wait_all(g_api->mappings,seq),"BAR1 result sequence timeout; retain workers and mappings");
    }
    auto tack=std::chrono::steady_clock::now();
    if(mode==3) {
        std::fill(out,out+kH,0.0f);
        auto us=[](auto a,auto b){return std::chrono::duration<double,std::micro>(b-a).count();};
        if(publish_us)*publish_us=us(t0,tpub);
        if(ack_us)*ack_us=us(tpub,tack);
        if(gather_us)*gather_us=0.0;
        if(join_us)*join_us=0.0;
        return 0;
    }
    std::array<DirectResultSnapshot,kWorkers> snapshot{};
    auto copy_one=[&](int i) {
        if(g_api->cpu_mailbox) {
            const auto &slot=cpu_mailbox_slot(*g_api,i);
            require(slot.status==0,"CPU mailbox returned error status at worker "+std::to_string(i));
            require(slot.worker==uint32_t(i),"CPU mailbox worker identity mismatch at slot "+std::to_string(i));
            require(slot.layer==layer,"CPU mailbox layer mismatch at worker "+std::to_string(i));
            std::memcpy(&snapshot[i],&slot,sizeof(snapshot[i]));
            return;
        }
        const auto &m=g_api->mappings[i];
        const auto *src=const_cast<const uint8_t*>(m.base)+kResultOff;
        std::memcpy(&snapshot[i],src,sizeof(snapshot[i]));
    };
    const unsigned nt=g_api->cpu_mailbox?1:std::min<unsigned>(g_api->gather_threads,kWorkers);
    if(nt==1) {
        for(int i=0;i<kWorkers;i++) copy_one(i);
    } else {
        std::vector<std::thread> readers;
        readers.reserve(nt);
        for(unsigned t=0;t<nt;t++) readers.emplace_back([&,t]{
            const int begin=static_cast<int>(t*kWorkers/nt);
            const int end=static_cast<int>((t+1)*kWorkers/nt);
            for(int i=begin;i<end;i++) copy_one(i);
        });
        for(auto &reader:readers) reader.join();
    }
    for(int i=0;i<kWorkers;i++) {
        const auto &m=g_api->mappings[i];
        require(snapshot[i].seq==seq,g_api->cpu_mailbox?
                "CPU mailbox sequence changed during snapshot at worker "+std::to_string(i):
                "BAR1 result sequence changed during snapshot at "+m.bdf);
        require(snapshot[i].status==0,g_api->cpu_mailbox?
                "CPU mailbox returned error status at worker "+std::to_string(i):
                "resident worker returned error status at "+m.bdf);
    }
    if(mode==5) {
        for(int i=0;i<kWorkers;i++) for(int j=0;j<kH;j++)
            require(snapshot[i].result[j]==r.x[j],"echo payload mismatch at worker "+
                    std::to_string(i)+" element "+std::to_string(j));
    }
    auto tgather=std::chrono::steady_clock::now();
    const float gate=snapshot[kRouted].gate;
    require(std::isfinite(gate),"shared gate result is not finite");
    for(int j=0;j<kH;j++) {
        float sum=0.0f;
        for(int i=0;i<kRouted;i++) sum+=snapshot[i].result[j];
        float shared_sum=0.0f;
        for(int i=kRouted;i<kWorkers;i++) shared_sum+=snapshot[i].result[j];
        out[j]=sum+gate*shared_sum;
    }
    auto tjoin=std::chrono::steady_clock::now();
    auto us=[](auto a,auto b){return std::chrono::duration<double,std::micro>(b-a).count();};
    if(publish_us)*publish_us=us(t0,tpub);
    if(ack_us)*ack_us=us(tpub,tack);
    if(gather_us)*gather_us=us(tack,tgather);
    if(join_us)*join_us=us(tgather,tjoin);
    return 0;
}
}

#ifdef Q38_DIRECT_LIBRARY
extern "C" int q38_direct_open(const char *groups_path,const char *maps_path,char *err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try { api_open(groups_path,maps_path);g_api_error.clear();return 0; }
    catch(const std::exception&e){g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;}
}
extern "C" int q38_direct_configure_ple(const char *groups_path,const char *manifest_path,
                                           char *err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try { api_configure_ple(groups_path,manifest_path);g_api_error.clear();return 0; }
    catch(const std::exception&e){g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;}
}
extern "C" int q38_direct_ple_lookup(const uint64_t *rows,uint32_t count,uint8_t *output,
                                      double *elapsed_us,char *err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try { int rc=api_ple_lookup(rows,count,output,elapsed_us);g_api_error.clear();return rc; }
    catch(const std::exception&e){g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;}
}
extern "C" int q38_direct_ple_lookup_diagnostic(const uint64_t *rows,uint32_t count,uint8_t *output,
                                      double *elapsed_us,double *phases,uint64_t *counters,
                                      char *err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try {
        PleLookupBreakdown b;
        int rc=api_ple_lookup(rows,count,output,elapsed_us,&b);
        if(phases){phases[0]=b.resolve_us;phases[1]=b.sequence_check_us;phases[2]=b.command_store_us;
          phases[3]=b.command_fence_us;phases[4]=b.sequence_publish_us;phases[5]=b.ack_wait_us;
          phases[6]=b.result_status_us;phases[7]=b.result_copy_us;phases[8]=b.total_us;}
        if(counters){counters[0]=b.batches;counters[1]=b.poll_passes;counters[2]=b.poll_reads;}
        g_api_error.clear();return rc;
    } catch(const std::exception&e) {
        g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;
    }
}
extern "C" int q38_direct_run(uint32_t layer,const float*h,const int32_t*ids,const float*routes,
                               float*out,double*publish_us,double*ack_us,double*gather_us,double*join_us,
                               char*err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try { int rc=api_run(layer,h,ids,routes,out,publish_us,ack_us,gather_us,join_us);g_api_error.clear();return rc; }
    catch(const std::exception&e){g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;}
}
extern "C" int q38_direct_transport_round(uint32_t layer,const float*h,const int32_t*ids,
                               const float*routes,double*publish_us,double*ack_us,
                               char*err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try {
        std::array<float,kH> unused{};
        int rc=api_run(layer,h,ids,routes,unused.data(),publish_us,ack_us,nullptr,nullptr,nullptr,3);
        g_api_error.clear();return rc;
    } catch(const std::exception&e) {
        g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;
    }
}
extern "C" int q38_direct_echo_round(uint32_t layer,const float*h,const int32_t*ids,
                               const float*routes,float*out,double*publish_us,double*ack_us,
                               double*gather_us,double*join_us,char*err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try {
        int rc=api_run(layer,h,ids,routes,out,publish_us,ack_us,gather_us,join_us,nullptr,5);
        g_api_error.clear();return rc;
    } catch(const std::exception&e) {
        g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;
    }
}
extern "C" int q38_direct_run_diagnostic(uint32_t layer,const float*h,const int32_t*ids,
                               const float*routes,float*out,double*publish_us,double*ack_us,
                               double*gather_us,double*join_us,double*payload_store_us,
                               double*wc_drain_us,double*request_readback_us,
                               double*sequence_publish_us,double*request_copy_us,
                               double*request_copy_cpu_us,double*input_validation_us,
                               double*input_validation_cpu_us,double*input_prepare_us,
                               double*sequence_check_us,char*err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try {
        Mode1PublishBreakdown breakdown;
        int rc=api_run(layer,h,ids,routes,out,publish_us,ack_us,gather_us,join_us,&breakdown);
        if(payload_store_us)*payload_store_us=breakdown.payload_store_us;
        if(wc_drain_us)*wc_drain_us=breakdown.wc_drain_us;
        if(request_readback_us)*request_readback_us=breakdown.request_readback_us;
        if(sequence_publish_us)*sequence_publish_us=breakdown.sequence_publish_us;
        if(request_copy_us)*request_copy_us=breakdown.request_copy_us;
        if(request_copy_cpu_us)*request_copy_cpu_us=breakdown.request_copy_cpu_us;
        if(input_validation_us)*input_validation_us=breakdown.input_validation_us;
        if(input_validation_cpu_us)*input_validation_cpu_us=breakdown.input_validation_cpu_us;
        if(input_prepare_us)*input_prepare_us=breakdown.input_prepare_us;
        if(sequence_check_us)*sequence_check_us=breakdown.sequence_check_us;
        g_api_error.clear();return rc;
    } catch(const std::exception&e) {
        g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;
    }
}
extern "C" int q38_direct_set_cpu_mailbox(void *base,size_t bytes,char*err,size_t errcap) {
    std::lock_guard<std::mutex> lock(g_api_mutex);
    try {
        require(bool(g_api),"direct mailbox API is not open");
        require(base!=nullptr,"CPU mailbox base pointer is null");
        require(bytes>=Q38_CPU_MAILBOX_BYTES,"CPU mailbox mapping is smaller than the ABI requires");
        require((reinterpret_cast<uintptr_t>(base)&(alignof(Q38CpuMailboxSlot)-1))==0,
                "CPU mailbox base is not naturally aligned");
        g_api->cpu_mailbox=static_cast<Q38CpuMailboxSlot*>(base);
        g_api_error.clear();
        return 0;
    } catch(const std::exception&e) {
        g_api_error=e.what();if(err&&errcap)std::snprintf(err,errcap,"%s",g_api_error.c_str());return 1;
    }
}
extern "C" void q38_direct_close() {
    std::lock_guard<std::mutex> lock(g_api_mutex);g_api.reset();
}
#endif

#ifndef Q38_DIRECT_LIBRARY
int main(int argc,char**argv) {
    if(argc!=10) {
        std::fprintf(stderr,"usage: %s GROUPS_TSV GPU_MAPPINGS.manifest REQUEST.bin OBSERVED_DIR ROUTED_PARTS.f32|- SHARED_PARTS.f32|- ROUTED_OUT.f32|- SHARED_OUT.f32|- SEQ\n",argv[0]); return 2;
    }
    try {
        auto groups=read_groups(argv[1]); auto maps=read_mappings(argv[2]); auto req=read_request(argv[3]);
        std::vector<float> route_ref,shared_ref,route_out,shared_out;
        if(std::string(argv[5])!="-") route_ref=read_floats(argv[5],size_t(kRouted)*kH);
        if(std::string(argv[6])!="-") shared_ref=read_floats(argv[6],size_t(kShared)*kH);
        if(std::string(argv[7])!="-") route_out=read_floats(argv[7],kH);
        if(std::string(argv[8])!="-") shared_out=read_floats(argv[8],kH);
        uint32_t seq=uint32_t(std::stoul(argv[9])); require(seq>=2,"sequence must follow bootstrap sequence 1");
        int fd=open("/dev/m10_qwen_mc_wc6",O_RDWR|O_CLOEXEC); require(fd>=0,std::string("open PEX mapper: ")+strerror(errno));
        void*mc[6]; for(int g=0;g<6;g++){mc[g]=mmap(nullptr,kWindow,PROT_READ|PROT_WRITE,MAP_SHARED,fd,g*kWindow);require(mc[g]!=MAP_FAILED,"mmap PEX multicast aperture");}
        for(const auto&x:maps) require(x.result->ready==1,"resident result mailbox not ready for "+x.bdf);
        auto t0=std::chrono::steady_clock::now(); write_request(mc,maps,req,seq); auto tpub=std::chrono::steady_clock::now();
        if(!wait_all(maps,seq)) throw std::runtime_error("BAR1 result sequence timeout; keep services/mappings alive for recovery");
        auto tdone=std::chrono::steady_clock::now();
        size_t mismatches=0; float max_abs=0; int first=-1; std::vector<float> agg(kH,0.0f);
        std::vector<float> route_obs(size_t(kRouted)*kH),shared_obs(size_t(kShared)*kH);
        for(int i=0;i<kWorkers;i++) {
            const auto&x=maps[i]; require(x.result->status==0,"resident returned nonzero status at worker "+std::to_string(i));
            if(i<kRouted) {
                for(int j=0;j<kH;j++){float got=x.result->result[j];route_obs[size_t(i)*kH+j]=got;if(!route_ref.empty()){float ref=route_ref[size_t(i)*kH+j],d=std::abs(got-ref);max_abs=std::max(max_abs,d);if(d>1e-6f+1e-6f*std::abs(ref)){if(first<0)first=i*kH+j;++mismatches;}}agg[j]+=got;}
            } else {
                int w=i-kRouted; for(int j=0;j<kH;j++){float got=x.result->result[j];shared_obs[size_t(w)*kH+j]=got;if(!shared_ref.empty()){float ref=shared_ref[size_t(w)*kH+j],d=std::abs(got-ref);max_abs=std::max(max_abs,d);if(d>1e-6f+1e-6f*std::abs(ref)){if(first<0)first=i*kH+j;++mismatches;}}agg[j]+=got*(w==0?x.result->gate:maps[kRouted].result->gate);}
                if(w==0) require(std::isfinite(x.result->gate),"shared gate is not finite");
            }
        }
        if(!route_out.empty()&&!shared_out.empty()) for(int j=0;j<kH;j++){float ref=route_out[j]+shared_out[j];float d=std::abs(agg[j]-ref);max_abs=std::max(max_abs,d);if(d>2e-5f+2e-5f*std::abs(ref)){if(first<0)first=kWorkers*kH+j;++mismatches;}}
        auto us=[](auto a,auto b){return std::chrono::duration<double,std::micro>(b-a).count();};
        std::printf("DIRECT_REQUEST,seq,%u,layer,%u,publish_us,%.3f,response_us,%.3f,checked_us,%.3f,mismatches,%zu,max_abs,%.9g,first,%d\n",seq,req.layer,us(t0,tpub),us(tpub,tdone),us(t0,tdone),mismatches,max_abs,first);
        auto write=[&](const std::string&name,const std::vector<float>&v){std::ofstream f(std::string(argv[4])+"/"+name,std::ios::binary);require(bool(f),"create observation file "+name);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));require(bool(f),"write observation file "+name);};
        write("routed_parts.f32",route_obs);write("shared_parts.f32",shared_obs);write("aggregate.f32",agg);
        if(mismatches) throw std::runtime_error("direct BAR1 results differ from same-request resident reference");
        for(int g=0;g<6;g++) munmap(mc[g],kWindow);
        close(fd);
        for(const auto&x:maps){munmap(const_cast<uint8_t*>(x.base),kWindow);close(x.fd);}
        std::puts("DIRECT_BAR1_LAYER_REQUEST,PASS"); return 0;
    } catch(const std::exception&e) { std::fprintf(stderr,"DIRECT_BAR1_LAYER_REQUEST,FAIL,%s\n",e.what()); return 1; }
}
#endif
