#pragma once
#include <cuda.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <inttypes.h>
#include <cerrno>
#include <cstring>
#include <sstream>
constexpr size_t QWEN_ALLOC = 2ull << 20;
constexpr size_t QWEN_MAP = 64ull << 10;
constexpr size_t QWEN_INPUT_OFF = 0x2000;
constexpr size_t QWEN_MC_OFF = 0x2000;
constexpr uint32_t QWEN_MAP_REFLECTED = 2u << 15;
constexpr uint64_t QWEN_MC_INDEX = 16;

struct Nvos33Q { uint32_t c,d,m,p; uint64_t off,len,addr; uint32_t status,flags; int32_t fd; uint32_t tail; };
struct Nvos34Q { uint32_t c,d,m,p; uint64_t addr; uint32_t status,flags; };
using ExportGetterQ = int(*)(uint32_t*,uint32_t*,uint32_t*,int*);

static void cucheck(CUresult e,const char *where) {
    if(e!=CUDA_SUCCESS){const char*n=nullptr,*s=nullptr;cuGetErrorName(e,&n);cuGetErrorString(e,&s);
        throw std::runtime_error(std::string(where)+": "+(n?n:"CUDA error")+": "+(s?s:"unknown"));}
}
#define CUKQ(x) cucheck((x),#x)
static void syscheck(int r,const char*w){if(r<0)throw std::runtime_error(std::string(w)+": "+strerror(errno));}
static std::string qhex(uint64_t x){char b[32];snprintf(b,sizeof(b),"0x%llx",(unsigned long long)x);return b;}
static std::vector<std::string> qsplit(const std::string&s,char c){std::vector<std::string>v;std::stringstream z(s);std::string t;while(std::getline(z,t,c))v.push_back(t);return v;}

struct QGroup { std::string serial,up; std::vector<std::string>ports,gpus; uint64_t lo=0,hi=0,base=0; };
static std::vector<QGroup> qgroups(const char *path){
    std::ifstream f(path);if(!f)throw std::runtime_error("open six-card group manifest");
    std::vector<QGroup> out;std::string l;
    while(std::getline(f,l)){if(l.empty())continue;auto x=qsplit(l,'\t');if(x.size()!=7)throw std::runtime_error("bad group manifest row");
        QGroup g;g.serial=x[0];g.up=x[1];g.ports=qsplit(x[2],',');g.gpus=qsplit(x[3],',');
        g.lo=strtoull(x[4].c_str(),nullptr,0);g.hi=strtoull(x[5].c_str(),nullptr,0);g.base=strtoull(x[6].c_str(),nullptr,0);
        if(g.ports.size()!=4||g.gpus.size()!=4||g.base%QWEN_MAP||g.base<g.lo||g.base+QWEN_MAP-1>g.hi)throw std::runtime_error("invalid group resource row");out.push_back(g);}
    if(out.size()!=6)throw std::runtime_error("expected six M10 groups");return out;
}
static int nvidia_minor(const std::string&bdf){
    // CUDA accepts either case for PCI IDs, but procfs GPU directory names
    // are lowercase. Late-enumerated M10s use BDFs such as (example only) 0000:0B:00.0;
    // using that spelling directly silently breaks the procfs lookup.
    std::string proc_bdf=bdf;
    for(char &c:proc_bdf)if(c>='A'&&c<='F')c=char(c-'A'+'a');
    std::ifstream f("/proc/driver/nvidia/gpus/"+proc_bdf+"/information");std::string l;
    while(std::getline(f,l))if(l.rfind("Device Minor:",0)==0){int n=-1;std::istringstream(l.substr(13))>>n;if(n>=0)return n;}
    throw std::runtime_error("NVIDIA device minor missing for "+bdf);
}
static std::pair<uint64_t,uint64_t> qbar1(const std::string&bdf){
    std::string sysfs_bdf=bdf;
    for(char &c:sysfs_bdf)if(c>='A'&&c<='F')c=char(c-'A'+'a');
    std::ifstream f("/sys/bus/pci/devices/"+sysfs_bdf+"/resource");std::string l;uint64_t a,b,flags;
    if(!std::getline(f,l)||!std::getline(f,l))throw std::runtime_error("read BAR1 resource");std::istringstream s(l);s>>std::hex>>a>>b>>flags;
    if(!s||!a||b<a)throw std::runtime_error("unassigned BAR1 resource for "+bdf);return {a,b};
}
static int qrun(const std::vector<std::string>&a){
    pid_t p=fork();if(p<0)throw std::runtime_error("fork config helper");if(!p){dup2(STDERR_FILENO,STDOUT_FILENO);std::vector<char*>v;for(const auto&s:a)v.push_back(const_cast<char*>(s.c_str()));v.push_back(nullptr);execvp(v[0],v.data());_exit(127);}
    int st;while(waitpid(p,&st,0)<0&&errno==EINTR){}return WIFEXITED(st)?WEXITSTATUS(st):128;
}

struct QMem {
 int group=-1,worker=-1,ordinal=-1,devfd=-1,rmfd=-1,exportfd=-1;
 CUdevice device{}; CUcontext ctx{}; CUdeviceptr va=0; CUmemGenericAllocationHandle handle{}; size_t gran=0,bytes=0;
 size_t map_bytes=QWEN_MAP;
 bool reserved=false,created=false,mapped=false; uint32_t client=0,hdev=0,hmem=0; Nvos33Q rmmap{}; void* cpu=MAP_FAILED;
 cudaStream_t stream{};
};
static void qcreate_mapping(QMem&m,const std::string&bdf,int group,int worker,ExportGetterQ get,size_t map_bytes=QWEN_MAP){
    if (!map_bytes || map_bytes % 4096) throw std::runtime_error("RM mapping size must be a nonzero multiple of 4 KiB");
    m.map_bytes=map_bytes;
    m.group=group;m.worker=worker;CK(cudaDeviceGetByPCIBusId(&m.ordinal,bdf.c_str()));CUKQ(cuDeviceGet(&m.device,m.ordinal));
    CUKQ(cuDevicePrimaryCtxRetain(&m.ctx,m.device));CUKQ(cuCtxSetCurrent(m.ctx));CK(cudaSetDevice(m.ordinal));CK(cudaStreamCreateWithFlags(&m.stream,cudaStreamNonBlocking));
    CUmemAllocationProp p{};p.type=CU_MEM_ALLOCATION_TYPE_PINNED;p.requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    p.location.type=CU_MEM_LOCATION_TYPE_DEVICE;p.location.id=m.ordinal;CUKQ(cuMemGetAllocationGranularity(&m.gran,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    const size_t requested_bytes=std::max(QWEN_ALLOC,map_bytes);
    m.bytes=((requested_bytes+m.gran-1)/m.gran)*m.gran;CUKQ(cuMemAddressReserve(&m.va,m.bytes,m.gran,0,0));m.reserved=true;
    CUKQ(cuMemCreate(&m.handle,m.bytes,&p,0));m.created=true;CUKQ(cuMemMap(m.va,m.bytes,0,m.handle,0));m.mapped=true;
    CUmemAccessDesc ac{};ac.location.type=CU_MEM_LOCATION_TYPE_DEVICE;ac.location.id=m.ordinal;ac.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;CUKQ(cuMemSetAccess(m.va,m.bytes,&ac,1));
    CUKQ(cuMemExportToShareableHandle(&m.exportfd,m.handle,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0));
    if(!get(&m.client,&m.hdev,&m.hmem,&m.rmfd))throw std::runtime_error("RM export capture failed for "+bdf);
    char path[64];snprintf(path,sizeof(path),"/dev/nvidia%d",nvidia_minor(bdf));m.devfd=open(path,O_RDWR|O_CLOEXEC);syscheck(m.devfd,"open BDF-matched NVIDIA node");
    int fd=m.rmfd;syscheck(ioctl(m.devfd,_IOWR('F',0xc9,int),&fd),"NV_ESC_REGISTER_FD");
    m.rmmap.c=m.client;m.rmmap.d=m.hdev;m.rmmap.m=m.hmem;m.rmmap.len=m.map_bytes;m.rmmap.flags=QWEN_MAP_REFLECTED;m.rmmap.fd=m.devfd;
    syscheck(ioctl(m.rmfd,_IOWR('F',0x4e,Nvos33Q),&m.rmmap),"NV_ESC_RM_MAP_MEMORY");if(m.rmmap.status)throw std::runtime_error("RM reflected map returned "+std::to_string(m.rmmap.status));
    auto r=qbar1(bdf);if(m.rmmap.addr%QWEN_MAP||m.rmmap.addr<r.first||m.rmmap.addr+m.map_bytes-1>r.second)throw std::runtime_error("RM BAR1 map outside resource or misaligned");
    m.cpu=mmap(nullptr,m.map_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,m.devfd,0);if(m.cpu==MAP_FAILED)throw std::runtime_error(std::string("mmap RM BAR1: ")+strerror(errno));
    fprintf(stderr,"GPU_MAP,group,%d,worker,%d,bdf,%s,cuda_ordinal,%d,nvidia_minor,%d,cuda_va,0x%016" PRIx64 ",bar1_bus,0x%016" PRIx64 "\n",group,worker,bdf.c_str(),m.ordinal,nvidia_minor(bdf),uint64_t(m.va),m.rmmap.addr);
}
static void qdestroy_mem(QMem&m){
 if(m.ctx){cuCtxSetCurrent(m.ctx);cudaSetDevice(m.ordinal);if(m.stream)cudaStreamDestroy(m.stream);}
 if(m.cpu!=MAP_FAILED){Nvos34Q u{};u.c=m.client;u.d=m.hdev;u.m=m.hmem;u.addr=reinterpret_cast<uint64_t>(m.cpu);u.flags=m.rmmap.flags;if(m.rmfd>=0)(void)ioctl(m.rmfd,_IOWR('F',0x4f,Nvos34Q),&u);munmap(m.cpu,m.map_bytes);}
 if(m.devfd>=0)close(m.devfd);if(m.exportfd>=0)close(m.exportfd);if(m.rmfd>=0)close(m.rmfd);
 if(m.ctx){cuCtxSetCurrent(m.ctx);if(m.mapped)cuMemUnmap(m.va,m.bytes);if(m.reserved)cuMemAddressFree(m.va,m.bytes);if(m.created)cuMemRelease(m.handle);cuDevicePrimaryCtxRelease(m.device);}
}
