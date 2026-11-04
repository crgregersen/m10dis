#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool ck(cudaError_t e,const char*where){
    if(e==cudaSuccess)return true;
    std::fprintf(stderr,"FAIL,%s,%s\n",where,cudaGetErrorString(e));return false;
}
int main(int argc,char**argv){
    int count=0;if(!ck(cudaGetDeviceCount(&count),"cudaGetDeviceCount"))return 2;
    if(argc!=count+1){std::fprintf(stderr,"usage: %s bytes_per_visible_device... (%d values)\n",argv[0],count);return 2;}
    std::vector<void*> allocations(count,nullptr);bool ok=true;
    for(int d=0;d<count;d++){
        unsigned long long bytes=std::strtoull(argv[d+1],nullptr,10);
        if(!ck(cudaSetDevice(d),"cudaSetDevice")){ok=false;break;}
        if(bytes){
            cudaError_t e=cudaMalloc(&allocations[d],static_cast<size_t>(bytes));
            if(e!=cudaSuccess){std::fprintf(stderr,"FAIL,ALLOC,gpu,%d,bytes,%llu,error,%s\n",d,bytes,cudaGetErrorString(e));ok=false;break;}
            e=cudaMemset(allocations[d],0,4096);
            if(e!=cudaSuccess){std::fprintf(stderr,"FAIL,TOUCH,gpu,%d,error,%s\n",d,cudaGetErrorString(e));ok=false;break;}
        }
        size_t free_b=0,total_b=0;if(!ck(cudaMemGetInfo(&free_b,&total_b),"cudaMemGetInfo")){ok=false;break;}
        std::printf("CAPACITY,gpu,%d,requested,%llu,free_mib,%zu,total_mib,%zu\n",d,bytes,free_b/(1024*1024),total_b/(1024*1024));
        std::fflush(stdout);
    }
    if(ok){std::puts("CAPACITY_ALL_DEVICES_ALLOCATED");std::fflush(stdout);}
    for(int d=count-1;d>=0;d--)if(allocations[d]){cudaSetDevice(d);cudaFree(allocations[d]);}
    return ok?0:1;
}
