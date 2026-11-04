#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <atomic>
#include <fcntl.h>
#include <mutex>
#include "qwen38_parallel_load.cuh"

struct Shard { int id,gpu,rows,width; std::string path; unsigned long long offset,bytes; };
struct Device { int host_index=-1; std::string bdf,uuid; unsigned long long expert=0,ple=0,scratch=0; void *reserve=nullptr,*table=nullptr,*scratch_ptr=nullptr;
 unsigned long long lm_offset=0,lm_rows=0,lm_row_start=0; };   // LM-head slice (PLE_LM_HEAD_FILE)
constexpr unsigned long long kLmRowBytes=2560ull*2;   // BF16 row of lm_head.weight
__global__ void ple_gather_row(const uint8_t *table,uint8_t *out,size_t byte_offset){
 for(int i=threadIdx.x;i<160;i+=blockDim.x)out[i]=table[byte_offset+i];
}
__global__ void ple_gather_rows(const uint8_t *table,const unsigned long long *offsets,uint8_t *out,int count){
 int row=blockIdx.x; if(row>=count)return;
 for(int i=threadIdx.x;i<160;i+=blockDim.x)out[size_t(row)*160+i]=table[offsets[row]+i];
}
static bool ck(cudaError_t e,const char*w){if(e==cudaSuccess)return true;std::fprintf(stderr,"CUDA_ERROR,%s,%s\n",w,cudaGetErrorString(e));return false;}
static std::vector<std::string> fields(const std::string&s){std::vector<std::string>v;size_t p=0;for(;;){auto q=s.find('\t',p);v.push_back(s.substr(p,q==std::string::npos?q:q-p));if(q==std::string::npos)break;p=q+1;}return v;}
static std::string canon_bdf(std::string s){auto p=s.find(':');return p==std::string::npos?s:s.substr(p+1);}
static bool recv_all(int fd,void *ptr,size_t n){auto*p=static_cast<uint8_t*>(ptr);while(n){ssize_t r=recv(fd,p,n,0);if(r<=0)return false;p+=r;n-=size_t(r);}return true;}
// Distinguish a normal client close between requests from a truncated frame.
// The vLLM worker may close its socket during engine shutdown without being
// able to send the explicit stop marker (the worker is spawned and shut down
// by vLLM independently of this parent process).
static int recv_request_count(int fd,uint32_t *value){
 auto*p=reinterpret_cast<uint8_t*>(value);size_t got=0;
 while(got<sizeof(*value)){
  ssize_t r=recv(fd,p+got,sizeof(*value)-got,0);
  if(r==0)return got==0?0:-1;
  if(r<0)return -1;
  got+=size_t(r);
 }
 return 1;
}
static bool send_all(int fd,const void *ptr,size_t n){auto*p=static_cast<const uint8_t*>(ptr);while(n){ssize_t r=send(fd,p,n,MSG_NOSIGNAL);if(r<=0)return false;p+=r;n-=size_t(r);}return true;}
int main(int argc,char**argv){
 if(argc<4||argc>5){std::fprintf(stderr,"usage: %s gpu_allocations.tsv ple_shards.tsv query_global_rows.txt [unix_socket]\n",argv[0]);return 2;}
 int count=0;if(!ck(cudaGetDeviceCount(&count),"cudaGetDeviceCount"))return 2;
 if(count!=25){std::fprintf(stderr,"expected 25 visible CUDA devices, got %d\n",count);return 2;}
 std::vector<Device>host_devices(count),d(count);std::ifstream af(argv[1]);std::string line;
 while(std::getline(af,line)){auto v=fields(line);if(v.size()!=6)return 2;int i=std::stoi(v[0]);if(i<0||i>=count)return 2;host_devices[i].host_index=i;host_devices[i].bdf=v[1];host_devices[i].uuid=v[2];host_devices[i].expert=std::stoull(v[3]);host_devices[i].ple=std::stoull(v[4]);host_devices[i].scratch=std::stoull(v[5]);}
 std::vector<int>host_to_cuda(count,-1);
 for(int ordinal=0;ordinal<count;ordinal++){
  char bdf[32]{};if(!ck(cudaDeviceGetPCIBusId(bdf,sizeof(bdf),ordinal),"cudaDeviceGetPCIBusId"))return 2;
  const std::string key=canon_bdf(bdf);int host=-1;
  for(int i=0;i<count;i++)if(!host_devices[i].bdf.empty()&&canon_bdf(host_devices[i].bdf)==key){host=i;break;}
  if(host<0||host_to_cuda[host]>=0){std::fprintf(stderr,"CUDA_BDF_MAP_FAIL,ordinal,%d,bdf,%s\n",ordinal,bdf);return 2;}
  host_to_cuda[host]=ordinal;d[ordinal]=host_devices[host];
  std::printf("DEVICE_MAP,cuda_ordinal,%d,host_index,%d,bdf,%s,uuid,%s,role,%s\n",ordinal,host,d[ordinal].bdf.c_str(),d[ordinal].uuid.c_str(),d[ordinal].expert?"routed":d[ordinal].ple?"shared":"unused");
 }
 for(int i=0;i<count;i++)if(host_to_cuda[i]<0){std::fprintf(stderr,"CUDA_HOST_DEVICE_MISSING,index,%d,bdf,%s\n",i,host_devices[i].bdf.c_str());return 2;}
 std::set<int> selected_shards;
 if(const char *selection=std::getenv("PLE_SHARD_IDS")){
  std::string ids(selection);size_t p=0;
  while(p<ids.size()){size_t q=ids.find(',',p);int id=std::stoi(ids.substr(p,q==std::string::npos?q:q-p));if(id<0||id>=128||!selected_shards.insert(id).second){std::fprintf(stderr,"PLE_SHARD_IDS_INVALID,%s\n",selection);return 2;}if(q==std::string::npos)break;p=q+1;}
  if(selected_shards.empty()){std::fprintf(stderr,"PLE_SHARD_IDS_EMPTY\n");return 2;}
  std::printf("PLE_PARTIAL_LOAD,selected_shards,%s\n",selection);
 }
 std::vector<Shard>shards;std::ifstream sf(argv[2]);
 while(std::getline(sf,line)){auto v=fields(line);if(v.size()!=7)return 2;int host=std::stoi(v[1]);if(host<0||host>=count||host_to_cuda[host]<0)return 2;shards.push_back({std::stoi(v[0]),host_to_cuda[host],std::stoi(v[5]),std::stoi(v[6]),v[2],std::stoull(v[3]),std::stoull(v[4])});}
 if(selected_shards.empty()){for(int id=0;id<128;id++)selected_shards.insert(id);}
 shards.erase(std::remove_if(shards.begin(),shards.end(),[&](const Shard&s){return !selected_shards.count(s.id);}),shards.end());
 if(shards.size()!=selected_shards.size()){std::fprintf(stderr,"PLE_SELECTED_SHARD_MISSING\n");return 2;}
 std::vector<unsigned long long> next(count,0);std::vector<std::vector<size_t>> offsets(count);
 std::vector<size_t> shard_offsets(128,SIZE_MAX);std::vector<int> shard_gpus(128,-1),shard_indices(128,-1);
 for(size_t si=0;si<shards.size();si++){const auto&s=shards[si];shard_offsets[s.id]=next[s.gpu];shard_gpus[s.id]=s.gpu;shard_indices[s.id]=int(si);offsets[s.gpu].push_back(next[s.gpu]);next[s.gpu]+=s.bytes;}
 // LM head (2026-11-03, PLE_LM_HEAD_FILE = raw BF16 lm_head.weight, row-major): every device that holds
 // a PLE table also holds one contiguous row slice of the LM head for worker mode 7 (PCI-cluster top-k
 // logits), appended at the END of its allocation: [slice weights: rows*5120][scratch: rows*4]. The
 // worker requires exactly this layout (region ends at the allocation end), so its scratch writes can
 // never touch PLE rows. Rows are split evenly in CUDA-ordinal order; the manifest records each slice.
 unsigned long long lm_total_rows=0;const char*lm_file=std::getenv("PLE_LM_HEAD_FILE");
 if(lm_file&&*lm_file){
  std::error_code ec;const auto fsz=std::filesystem::file_size(lm_file,ec);
  if(ec||!fsz||fsz%kLmRowBytes){std::fprintf(stderr,"LM_HEAD_FILE_INVALID,%s\n",lm_file);return 2;}
  lm_total_rows=fsz/kLmRowBytes;std::vector<int>lm_devs;for(int i=0;i<count;i++)if(next[i])lm_devs.push_back(i);
  if(lm_devs.empty()){std::fprintf(stderr,"LM_HEAD_NO_DEVICES\n");return 2;}
  const unsigned long long base=lm_total_rows/lm_devs.size(),extra=lm_total_rows%lm_devs.size();unsigned long long row=0;
  for(size_t k=0;k<lm_devs.size();k++){const int g=lm_devs[k];const unsigned long long rows=base+(k<extra?1:0);
   const unsigned long long off=(next[g]+4095ull)&~4095ull;d[g].lm_offset=off;d[g].lm_rows=rows;d[g].lm_row_start=row;row+=rows;
   next[g]=off+rows*kLmRowBytes+rows*4ull;
   std::printf("LM_HEAD_LAYOUT,gpu,%d,bdf,%s,row_start,%llu,rows,%llu,offset,%llu,scratch_offset,%llu,device_bytes,%llu\n",
               g,d[g].bdf.c_str(),d[g].lm_row_start,rows,off,off+rows*kLmRowBytes,next[g]);}
  if(row!=lm_total_rows){std::fprintf(stderr,"LM_HEAD_SPLIT_FAIL\n");return 2;}
 }
 for(int i=0;i<count;i++)d[i].ple=next[i];
 std::vector<size_t> baseline_free(count);bool ok=true;
 const char*skip_reserve_env=std::getenv("PLE_SKIP_EXPERT_RESERVE");
 const bool skip_expert_reserve=skip_reserve_env&&std::strcmp(skip_reserve_env,"1")==0;
 if(skip_expert_reserve)std::printf("EXPERT_RESERVE,skipped,actual_expert_service_will_be_loaded_separately\n");
 auto t0=std::chrono::steady_clock::now();
 const char*par_env=std::getenv("PLE_PARALLEL_LOAD");const bool parallel_load=!(par_env&&std::strcmp(par_env,"0")==0);
 std::printf("PLE_LOAD_MODE,%s\n",parallel_load?"parallel_per_card":"serial");
 for(int i=0;i<count;i++){
  if(!ck(cudaSetDevice(i),"cudaSetDevice")){ok=false;break;}
  size_t f=0,t=0;if(!ck(cudaMemGetInfo(&f,&t),"cudaMemGetInfo")){ok=false;break;}baseline_free[i]=f;
  const auto expert_request=skip_expert_reserve?0ull:d[i].expert;
  std::tuple<unsigned long long,void**,const char*> requests[]={{expert_request,&d[i].reserve,"expert_reserve"},{d[i].ple,&d[i].table,"ple_table"},{d[i].scratch,&d[i].scratch_ptr,"query_scratch"}};
  for(auto [bytes,ptr,label]:requests){
    if(bytes){cudaError_t e=cudaMalloc(ptr,static_cast<size_t>(bytes));if(e!=cudaSuccess){std::fprintf(stderr,"ALLOC_FAIL,gpu,%d,kind,%s,bytes,%llu,error,%s\n",i,label,bytes,cudaGetErrorString(e));ok=false;break;}}
  }
  if(!ok)break;
  size_t ff=0,tt=0;cudaMemGetInfo(&ff,&tt);std::printf("ALLOCATED,gpu,%d,ple_bytes,%llu,free_mib,%zu\n",i,d[i].ple,ff/(1024*1024));std::fflush(stdout);
  if(d[i].expert && ff < 512ull*1024*1024){std::fprintf(stderr,"HEADROOM_FAIL,gpu,%d,free_mib,%zu\n",i,ff/(1024*1024));ok=false;break;}
 }
 if(ok && !parallel_load){
  std::vector<uint8_t> host;
  for(size_t si=0;si<shards.size();si++){
   const auto&s=shards[si];int g=s.gpu;host.resize(s.bytes);
   std::ifstream f(s.path,std::ios::binary);if(!f){std::fprintf(stderr,"OPEN_FAIL,%s\n",s.path.c_str());ok=false;break;}
   f.seekg(static_cast<std::streamoff>(s.offset));f.read(reinterpret_cast<char*>(host.data()),static_cast<std::streamsize>(s.bytes));
   if(!f){std::fprintf(stderr,"READ_FAIL,shard,%d\n",s.id);ok=false;break;}
   if(!ck(cudaSetDevice(g),"cudaSetDevice_load")){ok=false;break;}
   size_t local_index=0;for(size_t j=0;j<si;j++)if(shards[j].gpu==g)local_index++;
   auto *dst=static_cast<uint8_t*>(d[g].table)+offsets[g][local_index];
   if(!ck(cudaMemcpy(dst,host.data(),s.bytes,cudaMemcpyHostToDevice),"cudaMemcpy_PLE")){ok=false;break;}
   // Verify the complete shard in device memory, not only a few sampled rows.
   // The D2H buffer is bounded to one 400 MB shard and released before the
   // next shard is processed.
   std::vector<uint8_t> readback(s.bytes);
   if(!ck(cudaMemcpy(readback.data(),dst,s.bytes,cudaMemcpyDeviceToHost),"cudaMemcpy_PLE_full_readback")){ok=false;break;}
   if(std::memcmp(readback.data(),host.data(),s.bytes)!=0){
    size_t first=0;while(first<s.bytes && readback[first]==host[first])++first;
    std::fprintf(stderr,"FULL_SHARD_MISMATCH,shard,%d,byte,%zu,expected,%u,observed,%u\n",
                 s.id,first,first<s.bytes?unsigned(host[first]):0u,first<s.bytes?unsigned(readback[first]):0u);
    ok=false;break;
   }
   std::printf("PLE_SHARD_FULL_READBACK_PASS,id,%d,bytes,%llu\n",s.id,s.bytes);std::fflush(stdout);
   const int test_rows[3]={0,s.rows/2,s.rows-1};
   for(int r:test_rows){std::array<uint8_t,160>got{};const size_t off=size_t(r)*s.width;
    if(!ck(cudaMemcpy(got.data(),dst+off,got.size(),cudaMemcpyDeviceToHost),"cudaMemcpy_sample")){ok=false;break;}
    if(std::memcmp(got.data(),host.data()+off,got.size())!=0){std::fprintf(stderr,"DATA_MISMATCH,shard,%d,row,%d\n",s.id,r);ok=false;break;}}
   if(!ok)break;
   if((s.id+1)%16==0){std::printf("PLE_SHARDS_LOADED,%d/128\n",s.id+1);std::fflush(stdout);}
  }
 }
 // Parallel path (default): one thread per M10 card, pinned O_DIRECT chunks, async H2D,
 // full D2H readback + memcmp of every chunk (qwen38_parallel_load.cuh), then the same
 // three sampled rows per shard compared against the file.
 if(ok && parallel_load){
  try{
   std::vector<int> devs;for(const auto&s:shards)if(std::find(devs.begin(),devs.end(),s.gpu)==devs.end())devs.push_back(s.gpu);
   auto groups=q38load::group_by_card(devs);
   std::map<int,size_t> group_of;for(size_t gi=0;gi<groups.size();gi++)for(int g:groups[gi])group_of[g]=gi;
   std::vector<std::vector<size_t>> work(groups.size());for(size_t si=0;si<shards.size();si++)work[group_of[shards[si].gpu]].push_back(si);
   q38load::Stats st;std::mutex logm;std::atomic<int> done{0};auto tp=std::chrono::steady_clock::now();
   q38load::per_card(groups,[&](size_t gi,const std::vector<int>&gdevs){
    q38load::Slots slots;std::map<int,cudaStream_t> streams;
    for(int g:gdevs){q38load::cuda_check(cudaSetDevice(g),"cudaSetDevice stream");cudaStream_t x;q38load::cuda_check(cudaStreamCreateWithFlags(&x,cudaStreamNonBlocking),"cudaStreamCreate");streams[g]=x;}
    for(size_t si:work[gi]){
     const auto&s=shards[si];int g=s.gpu;auto*dst=static_cast<uint8_t*>(d[g].table)+shard_offsets[s.id];
     q38load::upload_file(slots,st,g,streams[g],s.path,s.offset,s.bytes,dst);
     const int test_rows[3]={0,s.rows/2,s.rows-1};
     int fd=::open(s.path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("open for row check: "+s.path);
     for(int r:test_rows){std::array<uint8_t,160>got{},want{};const size_t off=size_t(r)*s.width;
      q38load::cuda_check(cudaSetDevice(g),"cudaSetDevice sample");q38load::cuda_check(cudaMemcpy(got.data(),dst+off,got.size(),cudaMemcpyDeviceToHost),"cudaMemcpy_sample");
      if(::pread(fd,want.data(),want.size(),static_cast<off_t>(s.offset+off))!=ssize_t(want.size())||std::memcmp(got.data(),want.data(),got.size())!=0){::close(fd);throw std::runtime_error("DATA_MISMATCH shard "+std::to_string(s.id)+" row "+std::to_string(r));}}
     ::close(fd);
     std::lock_guard<std::mutex> l(logm);std::printf("PLE_SHARD_FULL_READBACK_PASS,id,%d,bytes,%llu,gpu,%d,card_thread,%zu\n",s.id,s.bytes,g,gi);
     int n=++done;if(n%16==0)std::printf("PLE_SHARDS_LOADED,%d/%zu\n",n,shards.size());std::fflush(stdout);
    }
    for(auto&kv:streams){cudaSetDevice(kv.first);cudaStreamDestroy(kv.second);}
   });
   const double secs=std::chrono::duration<double>(std::chrono::steady_clock::now()-tp).count();
   std::printf("PLE_PARALLEL_LOAD,card_threads,%zu,bytes,%llu,seconds,%.3f,GBps,%.2f,direct_files,%llu,buffered_files,%llu,read_thread_s,%.2f,wait_thread_s,%.2f\n",
     groups.size(),st.bytes.load(),secs,st.bytes.load()/secs/1e9,st.direct_files.load(),st.buffered_files.load(),st.read_ns.load()/1e9,st.wait_ns.load()/1e9);
   std::fflush(stdout);
  }catch(const std::exception&e){std::fprintf(stderr,"PLE_PARALLEL_LOAD_ERROR,%s\n",e.what());ok=false;}
 }
 // LM-head slices: read from the raw file, copy, full readback compare, zero the scratch.
 if(ok&&lm_total_rows){
  auto tl=std::chrono::steady_clock::now();int fd=::open(lm_file,O_RDONLY|O_CLOEXEC);
  if(fd<0){std::fprintf(stderr,"LM_HEAD_OPEN_FAIL,%s\n",lm_file);ok=false;}
  std::vector<uint8_t>host,back;unsigned long long loaded=0;
  for(int g=0;ok&&g<count;g++)if(d[g].lm_rows){
   const size_t bytes=size_t(d[g].lm_rows*kLmRowBytes);host.resize(bytes);back.resize(bytes);size_t got=0;
   while(got<bytes){ssize_t r=::pread(fd,host.data()+got,bytes-got,off_t(d[g].lm_row_start*kLmRowBytes+got));if(r<=0)break;got+=size_t(r);}
   if(got!=bytes){std::fprintf(stderr,"LM_HEAD_READ_FAIL,gpu,%d\n",g);ok=false;break;}
   if(!ck(cudaSetDevice(g),"cudaSetDevice_lm")){ok=false;break;}
   auto*dst=static_cast<uint8_t*>(d[g].table)+d[g].lm_offset;
   if(!ck(cudaMemcpy(dst,host.data(),bytes,cudaMemcpyHostToDevice),"cudaMemcpy_LM_HEAD")){ok=false;break;}
   if(!ck(cudaMemcpy(back.data(),dst,bytes,cudaMemcpyDeviceToHost),"cudaMemcpy_LM_HEAD_readback")){ok=false;break;}
   if(std::memcmp(back.data(),host.data(),bytes)!=0){std::fprintf(stderr,"LM_HEAD_READBACK_MISMATCH,gpu,%d\n",g);ok=false;break;}
   if(!ck(cudaMemset(dst+bytes,0,size_t(d[g].lm_rows*4ull)),"cudaMemset_LM_HEAD_scratch")){ok=false;break;}
   loaded+=d[g].lm_rows;
   std::printf("LM_HEAD_SLICE_FULL_READBACK_PASS,gpu,%d,row_start,%llu,rows,%llu,bytes,%zu\n",g,d[g].lm_row_start,d[g].lm_rows,bytes);std::fflush(stdout);
  }
  if(fd>=0)::close(fd);
  if(ok&&loaded!=lm_total_rows){std::fprintf(stderr,"LM_HEAD_INCOMPLETE,%llu,%llu\n",loaded,lm_total_rows);ok=false;}
  if(ok)std::printf("LM_HEAD_LOADED,rows,%llu,seconds,%.2f,file,%s\n",loaded,std::chrono::duration<double>(std::chrono::steady_clock::now()-tl).count(),lm_file);
  std::fflush(stdout);
 }
 // Optional CUDA IPC export after every PLE device allocation is populated.
 // The PLE owner retains these allocations for all importing worker lifetimes.
 if(ok){
  const char*ipc_env=std::getenv("PLE_IPC_EXPORT_DIR");
  if(ipc_env&&*ipc_env){
   try{
    std::filesystem::path dir(ipc_env);std::filesystem::create_directories(dir);
    std::ofstream mf(dir/"ple_ipc_manifest.tsv");
    if(!mf)throw std::runtime_error("cannot create PLE IPC manifest");
    mf<<"DEVICE\tcuda_ordinal\tbdf\tbytes\thandle_file\n";
    for(int g=0;g<count;g++)if(d[g].table){
     if(!ck(cudaSetDevice(g),"cudaSetDevice_ipc_export")){ok=false;break;}cudaIpcMemHandle_t handle{};
     if(!ck(cudaIpcGetMemHandle(&handle,d[g].table),"cudaIpcGetMemHandle_PLE")){ok=false;break;}
     std::string safe=d[g].bdf;for(char&c:safe)if(c==':'||c=='.')c='_';
     std::string name="gpu_"+safe+".ipc";std::ofstream hf(dir/name,std::ios::binary);
     if(!hf)throw std::runtime_error("cannot create PLE IPC handle file");
     hf.write(reinterpret_cast<const char*>(&handle),sizeof(handle));hf.close();
     mf<<"DEVICE\t"<<g<<"\t"<<d[g].bdf<<"\t"<<next[g]<<"\t"<<name<<"\n";
    }
    mf<<"SHARD\tid\tcuda_ordinal\tbdf\tbytes\toffset\trows\twidth\n";
    for(const auto&s:shards)mf<<"SHARD\t"<<s.id<<"\t"<<s.gpu<<"\t"<<d[s.gpu].bdf<<"\t"<<s.bytes<<"\t"<<shard_offsets[s.id]<<"\t"<<s.rows<<"\t"<<s.width<<"\n";
    if(lm_total_rows){
     mf<<"LMHEAD\tcuda_ordinal\tbdf\toffset\trows\trow_start\trow_bytes\n";
     for(int g=0;g<count;g++)if(d[g].lm_rows)mf<<"LMHEAD\t"<<g<<"\t"<<d[g].bdf<<"\t"<<d[g].lm_offset<<"\t"<<d[g].lm_rows<<"\t"<<d[g].lm_row_start<<"\t"<<kLmRowBytes<<"\n";
    }
    mf.flush();if(!mf)throw std::runtime_error("failed writing PLE IPC manifest");
    std::printf("PLE_IPC_EXPORT_READY,%s,devices,%d,shards,%zu\n",dir.c_str(),count,shards.size());std::fflush(stdout);
   }catch(const std::exception&e){std::fprintf(stderr,"PLE_IPC_EXPORT_FAIL,%s\n",e.what());ok=false;}
  }
 }
 if(ok){
  std::ifstream qf(argv[3]);unsigned long long global_row=0;size_t query_count=0;
  std::vector<bool> touched(shards.size(),false);
  std::ofstream qout("stage9_ple_query_results.tsv");
  if(!qf){std::fprintf(stderr,"QUERY_FILE_OPEN_FAIL,%s\n",argv[3]);ok=false;}
  while(ok && qf>>global_row){
   const size_t shard=global_row/2500012ull, local_row=global_row%2500012ull;
   if(shard>=128||shard_indices[shard]<0){std::fprintf(stderr,"QUERY_ROW_UNLOADED,%llu,shard,%zu\n",global_row,shard);ok=false;break;}
   const size_t si=size_t(shard_indices[shard]);
   const int g=shard_gpus[shard];std::array<uint8_t,160>got{},expected{};
   if(!ck(cudaSetDevice(g),"cudaSetDevice_query")){ok=false;break;}
   ple_gather_row<<<1,32>>>(static_cast<const uint8_t*>(d[g].table),static_cast<uint8_t*>(d[g].scratch_ptr),shard_offsets[shard]+local_row*160);
   if(!ck(cudaGetLastError(),"ple_gather_row_launch") || !ck(cudaDeviceSynchronize(),"ple_gather_row_sync") || !ck(cudaMemcpy(got.data(),d[g].scratch_ptr,got.size(),cudaMemcpyDeviceToHost),"cudaMemcpy_query")){ok=false;break;}
   const auto&s=shards[si];std::ifstream f(s.path,std::ios::binary);
   f.seekg(static_cast<std::streamoff>(s.offset+local_row*160));f.read(reinterpret_cast<char*>(expected.data()),expected.size());
   if(!f || std::memcmp(got.data(),expected.data(),got.size())!=0){std::fprintf(stderr,"QUERY_MISMATCH,global_row,%llu,shard,%zu,local_row,%zu,gpu,%d\n",global_row,shard,local_row,g);ok=false;break;}
   touched[si]=true;qout<<global_row<<'\t'<<shard<<'\t'<<local_row<<'\t'<<g<<"\tMATCH\n";++query_count;
  }
  size_t distinct=0;for(bool v:touched)distinct+=v;
  if(ok){std::printf("PLE_LOOKUP_ROWS_PASS,count,%zu,distinct_shards,%zu\n",query_count,distinct);std::fflush(stdout);}
 }
 if(ok){
  for(int i=0;i<count;i++)if(d[i].expert){cudaSetDevice(i);size_t f=0,t=0;cudaMemGetInfo(&f,&t);if(f<512ull*1024*1024)ok=false;}
  if(ok){unsigned long long total_bytes=0;for(auto n:next)total_bytes+=n;std::printf("PASS_STAGE9_PLE_RESIDENT_LOAD_READBACK,shards,%zu,bytes,%llu,seconds,%.3f\n",shards.size(),total_bytes,std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());}
 }
 if(ok && argc==5){
  const char*path=argv[4];if(std::strlen(path)>=sizeof(sockaddr_un::sun_path)){std::fprintf(stderr,"SOCKET_PATH_TOO_LONG\n");ok=false;}
  int server=-1,client=-1;bool stop_requested=false;uint64_t client_sessions=0;struct sockaddr_un addr{};
  if(ok){unlink(path);server=socket(AF_UNIX,SOCK_STREAM,0);if(server<0){perror("socket");ok=false;}}
  if(ok){addr.sun_family=AF_UNIX;std::strncpy(addr.sun_path,path,sizeof(addr.sun_path)-1);if(bind(server,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0||listen(server,1)<0){perror("bind/listen");ok=false;}}
  if(ok){std::printf("PLE_LOOKUP_SERVER_READY,%s\n",path);std::fflush(stdout);}
  // A vLLM process is a replaceable client, not the owner of the PLE
  // allocations. Accept clients serially until the owner sends the explicit
  // stop marker. A clean client EOF must not release resident CUDA memory.
  while(ok && !stop_requested){
   client=accept(server,nullptr,nullptr);if(client<0){perror("accept");ok=false;break;}
   std::printf("PLE_LOOKUP_CLIENT_CONNECTED,session,%llu\n",static_cast<unsigned long long>(++client_sessions));std::fflush(stdout);
   bool client_broken=false;
   while(ok){
    uint32_t n=0;int header_state=recv_request_count(client,&n);
    if(header_state==0){std::printf("PLE_LOOKUP_CLIENT_EOF,clean\n");std::fflush(stdout);break;}
    if(header_state<0){std::fprintf(stderr,"CLIENT_DISCONNECTED,truncated_header\n");client_broken=true;break;}
    if(n==0xffffffffu){stop_requested=true;break;}
   const auto request_started=std::chrono::steady_clock::now();
   uint32_t status=0;if(n>65536u){status=1;uint32_t zero=0;if(!send_all(client,&status,4)||!send_all(client,&zero,4))ok=false;continue;}
   std::vector<unsigned long long> ids(n);std::vector<uint8_t> result(size_t(n)*160);
   if(n && !recv_all(client,ids.data(),size_t(n)*sizeof(ids[0]))){client_broken=true;break;}
   std::vector<std::vector<std::pair<size_t,unsigned long long>>> per_device(count);
   for(size_t i=0;i<n;i++){
    if(ids[i]>=320001536ull){status=1;break;}
    size_t sid=ids[i]/2500012ull,row=ids[i]%2500012ull;int g=shard_gpus[sid];
    per_device[g].push_back({i,shard_offsets[sid]+row*160ull});
   }
   for(int g=0;status==0 && g<24;g++)if(!per_device[g].empty()){
    size_t m=per_device[g].size(),offset_bytes=m*sizeof(unsigned long long),out_off=(offset_bytes+255)&~size_t(255);
    if(out_off+m*160> d[g].scratch){status=2;break;}
    std::vector<unsigned long long> offsets_g(m);for(size_t j=0;j<m;j++)offsets_g[j]=per_device[g][j].second;
    if(!ck(cudaSetDevice(g),"server_set_device")||!ck(cudaMemcpy(d[g].scratch_ptr,offsets_g.data(),offset_bytes,cudaMemcpyHostToDevice),"server_offsets_h2d")){status=2;break;}
    auto*out_dev=static_cast<uint8_t*>(d[g].scratch_ptr)+out_off;
    ple_gather_rows<<<unsigned(m),32>>>(static_cast<const uint8_t*>(d[g].table),static_cast<const unsigned long long*>(d[g].scratch_ptr),out_dev,int(m));
    if(!ck(cudaGetLastError(),"server_gather_launch")||!ck(cudaDeviceSynchronize(),"server_gather_sync")){status=2;break;}
    std::vector<uint8_t> rows_g(m*160);if(!ck(cudaMemcpy(rows_g.data(),out_dev,rows_g.size(),cudaMemcpyDeviceToHost),"server_rows_d2h")){status=2;break;}
    for(size_t j=0;j<m;j++)std::memcpy(result.data()+per_device[g][j].first*160,rows_g.data()+j*160,160);
   }
   if(!send_all(client,&status,sizeof(status))||!send_all(client,&n,sizeof(n))||(status==0&&n&&!send_all(client,result.data(),result.size()))){client_broken=true;break;}
   std::printf("PLE_LOOKUP_REQUEST,count,%u,status,%u\n",n,status);std::fflush(stdout);
   const double request_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-request_started).count();
   std::printf("PLE_LOOKUP_TIMING,rows,%u,ms,%.6f\n",n,request_ms);std::fflush(stdout);
   }
   if(client>=0){close(client);client=-1;}
   if(client_broken)std::fprintf(stderr,"PLE_CLIENT_SESSION_DROPPED,service_retained\n");
  }
  if(server>=0)close(server);unlink(path);
  if(ok)std::printf("PLE_LOOKUP_SERVER_STOPPED\n");
 }
 // In resident IPC mode this process is only the allocation owner. The
 // persistent routed/shared workers perform lookups; this process must keep
 // the exported CUDA allocations alive until the daemon has stopped them.
 if(ok){
  const char*hold=std::getenv("PLE_IPC_OWNER_HOLD");
  if(hold&&std::strcmp(hold,"1")==0){
   const char*ipc=std::getenv("PLE_IPC_EXPORT_DIR");
   if(!ipc||!*ipc||!std::filesystem::is_regular_file(std::filesystem::path(ipc)/"ple_ipc_manifest.tsv")){
    std::fprintf(stderr,"PLE_IPC_OWNER_HOLD requested without a complete exported manifest\n");ok=false;
   }else{
    std::printf("PLE_IPC_OWNER_READY,%s\n",ipc);std::fflush(stdout);
    std::string command;
    while(std::getline(std::cin,command)){
     if(command=="STOP"){std::printf("PLE_IPC_OWNER_STOP_ACK\n");std::fflush(stdout);break;}
    }
   }
  }
 }
 for(int i=count-1;i>=0;i--){cudaSetDevice(i);for(void*p:{d[i].scratch_ptr,d[i].table,d[i].reserve})if(p)cudaFree(p);}
 return ok?0:1;
}
