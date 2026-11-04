#define QWEN_M10_LIBRARY_ONLY
#include "bench_resident_multilayer_fp8.cu"
#include "resident_dynamic_routed.cuh"
#include "resident_multicast_mapping.cuh"
#include "resident_pdb_export.cuh"
#include "resident_cpu_mailbox.cuh"
#include "qwen38_parallel_load.cuh"
#include <utility>
#include <cuda.h>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <atomic>
#include <thread>
#include <mutex>
#include <xmmintrin.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <sstream>

// Experimental persistent routed worker with five PEX multicast windows.
struct DynamicWorker {
 ResidentDynamicControl *host=nullptr,*device=nullptr;
 ResidentDynamicReturn *out=nullptr,*out_device=nullptr;
 ResidentDynamicGrid *grid=nullptr; DeviceWeights *table=nullptr;
 float *act=nullptr,*local=nullptr; cudaStream_t stream=nullptr;
 void *batch_scratch=nullptr;   // worker mode 8 (batched MoE) device scratch, Q38_BATCH_SCRATCH_BYTES
 const uint8_t *ple_table=nullptr; uint64_t ple_bytes=0;
 unsigned seq=0; bool running=false;
};
static bool await_value(volatile unsigned* p,unsigned value,int timeout_ms){
 auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
 while(std::chrono::steady_clock::now()<end){if(*p==value){std::atomic_thread_fence(std::memory_order_seq_cst);return true;}std::this_thread::yield();}return false;
}
static void publish(DynamicWorker &w,unsigned seq){_mm_sfence();std::atomic_thread_fence(std::memory_order_seq_cst);w.host->seq=seq;_mm_sfence();}
struct RoutedKernelImage { CUmodule module=nullptr; CUfunction function=nullptr; };
struct PleDevice { uint64_t bytes=0; std::string handle; };
static std::string ple_canon(std::string s){auto p=s.find(':');if(p!=std::string::npos)s=s.substr(p+1);std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return char(std::tolower(c));});return s;}
#include "resident_pack_down_kernel.cuh"
static std::map<std::string,PleDevice> read_ple_devices(const char*path){std::map<std::string,PleDevice>m;if(!path||!*path)return m;std::ifstream f(path);if(!f)throw std::runtime_error("cannot open PLE IPC manifest");std::string l;while(std::getline(f,l)){std::istringstream s(l);std::string k,o,b,bytes,h;if(!(s>>k)||k!="DEVICE")continue;if(!(s>>o>>b>>bytes>>h))throw std::runtime_error("malformed PLE IPC device row");if(o=="cuda_ordinal")continue;m[ple_canon(b)]={std::stoull(bytes),h};}return m;}
static void import_ple_table(DynamicWorker&v,unsigned d,const std::map<std::string,PleDevice>&m){char bdf[32]{};CK(cudaDeviceGetPCIBusId(bdf,sizeof(bdf),int(d)));auto it=m.find(ple_canon(bdf));if(it==m.end()||!it->second.bytes)return;const char*manifest=std::getenv("Q38_PLE_IPC_MANIFEST");if(!manifest||!*manifest)throw std::runtime_error("PLE manifest path missing");auto hp=std::filesystem::path(manifest).parent_path()/it->second.handle;cudaIpcMemHandle_t h{};std::ifstream f(hp,std::ios::binary);if(!f.read(reinterpret_cast<char*>(&h),sizeof(h)))throw std::runtime_error("cannot read PLE IPC handle for "+std::string(bdf));void*p=nullptr;CK(cudaIpcOpenMemHandle(&p,h,cudaIpcMemLazyEnablePeerAccess));v.ple_table=static_cast<const uint8_t*>(p);v.ple_bytes=it->second.bytes;}
static void check_driver(CUresult r,const char*where){if(r!=CUDA_SUCCESS){const char*name="unknown";cuGetErrorName(r,&name);throw std::runtime_error(std::string(where)+": "+name);}}
static std::vector<RoutedKernelImage> load_kernel_image(const std::string&path,int count){
 std::vector<RoutedKernelImage> staged(count);
 try{for(int d=0;d<count;d++){CK(cudaSetDevice(d));check_driver(cuModuleLoad(&staged[d].module,path.c_str()),"cuModuleLoad routed kernel");check_driver(cuModuleGetFunction(&staged[d].function,staged[d].module,"resident_dynamic_routed"),"cuModuleGetFunction routed kernel");}}
 catch(...){for(int d=0;d<count;d++)if(staged[d].module){CK(cudaSetDevice(d));cuModuleUnload(staged[d].module);}throw;}
 return staged;
}
static void unload_kernel_image(std::vector<RoutedKernelImage>&image){for(size_t d=0;d<image.size();d++)if(image[d].module){CK(cudaSetDevice(int(d)));check_driver(cuModuleUnload(image[d].module),"cuModuleUnload routed kernel");image[d]={};}}
static void launch_kernel_image(RoutedKernelImage&image,DynamicWorker&v,unsigned d,unsigned long long deadline){
 int active=0;check_driver(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active,image.function,THREADS,0),"cuOccupancyMaxActiveBlocks routed kernel");cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,int(d)));if(active*prop.multiProcessorCount<10)throw std::runtime_error("reloaded routed kernel grid exceeds occupancy");
 const DeviceWeights*table=v.table;ResidentDynamicControl*control=v.device;ResidentDynamicGrid*grid=v.grid;float*act=v.act;float*local=v.local;ResidentDynamicReturn*out=v.out_device;const uint8_t*ple=v.ple_table;
 void*args[]={&table,&control,&grid,&act,&local,&out,&ple,&deadline};
 check_driver(cuLaunchKernel(image.function,10,1,1,THREADS,1,1,0,reinterpret_cast<CUstream>(v.stream),args,nullptr),"cuLaunchKernel routed image");
}

// 2026-11-03: a worker kernel may exit on its own (rejected request -> error path, exit_reason
// bits). The service must notice, or PARK waits for an acknowledgement that never comes and WAKE
// is a no-op because `started` is still true. A worker stream with no pending work means the
// persistent kernel has exited; then mark it stopped so WAKE relaunches on a reset grid.
template<class W> static bool q38_all_workers_exited(std::vector<W>&dynamic,const char*branch){
 bool any=false;for(size_t d=0;d<dynamic.size();d++){auto&v=dynamic[d];if(!v.running)continue;any=true;
  CK(cudaSetDevice(int(d)));cudaError_t q=cudaStreamQuery(v.stream);if(q==cudaErrorNotReady)return false;
  if(q!=cudaSuccess){fprintf(stderr,"%s_WORKER_STREAM_ERROR,device,%zu,%s\n",branch,d,cudaGetErrorString(q));return false;}}
 if(!any)return false;
 for(size_t d=0;d<dynamic.size();d++){auto&v=dynamic[d];if(!v.running)continue;
  fprintf(stderr,"%s_WORKER_ALREADY_EXITED,device,%zu,seq,%u,status,%u,exit_reason,0x%x\n",branch,d,v.out->seq,v.out->status,v.out->exit_reason);
  v.running=false;}
 return true;}
static void stop_resident_grid(std::vector<DynamicWorker>&dynamic){
 for(auto&v:dynamic)if(v.running){v.seq=v.host->seq;v.host->mode=2;publish(v,++v.seq);}
 for(size_t d=0;d<dynamic.size();d++){auto&v=dynamic[d];if(!v.running)continue;if(!await_value(&v.out->seq,v.seq,10000)||v.out->status||v.out->exit_reason!=2)throw std::runtime_error("resident routed kernel did not acknowledge reload stop");CK(cudaSetDevice(int(d)));CK(cudaStreamSynchronize(v.stream));v.running=false;}
}
static void reset_resident_grid(std::vector<DynamicWorker>&dynamic){for(size_t d=0;d<dynamic.size();d++){auto&v=dynamic[d];CK(cudaSetDevice(int(d)));CK(cudaMemset(v.grid,0,sizeof(*v.grid)));std::memset(v.out,0,sizeof(*v.out));v.host->mode=0;_mm_sfence();std::atomic_thread_fence(std::memory_order_seq_cst);}}

static bool valid_bundle(int layer,const std::string&dir){
 try {
  auto mpath=dir+"/resident_layer_manifest.json";std::ifstream mf(mpath);std::string text((std::istreambuf_iterator<char>(mf)),{});
  if(text.find("\"status\": \"PACKED_LAYER_RESIDENT_EXPERTS\"")==std::string::npos||text.find("\"layer\": "+std::to_string(layer))==std::string::npos)return false;
  for(int d=0;d<20;d++){std::string b=dir+"/gpu_"+(d<10?"0":"")+std::to_string(d)+"/";for(auto n:{"gate.u8","up.u8","down.u8","gate_scale.f32","up_scale.f32","down_scale.f32"})if(!std::filesystem::is_regular_file(b+n))return false;}
  return true;
 }catch(...){return false;}
}
static void emit_error(const std::string&id,const char*code,int layer){
 if(!strcmp(code,"INVALID_LAYER"))printf("{\"request_id\":\"%s\",\"status\":\"error\",\"code\":\"INVALID_LAYER\",\"layer_id\":%d,\"valid_layer_range\":[0,47]}\n",id.c_str(),layer);
 else if(!strcmp(code,"LAYER_LOAD_FAILED"))printf("{\"request_id\":\"%s\",\"status\":\"error\",\"code\":\"LAYER_LOAD_FAILED\",\"layer_id\":%d,\"retryable\":false}\n",id.c_str(),layer);
 else printf("{\"request_id\":\"%s\",\"status\":\"error\",\"code\":\"LAYER_NOT_READY\",\"layer_id\":%d,\"layer_state\":\"NOT_LOADED\",\"retryable\":true}\n",id.c_str(),layer);
}
int main(){
 setvbuf(stdout,nullptr,_IONBF,0);
 check_driver(cuInit(0),"cuInit");
 int count=0;CK(cudaGetDeviceCount(&count));if(count!=20){fprintf(stderr,"expected 20 routed GM107s, found %d\n",count);return 2;}
 std::vector<DeviceWork>work(count);for(int d=0;d<count;d++){CK(cudaSetDevice(d));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,d));if(p.major!=5||p.minor!=0)return 2;auto&w=work[d];CK(cudaMalloc(&w.x,H*4));CK(cudaMalloc(&w.ids,K*4));CK(cudaMalloc(&w.routes,K*4));CK(cudaMalloc(&w.act,K*W*4));CK(cudaMalloc(&w.out,H*4));CK(cudaStreamCreate(&w.stream));}
 std::map<int,std::string>states;std::map<int,std::vector<DeviceWeights>>resident;std::vector<DynamicWorker> dynamic(count);std::vector<RoutedKernelImage> kernel_image(count);std::string kernel_image_path;unsigned kernel_generation=0;bool started=false,parked=false;Q38CpuMailboxRegion cpu_mailbox;std::string line;
 const auto ple_devices=read_ple_devices(std::getenv("Q38_PLE_IPC_MANIFEST"));
 std::vector<QMem> qmem(count);std::vector<std::string> snapshots;int mcfd=-1;void* mcpu[5]={MAP_FAILED,MAP_FAILED,MAP_FAILED,MAP_FAILED,MAP_FAILED};
 const std::string config=getenv("Q38_PEX_GROUP_CONFIG")?getenv("Q38_PEX_GROUP_CONFIG"):Q38_SRC_DIR "/pex_group_config.py";
 auto* groups_path=getenv("Q38_MULTICAST_GROUPS");auto* snapshot_root=getenv("Q38_MULTICAST_SNAPSHOT");
 if(!groups_path||!snapshot_root){fprintf(stderr,"multicast group and snapshot paths required\n");return 2;}
 auto groups=qgroups(groups_path);
 std::mutex command_mutex;std::atomic<bool> heartbeat_stop{false},heartbeat_failed{false};std::thread heartbeat;int heartbeat_seconds=30;if(auto* h=getenv("Q38_HEARTBEAT_SECONDS")){heartbeat_seconds=atoi(h);if(heartbeat_seconds<0){fprintf(stderr,"invalid Q38_HEARTBEAT_SECONDS\n");return 2;}}
 unsigned long long resident_idle_deadline_cycles=120000000000ULL;
 if(const char* value=getenv("Q38_RESIDENT_IDLE_DEADLINE_CYCLES")){char* end=nullptr;auto parsed=strtoull(value,&end,10);if(!end||*end||parsed==0){fprintf(stderr,"invalid Q38_RESIDENT_IDLE_DEADLINE_CYCLES\n");return 2;}resident_idle_deadline_cycles=parsed;}

 while(std::getline(std::cin,line)){std::istringstream in(line);std::string cmd,id;int layer=-1;in>>cmd;
  if(cmd=="STOP"){puts("{\"status\":\"ok\",\"code\":\"STOPPED\"}");break;}
  if(cmd=="PARK_WORKERS"){
   heartbeat_stop=true;if(heartbeat.joinable())heartbeat.join();
   try{std::lock_guard<std::mutex>guard(command_mutex);if(started&&q38_all_workers_exited(dynamic,"ROUTED")){started=false;parked=true;}else if(started){stop_resident_grid(dynamic);started=false;parked=true;}else if(!parked)throw std::runtime_error("workers have not been started");puts("{\"status\":\"ok\",\"code\":\"WORKERS_PARKED\",\"branch\":\"routed\",\"weights_retained\":true}");}
   catch(const std::exception&e){printf("{\"status\":\"error\",\"code\":\"WORKER_PARK_FAILED\",\"message\":\"%s\"}\n",e.what());}continue;
  }
  if(cmd=="WAKE_WORKERS"){
   try{std::lock_guard<std::mutex>guard(command_mutex);if(started&&q38_all_workers_exited(dynamic,"ROUTED")){started=false;parked=true;}if(!started){if(!parked)throw std::runtime_error("workers have not been started");reset_resident_grid(dynamic);for(int d=0;d<count;d++){auto&v=dynamic[d];CK(cudaSetDevice(d));if(kernel_image[d].function)launch_kernel_image(kernel_image[d],v,unsigned(d),resident_idle_deadline_cycles);else{resident_dynamic_routed<<<10,THREADS,0,v.stream>>>(v.table,v.device,v.grid,v.act,v.local,v.out_device,v.ple_table,resident_idle_deadline_cycles);CK(cudaGetLastError());}v.running=true;}for(auto&v:dynamic)if(!await_value(&v.out->ready,1,10000))throw std::runtime_error("routed worker wake readiness timeout");started=true;parked=false;}puts("{\"status\":\"ok\",\"code\":\"WORKERS_ACTIVE\",\"branch\":\"routed\",\"weights_retained\":true}");}
   catch(const std::exception&e){printf("{\"status\":\"error\",\"code\":\"WORKER_WAKE_FAILED\",\"message\":\"%s\"}\n",e.what());}continue;
  }
  if(cmd=="LOAD_CPU_MAILBOX_MANIFEST"){
   std::string path;in>>path;if(path.empty()||path.find_first_of(" \t\r\n")!=std::string::npos){puts("{\"status\":\"error\",\"code\":\"INVALID_CPU_MAILBOX_MANIFEST_PATH\"}");continue;}
   auto emit_mailbox_aliases=[&](){printf("{\"status\":\"ok\",\"code\":\"CPU_MAILBOX_LOADED\",\"run_id\":\"%s\",\"workers\":20,\"device_slots\":[",cpu_mailbox.run_id.c_str());for(int d=0;d<count;d++){if(d)putchar(',');printf("{\"group\":%d,\"worker\":%d,\"bdf\":\"%s\",\"device_va\":\"0x%llx\"}",d/4,d%4,groups[d/4].gpus[d%4].c_str(),(unsigned long long)reinterpret_cast<uintptr_t>(cpu_mailbox.device_slots[d]));}puts("]}");};
   bool stopped=false;try{std::lock_guard<std::mutex> guard(command_mutex);auto manifest=q38_read_cpu_mailbox_manifest(path);
    if(!cpu_mailbox.run_id.empty()){if(cpu_mailbox.run_id!=manifest.run_id)throw std::runtime_error("a different CPU mailbox is already registered; unload it first");emit_mailbox_aliases();continue;}
    std::vector<unsigned> workers(count);for(unsigned d=0;d<unsigned(count);d++)workers[d]=d;
    const bool restart=started;if(restart){stop_resident_grid(dynamic);stopped=true;}
    auto staged=q38_register_cpu_mailbox(manifest,workers);cpu_mailbox=std::move(staged);for(int d=0;d<int(dynamic.size());d++)if(dynamic[d].host){dynamic[d].host->batch_area=reinterpret_cast<uintptr_t>(cpu_mailbox.device_slots[d])-size_t(d)*Q38_CPU_MAILBOX_SLOT_STRIDE+Q38_BATCH_OFF;_mm_sfence();}
    if(restart)for(int d=0;d<count;d++){CK(cudaSetDevice(d));auto&v=dynamic[d];v.host->cpu_mailbox_worker=unsigned(d);v.host->cpu_mailbox_reserved=0;v.host->cpu_mailbox_device_ptr=reinterpret_cast<uintptr_t>(cpu_mailbox.device_slots[d]);v.host->batch_area=reinterpret_cast<uintptr_t>(cpu_mailbox.device_slots[d])-size_t(d)*Q38_CPU_MAILBOX_SLOT_STRIDE+Q38_BATCH_OFF;_mm_sfence();}
    if(restart){reset_resident_grid(dynamic);for(int d=0;d<count;d++){CK(cudaSetDevice(d));if(kernel_image[d].function)launch_kernel_image(kernel_image[d],dynamic[d],unsigned(d),resident_idle_deadline_cycles);else{resident_dynamic_routed<<<10,THREADS,0,dynamic[d].stream>>>(dynamic[d].table,dynamic[d].device,dynamic[d].grid,dynamic[d].act,dynamic[d].local,dynamic[d].out_device,dynamic[d].ple_table,resident_idle_deadline_cycles);CK(cudaGetLastError());}dynamic[d].running=true;}for(auto&v:dynamic)if(!await_value(&v.out->ready,1,10000))throw std::runtime_error("CPU mailbox kernel restart readiness timeout");}
    stopped=false;emit_mailbox_aliases();
   }catch(const std::exception&e){fprintf(stderr,"ROUTED_CPU_MAILBOX_ERROR,%s\n",e.what());if(stopped&&cpu_mailbox.run_id.empty()){try{reset_resident_grid(dynamic);for(int d=0;d<count;d++){CK(cudaSetDevice(d));if(kernel_image[d].function)launch_kernel_image(kernel_image[d],dynamic[d],unsigned(d),resident_idle_deadline_cycles);else{resident_dynamic_routed<<<10,THREADS,0,dynamic[d].stream>>>(dynamic[d].table,dynamic[d].device,dynamic[d].grid,dynamic[d].act,dynamic[d].local,dynamic[d].out_device,dynamic[d].ple_table,resident_idle_deadline_cycles);CK(cudaGetLastError());}dynamic[d].running=true;}for(auto&v:dynamic)if(!await_value(&v.out->ready,1,10000))throw std::runtime_error("rollback kernel restart timed out");}catch(...){fprintf(stderr,"CPU mailbox rollback could not prove worker recovery; retaining CUDA resources\n");for(;;)std::this_thread::sleep_for(std::chrono::minutes(1));}}else if(stopped){fprintf(stderr,"CPU mailbox setup failed after commit; retaining worker resources for recovery\n");for(;;)std::this_thread::sleep_for(std::chrono::minutes(1));}puts("{\"status\":\"error\",\"code\":\"CPU_MAILBOX_LOAD_FAILED\"}");}continue;
  }
  if(cmd=="UNLOAD_CPU_MAILBOX_MANIFEST"){
   try{std::lock_guard<std::mutex> guard(command_mutex);const bool restart=started;if(restart){stop_resident_grid(dynamic);started=false;}if(restart)for(int d=0;d<count;d++){CK(cudaSetDevice(d));dynamic[d].host->cpu_mailbox_device_ptr=0;dynamic[d].host->batch_area=0;dynamic[d].host->cpu_mailbox_worker=0;_mm_sfence();}q38_unregister_cpu_mailbox(cpu_mailbox);if(restart){reset_resident_grid(dynamic);for(int d=0;d<count;d++){CK(cudaSetDevice(d));if(kernel_image[d].function)launch_kernel_image(kernel_image[d],dynamic[d],unsigned(d),resident_idle_deadline_cycles);else{resident_dynamic_routed<<<10,THREADS,0,dynamic[d].stream>>>(dynamic[d].table,dynamic[d].device,dynamic[d].grid,dynamic[d].act,dynamic[d].local,dynamic[d].out_device,dynamic[d].ple_table,resident_idle_deadline_cycles);CK(cudaGetLastError());}dynamic[d].running=true;}for(auto&v:dynamic)if(!await_value(&v.out->ready,1,10000))throw std::runtime_error("kernel restart after CPU mailbox unload timed out");started=true;}puts("{\"status\":\"ok\",\"code\":\"CPU_MAILBOX_UNLOADED\"}");}
   catch(const std::exception&e){fprintf(stderr,"ROUTED_CPU_MAILBOX_UNLOAD_ERROR,%s\n",e.what());puts("{\"status\":\"error\",\"code\":\"CPU_MAILBOX_UNLOAD_FAILED\"}");}continue;
  }
  if(cmd=="RELOAD_KERNELS"){
   std::string path;in>>path;if(path.empty()||path.find_first_of(" \t\r\n")!=std::string::npos){puts("{\"status\":\"error\",\"code\":\"INVALID_KERNEL_IMAGE_PATH\"}");continue;}
   bool was_started=started,quiesce_attempted=false;
   try{std::lock_guard<std::mutex> guard(command_mutex);
    // Stop the resident grid before loading into these CUDA contexts. Module
    // loading may synchronize with active work, while the old grid polls until
    // explicitly stopped.
    if(was_started){quiesce_attempted=true;stop_resident_grid(dynamic);}
    auto staged=load_kernel_image(path,count);
    auto previous=std::move(kernel_image);kernel_image=std::move(staged);kernel_image_path=path;
    // Unload the old module while its kernels are stopped. Unloading after
    // relaunch can synchronize the CUDA context and wait forever on the new
    // resident grid.
    unload_kernel_image(previous);
    if(was_started){reset_resident_grid(dynamic);for(int d=0;d<count;d++){CK(cudaSetDevice(d));launch_kernel_image(kernel_image[d],dynamic[d],unsigned(d),resident_idle_deadline_cycles);dynamic[d].running=true;}for(auto&v:dynamic)if(!await_value(&v.out->ready,1,10000))throw std::runtime_error("reloaded routed kernel startup timeout");}
    ++kernel_generation;printf("{\"status\":\"ok\",\"code\":\"KERNELS_RELOADED\",\"generation\":%u,\"running\":%s,\"image\":\"%s\"}\n",kernel_generation,was_started?"true":"false",path.c_str());
   }catch(const std::exception&e){fprintf(stderr,"ROUTED_KERNEL_RELOAD_ERROR,%s\n",e.what());printf("{\"status\":\"error\",\"code\":\"KERNEL_RELOAD_FAILED\",\"message\":\"%s\"}\n",e.what());if(quiesce_attempted){fflush(stdout);fprintf(stderr,"reload could not prove a clean restart; retaining CUDA resources\n");for(;;)std::this_thread::sleep_for(std::chrono::minutes(1));}}continue;
  }
  if(cmd=="LOAD"){
   std::string dir;in>>layer>>dir;if(started){printf("{\"status\":\"error\",\"code\":\"LOAD_AFTER_START\"}\n");continue;}if(layer<0||layer>=48){emit_error("load","INVALID_LAYER",layer);continue;}
   if(states[layer]=="READY"){printf("{\"status\":\"ok\",\"code\":\"LAYER_READY\",\"layer_id\":%d,\"already_ready\":true}\n",layer);continue;}
   states[layer]="LOADING";g_file_read_s=0;g_cpu_pack_s=0;g_cuda_upload_s=0;fflush(stdout);
   if(!valid_bundle(layer,dir)){states[layer]="ERROR";printf("{\"status\":\"error\",\"code\":\"LAYER_LOAD_FAILED\",\"layer_id\":%d,\"reason\":\"bundle_missing_or_manifest_mismatch\"}\n",layer);continue;}
   std::vector<DeviceWeights> layerw(count);bool ok=true;
   const char*par_env=getenv("Q38_PARALLEL_LOAD");const bool parallel_load=!(par_env&&strcmp(par_env,"0")==0);
   if(!parallel_load)for(int d=0;d<count&&ok;d++){CK(cudaSetDevice(d));std::string b=dir+"/gpu_"+(d<10?"0":"")+std::to_string(d)+"/";auto rg=read_file<uint8_t>(b+"gate.u8",size_t(E)*W*H),ru=read_file<uint8_t>(b+"up.u8",size_t(E)*W*H),rd=read_file<uint8_t>(b+"down.u8",size_t(E)*W*H);auto pg=pack_gate(rg),pu=pack_gate(ru),pd=pack_down(rd);auto gs=read_file<float>(b+"gate_scale.f32",size_t(E)*W*(H/GROUP)),us=read_file<float>(b+"up_scale.f32",size_t(E)*W*(H/GROUP)),ds=read_file<float>(b+"down_scale.f32",size_t(E)*H);auto&w=layerw[d];w.g=upload(pg);w.u=upload(pu);w.d=upload(pd);w.gs=upload(gs);w.us=upload(us);w.ds=upload(ds);}
   double load_wall_s=0;
   if(parallel_load){
    // One thread per M10 card: pinned O_DIRECT chunks, async H2D, full readback compare
    // (qwen38_parallel_load.cuh). gate/up: pack_gate is a little-endian identity, so the raw
    // bytes are the packed words. down: raw bytes uploaded and verified, then transposed on
    // the GPU by pack_down_kernel (same index map as pack_down).
    auto tp=std::chrono::steady_clock::now();q38load::Stats st;
    try{
     std::vector<int> devs(count);for(int d=0;d<count;d++)devs[d]=d;
     q38load::per_card(q38load::group_by_card(devs),[&](size_t,const std::vector<int>&gdevs){
      q38load::Slots slots;
      for(int d:gdevs){
       q38load::cuda_check(cudaSetDevice(d),"cudaSetDevice load");cudaStream_t sm;q38load::cuda_check(cudaStreamCreateWithFlags(&sm,cudaStreamNonBlocking),"cudaStreamCreate");
       std::string b=dir+"/gpu_"+(d<10?"0":"")+std::to_string(d)+"/";auto&w=layerw[d];
       const size_t pb=size_t(E)*W*H,sb=size_t(E)*W*(H/GROUP)*sizeof(float),db=size_t(E)*H*sizeof(float);
       uint8_t*raw_down=nullptr;
       q38load::cuda_check(cudaMalloc(&w.g,pb),"cudaMalloc g");q38load::cuda_check(cudaMalloc(&w.u,pb),"cudaMalloc u");q38load::cuda_check(cudaMalloc(&w.d,pb),"cudaMalloc d");
       q38load::cuda_check(cudaMalloc(&w.gs,sb),"cudaMalloc gs");q38load::cuda_check(cudaMalloc(&w.us,sb),"cudaMalloc us");q38load::cuda_check(cudaMalloc(&w.ds,db),"cudaMalloc ds");
       q38load::cuda_check(cudaMalloc(&raw_down,pb),"cudaMalloc raw_down");
       q38load::upload_file(slots,st,d,sm,b+"gate.u8",0,pb,w.g);
       q38load::upload_file(slots,st,d,sm,b+"up.u8",0,pb,w.u);
       q38load::upload_file(slots,st,d,sm,b+"down.u8",0,pb,raw_down);
       q38load::upload_file(slots,st,d,sm,b+"gate_scale.f32",0,sb,w.gs);
       q38load::upload_file(slots,st,d,sm,b+"up_scale.f32",0,sb,w.us);
       q38load::upload_file(slots,st,d,sm,b+"down_scale.f32",0,db,w.ds);
       q38load::cuda_check(cudaSetDevice(d),"cudaSetDevice transpose");
       const size_t words=size_t(E)*(W/4)*H;pack_down_kernel<<<unsigned((words+255)/256),256,0,sm>>>(raw_down,w.d);
       q38load::cuda_check(cudaGetLastError(),"pack_down_kernel launch");q38load::cuda_check(cudaStreamSynchronize(sm),"pack_down_kernel sync");
       q38load::cuda_check(cudaFree(raw_down),"cudaFree raw_down");cudaStreamDestroy(sm);
      }
     });
    }catch(const std::exception&e){fprintf(stderr,"ROUTED_PARALLEL_LOAD_ERROR,layer,%d,%s\n",layer,e.what());ok=false;}
    load_wall_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-tp).count();
    g_file_read_s=st.read_ns.load()/1e9;g_cpu_pack_s=0;g_cuda_upload_s=load_wall_s;
    fprintf(stderr,"ROUTED_PARALLEL_LOAD,layer,%d,seconds,%.3f,bytes,%llu,GBps,%.2f,direct_files,%llu,buffered_files,%llu,read_thread_s,%.2f,wait_thread_s,%.2f\n",layer,load_wall_s,st.bytes.load(),st.bytes.load()/load_wall_s/1e9,st.direct_files.load(),st.buffered_files.load(),st.read_ns.load()/1e9,st.wait_ns.load()/1e9);
   }
   if(ok){resident[layer]=std::move(layerw);states[layer]="READY";printf("{\"status\":\"ok\",\"code\":\"LAYER_READY\",\"layer_id\":%d,\"file_read_seconds\":%.6f,\"cpu_pack_seconds\":%.6f,\"cuda_alloc_copy_seconds\":%.6f,\"load_mode\":\"%s\",\"load_wall_seconds\":%.6f}\n",layer,g_file_read_s,g_cpu_pack_s,g_cuda_upload_s,parallel_load?"parallel_per_card":"serial",load_wall_s);}else{states[layer]="ERROR";emit_error("load","LAYER_LOAD_FAILED",layer);}continue;
  }
  if(cmd=="STATUS"){in>>layer;if(layer<0||layer>=48){emit_error("status","INVALID_LAYER",layer);continue;}const char*s=states.count(layer)?states[layer].c_str():"NOT_LOADED";printf("{\"status\":\"ok\",\"layer_id\":%d,\"layer_state\":\"%s\"}\n",layer,s);continue;}
  if(cmd=="INFER"){
   std::string reqpath,outpath,partpath;in>>id>>layer>>reqpath>>outpath>>partpath;
   if(layer<0||layer>=48){emit_error(id,"INVALID_LAYER",layer);continue;}
   if(!states.count(layer)||states[layer]=="NOT_LOADED"||states[layer]=="LOADING"){emit_error(id,"LAYER_NOT_READY",layer);continue;}
   if(states[layer]=="ERROR"){emit_error(id,"LAYER_LOAD_FAILED",layer);continue;}
   try{std::lock_guard<std::mutex> command_guard(command_mutex);if(heartbeat_failed)throw std::runtime_error("heartbeat failure");auto bytes=std::filesystem::file_size(reqpath);if(bytes!=sizeof(uint32_t)+size_t(H)*sizeof(float)+size_t(K)*sizeof(int32_t)+size_t(K)*sizeof(float))throw std::runtime_error("request byte length mismatch: "+std::to_string(bytes));std::ifstream f(reqpath,std::ios::binary);if(!f)throw std::runtime_error("request file open failed: "+reqpath);Request r=read_request(f);if(r.layer!=layer)throw std::runtime_error("request layer mismatch");for(int k=0;k<K;k++)if(r.ids[k]<0||r.ids[k]>=E||!std::isfinite(r.routes[k]))throw std::runtime_error("invalid expert ID or route");for(float x:r.x)if(!std::isfinite(x))throw std::runtime_error("invalid input");std::vector<float>parts(size_t(count)*H),out(H,0.0f);
    if(!started){
     for(int d=0;d<count;d++){
      CK(cudaSetDevice(d)); auto &v=dynamic[d]; import_ple_table(v,unsigned(d),ple_devices); std::vector<DeviceWeights>table(48);
      for(auto &entry:resident) table[entry.first]=entry.second[d]; v.table=upload(table);
      auto get=reinterpret_cast<ExportGetterQ>(dlsym(RTLD_DEFAULT,"rm_trace_get_last_export"));if(!get)throw std::runtime_error("RM trace preload missing");
      qcreate_mapping(qmem[d],groups[d/4].gpus[d%4],d/4,d%4,get,Q38_WORKER_BAR_MAP_BYTES);v.host=reinterpret_cast<ResidentDynamicControl*>(qmem[d].cpu);v.device=reinterpret_cast<ResidentDynamicControl*>(qmem[d].va);
      CK(cudaMemsetAsync(reinterpret_cast<void*>(qmem[d].va),0,sizeof(*v.host),qmem[d].stream));CK(cudaStreamSynchronize(qmem[d].stream));CK(cudaMalloc(&v.batch_scratch,Q38_BATCH_SCRATCH_BYTES));v.host->batch_scratch=reinterpret_cast<uintptr_t>(v.batch_scratch);v.host->batch_scratch_bytes=unsigned(Q38_BATCH_SCRATCH_BYTES);_mm_sfence();resident_emit_pdb_record(d/4,d%4,groups[d/4].gpus[d%4].c_str(),qmem[d].device,qmem[d].va);if(!cpu_mailbox.run_id.empty()){v.host->cpu_mailbox_worker=unsigned(d);v.host->cpu_mailbox_reserved=0;v.host->cpu_mailbox_device_ptr=reinterpret_cast<uintptr_t>(cpu_mailbox.device_slots[d]);v.host->batch_area=reinterpret_cast<uintptr_t>(cpu_mailbox.device_slots[d])-size_t(d)*Q38_CPU_MAILBOX_SLOT_STRIDE+Q38_BATCH_OFF;_mm_sfence();}
      CK(cudaHostAlloc(&v.out,sizeof(*v.out),cudaHostAllocMapped|cudaHostAllocPortable)); CK(cudaHostGetDevicePointer(&v.out_device,v.out,0)); memset(v.out,0,sizeof(*v.out));
      CK(cudaMalloc(&v.grid,sizeof(*v.grid)));CK(cudaMemset(v.grid,0,sizeof(*v.grid)));CK(cudaMalloc(&v.act,K*W*sizeof(float)));CK(cudaMalloc(&v.local,H*sizeof(float)));
      CK(cudaStreamCreateWithFlags(&v.stream,cudaStreamNonBlocking));cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,d));int active=0;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,resident_dynamic_routed,THREADS,0));if(active*prop.multiProcessorCount<10)throw std::runtime_error("resident grid exceeds occupancy");
     }
     for(int g=0;g<5;g++){
      auto snap=std::string(snapshot_root)+"/group"+std::to_string(g);const auto& group=groups[g];
      auto cap=std::vector<std::string>{"python3",config,"capture","--snapshot",snap,"--upstream",group.up,"--ports"};cap.insert(cap.end(),group.ports.begin(),group.ports.end());cap.push_back("--gpus");cap.insert(cap.end(),group.gpus.begin(),group.gpus.end());
      if(qrun(cap))throw std::runtime_error("PEX capture failed");snapshots.push_back(snap);
      auto apply=std::vector<std::string>{"python3",config,"apply","--snapshot",snap,"--base",qhex(group.base),"--map-bus"};for(int i=0;i<4;i++)apply.push_back(qhex(qmem[g*4+i].rmmap.addr));if(qrun(apply))throw std::runtime_error("PEX apply failed");
     }
     mcfd=open("/dev/m10_qwen_mc_wc6",O_RDWR|O_CLOEXEC);syscheck(mcfd,"open multicast mapper");
     for(int g=0;g<5;g++){mcpu[g]=mmap(nullptr,QWEN_MAP,PROT_READ|PROT_WRITE,MAP_SHARED,mcfd,g*QWEN_MAP);if(mcpu[g]==MAP_FAILED)throw std::runtime_error("multicast mmap failed");}
     for(int d=0;d<count;d++){auto&v=dynamic[d];CK(cudaSetDevice(d));if(kernel_image[d].function)launch_kernel_image(kernel_image[d],v,unsigned(d),resident_idle_deadline_cycles);else{resident_dynamic_routed<<<10,THREADS,0,v.stream>>>(v.table,v.device,v.grid,v.act,v.local,v.out_device,v.ple_table,resident_idle_deadline_cycles);CK(cudaGetLastError());}v.running=true;}
     for(auto &v:dynamic)if(!await_value(&v.out->ready,1,10000))throw std::runtime_error("resident startup timeout"); started=true;
     if(heartbeat_seconds>0) heartbeat=std::thread([&]{while(!heartbeat_stop){for(int i=0;i<heartbeat_seconds&&!heartbeat_stop;i++)std::this_thread::sleep_for(std::chrono::seconds(1));if(heartbeat_stop)break;std::lock_guard<std::mutex> lock(command_mutex);if(heartbeat_stop)break;
      for(auto&v:dynamic)v.seq=v.host->seq;unsigned seq=dynamic[0].seq+1;for(auto&v:dynamic)if(v.seq+1!=seq){heartbeat_failed=true;return;}
      for(int g=0;g<5;g++)reinterpret_cast<volatile uint32_t*>(mcpu[g])[2]=3;_mm_sfence();std::atomic_thread_fence(std::memory_order_seq_cst);for(int g=0;g<5;g++)reinterpret_cast<volatile uint32_t*>(mcpu[g])[0]=seq;_mm_sfence();for(auto&v:dynamic)v.seq=seq;
      for(auto&v:dynamic)if(!await_value(&v.out->seq,seq,10000)||v.out->status){fprintf(stderr,"routed heartbeat failed; retaining resources\n");heartbeat_failed=true;return;}
      fprintf(stderr,"ROUTED_HEARTBEAT,%u\n",seq);
     }});
    }
    ResidentDynamicControl payload{};payload.layer=layer;payload.mode=1;memcpy(payload.ids,r.ids.data(),K*sizeof(int32_t));memcpy(payload.routes,r.routes.data(),K*sizeof(float));memcpy(payload.x,r.x.data(),H*sizeof(float));
    auto* words=reinterpret_cast<const uint32_t*>(&payload);static_assert(sizeof(ResidentDynamicControl)%4==0);constexpr size_t mailbox_word=offsetof(ResidentDynamicControl,cpu_mailbox_worker)/4;
    for(int g=0;g<5;g++){auto* dst=reinterpret_cast<volatile uint32_t*>(mcpu[g]);for(size_t j=1;j<mailbox_word;j++)dst[j]=words[j]; /* request fields only: mailbox + batch fields (and later appended config) are service-owned */}
    _mm_sfence();std::atomic_thread_fence(std::memory_order_seq_cst);
    for(int d=0;d<count;d++){auto* p=reinterpret_cast<volatile uint32_t*>(qmem[d].cpu);if(p[1]!=uint32_t(layer)||p[4]!=words[4]||p[24]!=words[24])throw std::runtime_error("multicast readback mismatch");}
    for(auto&v:dynamic)v.seq=v.host->seq;const unsigned seq=dynamic[0].seq+1;for(auto&v:dynamic)if(v.seq+1!=seq)throw std::runtime_error("worker sequence divergence");
    for(int g=0;g<5;g++)reinterpret_cast<volatile uint32_t*>(mcpu[g])[0]=seq;
    _mm_sfence();std::atomic_thread_fence(std::memory_order_seq_cst);for(auto&v:dynamic)v.seq=seq;
    for(int d=0;d<count;d++){
     auto&v=dynamic[d];if(!await_value(&v.out->seq,v.seq,10000))throw std::runtime_error("resident result timeout");if(v.out->status)throw std::runtime_error("resident worker failure");
     for(int j=0;j<H;j++){float x=v.out->result[j];parts[size_t(d)*H+j]=x;out[j]+=x;}
    }
    std::ofstream of(outpath,std::ios::binary);of.write((char*)out.data(),H*4);std::ofstream pf(partpath,std::ios::binary);pf.write((char*)parts.data(),parts.size()*4);printf("{\"request_id\":\"%s\",\"status\":\"ok\",\"layer_id\":%d}\n",id.c_str(),layer);
   }catch(const std::exception&e){bool live=false;for(auto &v:dynamic)live=live||v.running;if(live){fprintf(stderr,"resident failure: %s; retaining resources for recovery\n",e.what());for(;;)std::this_thread::sleep_for(std::chrono::minutes(1));}fprintf(stderr,"ROUTED_INFER_ERROR,request,%s,layer,%d,error,%s\n",id.c_str(),layer,e.what());fflush(stderr);printf("{\"request_id\":\"%s\",\"status\":\"error\",\"code\":\"INVALID_REQUEST\",\"message\":\"%s\"}\n",id.c_str(),e.what());}continue;
  }
  puts("{\"status\":\"error\",\"code\":\"UNKNOWN_COMMAND\"}");
 }
 heartbeat_stop=true;if(heartbeat.joinable())heartbeat.join();if(heartbeat_failed){fprintf(stderr,"heartbeat failure; retaining resources\n");for(;;)std::this_thread::sleep_for(std::chrono::minutes(1));}
 if(started){
  for(auto&v:dynamic){v.seq=v.host->seq;v.host->mode=2;publish(v,++v.seq);}
  for(int d=0;d<count;d++){auto&v=dynamic[d];if(!await_value(&v.out->seq,v.seq,10000)){fprintf(stderr,"resident stop acknowledgement timeout; retaining resources\n");for(;;)std::this_thread::sleep_for(std::chrono::minutes(1));}CK(cudaSetDevice(d));CK(cudaStreamSynchronize(v.stream));v.running=false;}
  for(int d=0;d<count;d++){auto&v=dynamic[d];CK(cudaSetDevice(d));if(v.ple_table)CK(cudaIpcCloseMemHandle(const_cast<uint8_t*>(v.ple_table)));CK(cudaFree(v.table));CK(cudaFree(v.grid));CK(cudaFree(v.act));CK(cudaFree(v.local));CK(cudaFreeHost(v.out));CK(cudaStreamDestroy(v.stream));}
 }
 for(void* p:mcpu)if(p!=MAP_FAILED)munmap(p,QWEN_MAP);if(mcfd>=0)close(mcfd);
 for(auto it=snapshots.rbegin();it!=snapshots.rend();++it)if(qrun({"python3",config,"restore","--snapshot",*it})){fprintf(stderr,"PEX restore failed\n");return 3;}
 q38_unregister_cpu_mailbox(cpu_mailbox);for(auto&m:qmem)qdestroy_mem(m);resident_close_pdb_export();
 for(int d=0;d<count;d++){CK(cudaSetDevice(d));for(void*p:{(void*)work[d].x,(void*)work[d].ids,(void*)work[d].routes,(void*)work[d].act,(void*)work[d].out})CK(cudaFree(p));CK(cudaStreamDestroy(work[d].stream));for(auto&entry:resident){auto&w=entry.second[d];for(void*p:{(void*)w.g,(void*)w.u,(void*)w.d,(void*)w.gs,(void*)w.us,(void*)w.ds})if(p)CK(cudaFree(p));}}
 return 0;
}
