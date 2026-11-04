#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

constexpr int H=2560, W=32, K=10, THREADS=256;
#ifndef MAGIC_INT8
#define MAGIC_INT8 0
#endif
#ifndef DOWN_UNROLL
#define DOWN_UNROLL 1
#endif
#ifndef FP8_GROUP
#define FP8_GROUP 128
#endif
constexpr int kDownUnroll = DOWN_UNROLL;
void ck(cudaError_t e,const char *s){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",s,cudaGetErrorString(e));exit(1);}}
#define CK(x) ck(x,#x)
template<class T> std::vector<T> read(const std::string &p,size_t n){
 std::vector<T> v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*sizeof(T));
 if(!f || f.peek()!=EOF){fprintf(stderr,"bad file %s\n",p.c_str());exit(1);}return v;
}
template<class T> T* upload(const std::vector<T>&v){T*p;CK(cudaMalloc(&p,v.size()*sizeof(T)));CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
struct Data{uint32_t *g,*u,*d;float *gs,*us,*ds,*x,*routes,*act,*out;};
struct Grid{unsigned count,generation,error;};
__device__ bool barrier(Grid*s,unsigned generation){
 __shared__ int failed;
 __syncthreads();
 if(threadIdx.x==0){
  __threadfence();unsigned ticket=atomicAdd(&s->count,1u);
  if(ticket==gridDim.x-1){atomicExch(&s->count,0u);__threadfence();atomicExch(&s->generation,generation);}
  unsigned long long start=clock64();
  while(atomicAdd(&s->generation,0u)!=generation && !atomicAdd(&s->error,0u))
   if(clock64()-start>1000000000ULL) atomicExch(&s->error,1u);
  failed=atomicAdd(&s->error,0u)!=0;
 }
 __syncthreads();return !failed;
}
template<int FP> __device__ __forceinline__ float dec(unsigned b){
 if(!FP){
  #if MAGIC_INT8
  // Exactly maps every signed byte to FP32 with an exponent/mantissa bias.
  return __uint_as_float(0x4b000000u | (b ^ 0x80u)) - 8388736.0f;
  #else
  return float((int)(int8_t)b);
  #endif
 }
 unsigned a=b&127;
 float v=(a<8)?float(a)*(1.0f/512):__uint_as_float((a<<20)+(120u<<23));
 return (b&128)?-v:v;
}
__device__ float sumwarp(float x){for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
template<int FP> __device__ void gate(Data p){
 __shared__ float gsums[8],usums[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int task=blockIdx.x;task<K*W;task+=gridDim.x){
  float a=0,b=0;
  for(int word=threadIdx.x;word<H/4;word+=blockDim.x){
   unsigned g=p.g[task*(H/4)+word],u=p.u[task*(H/4)+word];
   float sg=p.gs[task*(H/FP8_GROUP)+word/(FP8_GROUP/4)],su=p.us[task*(H/FP8_GROUP)+word/(FP8_GROUP/4)];
   float4 x=((const float4*)p.x)[word];float xs[4]={x.x,x.y,x.z,x.w};
   #pragma unroll
   for(int j=0;j<4;j++){a=fmaf(dec<FP>((g>>(8*j))&255)*sg,xs[j],a);b=fmaf(dec<FP>((u>>(8*j))&255)*su,xs[j],b);}
  }
  a=sumwarp(a);b=sumwarp(b);
  if(!lane){gsums[warp]=a;usums[warp]=b;}__syncthreads();
  if(!warp){a=lane<8?gsums[lane]:0;b=lane<8?usums[lane]:0;a=sumwarp(a);b=sumwarp(b);
   if(!lane)p.act[task]=(a/(1+expf(-a)))*b;}
  __syncthreads();
 }
}
template<int FP> __device__ void down(Data p){
 __shared__ float acts[K*W];
 for(int i=threadIdx.x;i<K*W;i+=blockDim.x)acts[i]=p.act[i];__syncthreads();
 for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<H;row+=gridDim.x*blockDim.x){
  float sum=0;
  #pragma unroll 1
  for(int e=0;e<K;e++){
   float tmp=0;
   #pragma unroll kDownUnroll
   for(int word=0;word<W/4;word++){
    unsigned q=p.d[(e*(W/4)+word)*H+row];
    #pragma unroll
    for(int j=0;j<4;j++)tmp=fmaf(dec<FP>((q>>(8*j))&255),acts[e*W+word*4+j],tmp);
   }
   sum=fmaf(tmp*p.ds[e*H+row],p.routes[e],sum);
  }
  p.out[row]=sum;
 }
}
__device__ __forceinline__ unsigned long long gpu_ns(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
template<int FP> __global__ void resident(Data p,Grid *state,unsigned long long *ticks,int warm,int rounds){
 unsigned gen=0;unsigned long long start=0;
 for(int r=0;r<warm+rounds;r++){
  if(!blockIdx.x&&!threadIdx.x)start=gpu_ns();
  if(!barrier(state,++gen))return;
  gate<FP>(p);
  if(!barrier(state,++gen))return;
  down<FP>(p);
  if(!barrier(state,++gen))return;
  if(!blockIdx.x&&!threadIdx.x&&r>=warm)ticks[r-warm]=gpu_ns()-start;
 }
}
struct Worker{int device,shard,blocks,clock;Data p;Grid *state;unsigned long long *ticks;cudaStream_t stream;cudaEvent_t begin,end;std::vector<float> ref;};
double pct(std::vector<double>v,double q){std::sort(v.begin(),v.end());return v[std::min(v.size()-1,(size_t)(q*(v.size()-1)))];}
int main(int argc,char**argv){
 if(argc!=7){fprintf(stderr,"usage: bench DIR int8|fp8 DEVICES ROUNDS CSV OUTPUT_F32\n");return 2;}
 std::string dir=argv[1],fmt=argv[2];int devices=atoi(argv[3]),rounds=atoi(argv[4]);bool fp=fmt=="fp8";
 if((fmt!="int8"&&!fp)||devices<1||devices>20||rounds<1||rounds>10000)return 2;
 std::string prefix=fp?"official_fp8":"int8";int available;CK(cudaGetDeviceCount(&available));if(available<devices)return 2;
 std::vector<Worker> workers;
 for(int dev=0;dev<devices;dev++){
  CK(cudaSetDevice(dev));cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,dev));
  if(prop.major!=5||prop.minor!=0){fprintf(stderr,"device %d is not sm50\n",dev);return 2;}
  Worker a{};a.device=dev;a.shard=dev;a.clock=prop.clockRate;
  std::string b=dir+"/"+prefix+"."+std::to_string(dev)+".";
  a.p.g=upload(read<uint32_t>(b+"gate.bin",K*W*H/4));a.p.u=upload(read<uint32_t>(b+"up.bin",K*W*H/4));
  auto raw=read<uint8_t>(b+"down.bin",K*W*H);std::vector<uint32_t>packed(K*(W/4)*H);
  for(int e=0;e<K;e++)for(int w=0;w<W/4;w++)for(int row=0;row<H;row++){
   uint32_t v=0;for(int j=0;j<4;j++)v|=uint32_t(raw[(e*W+w*4+j)*H+row])<<(j*8);
   packed[(e*(W/4)+w)*H+row]=v;
  }
  a.p.d=upload(packed);a.p.gs=upload(read<float>(b+"gs.bin",K*W*H/FP8_GROUP));a.p.us=upload(read<float>(b+"us.bin",K*W*H/FP8_GROUP));a.p.ds=upload(read<float>(b+"ds.bin",K*H));
  a.p.x=upload(read<float>(dir+"/input.bin",H));a.p.routes=upload(read<float>(dir+"/routes.bin",K));
  a.ref=read<float>(b+"ref.bin",H);
  CK(cudaMalloc(&a.p.act,K*W*sizeof(float)));CK(cudaMalloc(&a.p.out,H*sizeof(float)));
  CK(cudaMalloc(&a.state,sizeof(Grid)));CK(cudaMemset(a.state,0,sizeof(Grid)));
  CK(cudaMalloc(&a.ticks,rounds*sizeof(unsigned long long)));
  CK(cudaStreamCreate(&a.stream));CK(cudaEventCreate(&a.begin));CK(cudaEventCreate(&a.end));
  int active;if(fp)CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,resident<1>,THREADS,0));else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,resident<0>,THREADS,0));
  a.blocks=std::min(25,active*prop.multiProcessorCount);
  printf("DEVICE,%d,shard,%d,blocks,%d,clock_khz,%d,format,%s\n",dev,dev,a.blocks,a.clock,fmt.c_str());
  workers.push_back(a);
 }
 for(auto &a:workers){CK(cudaSetDevice(a.device));CK(cudaEventRecord(a.begin,a.stream));
  if(fp)resident<1><<<a.blocks,THREADS,0,a.stream>>>(a.p,a.state,a.ticks,100,rounds);
  else resident<0><<<a.blocks,THREADS,0,a.stream>>>(a.p,a.state,a.ticks,100,rounds);
  CK(cudaGetLastError());CK(cudaEventRecord(a.end,a.stream));
 }
 std::ofstream csv(argv[5]);csv<<"format,gpu,shard,round,gpu_ns,gpu_us\n";
 std::vector<float>combined(H,0);bool ok=true;
 for(auto &a:workers){CK(cudaSetDevice(a.device));CK(cudaStreamSynchronize(a.stream));
  Grid state;CK(cudaMemcpy(&state,a.state,sizeof(state),cudaMemcpyDeviceToHost));if(state.error){fprintf(stderr,"barrier timeout GPU %d\n",a.device);return 1;}
  std::vector<unsigned long long>ts(rounds);CK(cudaMemcpy(ts.data(),a.ticks,ts.size()*8,cudaMemcpyDeviceToHost));
  std::vector<float>out(H);CK(cudaMemcpy(out.data(),a.p.out,H*4,cudaMemcpyDeviceToHost));
  double se=0,rr=0,maxerr=0;int bad=0;
  for(int i=0;i<H;i++){double d=out[i]-a.ref[i];se+=d*d;rr+=double(a.ref[i])*a.ref[i];maxerr=std::max(maxerr,std::abs(d));if(!std::isfinite(out[i])||std::abs(d)>2e-7+std::abs(a.ref[i])*1e-3)bad++;combined[i]+=out[i];}
  ok &= bad==0;std::vector<double>us;double sum=0;
  for(int i=0;i<rounds;i++){double t=double(ts[i])/1000;us.push_back(t);sum+=t;csv<<fmt<<','<<a.device<<','<<a.shard<<','<<i<<','<<ts[i]<<','<<t<<'\n';}
  float elapsed;CK(cudaEventElapsedTime(&elapsed,a.begin,a.end));
  printf("RESULT,%s,gpu,%d,bad,%d,rel_rmse,%.9g,max_abs,%.9g,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,event_us_per_round_including_warmup,%.3f\n",fmt.c_str(),a.device,bad,sqrt(se/std::max(rr,1e-40)),maxerr,sum/rounds,pct(us,.5),pct(us,.95),pct(us,.99),elapsed*1000/(rounds+100));
 }
 if(devices==20){auto ref=read<float>(dir+"/bf16_source_reference.bin",H);double se=0,rr=0;for(int i=0;i<H;i++){double d=combined[i]-ref[i];se+=d*d;rr+=double(ref[i])*ref[i];}printf("FULL_20_SLICE_REFERENCE_REL_RMSE,%.9g\n",sqrt(se/rr));}
 if(ok){std::ofstream output(argv[6],std::ios::binary);output.write(reinterpret_cast<const char*>(combined.data()),combined.size()*sizeof(float));if(!output){fprintf(stderr,"failed writing output %s\n",argv[6]);return 1;}}
 for(auto&a:workers){CK(cudaSetDevice(a.device));for(void*p:{(void*)a.p.g,(void*)a.p.u,(void*)a.p.d,(void*)a.p.gs,(void*)a.p.us,(void*)a.p.ds,(void*)a.p.x,(void*)a.p.routes,(void*)a.p.act,(void*)a.p.out,(void*)a.state,(void*)a.ticks})CK(cudaFree(p));CK(cudaEventDestroy(a.begin));CK(cudaEventDestroy(a.end));CK(cudaStreamDestroy(a.stream));}
 printf("STATUS,%s\n",ok?"PASS":"FAIL");return ok?0:1;
}
