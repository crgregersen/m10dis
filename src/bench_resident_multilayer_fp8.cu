#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int H=2560,W=32,K=10,E=512,THREADS=256,GROUP=128;
struct DeviceWeights{uint32_t *g,*u,*d;float *gs,*us,*ds;};
struct DeviceWork{float *x,*routes,*act,*out;int32_t *ids;cudaStream_t stream;};
struct Request{int layer;std::vector<float>x,routes;std::vector<int32_t>ids;};
static double g_file_read_s=0.0,g_cpu_pack_s=0.0,g_cuda_upload_s=0.0;
void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(1);}}
#define CK(x) ck((x),#x)
template<class T>std::vector<T> read_file(const std::string&p,size_t n){auto t=std::chrono::steady_clock::now();std::vector<T>v(n);std::ifstream f(p,std::ios::binary);f.read((char*)v.data(),n*sizeof(T));if(!f||f.peek()!=EOF){fprintf(stderr,"bad file or size: %s\n",p.c_str());std::exit(1);}g_file_read_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return v;}
template<class T>T*upload(const std::vector<T>&v){auto t=std::chrono::steady_clock::now();T*p=nullptr;CK(cudaMalloc(&p,v.size()*sizeof(T)));CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));g_cuda_upload_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return p;}
std::vector<uint32_t> pack_gate(const std::vector<uint8_t>&r){auto t=std::chrono::steady_clock::now();std::vector<uint32_t>p(r.size()/4);for(size_t i=0;i<p.size();i++)p[i]=uint32_t(r[4*i])|(uint32_t(r[4*i+1])<<8)|(uint32_t(r[4*i+2])<<16)|(uint32_t(r[4*i+3])<<24);g_cpu_pack_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return p;}
std::vector<uint32_t> pack_down(const std::vector<uint8_t>&r){auto t=std::chrono::steady_clock::now();std::vector<uint32_t>p(size_t(E)*(W/4)*H);for(int e=0;e<E;e++)for(int g=0;g<W/4;g++)for(int row=0;row<H;row++){uint32_t v=0;for(int j=0;j<4;j++)v|=uint32_t(r[(size_t(e)*W+g*4+j)*H+row])<<(8*j);p[(size_t(e)*(W/4)+g)*H+row]=v;}g_cpu_pack_s+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return p;}
__device__ __forceinline__ float fp8e4m3(uint32_t x){unsigned b=x&255,a=b&127;float v=(a<8)?float(a)*(1.0f/512.0f):__uint_as_float((a<<20)+(120u<<23));return(b&128)?-v:v;}
__device__ float sumwarp(float x){for(int d=16;d;d>>=1)x+=__shfl_down_sync(0xffffffff,x,d);return x;}
struct Params{DeviceWeights w;const float*x,*routes;const int32_t*ids;float*act,*out;};
__global__ void gate_layer(Params p){__shared__ float gs[8],us[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int task=blockIdx.x;task<K*W;task+=gridDim.x){int k=task/W,r=task%W,e=p.ids[k];float a=0,b=0;size_t wb=(size_t(e)*W+r)*(H/4),sb=(size_t(e)*W+r)*(H/GROUP);for(int q=threadIdx.x;q<H/4;q+=blockDim.x){uint32_t gw=p.w.g[wb+q],uw=p.w.u[wb+q];float sg=p.w.gs[sb+q/(GROUP/4)],su=p.w.us[sb+q/(GROUP/4)];float4 xv=((const float4*)p.x)[q];float xs[4]={xv.x,xv.y,xv.z,xv.w};
#pragma unroll
for(int j=0;j<4;j++){a=fmaf(fp8e4m3(gw>>(8*j))*sg,xs[j],a);b=fmaf(fp8e4m3(uw>>(8*j))*su,xs[j],b);}}a=sumwarp(a);b=sumwarp(b);if(!lane){gs[warp]=a;us[warp]=b;}__syncthreads();if(!warp){a=lane<8?gs[lane]:0;b=lane<8?us[lane]:0;a=sumwarp(a);b=sumwarp(b);if(!lane)p.act[task]=(a/(1+expf(-a)))*b;}__syncthreads();}}
__global__ void down_layer(Params p){__shared__ float acts[K*W];for(int i=threadIdx.x;i<K*W;i+=blockDim.x)acts[i]=p.act[i];__syncthreads();for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<H;row+=gridDim.x*blockDim.x){float sum=0;for(int k=0;k<K;k++){int e=p.ids[k];float part=0;
#pragma unroll
for(int q=0;q<W/4;q++){uint32_t v=p.w.d[(size_t(e)*(W/4)+q)*H+row];
#pragma unroll
for(int j=0;j<4;j++)part=fmaf(fp8e4m3(v>>(8*j)),acts[k*W+q*4+j],part);}sum=fmaf(part*p.w.ds[size_t(e)*H+row],p.routes[k],sum);}p.out[row]=sum;}}
Request read_request(std::ifstream&f){Request r;uint32_t layer;f.read((char*)&layer,4);r.layer=(int)layer;r.x.resize(H);r.ids.resize(K);r.routes.resize(K);f.read((char*)r.x.data(),H*4);f.read((char*)r.ids.data(),K*4);f.read((char*)r.routes.data(),K*4);if(!f)throw std::runtime_error("truncated request");return r;}
Request*unused_request_symbol();
#ifndef QWEN_M10_LIBRARY_ONLY
int main(int argc,char**argv){
 if(argc!=5){fprintf(stderr,"usage: resident_multilayer MANIFEST REQUESTS OUTPUT_F32 PARTIALS_F32\n");return 2;}
 std::map<int,std::string> dirs;{std::ifstream m(argv[1]);int layer;std::string path;while(m>>layer>>path){if(layer<0||layer>=48||dirs.count(layer)){fprintf(stderr,"invalid/duplicate manifest layer\n");return 2;}dirs[layer]=path;}}if(dirs.empty())return 2;
 std::ifstream rf(argv[2],std::ios::binary);uint32_t n=0;rf.read((char*)&n,4);if(!rf||!n||n>10000)return 2;std::vector<Request>req;req.reserve(n);for(uint32_t i=0;i<n;i++){req.push_back(read_request(rf));if(!dirs.count(req.back().layer)){fprintf(stderr,"request layer not READY: %d\n",req.back().layer);return 2;}}if(rf.peek()!=EOF)return 2;
 int count=0;CK(cudaGetDeviceCount(&count));if(count!=20)return 2;std::vector<int> layer_ids;for(auto &kv:dirs)layer_ids.push_back(kv.first);std::vector<std::vector<DeviceWeights>>weights(layer_ids.size(),std::vector<DeviceWeights>(count));std::vector<DeviceWork>work(count);std::map<int,size_t>index;for(size_t i=0;i<layer_ids.size();i++)index[layer_ids[i]]=i;
 for(int dev=0;dev<count;dev++){CK(cudaSetDevice(dev));cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,dev));if(prop.major!=5||prop.minor!=0)return 2;for(size_t li=0;li<layer_ids.size();li++){std::string base=dirs[layer_ids[li]]+"/gpu_"+(dev<10?"0":"")+std::to_string(dev)+"/";auto rg=read_file<uint8_t>(base+"gate.u8",size_t(E)*W*H),ru=read_file<uint8_t>(base+"up.u8",size_t(E)*W*H),rd=read_file<uint8_t>(base+"down.u8",size_t(E)*W*H);auto pg=pack_gate(rg),pu=pack_gate(ru),pd=pack_down(rd);auto gs=read_file<float>(base+"gate_scale.f32",size_t(E)*W*(H/GROUP)),us=read_file<float>(base+"up_scale.f32",size_t(E)*W*(H/GROUP)),ds=read_file<float>(base+"down_scale.f32",size_t(E)*H);auto &w=weights[li][dev];w.g=upload(pg);w.u=upload(pu);w.d=upload(pd);w.gs=upload(gs);w.us=upload(us);w.ds=upload(ds);}
  auto&w=work[dev];CK(cudaMalloc(&w.x,H*4));CK(cudaMalloc(&w.ids,K*4));CK(cudaMalloc(&w.routes,K*4));CK(cudaMalloc(&w.act,K*W*4));CK(cudaMalloc(&w.out,H*4));CK(cudaStreamCreate(&w.stream));size_t free_b,total_b;CK(cudaMemGetInfo(&free_b,&total_b));printf("MULTILAYER_DEVICE,%d,layers,%zu,free,%zu,total,%zu\n",dev,layer_ids.size(),free_b,total_b);}
 printf("MULTILAYER_READY,layers,%zu,requests,%u\n",layer_ids.size(),n);
 std::vector<float>out(size_t(n)*H),parts(size_t(n)*count*H);for(uint32_t r=0;r<n;r++){auto&rq=req[r];for(int dev=0;dev<count;dev++){CK(cudaSetDevice(dev));auto&w=work[dev];CK(cudaMemcpyAsync(w.x,rq.x.data(),H*4,cudaMemcpyHostToDevice,w.stream));CK(cudaMemcpyAsync(w.ids,rq.ids.data(),K*4,cudaMemcpyHostToDevice,w.stream));CK(cudaMemcpyAsync(w.routes,rq.routes.data(),K*4,cudaMemcpyHostToDevice,w.stream));Params p{weights[index[rq.layer]][dev],w.x,w.routes,w.ids,w.act,w.out};gate_layer<<<25,THREADS,0,w.stream>>>(p);CK(cudaGetLastError());down_layer<<<25,THREADS,0,w.stream>>>(p);CK(cudaGetLastError());}for(int dev=0;dev<count;dev++){CK(cudaSetDevice(dev));auto&w=work[dev];CK(cudaStreamSynchronize(w.stream));CK(cudaMemcpy(parts.data()+(size_t(r)*count+dev)*H,w.out,H*4,cudaMemcpyDeviceToHost));}float*dst=out.data()+size_t(r)*H;std::fill(dst,dst+H,0);for(int dev=0;dev<count;dev++)for(int j=0;j<H;j++)dst[j]+=parts[(size_t(r)*count+dev)*H+j];printf("MULTILAYER_REQUEST,%u,layer,%d\n",r,rq.layer);}
 {std::ofstream f(argv[3],std::ios::binary);f.write((char*)out.data(),out.size()*4);} {std::ofstream f(argv[4],std::ios::binary);f.write((char*)parts.data(),parts.size()*4);}
 for(int dev=0;dev<count;dev++){CK(cudaSetDevice(dev));auto&w=work[dev];for(void*p:{(void*)w.x,(void*)w.ids,(void*)w.routes,(void*)w.act,(void*)w.out})CK(cudaFree(p));CK(cudaStreamDestroy(w.stream));for(size_t li=0;li<weights.size();li++){auto&q=weights[li][dev];for(void*p:{(void*)q.g,(void*)q.u,(void*)q.d,(void*)q.gs,(void*)q.us,(void*)q.ds})CK(cudaFree(p));}}
 printf("MULTILAYER_STATUS,PASS\n");return 0;
}
#endif
