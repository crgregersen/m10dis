// Routed worker kernel v1: compiled into the routed service; the daemon start replaces it with the v3 cubin.
// Requires bench_resident_multilayer_fp8.cu types.
// Control/input locations must be established by the owning service.
#pragma once
#include "resident_direct_mailbox.cuh"
#include "resident_cpu_mailbox_abi.cuh"
#define RESIDENT_PLE_ROW_BYTES 160
#define RESIDENT_PLE_MAX_ROWS 4096
// mode=4 uses ple_count/ple_offsets; offsets are byte offsets into this
// device's imported CUDA-IPC PLE allocation. MoE fields remain unchanged.
struct ResidentDynamicControl {
 volatile unsigned seq, layer, mode, reserved;
 unsigned ple_count, ple_reserved[3];
 unsigned long long ple_offsets[RESIDENT_PLE_MAX_ROWS];
 int32_t ids[K]; float routes[K]; float x[H];
 unsigned cpu_mailbox_worker, cpu_mailbox_reserved;
 unsigned long long cpu_mailbox_device_ptr;
 unsigned long long batch_scratch;          // mode 8 device scratch (routed service, 2026-11-03)
 unsigned batch_scratch_bytes, batch_reserved;
 unsigned long long batch_area;             // mode 8: worker VA of the host batch area (services, at mailbox load)
};
struct ResidentDynamicGrid { unsigned count, generation, error; };
struct ResidentDynamicReturn {
 volatile unsigned seq, status, exit_reason, ready;
 float result[H];
 uint8_t ple_rows[RESIDENT_PLE_MAX_ROWS * RESIDENT_PLE_ROW_BYTES];
};
static_assert(sizeof(ResidentDynamicControl) <= Q38_DIRECT_RESULT_OFF,
              "direct result mailbox overlaps routed command/control payload");
__device__ bool dynamic_grid_barrier(ResidentDynamicGrid *s, unsigned generation) {
 __shared__ unsigned failed; __syncthreads();
 if (!threadIdx.x) { __threadfence(); unsigned ticket=atomicAdd(&s->count,1u);
  if(ticket==gridDim.x-1){atomicExch(&s->count,0u);__threadfence();atomicExch(&s->generation,generation);}
  unsigned long long start=clock64();
  while(atomicAdd(&s->generation,0u)!=generation && !atomicAdd(&s->error,0u)) if(clock64()-start>1000000000ULL) atomicExch(&s->error,1u);
  failed=atomicAdd(&s->error,0u); }
 __syncthreads(); return failed==0;
}
__device__ void dynamic_gate_layer(Params p){__shared__ float gs[8],us[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int task=blockIdx.x;task<K*W;task+=gridDim.x){int k=task/W,r=task%W,e=p.ids[k];float a=0,b=0;size_t wb=(size_t(e)*W+r)*(H/4),sb=(size_t(e)*W+r)*(H/GROUP);for(int q=threadIdx.x;q<H/4;q+=blockDim.x){uint32_t gw=p.w.g[wb+q],uw=p.w.u[wb+q];float sg=p.w.gs[sb+q/(GROUP/4)],su=p.w.us[sb+q/(GROUP/4)];float4 xv=((const float4*)p.x)[q];float xs[4]={xv.x,xv.y,xv.z,xv.w};
#pragma unroll
for(int j=0;j<4;j++){a=fmaf(fp8e4m3(gw>>(8*j))*sg,xs[j],a);b=fmaf(fp8e4m3(uw>>(8*j))*su,xs[j],b);}}a=sumwarp(a);b=sumwarp(b);if(!lane){gs[warp]=a;us[warp]=b;}__syncthreads();if(!warp){a=lane<8?gs[lane]:0;b=lane<8?us[lane]:0;a=sumwarp(a);b=sumwarp(b);if(!lane)p.act[task]=(a/(1+expf(-a)))*b;}__syncthreads();}}
__device__ void dynamic_down_layer(Params p){__shared__ float acts[K*W];for(int i=threadIdx.x;i<K*W;i+=blockDim.x)acts[i]=p.act[i];__syncthreads();for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<H;row+=gridDim.x*blockDim.x){float sum=0;for(int k=0;k<K;k++){int e=p.ids[k];float part=0;
#pragma unroll
for(int q=0;q<W/4;q++){uint32_t v=p.w.d[(size_t(e)*(W/4)+q)*H+row];
#pragma unroll
for(int j=0;j<4;j++)part=fmaf(fp8e4m3(v>>(8*j)),acts[k*W+q*4+j],part);}sum=fmaf(part*p.w.ds[size_t(e)*H+row],p.routes[k],sum);}p.out[row]=sum;}}

// One grid per GPU. The host must publish payload before seq and wait for
// the exact acknowledged seq before reusing control or the result buffer.
// A GPU clock deadline bounds idle polling independently of the host.
extern "C" __global__ void resident_dynamic_routed(const DeviceWeights* layers, ResidentDynamicControl* control, ResidentDynamicGrid* grid, float* activation, float* result_local, ResidentDynamicReturn* output, const uint8_t* ple_table, unsigned long long deadline_cycles) {
 auto *direct = reinterpret_cast<ResidentDirectResult*>(reinterpret_cast<char*>(control) + Q38_DIRECT_RESULT_OFF);
 unsigned last=control->seq, gen=0;
 if(!dynamic_grid_barrier(grid,++gen)) return;
 if(!blockIdx.x&&!threadIdx.x){
  // Touch each page in the daemon-registered host result slot before the
  // RTX route installer queries this VA. cudaHostGetDevicePointer may return
  // the VA before UVM has materialized its leaf PTEs.
  auto *cpu_slot=reinterpret_cast<volatile unsigned*>(control->cpu_mailbox_device_ptr);
  if(cpu_slot){
   unsigned observed=0;
   for(unsigned byte=0;byte<Q38_CPU_MAILBOX_SLOT_BYTES;byte+=4096){
    observed^=cpu_slot[byte/sizeof(unsigned)];
   }
   asm volatile("" :: "r"(observed) : "memory");
  }
  // Materialize every 4-KiB page in the direct result record before the
  // external RTX mapper queries this worker VA. The RM/VMM allocation is
  // retained, but CUDA can leave untouched portions without leaf PTEs.
  direct->result[1024]=0.0f;
  direct->result[2048]=0.0f;
  __threadfence_system();
  output->ready=1;direct->ready=1;__threadfence_system();
 }
 for (;;) {
  __shared__ unsigned seq, layer, mode, timeout;
  if(!threadIdx.x) { unsigned long long start=clock64(); unsigned v=last; while(v==last && clock64()-start<deadline_cycles) v=control->seq; seq=v; timeout=(v==last); layer=control->layer; mode=control->mode; }
  __syncthreads();
  unsigned reason=0;
  if(timeout)reason|=1u;
  if(mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=5)reason|=2u;
  if(mode==1&&layer>=48)reason|=4u;
  if(mode==4&&control->ple_count&&!ple_table)reason|=8u;
  if(mode==4&&control->ple_count>RESIDENT_PLE_MAX_ROWS)reason|=16u;
  if(seq!=last+1)reason|=32u;
  if(reason&&!threadIdx.x)atomicOr(&grid->error,reason);
  if(!dynamic_grid_barrier(grid,++gen)) { if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;} return; }
  if(mode==3) { if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;} last=seq; __syncthreads(); continue; }
  if(mode==5) {
   volatile float *src=control->x;
   for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<H;i+=gridDim.x*blockDim.x) {
    const float value=src[i]; output->result[i]=value; direct->result[i]=value;
   }
   // Every producer thread orders its own payload stores before the grid
   // rendezvous. The publishing thread then exposes the completion sequence.
   __threadfence_system();
   if(!dynamic_grid_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){direct->gate=0.0f;output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;}
   last=seq;continue;
  }
  if(mode==2) { if(!blockIdx.x&&!threadIdx.x){output->status=0;output->exit_reason=2;direct->status=0;direct->exit_reason=2;__threadfence_system();output->seq=seq;direct->seq=seq;} return; }
  if(mode==4) {
   const unsigned n=control->ple_count;
   for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n*RESIDENT_PLE_ROW_BYTES;i+=gridDim.x*blockDim.x){
    const unsigned row=i/RESIDENT_PLE_ROW_BYTES, byte=i%RESIDENT_PLE_ROW_BYTES;
    output->ple_rows[i]=ple_table[control->ple_offsets[row]+byte];
    // The resident vLLM client reads this RM-backed mailbox through the
    // owner's BAR1 mapping. Keep the pinned-host return path for the older
    // diagnostic service, while publishing the same bytes into the bounded
    // direct mailbox (64 rows per transaction).
    reinterpret_cast<uint8_t*>(direct->result)[i]=ple_table[control->ple_offsets[row]+byte];
   }
   __threadfence_system();
   if(!dynamic_grid_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;}
   last=seq;continue;
  }
  Params p{layers[layer],control->x,control->routes,control->ids,activation,result_local};
  dynamic_gate_layer(p); if(!dynamic_grid_barrier(grid,++gen)) return;
  dynamic_down_layer(p); if(!dynamic_grid_barrier(grid,++gen)) return;
  auto *cpu_slot=reinterpret_cast<Q38CpuMailboxSlot*>(control->cpu_mailbox_device_ptr);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<H;i+=gridDim.x*blockDim.x) {
   const float value=result_local[i]; output->result[i]=value; direct->result[i]=value;
   if(cpu_slot)cpu_slot->result[i]=value;
  }
  __threadfence_system(); if(!dynamic_grid_barrier(grid,++gen)) return;
  if(!blockIdx.x&&!threadIdx.x) { output->status=0;direct->status=0;if(cpu_slot){cpu_slot->worker=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?0:control->cpu_mailbox_worker;cpu_slot->layer=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?1:layer;cpu_slot->status=0;}__threadfence_system();if(cpu_slot)cpu_slot->seq=seq;output->seq=seq;direct->seq=seq; }
  last=seq;
 }
}
