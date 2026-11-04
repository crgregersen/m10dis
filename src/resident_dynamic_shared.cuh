// Shared worker kernel v1: compiled into the shared service; the daemon start replaces it with the v2 cubin.
#pragma once
#include "resident_direct_mailbox.cuh"
#include "resident_cpu_mailbox_abi.cuh"
#define SHARED_RESIDENT_PLE_ROW_BYTES 160
#define SHARED_RESIDENT_PLE_MAX_ROWS 4096
struct SharedResidentControl { volatile unsigned seq,layer,mode,reserved; unsigned ple_count,ple_reserved[3]; unsigned long long ple_offsets[SHARED_RESIDENT_PLE_MAX_ROWS]; int32_t ids[K]; float routes[K]; float x[H]; unsigned cpu_mailbox_worker,cpu_mailbox_reserved; unsigned long long cpu_mailbox_device_ptr; unsigned long long batch_scratch; unsigned batch_scratch_bytes,batch_reserved; unsigned long long batch_area; };  /* batch_*: mode 8 (2026-11-03) */
struct SharedResidentGrid { unsigned count,generation,error; };
struct SharedResidentReturn { volatile unsigned seq,status,exit_reason,ready; float result[H]; float gate; uint8_t ple_rows[SHARED_RESIDENT_PLE_MAX_ROWS*SHARED_RESIDENT_PLE_ROW_BYTES]; };
static_assert(sizeof(SharedResidentControl) <= Q38_DIRECT_RESULT_OFF,
              "direct result mailbox overlaps shared command/control payload");
__device__ bool shared_resident_barrier(SharedResidentGrid*s,unsigned generation){
 __shared__ unsigned failed;__syncthreads();if(!threadIdx.x){__threadfence();auto t=atomicAdd(&s->count,1u);if(t==gridDim.x-1){atomicExch(&s->count,0u);__threadfence();atomicExch(&s->generation,generation);}unsigned long long start=clock64();while(atomicAdd(&s->generation,0u)!=generation&&!atomicAdd(&s->error,0u))if(clock64()-start>1000000000ULL)atomicExch(&s->error,1u);failed=atomicAdd(&s->error,0u);}__syncthreads();return failed==0;
}
extern "C" __global__ void resident_dynamic_shared(const Weights* layers,SharedResidentControl* control,SharedResidentGrid* grid,float* partial,SharedResidentReturn* output,const uint8_t* ple_table,unsigned worker,unsigned long long deadline){
 auto *direct=reinterpret_cast<ResidentDirectResult*>(reinterpret_cast<char*>(control)+Q38_DIRECT_RESULT_OFF);
 __shared__ unsigned seq,layer,mode,timeout;__shared__ float hidden[(S+9)/10];__shared__ float reduction[T];unsigned last=control->seq,gen=0;
 if(!shared_resident_barrier(grid,++gen))return;if(!blockIdx.x&&!threadIdx.x){
  // Materialize the daemon-registered host result alias before the RTX
  // installs a BAR1 PTE over it. cudaHostGetDevicePointer alone can leave
  // untouched pages without a queryable UVM leaf.
  auto *cpu_slot=reinterpret_cast<volatile unsigned*>(control->cpu_mailbox_device_ptr);
  if(cpu_slot){unsigned observed=0;for(unsigned byte=0;byte<Q38_CPU_MAILBOX_SLOT_BYTES;byte+=4096)observed^=cpu_slot[byte/sizeof(unsigned)];asm volatile("" :: "r"(observed) : "memory");}
  // Materialize all leaf pages covering the direct result record before an
  // external requester validates and installs return aliases.
  direct->result[1024]=0.0f;
  direct->result[2048]=0.0f;
  __threadfence_system();
  output->ready=1;direct->ready=1;__threadfence_system();
 }
 for(;;){
  if(!threadIdx.x){unsigned long long start=clock64();unsigned v=last;while(v==last&&clock64()-start<deadline)v=control->seq;seq=v;timeout=v==last;layer=control->layer;mode=control->mode;}
  __syncthreads();unsigned reason=0;if(timeout)reason|=1u;if(mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=5)reason|=2u;if(mode==1&&layer>=48)reason|=4u;if(mode==4&&control->ple_count&&!ple_table)reason|=8u;if(mode==4&&control->ple_count>SHARED_RESIDENT_PLE_MAX_ROWS)reason|=16u;if(seq!=last+1)reason|=32u;if(reason&&!threadIdx.x)atomicOr(&grid->error,reason);
  if(!shared_resident_barrier(grid,++gen)){if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;}return;}
  if(mode==3) { if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;} last=seq; __syncthreads(); continue; }
  if(mode==5) {
   volatile float *src=control->x;
   for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<H;i+=gridDim.x*blockDim.x) {
    const float value=src[i]; output->result[i]=value; direct->result[i]=value;
   }
   __threadfence_system();
   if(!shared_resident_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){output->gate=0.0f;direct->gate=0.0f;output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;}
   last=seq;continue;
  }
  if(mode==2){if(!blockIdx.x&&!threadIdx.x){output->status=0;output->exit_reason=2;direct->status=0;direct->exit_reason=2;__threadfence_system();output->seq=seq;direct->seq=seq;}return;}
  if(mode==4){const unsigned n=control->ple_count;auto*direct_bytes=reinterpret_cast<uint8_t*>(direct->result);for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n*SHARED_RESIDENT_PLE_ROW_BYTES;i+=gridDim.x*blockDim.x){const unsigned row=i/SHARED_RESIDENT_PLE_ROW_BYTES,byte=i%SHARED_RESIDENT_PLE_ROW_BYTES;const uint8_t value=ple_table[control->ple_offsets[row]+byte];output->ple_rows[i]=value;direct_bytes[i]=value;}__threadfence_system();if(!shared_resident_barrier(grid,++gen))return;if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;}last=seq;continue;}
  Weights w=layers[layer];volatile float* x=control->x;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
  for(int row0=warp;row0<(S+9)/10;row0+=WARPS){int row=int(blockIdx.x)+row0*10;if(row<S){size_t base=size_t(row)*H;float a=0,b=0;for(int c=lane;c<H;c+=32){float z=x[c];a=fmaf(bf(w.g[base+c]),z,a);b=fmaf(bf(w.u[base+c]),z,b);}a=ws(a);b=ws(b);if(!lane)hidden[row0]=(a/(1.0f+expf(-a)))*b;}}
  __syncthreads();for(int row0=0;row0<H;row0+=WARPS){int row=row0+warp;if(row<H){float sum=0;for(int c=lane;c<(S+9)/10;c+=32){int channel=int(blockIdx.x)+c*10;if(channel<S)sum=fmaf(bf(w.d[size_t(row)*S+channel]),hidden[c],sum);}sum=ws(sum);if(!lane)partial[size_t(blockIdx.x)*H+row]=sum;}}
  __syncthreads();if(worker==0&&blockIdx.x==0){float acc=0;for(int c=threadIdx.x;c<H;c+=blockDim.x)acc=fmaf(bf(w.gate[c]),x[c],acc);reduction[threadIdx.x]=acc;__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}if(!threadIdx.x)output->gate=1.0f/(1.0f+expf(-reduction[0]));__syncthreads();}
  __threadfence();if(!shared_resident_barrier(grid,++gen))return;
  auto*cpu_slot=reinterpret_cast<Q38CpuMailboxSlot*>(control->cpu_mailbox_device_ptr);
  if(!blockIdx.x)for(int j=threadIdx.x;j<H;j+=blockDim.x){float sum=0;for(unsigned b=0;b<gridDim.x;b++)sum+=partial[size_t(b)*H+j];output->result[j]=sum;direct->result[j]=sum;if(cpu_slot)cpu_slot->result[j]=sum;}
  if(worker==0&&!blockIdx.x&&!threadIdx.x)direct->gate=output->gate;
  __threadfence_system();if(!shared_resident_barrier(grid,++gen))return;
  if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;if(cpu_slot){cpu_slot->worker=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?0:control->cpu_mailbox_worker;cpu_slot->layer=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?1:layer;cpu_slot->gate=output->gate;cpu_slot->status=0;}__threadfence_system();if(cpu_slot)cpu_slot->seq=seq;output->seq=seq;direct->seq=seq;}last=seq;
 }
}
