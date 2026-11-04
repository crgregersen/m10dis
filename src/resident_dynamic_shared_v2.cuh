// Shared worker kernel v2 (deployed): loaded as a cubin (resident_shared_kernel_v2_module.cu) by RELOAD_KERNELS.
#pragma once
// v2 of resident_dynamic_shared.cuh: only the mode-1 (MoE) path differs.
#include "resident_direct_mailbox.cuh"
#include "resident_cpu_mailbox_abi.cuh"
#include "resident_shared_v2_math.cuh"
#include "pci_cluster_exchange.cuh"   // mode 6 row gather (2026-11-03)
#include "resident_batch_layout.cuh"   // mode 8 batch layout helpers (2026-11-03)
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
// __launch_bounds__(256, 2): the persistent grid needs 2 blocks per SM (10 blocks on 5 SMs, occupancy-checked at load).
extern "C" __global__ void __launch_bounds__(256, 2) resident_dynamic_shared(const Weights* layers,SharedResidentControl* control,SharedResidentGrid* grid,float* partial,SharedResidentReturn* output,const uint8_t* ple_table,unsigned worker,unsigned long long deadline){
 auto *direct=reinterpret_cast<ResidentDirectResult*>(reinterpret_cast<char*>(control)+Q38_DIRECT_RESULT_OFF);
 __shared__ unsigned seq,layer,mode,timeout,xfmt,prow,pok;__shared__ float hidden[(S+9)/10];__shared__ float reduction[T];__shared__ __align__(16) float q38_xs[H];__shared__ __align__(16) float q38_xs2[H];__shared__ __align__(16) float q38_hs[S];unsigned last=control->seq,gen=0;
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
  // 2026-11-03: a relaunch after an error exit must not keep the old error in the direct record
  // (the services' relaunch only clears the host return struct); the RTX wait checks both words.
  output->status=0;output->exit_reason=0;direct->status=0;direct->exit_reason=0;
  output->ready=1;direct->ready=1;__threadfence_system();
 }
 unsigned long long q38_t_seen=0;   // worker timing (block 0 thread 0)
 for(;;){
  if(!threadIdx.x){unsigned long long start=clock64();unsigned v=last;while(v==last&&clock64()-start<deadline)v=control->seq;seq=v;if(!blockIdx.x)q38_t_seen=pci_cluster_gtimer();timeout=v==last;layer=control->layer;mode=control->mode;xfmt=control->reserved;
   /* mode 9 (pipelined batch, 2026-11-03): next request = last+1 in host slot (last+1)&1, see the routed kernel */
   prow=0;pok=0;if(mode==Q38_BATCH_PIPE_MODE&&!timeout&&control->batch_area){if(v-last<=Q38_BATCH_PIPE_SLOTS)seq=last+1;const volatile unsigned*hd=q38_batch_req_hdr(q38_batch_area(control),seq&1u);prow=hd[1];layer=hd[2];pok=hd[3]==seq;}}
  __syncthreads();unsigned reason=0;if(timeout)reason|=1u;if(mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=5&&mode!=PCI_CLUSTER_MODE_ROW_GATHER&&mode!=PCI_CLUSTER_MODE_LM_TOPK&&mode!=Q38_BATCH_MODE&&mode!=Q38_BATCH_PIPE_MODE)reason|=2u;if(mode==1&&layer>=48)reason|=4u;if(mode==4&&control->ple_count&&!ple_table)reason|=8u;if(mode==4&&control->ple_count>SHARED_RESIDENT_PLE_MAX_ROWS)reason|=16u;if(mode==PCI_CLUSTER_MODE_ROW_GATHER&&!threadIdx.x)reason|=pci_cluster_row_gather_validate(control,ple_table);if(mode==PCI_CLUSTER_MODE_LM_TOPK&&!threadIdx.x)reason|=pci_cluster_lm_validate(control,ple_table);if(mode==Q38_BATCH_MODE&&(control->ple_count<1||control->ple_count>Q38_BATCH_MAX_ROWS||!control->batch_area||!control->batch_scratch||control->batch_scratch_bytes<Q38_SBATCH_SCRATCH_BYTES||control->cpu_mailbox_worker>=Q38_CPU_MAILBOX_WORKERS||!(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)||layer>=48))reason|=Q38_BATCH_REASON_CONFIG;if(mode==Q38_BATCH_PIPE_MODE&&(!pok||prow<1||prow>Q38_BATCH_PIPE_ROWS||!control->batch_area||!control->batch_scratch||control->batch_scratch_bytes<Q38_SBATCH_SCRATCH_BYTES||control->cpu_mailbox_worker>=Q38_CPU_MAILBOX_WORKERS||!(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)||layer>=48))reason|=Q38_BATCH_REASON_CONFIG;if(seq!=last+1)reason|=32u;if(reason&&!threadIdx.x)atomicOr(&grid->error,reason);
  if(!shared_resident_barrier(grid,++gen)){if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;pci_cluster_slot_error(pci_cluster_rtx_slot(control),seq,why);}return;}
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
  if(mode==PCI_CLUSTER_MODE_LM_TOPK){
   // LM-head top-k (pci_cluster_exchange.cuh): slice matvec on all blocks, exact top-K + LSE in block 0.
   pci_cluster_lm_matvec(control,ple_table,q38_xs);
   if(!shared_resident_barrier(grid,++gen))return;
   float lse=0.0f;
   if(!blockIdx.x)lse=pci_cluster_lm_select(control,ple_table,direct);
   if(!shared_resident_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){direct->gate=lse;output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,lse);}
   last=seq;continue;
  }
  if(mode==PCI_CLUSTER_MODE_ROW_GATHER){
   // Broadcast row gather (pci_cluster_exchange.cuh): copy only this worker's rows, report the count.
   const unsigned owned=pci_cluster_row_gather(control,direct,ple_table);
   if(!shared_resident_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){direct->gate=__uint_as_float(owned);output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,__uint_as_float(owned));}
   last=seq;continue;
  }
  if(mode==4){const unsigned n=control->ple_count;auto*direct_bytes=reinterpret_cast<uint8_t*>(direct->result);for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<n*SHARED_RESIDENT_PLE_ROW_BYTES;i+=gridDim.x*blockDim.x){const unsigned row=i/SHARED_RESIDENT_PLE_ROW_BYTES,byte=i%SHARED_RESIDENT_PLE_ROW_BYTES;const uint8_t value=ple_table[control->ple_offsets[row]+byte];output->ple_rows[i]=value;direct_bytes[i]=value;}__threadfence_system();if(!shared_resident_barrier(grid,++gen))return;if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;}last=seq;continue;}
  if(mode==Q38_BATCH_MODE||mode==Q38_BATCH_PIPE_MODE){   /* mode 9: same body on host slot seq&1 */
   // Batched MoE, shared expert, expert-major (2026-11-03 v2): stage all R rows to device scratch; hidden
   // in tiles of 2 x rows held in shared memory (each g/u vector loaded once per tile); worker 0 computes
   // row t's gate on block t % grid; down in tiles of 16 hidden rows (each d vector loaded once per tile);
   // one coalesced copy to the host part area. Per row the arithmetic is mode 1's (bit-identical).
   // Worker 0 then reduces ((0+s20)+s21)+s22)+s23 like the RTX.
#define Q38_SBATCH_FAIL { if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;pci_cluster_slot_error(pci_cluster_rtx_slot(control),seq,why);} return; }
   const bool bpipe=mode==Q38_BATCH_PIPE_MODE;
   const unsigned R=bpipe?prow:control->ple_count, me=control->cpu_mailbox_worker, bslot=bpipe?(seq&1u):0u;
   char *bt=q38_batch_area(control);
   Weights w=layers[layer];int warp=threadIdx.x>>5,lane=threadIdx.x&31;
   const int gwarp=int(blockIdx.x)*WARPS+warp,nwarps=int(gridDim.x)*WARPS;
   float *mine=q38_batch_part(bt,me,bslot);
   char *sc=reinterpret_cast<char*>(control->batch_scratch);
   float *sx=reinterpret_cast<float*>(sc+Q38_SBATCH_SCRATCH_X),*shid=reinterpret_cast<float*>(sc+Q38_SBATCH_SCRATCH_HID),*sdn=reinterpret_cast<float*>(sc+Q38_SBATCH_SCRATCH_DN);
   unsigned long long q38_t_stage=0,q38_t_hid=0,q38_t_dn=0;
   const bool xbf=q38_batch_stage_x(bt,R,H,sx,bslot);   // FP32 or exact BF16 request rows (Q38_BATCH_REQ_FMT)
   __threadfence();if(!shared_resident_barrier(grid,++gen)) Q38_SBATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x)q38_t_stage=pci_cluster_gtimer();
   // BF16 requests (2026-11-03): 4-row tiles with x as BF16 pairs in q38_xs (rows 0,1) and q38_xs2 (rows 2,3):
   // the gate/up weights (1.64 MB per worker) are read once per 4 rows instead of 2. FP32 requests: 2-row tiles.
   if(xbf)for(unsigned t0=0;t0<R;t0+=4){
    const unsigned n=R-t0<4u?R-t0:4u;
    uint32_t*xw0=reinterpret_cast<uint32_t*>(q38_xs),*xw2=reinterpret_cast<uint32_t*>(q38_xs2);
    const uint32_t*const xr[4]={xw0,xw0+H/2,n>2?xw2:xw0+(n-1)*(H/2),n>3?xw2+H/2:(n>2?xw2:xw0+(n-1)*(H/2))};
    {const volatile float*xv=sx;for(unsigned r=0;r<n;r++){uint32_t*dst=r<2?xw0+r*(H/2):xw2+(r-2)*(H/2);
      for(int i=threadIdx.x;i<H/2;i+=blockDim.x){const float a=xv[size_t(t0+r)*H+2*i],b=xv[size_t(t0+r)*H+2*i+1];dst[i]=(__float_as_uint(a)>>16)|(__float_as_uint(b)&0xFFFF0000u);}}}
    __syncthreads();
    const uint32_t*const xq[4]={xr[0],n>1?xr[1]:xr[0],xr[2],xr[3]};
    float*const hq[4]={shid+size_t(t0)*S,shid+size_t(t0+(n>1?1:0))*S,shid+size_t(t0+(n>2?2:0))*S,shid+size_t(t0+(n>3?3:0))*S};
    q38_shared_v2_hidden4_bfx(w.g,w.u,xq,n,hq,gwarp,nwarps,lane,H,S);
    if(worker==0)for(unsigned j=0;j<n;j++){const unsigned t=t0+j;if(blockIdx.x==t%gridDim.x){
     reduction[threadIdx.x]=q38_shared_v2_gate_partial_bfx(w.gate,xq[j],threadIdx.x,blockDim.x,H);__syncthreads();
     for(int stride=blockDim.x/2;stride;stride>>=1){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
     if(!threadIdx.x)q38_batch_gate(bt,bslot)[t]=1.0f/(1.0f+expf(-reduction[0]));__syncthreads();}}
    __syncthreads();
   }
   else for(unsigned t0=0;t0<R;t0+=2){
    const bool two=t0+1<R;
    {const volatile float*xv=sx;for(int i=threadIdx.x;i<H;i+=blockDim.x){q38_xs[i]=xv[size_t(t0)*H+i];if(two)q38_xs2[i]=xv[size_t(t0+1)*H+i];}}
    __syncthreads();
    q38_shared_v2_hidden2(w.g,w.u,q38_xs,two?q38_xs2:nullptr,shid+size_t(t0)*S,shid+size_t(t0+1)*S,gwarp,nwarps,lane,H,S);
    if(worker==0)for(unsigned j=0;j<(two?2u:1u);j++){const unsigned t=t0+j;if(blockIdx.x==t%gridDim.x){
     reduction[threadIdx.x]=q38_shared_v2_gate_partial(w.gate,j?q38_xs2:q38_xs,threadIdx.x,blockDim.x,H);__syncthreads();
     for(int stride=blockDim.x/2;stride;stride>>=1){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}
     if(!threadIdx.x)q38_batch_gate(bt,bslot)[t]=1.0f/(1.0f+expf(-reduction[0]));__syncthreads();}}
    __syncthreads();
   }
   __threadfence();if(!shared_resident_barrier(grid,++gen)) Q38_SBATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x)q38_t_hid=pci_cluster_gtimer();
   for(unsigned t0=0;t0<R;t0+=16){
    const unsigned nt=R-t0<16u?R-t0:16u;
    {const volatile float*hv=shid;for(unsigned i=threadIdx.x;i<nt*unsigned(S);i+=blockDim.x)q38_xs[i]=hv[size_t(t0)*S+i];}
    __syncthreads();
    for(int base=gwarp*32;base<H;base+=nwarps*32)q38_shared_v2_down_tile(w.d,q38_xs,nt,sdn+size_t(t0)*H,base,lane,H,S);
    __syncthreads();
   }
   __threadfence();if(!shared_resident_barrier(grid,++gen)) Q38_SBATCH_FAIL
   {const volatile float*dv=sdn;for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<R*unsigned(H);i+=gridDim.x*blockDim.x)mine[i]=dv[i];}
   __threadfence_system();if(!shared_resident_barrier(grid,++gen)) Q38_SBATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x)q38_t_dn=pci_cluster_gtimer();
   if(worker==0){
    if(!threadIdx.x){const unsigned long long t0=clock64();
     for(unsigned k=1;k<4;k++){volatile unsigned *f=q38_batch_flag(bt,me+k,bslot);while(*f!=seq){if(clock64()-t0>Q38_AGG_DEADLINE){atomicOr(&grid->error,64u);break;}}}}
    __syncthreads();__threadfence();
    for(unsigned i=blockIdx.x*blockDim.x+threadIdx.x;i<R*unsigned(H);i+=gridDim.x*blockDim.x){
     float sh=0.0f;
     for(unsigned k=0;k<4;k++)sh+=reinterpret_cast<const volatile float*>(q38_batch_part(bt,me+k,bslot))[i];
     mine[i]=sh;}
    __threadfence_system();if(!shared_resident_barrier(grid,++gen)) Q38_SBATCH_FAIL
   }
   if(!blockIdx.x&&!threadIdx.x){*q38_batch_flag(bt,me,bslot)=seq;__threadfence_system();
    pci_cluster_wtiming_write(control,seq,layer,0u,bpipe?6u:4u,q38_t_seen,q38_t_stage,q38_t_hid,q38_t_dn);
    output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;
    pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,0.0f);}
#undef Q38_SBATCH_FAIL
   last=seq;continue;
  }
  // ---- Q38 shared v2 (2026-11-02): same grid, buffers and protocol as v1; see resident_shared_v2_math.cuh ----
  Weights w=layers[layer];int warp=threadIdx.x>>5,lane=threadIdx.x&31;
  const int gwarp=int(blockIdx.x)*WARPS+warp,nwarps=int(gridDim.x)*WARPS;
  if(xfmt==1u){volatile uint32_t*xw=reinterpret_cast<volatile uint32_t*>(control->x);for(int i=threadIdx.x;i<H/2;i+=blockDim.x){const uint32_t w=xw[i];q38_xs[2*i]=__uint_as_float(w<<16);q38_xs[2*i+1]=__uint_as_float(w&0xFFFF0000u);}}   /* mode-1 BF16 x (control->reserved == 1, 2026-11-03), exact */
  else{volatile float* x=control->x;for(int i=threadIdx.x;i<H;i+=blockDim.x)q38_xs[i]=x[i];}
  __syncthreads();
  // hidden goes to partial[0..S): every block needs all S values for the down projection.
  q38_shared_v2_hidden(w.g,w.u,q38_xs,partial,gwarp,nwarps,lane,H,S);
  if(worker==0&&blockIdx.x==0){reduction[threadIdx.x]=q38_shared_v2_gate_partial(w.gate,q38_xs,threadIdx.x,blockDim.x,H);__syncthreads();for(int stride=blockDim.x/2;stride;stride>>=1){if(threadIdx.x<stride)reduction[threadIdx.x]+=reduction[threadIdx.x+stride];__syncthreads();}if(!threadIdx.x)output->gate=1.0f/(1.0f+expf(-reduction[0]));__syncthreads();}
  __threadfence();if(!shared_resident_barrier(grid,++gen))return;
  auto*cpu_slot=reinterpret_cast<Q38CpuMailboxSlot*>(control->cpu_mailbox_device_ptr);
  {volatile float* hp=partial;for(int i=threadIdx.x;i<S;i+=blockDim.x)q38_hs[i]=hp[i];}
  __syncthreads();
  for(int base=gwarp*32;base<H;base+=nwarps*32){const float mine=q38_shared_v2_down_chunk(w.d,q38_hs,base,lane,S);const int row=base+lane;if(!(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN))output->result[row]=mine;direct->result[row]=mine;if(cpu_slot)cpu_slot->result[row]=mine;}
  if(worker==0&&!blockIdx.x&&!threadIdx.x)direct->gate=output->gate;
  __threadfence_system();if(!shared_resident_barrier(grid,++gen))return;
  if(!blockIdx.x&&!threadIdx.x)pci_cluster_wtiming_write(control,seq,layer,0u,2u,q38_t_seen,0ull,0ull,0ull);
  if(!blockIdx.x&&!threadIdx.x){output->status=0;direct->status=0;if(cpu_slot){cpu_slot->worker=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?0:control->cpu_mailbox_worker;cpu_slot->layer=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?1:layer;cpu_slot->gate=output->gate;cpu_slot->status=0;}__threadfence_system();if(cpu_slot)cpu_slot->seq=seq;output->seq=seq;direct->seq=seq;}last=seq;
 }
}
