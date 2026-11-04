// Routed worker kernel v3 (deployed): loaded as a cubin (resident_routed_kernel_v3_module.cu) by RELOAD_KERNELS.
// Requires bench_resident_multilayer_fp8.cu types.
// Control/input locations must be established by the owning service.
#pragma once
// v3 of resident_dynamic_routed.cuh = v2 + optional on-card leader push aggregation (2026-11-02).
#include "resident_direct_mailbox.cuh"
#include "resident_cpu_mailbox_abi.cuh"
#include "resident_routed_v2_math.cuh"
#include "pci_cluster_exchange.cuh"   // mode 6 row gather (2026-11-03)
#include "resident_routed_batch.cuh"   // mode 8 batched MoE (2026-11-03)
// Leader push aggregation block, written by the RTX peer library (Q38_RTX_PEER_LEADER_AGG)
// through the owner CPU BAR1 mapping while workers are parked; absent/zero = role 0 (v2 behaviour).
// role 1 leader: wait for n_peers flags at control+slot_offset+k*slot_stride+H*4 == seq, add the
//   n_peers pushed vectors to its own result, then return the sum to the RTX as usual.
// role 2 peer: write the result to push_va (remapped to the leader slot), then the flag, and return
//   only the RTX slot header (the RTX reducer skips its vector).
struct Q38AggBlock { unsigned magic, role, peer_index, n_peers; unsigned long long push_va; unsigned slot_offset, slot_stride; };
#define Q38_AGG_OFF 0xB000u
#define Q38_AGG_MAGIC 0x51384147u
#define Q38_AGG_DEADLINE 2000000000ULL
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
static_assert(sizeof(ResidentDynamicControl) <= Q38_AGG_OFF, "aggregation block overlaps control");
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
// Q38 routed v2 (2026-11-02): warp per (expert,row) task; see resident_routed_v2_math.cuh.
// One shared x buffer for the whole kernel (a __shared__ in a device function is one per-kernel allocation):
// mode 1 gate/up (first H), LM top-k matvec (first H) and the mode-8 gate/up tile (2*H). Keeps 2 blocks/SM.
__device__ __forceinline__ float *q38_routed_xbuf(){__shared__ __align__(16) float buf[2*H];return buf;}
// Mode-1 x format (2026-11-03): control->reserved == Q38_MODE1_X_BF16 means x[] holds H/2 words of packed BF16
// (element 2m in the low half), written by the RTX for BF16 hidden states (half the multicast bytes); widened
// exactly. 0 = FP32 (CPU-driven requests, other h types).
#define Q38_MODE1_X_BF16 1u
__device__ void dynamic_gate_layer(Params p,bool xbf16){float *q38_rx=q38_routed_xbuf();if(xbf16){const uint32_t*xw=reinterpret_cast<const uint32_t*>(p.x);for(int i=threadIdx.x;i<H/2;i+=blockDim.x){const uint32_t w=xw[i];reinterpret_cast<float2*>(q38_rx)[i]=make_float2(__uint_as_float(w<<16),__uint_as_float(w&0xFFFF0000u));}}else for(int i=threadIdx.x;i<H/4;i+=blockDim.x)reinterpret_cast<float4*>(q38_rx)[i]=reinterpret_cast<const float4*>(p.x)[i];__syncthreads();const int lane=threadIdx.x&31,gwarp=int(blockIdx.x)*int(blockDim.x>>5)+int(threadIdx.x>>5),nwarps=int(gridDim.x)*int(blockDim.x>>5);q38_routed_v2_gate_up(p.w.g,p.w.u,p.w.gs,p.w.us,p.ids,q38_rx,p.act,gwarp,nwarps,lane,H,W,K,GROUP);}
__device__ void dynamic_down_layer(Params p){__shared__ float acts[K*W];for(int i=threadIdx.x;i<K*W;i+=blockDim.x)acts[i]=p.act[i];__syncthreads();for(int row=blockIdx.x*blockDim.x+threadIdx.x;row<H;row+=gridDim.x*blockDim.x){float sum=0;for(int k=0;k<K;k++){int e=p.ids[k];float part=0;
#pragma unroll
for(int q=0;q<W/4;q++){uint32_t v=p.w.d[(size_t(e)*(W/4)+q)*H+row];
#pragma unroll
for(int j=0;j<4;j++)part=fmaf(fp8e4m3(v>>(8*j)),acts[k*W+q*4+j],part);}   /* original decode: the fast decode made this latency-bound loop slower (t17: 22.5 -> 32.8 us) */sum=fmaf(part*p.w.ds[size_t(e)*H+row],p.routes[k],sum);}p.out[row]=sum;}}

// One grid per GPU. The host must publish payload before seq and wait for
// the exact acknowledged seq before reusing control or the result buffer.
// A GPU clock deadline bounds idle polling independently of the host.
// __launch_bounds__(256, 2): the persistent grid needs 2 blocks per SM (10 blocks on 5 SMs, occupancy-checked at
// load); without it ptxas may exceed 128 registers on any change (168 with the parallel list build, 2026-11-03).
extern "C" __global__ void __launch_bounds__(256, 2) resident_dynamic_routed(const DeviceWeights* layers, ResidentDynamicControl* control, ResidentDynamicGrid* grid, float* activation, float* result_local, ResidentDynamicReturn* output, const uint8_t* ple_table, unsigned long long deadline_cycles) {
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
  // 2026-11-03: a relaunch after an error exit must not keep the old error in the direct record
  // (the services' relaunch only clears the host return struct); the RTX wait checks both words.
  output->status=0;output->exit_reason=0;direct->status=0;direct->exit_reason=0;
  output->ready=1;direct->ready=1;__threadfence_system();
 }
 unsigned long long q38_t_seen=0,q38_t_chk=0,q38_t_gu=0,q38_t_dn=0;   // worker timing (block 0 thread 0)
 for (;;) {
  __shared__ unsigned seq, layer, mode, timeout, xfmt, prow, pok;
  if(!threadIdx.x) { unsigned long long start=clock64(); unsigned v=last; while(v==last && clock64()-start<deadline_cycles) v=control->seq; seq=v; timeout=(v==last); if(!blockIdx.x)q38_t_seen=pci_cluster_gtimer(); layer=control->layer; mode=control->mode; xfmt=control->reserved;
   // mode 9 (pipelined batch, 2026-11-03): up to two requests are published ahead, so the doorbell may already
   // read last+2; the next request is always last+1 and its rows / layer / seq are in host slot (last+1) & 1.
   // A doorbell beyond last+2 keeps seq = v and fails the seq check below.
   prow=0; pok=0;
   if(mode==Q38_BATCH_PIPE_MODE&&!timeout&&control->batch_area){ if(v-last<=Q38_BATCH_PIPE_SLOTS)seq=last+1;
    const volatile unsigned *hd=q38_batch_req_hdr(q38_batch_area(control),seq&1u); prow=hd[1]; layer=hd[2]; pok=hd[3]==seq; } }
  __syncthreads();
  unsigned reason=0;
  if(timeout)reason|=1u;
  if(mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=5&&mode!=PCI_CLUSTER_MODE_ROW_GATHER&&mode!=PCI_CLUSTER_MODE_LM_TOPK&&mode!=Q38_BATCH_MODE&&mode!=Q38_BATCH_PIPE_MODE)reason|=2u;
  if(mode==1&&layer>=48)reason|=4u;
  if(mode==4&&control->ple_count&&!ple_table)reason|=8u;
  if(mode==4&&control->ple_count>RESIDENT_PLE_MAX_ROWS)reason|=16u;
  if(mode==PCI_CLUSTER_MODE_ROW_GATHER&&!threadIdx.x)reason|=pci_cluster_row_gather_validate(control,ple_table);
  if(mode==PCI_CLUSTER_MODE_LM_TOPK&&!threadIdx.x)reason|=pci_cluster_lm_validate(control,ple_table);
  if(seq!=last+1)reason|=32u;
  // 2026-11-03: validate the request before any weight access. A capture-time request with
  // uncomputed inputs carried garbage expert ids and faulted every routed GM107 (Xid 31).
  // Bad ids/routes now take the existing error path (status=1, exit_reason bits) instead.
  if(mode==1){const volatile int32_t*vid=control->ids;const volatile float*vr=control->routes;
    for(int k=0;k<K;k++){const int e=vid[k];const float r=vr[k];
    if(e<0||e>=E)reason|=128u;if(!(r==r)||r>3.0e38f||r< -3.0e38f)reason|=256u;}}
  if(reason&&!threadIdx.x)atomicOr(&grid->error,reason);
  if(mode==Q38_BATCH_MODE){const unsigned br=q38_batch_validate_routed(control,layer,E,control->ple_count,Q38_BATCH_MAX_ROWS,0u);if(br)atomicOr(&grid->error,br);}
  if(mode==Q38_BATCH_PIPE_MODE){const unsigned br=pok?q38_batch_validate_routed(control,layer,E,prow,Q38_BATCH_PIPE_ROWS,seq&1u):Q38_BATCH_REASON_CONFIG;if(br)atomicOr(&grid->error,br);}
  if(!dynamic_grid_barrier(grid,++gen)) { if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;pci_cluster_slot_error(pci_cluster_rtx_slot(control),seq,why);} return; }
  if(!blockIdx.x&&!threadIdx.x)q38_t_chk=pci_cluster_gtimer();
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
  if(mode==PCI_CLUSTER_MODE_LM_TOPK) {
   // LM-head top-k (pci_cluster_exchange.cuh): slice matvec on all blocks, exact top-K + LSE in block 0.
   pci_cluster_lm_matvec(control,ple_table,q38_routed_xbuf());
   if(!dynamic_grid_barrier(grid,++gen))return;
   float lse=0.0f;
   if(!blockIdx.x)lse=pci_cluster_lm_select(control,ple_table,direct);
   if(!dynamic_grid_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){direct->gate=lse;output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,lse);}
   last=seq;continue;
  }
  if(mode==PCI_CLUSTER_MODE_ROW_GATHER) {
   // Broadcast row gather (pci_cluster_exchange.cuh): copy only this worker's rows, report the count.
   const unsigned owned=pci_cluster_row_gather(control,direct,ple_table);
   if(!dynamic_grid_barrier(grid,++gen))return;
   if(!blockIdx.x&&!threadIdx.x){direct->gate=__uint_as_float(owned);output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,__uint_as_float(owned));}
   last=seq;continue;
  }
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
  if(mode==Q38_BATCH_MODE||mode==Q38_BATCH_PIPE_MODE) {
   // Batched MoE (resident_routed_batch.cuh): expert-major over R rows, bit-identical per row to mode 1.
   // Mode 9 = the same body on host slot seq & 1 (rows and layer from the slot header).
#define Q38_BATCH_FAIL { if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;pci_cluster_slot_error(pci_cluster_rtx_slot(control),seq,why);} return; }
   const unsigned R=mode==Q38_BATCH_PIPE_MODE?prow:control->ple_count, me=control->cpu_mailbox_worker;
   const Q38BatchView bv=q38_batch_view(control,mode==Q38_BATCH_PIPE_MODE?(seq&1u):0u);   // shared seq/mode: no extra live registers
   const DeviceWeights lw=layers[layer];
   static_assert(2*H>=E+2*int(Q38_BATCH_MAX_ROWS*Q38_BATCH_K),"stage lists exceed the shared x buffer");
   // timing extra (kind 3): high 32 bits = validation end - t_seen, low 32 = block 0's list build (ns)
   const unsigned long long q38_t_val=q38_t_chk;
   const unsigned long long q38_lists=q38_batch_stage(bv,R,E,H,reinterpret_cast<unsigned*>(q38_routed_xbuf()),seq); if(!dynamic_grid_barrier(grid,++gen)) Q38_BATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x)q38_t_chk=pci_cluster_gtimer();   // mode 8: t_checked = after check + host x staging
   static_assert(4*H*2<=2*H*4,"4 BF16 rows exceed the shared x buffer");
   if(*reinterpret_cast<volatile unsigned*>(bv.x_inexact)!=seq)q38_batch_gate_up_bf16(bv,lw.g,lw.u,lw.gs,lw.us,q38_routed_xbuf(),H,W,GROUP);
   else q38_batch_gate_up(bv,lw.g,lw.u,lw.gs,lw.us,q38_routed_xbuf(),H,W,GROUP); if(!dynamic_grid_barrier(grid,++gen)) Q38_BATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x)q38_t_gu=pci_cluster_gtimer();
   q38_batch_down_terms(bv,lw.d,lw.ds,H,W); if(!dynamic_grid_barrier(grid,++gen)) Q38_BATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x)q38_t_dn=pci_cluster_gtimer();
   const volatile Q38AggBlock *ag=reinterpret_cast<const volatile Q38AggBlock*>(reinterpret_cast<const char*>(control)+Q38_AGG_OFF);
   const unsigned brole=(ag->magic==Q38_AGG_MAGIC)?ag->role:0u, bnp=(brole==1u)?ag->n_peers:0u;
   // v2 step 3 (2026-11-03): on-card reduce streamed through the mode-1 aggregation windows (block 0; see
   // q38_batch_card_leader/peer). A leader leaves own + peers (mode-1 order) in its part area, a peer writes
   // no part area, a worker without aggregation writes its own rows.
   float *mine=q38_batch_part(bv.batch,me,bv.slot);
   q38_batch_sum_rows(bv,R,H,brole?bv.x_stage:mine);
   if(!dynamic_grid_barrier(grid,++gen)) Q38_BATCH_FAIL
   if(brole==1u&&!blockIdx.x){if(!q38_batch_card_leader(bv,R,H,seq,me,reinterpret_cast<const char*>(control)+ag->slot_offset,ag->slot_stride,bnp,mine)&&!threadIdx.x)atomicOr(&grid->error,64u);}
   if(brole==2u&&!blockIdx.x){if(!q38_batch_card_peer(bv,R,H,seq,me,reinterpret_cast<volatile float*>(ag->push_va))&&!threadIdx.x)atomicOr(&grid->error,64u);}
   if(brole&&!dynamic_grid_barrier(grid,++gen)) Q38_BATCH_FAIL
   if(!blockIdx.x&&!threadIdx.x){
    *q38_batch_flag(bv.batch,me,bv.slot)=seq; __threadfence_system();
    pci_cluster_wtiming_write(control,seq,layer,brole,mode==Q38_BATCH_PIPE_MODE?5u:3u,q38_t_seen,q38_t_chk,q38_t_gu,q38_t_dn,
      (min(q38_t_val-q38_t_seen,0xFFFFFFFFull)<<32)|min(q38_lists,0xFFFFFFFFull));
    output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;
    pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,0.0f);}
#undef Q38_BATCH_FAIL
   last=seq;continue;
  }
  Params p{layers[layer],control->x,control->routes,control->ids,activation,result_local};
  dynamic_gate_layer(p,xfmt==Q38_MODE1_X_BF16); if(!dynamic_grid_barrier(grid,++gen)) return;
  if(!blockIdx.x&&!threadIdx.x)q38_t_gu=pci_cluster_gtimer();
  dynamic_down_layer(p); if(!dynamic_grid_barrier(grid,++gen)) return;
  if(!blockIdx.x&&!threadIdx.x)q38_t_dn=pci_cluster_gtimer();
  auto *cpu_slot=reinterpret_cast<Q38CpuMailboxSlot*>(control->cpu_mailbox_device_ptr);
  // v2: in RTX-return mode nobody reads the pinned-host output->result copy; skip that PCIe write.
  const bool q38_rtx_return=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)!=0;
  const volatile Q38AggBlock *q38_agg=reinterpret_cast<const volatile Q38AggBlock*>(reinterpret_cast<const char*>(control)+Q38_AGG_OFF);
  const unsigned q38_role=(q38_agg->magic==Q38_AGG_MAGIC)?q38_agg->role:0u;
  const unsigned q38_np=q38_agg->n_peers,q38_off=q38_agg->slot_offset,q38_stride=q38_agg->slot_stride;
  volatile float *q38_push=q38_role==2u?reinterpret_cast<volatile float*>(q38_agg->push_va):nullptr;
  if(q38_role==1u){
   if(!threadIdx.x){const unsigned long long t0=clock64();
    for(unsigned k=0;k<q38_np;k++){auto *f=reinterpret_cast<const volatile unsigned*>(reinterpret_cast<const char*>(control)+q38_off+k*q38_stride+H*4u);
     while(*f!=seq){if(clock64()-t0>Q38_AGG_DEADLINE){atomicOr(&grid->error,64u);break;}}}}
   __syncthreads();__threadfence();
  }
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<H;i+=gridDim.x*blockDim.x) {
   float value=result_local[i];
   if(q38_role==1u)for(unsigned k=0;k<q38_np;k++)value+=reinterpret_cast<const volatile float*>(reinterpret_cast<const char*>(control)+q38_off+k*q38_stride)[i];
   if(q38_role==2u){q38_push[i]=value;direct->result[i]=value;continue;}
   if(!q38_rtx_return)output->result[i]=value; direct->result[i]=value;
   if(cpu_slot)cpu_slot->result[i]=value;
  }
  __threadfence_system(); if(!dynamic_grid_barrier(grid,++gen)) return;
  if(!blockIdx.x&&!threadIdx.x) pci_cluster_wtiming_write(control,seq,layer,q38_role,1u,q38_t_seen,q38_t_chk,q38_t_gu,q38_t_dn);
  if(!blockIdx.x&&!threadIdx.x) { output->status=0;direct->status=0;if(cpu_slot){cpu_slot->worker=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?0:control->cpu_mailbox_worker;cpu_slot->layer=(control->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)?1:layer;cpu_slot->status=0;}if(q38_role==2u)q38_push[H]=__uint_as_float(seq);__threadfence_system();if(cpu_slot)cpu_slot->seq=seq;output->seq=seq;direct->seq=seq; }
  last=seq;
 }
}
