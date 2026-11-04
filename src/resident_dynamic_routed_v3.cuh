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
#include "resident_multirow_abi.cuh"    // mode 10 multi-row decode MoE (2026-11-04)
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

// Mode 10 checks (thread 0, before any weight access; resident_multirow_abi.cuh). Routed workers need leader
// aggregation (role 1 leader / role 2 peer) with a matching multi-row block, the RTX return route and the
// routed scratch (row-1 act and down output).
__device__ __noinline__ unsigned q38_mr_validate_routed(const ResidentDynamicControl *c, unsigned layer, unsigned xfmt) {
 const volatile ResidentDynamicControl *v=c;
 const volatile Q38AggBlock *ag=reinterpret_cast<const volatile Q38AggBlock*>(reinterpret_cast<const char*>(c)+Q38_AGG_OFF);
 const volatile Q38MrBlock *mb=reinterpret_cast<const volatile Q38MrBlock*>(reinterpret_cast<const char*>(c)+Q38_MR_BLOCK_OFF);
 const unsigned R=v->ple_count, arole=ag->magic==Q38_AGG_MAGIC?ag->role:0u, mrole=mb->role&0xFFu, maxr=mb->role>>8;
 unsigned why=0;
 // maxr: rows the RTX routes cover (set at peer open); rows above it would write outside the remapped pages
 if(R<1||R>Q38_MR_MAX_ROWS||R>maxr||xfmt!=Q38_MODE1_X_BF16||layer>=Q38_LAYERS_WITH_MTP||mb->magic!=Q38_MR_MAGIC)why|=Q38_MR_REASON_CONFIG;
 if(!(v->cpu_mailbox_reserved&Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)||!v->cpu_mailbox_device_ptr)why|=Q38_MR_REASON_CONFIG;
 if(!v->batch_scratch||v->batch_scratch_bytes<Q38_BATCH_SCRATCH_PD)why|=Q38_MR_REASON_CONFIG;
 if(arole==1u){if(mrole!=Q38_MR_ROLE_LEADER||(R>1&&(!mb->ret1_va||!mb->slot1_stride)))why|=Q38_MR_REASON_CONFIG;}
 else if(arole==2u){if(mrole!=Q38_MR_ROLE_PEER||(R>1&&!mb->push1_va))why|=Q38_MR_REASON_CONFIG;}
 else why|=Q38_MR_REASON_CONFIG;
 for(unsigned r=0;r<R&&r<Q38_MR_MAX_ROWS;r++){
  const volatile int32_t *ir=reinterpret_cast<const volatile int32_t*>(reinterpret_cast<const char*>(c)+q38_mr_ids_off(r));
  const volatile float *rr=reinterpret_cast<const volatile float*>(reinterpret_cast<const char*>(c)+q38_mr_routes_off(r));
  for(int k=0;k<K;k++){const int e=ir[k];const float x=rr[k];if(e<0||e>=E||!(x==x)||x>3.0e38f||x< -3.0e38f)why|=Q38_MR_REASON_ROWS;}}
 return why;
}

// Mode 10, both rows in the same loops (2026-11-04): each warp/thread runs the row-0 and row-1 chains of the
// mode-1 gate/up and down side by side (independent loads overlap), with exactly mode 1's per-row operation order,
// so each row stays bit-identical to dynamic_gate_layer / dynamic_down_layer.
__device__ __forceinline__ void q38_mr_gate_up2(const DeviceWeights &w, const int32_t *ids0, const int32_t *ids1,
    const float *x0, const float *x1, float *act0, float *act1, int gwarp, int nwarps, int lane) {
 for (int task = gwarp; task < K * W; task += nwarps) {
  const int k = task / W, r = task % W;
  const size_t row0 = size_t(ids0[k]) * W + r, row1 = size_t(ids1[k]) * W + r;
  const uint4 *g0 = reinterpret_cast<const uint4 *>(w.g + row0 * (H / 4)), *u0 = reinterpret_cast<const uint4 *>(w.u + row0 * (H / 4));
  const uint4 *g1 = reinterpret_cast<const uint4 *>(w.g + row1 * (H / 4)), *u1 = reinterpret_cast<const uint4 *>(w.u + row1 * (H / 4));
  const float *gs0 = w.gs + row0 * (H / GROUP), *us0 = w.us + row0 * (H / GROUP);
  const float *gs1 = w.gs + row1 * (H / GROUP), *us1 = w.us + row1 * (H / GROUP);
  float a0 = 0.0f, b0 = 0.0f, a1 = 0.0f, b1 = 0.0f;
  for (int v = lane; v < H / 16; v += 32) {
   const int grp = (v * 16) / GROUP;
   const uint4 wg0 = __ldg(g0 + v), wu0 = __ldg(u0 + v), wg1 = __ldg(g1 + v), wu1 = __ldg(u1 + v);
   a0 = fmaf(__ldg(gs0 + grp), q38_fp8dot16(wg0, x0 + v * 16), a0);
   b0 = fmaf(__ldg(us0 + grp), q38_fp8dot16(wu0, x0 + v * 16), b0);
   a1 = fmaf(__ldg(gs1 + grp), q38_fp8dot16(wg1, x1 + v * 16), a1);
   b1 = fmaf(__ldg(us1 + grp), q38_fp8dot16(wu1, x1 + v * 16), b1);
  }
  a0 = q38_routed_warp_sum(a0); b0 = q38_routed_warp_sum(b0); a1 = q38_routed_warp_sum(a1); b1 = q38_routed_warp_sum(b1);
  if (!lane) { act0[task] = (a0 / (1.0f + expf(-a0))) * b0; act1[task] = (a1 / (1.0f + expf(-a1))) * b1; }
 }
}
// acts: 2*K*W floats in shared memory (row 0, then row 1), loaded by the caller.
__device__ __forceinline__ void q38_mr_down2(const DeviceWeights &w, const int32_t *ids0, const int32_t *ids1,
    const float *rt0, const float *rt1, const float *acts, float *out0, float *out1) {
 for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < H; row += gridDim.x * blockDim.x) {
  float s0 = 0, s1 = 0;
  for (int k = 0; k < K; k++) {
   const int e0 = ids0[k], e1 = ids1[k];
   float p0 = 0, p1 = 0;
#pragma unroll
   for (int q = 0; q < W / 4; q++) {
    const uint32_t v0 = w.d[(size_t(e0) * (W / 4) + q) * H + row], v1 = w.d[(size_t(e1) * (W / 4) + q) * H + row];
#pragma unroll
    for (int j = 0; j < 4; j++) {
     p0 = fmaf(q38_fp8_fast(v0, j), acts[k * W + q * 4 + j], p0);                 // = fp8e4m3, faster for 2+ rows
     p1 = fmaf(q38_fp8_fast(v1, j), acts[K * W + k * W + q * 4 + j], p1);
    }
   }
   s0 = fmaf(p0 * w.ds[size_t(e0) * H + row], rt0[k], s0);
   s1 = fmaf(p1 * w.ds[size_t(e1) * H + row], rt1[k], s1);
  }
  out0[row] = s0; out1[row] = s1;
 }
}

// q38_fp8dot16 with x as 8 words of packed BF16 pairs (element 2m in the low half): exact widening and the same
// fmaf order, so the result is bit-identical to q38_fp8dot16 on the widened row.
__device__ __forceinline__ float q38_fp8dot16_bfx(uint4 w, const uint32_t *xw) {
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint2 xv = reinterpret_cast<const uint2 *>(xw)[i];
        acc = fmaf(q38_fp8_fast(words[i], 0), __uint_as_float(xv.x << 16), acc);
        acc = fmaf(q38_fp8_fast(words[i], 1), __uint_as_float(xv.x & 0xFFFF0000u), acc);
        acc = fmaf(q38_fp8_fast(words[i], 2), __uint_as_float(xv.y << 16), acc);
        acc = fmaf(q38_fp8_fast(words[i], 3), __uint_as_float(xv.y & 0xFFFF0000u), acc);
    }
    return acc;
}
// Row pointers of a mode-10 request, computed from the row index (no pointer arrays, which would live in local
// memory): ids / routes in the control block, act / down output = the caller's buffers for row 0, the routed scratch
// for rows >= 1.
struct Q38MrRows {
 const char *cb; float *act0, *actb, *out0, *outb;
 __device__ __forceinline__ const int32_t *ids(unsigned i) const { return reinterpret_cast<const int32_t *>(cb + q38_mr_ids_off(i)); }
 __device__ __forceinline__ const float *rt(unsigned i) const { return reinterpret_cast<const float *>(cb + q38_mr_routes_off(i)); }
 __device__ __forceinline__ float *act(unsigned i) const { return i ? actb + size_t(i - 1) * K * W : act0; }
 __device__ __forceinline__ float *out(unsigned i) const { return i ? outb + size_t(i - 1) * H : out0; }
};
// Mode 10 with N = 3 or 4 rows: the 2-row structure (all rows of one (expert slot k, slice row r) task in one warp,
// independent load chains), x rows as BF16 words in shared memory (xw: N rows of H/2 words).
template <int N>
__device__ __forceinline__ void q38_mr_gate_upN(const DeviceWeights &w, const Q38MrRows &m, const uint32_t *xw,
    int gwarp, int nwarps, int lane) {
 for (int task = gwarp; task < K * W; task += nwarps) {
  const int k = task / W, r = task % W;
  size_t row[N];
#pragma unroll
  for (int i = 0; i < N; i++) row[i] = size_t(m.ids(i)[k]) * W + r;
  float a[N], b[N];
#pragma unroll
  for (int i = 0; i < N; i++) { a[i] = 0.0f; b[i] = 0.0f; }
  for (int v = lane; v < H / 16; v += 32) {
   const int grp = (v * 16) / GROUP;
#pragma unroll
   for (int i = 0; i < N; i++) {
    const uint4 wg = __ldg(reinterpret_cast<const uint4 *>(w.g + row[i] * (H / 4)) + v);
    const uint4 wu = __ldg(reinterpret_cast<const uint4 *>(w.u + row[i] * (H / 4)) + v);
    a[i] = fmaf(__ldg(w.gs + row[i] * (H / GROUP) + grp), q38_fp8dot16_bfx(wg, xw + i * (H / 2) + v * 8), a[i]);
    b[i] = fmaf(__ldg(w.us + row[i] * (H / GROUP) + grp), q38_fp8dot16_bfx(wu, xw + i * (H / 2) + v * 8), b[i]);
   }
  }
#pragma unroll
  for (int i = 0; i < N; i++) {
   a[i] = q38_routed_warp_sum(a[i]); b[i] = q38_routed_warp_sum(b[i]);
   if (!lane) m.act(i)[task] = (a[i] / (1.0f + expf(-a[i]))) * b[i];
  }
 }
}
// acts: N*K*W floats in shared memory (row i at i*K*W).
template <int N>
__device__ __forceinline__ void q38_mr_downN(const DeviceWeights &w, const Q38MrRows &m, const float *acts) {
 for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < H; row += gridDim.x * blockDim.x) {
  float s[N];
#pragma unroll
  for (int i = 0; i < N; i++) s[i] = 0.0f;
  for (int k = 0; k < K; k++) {
   float p[N]; int e[N];
#pragma unroll
   for (int i = 0; i < N; i++) { e[i] = m.ids(i)[k]; p[i] = 0.0f; }
#pragma unroll
   for (int q = 0; q < W / 4; q++) {
#pragma unroll
    for (int i = 0; i < N; i++) {
     const uint32_t v = w.d[(size_t(e[i]) * (W / 4) + q) * H + row];
#pragma unroll
     for (int j = 0; j < 4; j++) p[i] = fmaf(q38_fp8_fast(v, j), acts[i * K * W + k * W + q * 4 + j], p[i]);   // = fp8e4m3, faster
    }
   }
#pragma unroll
   for (int i = 0; i < N; i++) s[i] = fmaf(p[i] * w.ds[size_t(e[i]) * H + row], m.rt(i)[k], s[i]);
  }
#pragma unroll
  for (int i = 0; i < N; i++) m.out(i)[row] = s[i];
 }
}

// 3-4 row gate/up (stage + compute) in its own non-inlined function (keeps the register use of the 1-2 row path
// and of the kernel's call site unchanged).
__device__ __noinline__ void q38_mr_gate_up_rows(const DeviceWeights lw, const Q38MrRows m, unsigned R, float *xs) {
 const int lane=threadIdx.x&31,gwarp=int(blockIdx.x)*int(blockDim.x>>5)+int(threadIdx.x>>5),nwarps=int(gridDim.x)*int(blockDim.x>>5);
 static_assert(4*(H/2)<=2*H,"four BF16 rows exceed the shared x buffer");
 uint32_t *xw=reinterpret_cast<uint32_t*>(xs);
 for(unsigned r=0;r<R;r++){const uint32_t *src=reinterpret_cast<const uint32_t*>(m.cb+q38_mr_x_off(r));for(int i=threadIdx.x;i<H/2;i+=blockDim.x)xw[r*(H/2)+i]=src[i];}
 __syncthreads();
 if(R==3)q38_mr_gate_upN<3>(lw,m,xw,gwarp,nwarps,lane);else q38_mr_gate_upN<4>(lw,m,xw,gwarp,nwarps,lane);
}
__device__ __noinline__ void q38_mr_down_rows(const DeviceWeights lw, const Q38MrRows m, unsigned R, const float *acts) {
 if(R==3)q38_mr_downN<3>(lw,m,acts);else q38_mr_downN<4>(lw,m,acts);
}

// Mode 10 body (all threads of the grid, after the request checks passed). Returns false after an error exit
// (reported to the RTX slot like the other modes); gen is the kernel's grid-barrier generation.
__device__ __noinline__ bool q38_mr_routed(const DeviceWeights* layers, ResidentDynamicControl* control, ResidentDynamicGrid* grid,
    float* activation, float* result_local, ResidentDynamicReturn* output, ResidentDirectResult* direct, unsigned seq, unsigned layer,
    unsigned &gen, unsigned long long t_seen, unsigned long long t_chk) {
   unsigned long long t_gu=0,t_dn=0;
   // Multi-row decode MoE (resident_multirow_abi.cuh): R = 1..4 BF16 rows; per row mode 1's gate/up, down and card
   // sum (same order), so each row is bit-identical to a mode-1 call. Rows >= 1: act / down output in the routed
   // scratch; results to the row-r return records (leader) or the packed row push window (peer).
#define Q38_MR_FAIL { if(!blockIdx.x&&!threadIdx.x){unsigned why=atomicAdd(&grid->error,0u);output->status=1;output->exit_reason=why;direct->status=1;direct->exit_reason=why;__threadfence_system();direct->seq=seq;output->seq=seq;pci_cluster_slot_error(pci_cluster_rtx_slot(control),seq,why);} return false; }
   const unsigned R=control->ple_count;
   const DeviceWeights lw=layers[layer];
   const char *cb=reinterpret_cast<const char*>(control);
   const Q38MrRows m{cb,activation,reinterpret_cast<float*>(control->batch_scratch+Q38_BATCH_SCRATCH_ACT),
                     result_local,reinterpret_cast<float*>(control->batch_scratch+Q38_BATCH_SCRATCH_X)};
   {float *xs=q38_routed_xbuf();
    const int lane=threadIdx.x&31,gwarp=int(blockIdx.x)*int(blockDim.x>>5)+int(threadIdx.x>>5),nwarps=int(gridDim.x)*int(blockDim.x>>5);
    if(R<=2){
     const uint32_t *xw0=reinterpret_cast<const uint32_t*>(cb+q38_mr_x_off(0)),*xw1=reinterpret_cast<const uint32_t*>(cb+q38_mr_x_off(1));
     for(int i=threadIdx.x;i<H/2;i+=blockDim.x){const uint32_t w=xw0[i];reinterpret_cast<float2*>(xs)[i]=make_float2(__uint_as_float(w<<16),__uint_as_float(w&0xFFFF0000u));}
     if(R>1)for(int i=threadIdx.x;i<H/2;i+=blockDim.x){const uint32_t w=xw1[i];reinterpret_cast<float2*>(xs+H)[i]=make_float2(__uint_as_float(w<<16),__uint_as_float(w&0xFFFF0000u));}
     __syncthreads();
     if(R>1)q38_mr_gate_up2(lw,m.ids(0),m.ids(1),xs,xs+H,m.act(0),m.act(1),gwarp,nwarps,lane);
     else q38_routed_v2_gate_up(lw.g,lw.u,lw.gs,lw.us,m.ids(0),xs,m.act(0),gwarp,nwarps,lane,H,W,K,GROUP);
    } else q38_mr_gate_up_rows(lw,m,R,xs);   // 3-4 rows: BF16 words in shared (4 rows x H/2 words = the 2*H-float buffer)
   }
   if(!dynamic_grid_barrier(grid,++gen)) Q38_MR_FAIL
   if(!blockIdx.x&&!threadIdx.x)t_gu=pci_cluster_gtimer();
   if(R>1){float *acts=q38_routed_xbuf();   // x is no longer needed: every row's activations in shared memory
    for(unsigned r=0;r<R;r++){const float *a=m.act(r);for(int i=threadIdx.x;i<K*W;i+=blockDim.x)acts[r*K*W+i]=a[i];}
    __syncthreads();
    if(R==2)q38_mr_down2(lw,m.ids(0),m.ids(1),m.rt(0),m.rt(1),acts,m.out(0),m.out(1));
    else q38_mr_down_rows(lw,m,R,acts);}
   else dynamic_down_layer(Params{lw,control->x,m.rt(0),m.ids(0),m.act(0),m.out(0)});
   if(!dynamic_grid_barrier(grid,++gen)) Q38_MR_FAIL
   if(!blockIdx.x&&!threadIdx.x)t_dn=pci_cluster_gtimer();
   const volatile Q38AggBlock *ag=reinterpret_cast<const volatile Q38AggBlock*>(cb+Q38_AGG_OFF);
   const volatile Q38MrBlock *mb=reinterpret_cast<const volatile Q38MrBlock*>(cb+Q38_MR_BLOCK_OFF);
   const unsigned arole=ag->role, np=arole==1u?ag->n_peers:0u;   // validated: role 1 or 2 with matching MR block
   const char *sl0=cb+ag->slot_offset; const unsigned st0=ag->slot_stride;
   const int step=static_cast<int>(mb->slot1_stride);              // reducer record step (+/- 0x3000)
   auto *cpu_slot=reinterpret_cast<Q38CpuMailboxSlot*>(control->cpu_mailbox_device_ptr);
   const unsigned long long ret1=mb->ret1_va;
   volatile float *push0=arole==2u?reinterpret_cast<volatile float*>(ag->push_va):nullptr;
   char *push1=arole==2u?reinterpret_cast<char*>(mb->push1_va):nullptr;
   if(arole==1u){
    if(!threadIdx.x){const unsigned long long t0=clock64();
     for(unsigned k=0;k<np;k++){
      auto *f0=reinterpret_cast<const volatile unsigned*>(sl0+k*st0+H*4u);
      while(*f0!=seq){if(clock64()-t0>Q38_AGG_DEADLINE){atomicOr(&grid->error,64u);break;}}
      for(unsigned r=1;r<R;r++){auto *f=reinterpret_cast<const volatile unsigned*>(cb+Q38_MR_PEER_BASE(k)+(r-1)*Q38_MR_PUSH_ROW_BYTES+H*4u);
       while(*f!=seq){if(clock64()-t0>Q38_AGG_DEADLINE){atomicOr(&grid->error,64u);break;}}}}}
    __syncthreads();__threadfence();
   }
   for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<H;i+=gridDim.x*blockDim.x) {
    float v0=m.out(0)[i];
    if(arole==1u)for(unsigned k=0;k<np;k++)v0+=reinterpret_cast<const volatile float*>(sl0+k*st0)[i];
    direct->result[i]=v0;
    if(arole==2u)push0[i]=v0; else cpu_slot->result[i]=v0;
    for(unsigned r=1;r<R;r++){float v=m.out(r)[i];
     if(arole==1u)for(unsigned k=0;k<np;k++)v+=reinterpret_cast<const volatile float*>(cb+Q38_MR_PEER_BASE(k)+(r-1)*Q38_MR_PUSH_ROW_BYTES)[i];
     if(arole==2u)reinterpret_cast<volatile float*>(push1+(r-1)*Q38_MR_PUSH_ROW_BYTES)[i]=v;
     else reinterpret_cast<Q38CpuMailboxSlot*>(ret1+(long long)(r-1)*step)->result[i]=v;}
   }
   __threadfence_system(); if(!dynamic_grid_barrier(grid,++gen)) Q38_MR_FAIL
   if(!blockIdx.x&&!threadIdx.x){
    pci_cluster_wtiming_write(control,seq,layer,arole,7u,t_seen,t_chk,t_gu,t_dn,R);
    output->status=0;direct->status=0;
    if(arole==2u){for(unsigned r=R-1;r>=1;r--)reinterpret_cast<volatile float*>(push1+(r-1)*Q38_MR_PUSH_ROW_BYTES)[H]=__uint_as_float(seq);
     push0[H]=__uint_as_float(seq);}
    if(arole==1u)for(unsigned r=1;r<R;r++){auto *rec=reinterpret_cast<Q38CpuMailboxSlot*>(ret1+(long long)(r-1)*step);
     rec->gate=0.0f;rec->worker=0;rec->layer=1;rec->status=0;__threadfence_system();rec->seq=seq;}
    cpu_slot->worker=0;cpu_slot->layer=1;cpu_slot->status=0;
    __threadfence_system();cpu_slot->seq=seq;output->seq=seq;direct->seq=seq;}
#undef Q38_MR_FAIL
   return true;
}

// LM-head top-k for two BF16 rows (2026-11-04): one slice read, row 0 selected on block 0, row 1 on block 1.
__device__ __noinline__ bool q38_lm2_routed(ResidentDynamicControl* control, ResidentDynamicGrid* grid, const uint8_t* ple_table,
    ResidentDynamicReturn* output, ResidentDirectResult* direct, unsigned seq, unsigned &gen) {
 // LM-head top-k for N = 2..4 BF16 rows (2026-11-04): one slice read, row r selected on block r.
 float *xb=q38_routed_xbuf();
 const unsigned n=control->reserved;
 if(n==2)pci_cluster_lm_matvec2(control,ple_table,xb,xb+H);
 else if(n==3)pci_cluster_lm_matvecN<3>(control,ple_table,reinterpret_cast<uint32_t*>(xb),reinterpret_cast<uint32_t*>(xb+H));
 else pci_cluster_lm_matvecN<4>(control,ple_table,reinterpret_cast<uint32_t*>(xb),reinterpret_cast<uint32_t*>(xb+H));
 if(!dynamic_grid_barrier(grid,++gen))return false;
 const unsigned rows=static_cast<unsigned>(control->ple_offsets[2*control->cpu_mailbox_worker+1]);
 const unsigned k=min(static_cast<unsigned>(control->ple_count),rows);
 float lse=0.0f;
 if(blockIdx.x==0)lse=pci_cluster_lm_select(control,ple_table,direct);
 if(blockIdx.x>=1&&blockIdx.x<n){const unsigned r=blockIdx.x;
  const float l=pci_cluster_lm_select(control,ple_table,direct,reinterpret_cast<const unsigned*>(control->batch_scratch)+size_t(r-1)*rows,r*(k+1u));
  if(!threadIdx.x){reinterpret_cast<volatile float*>(direct->result)[PCI_CLUSTER_LM_LSE1_WORD-(r-1)]=l;Q38CpuMailboxSlot*s=pci_cluster_rtx_slot(control);
   if(s)reinterpret_cast<volatile float*>(s->result)[PCI_CLUSTER_LM_LSE1_WORD-(r-1)]=l;__threadfence_system();}}
 if(!dynamic_grid_barrier(grid,++gen))return false;
 if(!blockIdx.x&&!threadIdx.x){direct->gate=lse;output->status=0;direct->status=0;__threadfence_system();output->seq=seq;direct->seq=seq;pci_cluster_slot_ack(pci_cluster_rtx_slot(control),seq,lse);}
 return true;
}

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
   // mode 10 (2026-11-04): the row-1 return record / row-1 push window may be remapped from the next pages
   // (own slot page 3 and the next two slots, inside the registered mailbox region); reads only.
   for(unsigned byte=0x3000;byte<0xC000;byte+=4096)observed^=cpu_slot[byte/sizeof(unsigned)];
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
  if(mode!=1&&mode!=2&&mode!=3&&mode!=4&&mode!=5&&mode!=PCI_CLUSTER_MODE_ROW_GATHER&&mode!=PCI_CLUSTER_MODE_LM_TOPK&&mode!=Q38_BATCH_MODE&&mode!=Q38_BATCH_PIPE_MODE&&mode!=Q38_MR_MODE)reason|=2u;
  if(mode==Q38_MR_MODE&&!threadIdx.x)reason|=q38_mr_validate_routed(control,layer,xfmt);
  if(mode==1&&layer>=Q38_LAYERS_WITH_MTP)reason|=4u;
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
  if(mode==PCI_CLUSTER_MODE_LM_TOPK&&control->reserved>=PCI_CLUSTER_LM_FMT_BF16X2) {
   if(!q38_lm2_routed(control,grid,ple_table,output,direct,seq,gen)) return;   // two rows (non-inlined)
   last=seq;continue;
  }
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
  if(mode==Q38_MR_MODE) {
   // Multi-row decode MoE (resident_multirow_abi.cuh), in its own non-inlined function so its registers do not
   // change the allocation of the mode-1 / mode-8 paths of this persistent kernel.
   if(!q38_mr_routed(layers,control,grid,activation,result_local,output,direct,seq,layer,gen,q38_t_seen,q38_t_chk)) return;
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
