// pci_cluster_exchange.cuh - worker (M10) side of the PCI-cluster exchange.
//
// The exchange is one pattern for every RTX -> M10 request:
//   1. the RTX broadcasts ONE request to all 24 workers (PEX hardware multicast, identical bytes on
//      every worker of a card: header {seq, layer, mode, count} + op payload),
//   2. every worker runs the op on its own resident data and writes its part into its direct
//      result record (ResidentDirectResult, read by the RTX through the return route),
//   3. the RTX waits for all 24 acknowledgements and combines the parts (qwen38_vllm_rtx_peer.cu,
//      pci_cluster_* kernels).
// Ops: mode 1 = MoE (each worker computes its experts; RTX sums) - in the worker kernels.
//      mode 6 = row gather (this file; each worker copies the rows it owns; RTX picks them up).
//
// Mode 6 (row gather), request in multicast page 0 of the control block:
//   ple_count            n rows, n <= PCI_CLUSTER_ROW_GATHER_MAX_ROWS
//   ple_offsets[r]       (owner << 40) | byte offset of row r inside the owner's table
//                        owner = peer worker index (0..23), resolved on the RTX from the shard manifest
// Per-worker configuration, written once per worker (unicast) by the RTX peer open, never multicast:
//   cpu_mailbox_worker   this worker's peer index (already set by the peer open)
//   cpu_mailbox_reserved must carry Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN (peer return route active)
//   ple_reserved[1..2]   size in bytes of this worker's table (low, high word); 0 = not configured
// Result: the worker's own rows, compacted in request order, in direct->result (40 words per row);
//   direct->gate carries the owned-row count as raw bits, so the RTX can cross-check it.
//   The RTX does NOT read `direct`: during a peer session cpu_mailbox_device_ptr points at a VA whose
//   PTE the peer open remapped onto this worker's RTX return slot (Q38CpuMailboxSlot). Rows, count and
//   the acknowledgement therefore also go to that slot, with the MoE header convention for the RTX
//   path: worker = 0, layer = 1, status = 0, then seq (2026-11-03 r18: acking only `direct` timed out).
// Every check runs before the grid barrier, i.e. before any table read; a failed check takes the
// kernel's normal error exit (status 1, exit_reason bits PCI_CLUSTER_REASON_*).
#pragma once
#include <stdint.h>
#include "resident_direct_mailbox.cuh"
#include "resident_cpu_mailbox_abi.cuh"

#define PCI_CLUSTER_MODE_ROW_GATHER 6u
#define PCI_CLUSTER_ROW_BYTES 160u
#define PCI_CLUSTER_ROW_WORDS (PCI_CLUSTER_ROW_BYTES / 4u)
// ple_offsets starts at byte 32; only the first 4-KiB page of the control block is multicast-aliased.
#define PCI_CLUSTER_ROW_GATHER_MAX_ROWS ((4096u - 32u) / 8u)
// One direct result record holds 10240 payload bytes = 64 rows of 160 bytes.
#define PCI_CLUSTER_ROWS_PER_WORKER 64u
#define PCI_CLUSTER_OWNER_SHIFT 40
#define PCI_CLUSTER_OFFSET_MASK ((1ull << PCI_CLUSTER_OWNER_SHIFT) - 1ull)
#define PCI_CLUSTER_REASON_ROW_GATHER_CONFIG 512u   // no table / size / return flag, or too many rows
#define PCI_CLUSTER_REASON_ROW_GATHER_ROW 1024u     // an owned row is out of bounds, misaligned, or > 64 owned
static_assert(PCI_CLUSTER_ROWS_PER_WORKER * PCI_CLUSTER_ROW_BYTES <= sizeof(float) * 2560u,
              "row-gather capacity exceeds the direct result payload");
static_assert(PCI_CLUSTER_ROW_BYTES % 4u == 0u, "rows are copied as 32-bit words");
static_assert(PCI_CLUSTER_ROWS_PER_WORKER * PCI_CLUSTER_ROW_BYTES <= sizeof(float) * Q38_CPU_MAILBOX_HIDDEN,
              "row-gather capacity exceeds the RTX return slot payload");

__host__ __device__ __forceinline__ unsigned long long pci_cluster_row_desc(unsigned owner, unsigned long long offset) {
    return (static_cast<unsigned long long>(owner) << PCI_CLUSTER_OWNER_SHIFT) | (offset & PCI_CLUSTER_OFFSET_MASK);
}

template <class Control>
__device__ __forceinline__ unsigned long long pci_cluster_table_bytes(const Control *c) {
    const volatile unsigned *r = c->ple_reserved;
    return static_cast<unsigned long long>(r[1]) | (static_cast<unsigned long long>(r[2]) << 32);
}

// One thread, before the grid barrier. Returns error reason bits (0 = request is safe to execute).
template <class Control>
__device__ unsigned pci_cluster_row_gather_validate(const Control *c, const uint8_t *table) {
    const volatile Control *v = c;
    const unsigned n = v->ple_count;
    const unsigned long long bytes = pci_cluster_table_bytes(c);
    if (!table || !bytes || n > PCI_CLUSTER_ROW_GATHER_MAX_ROWS ||
        !(v->cpu_mailbox_reserved & Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN))
        return PCI_CLUSTER_REASON_ROW_GATHER_CONFIG;
    const unsigned me = v->cpu_mailbox_worker;
    unsigned owned = 0;
    for (unsigned r = 0; r < n; r++) {
        const unsigned long long e = v->ple_offsets[r];
        if (static_cast<unsigned>(e >> PCI_CLUSTER_OWNER_SHIFT) != me) continue;
        const unsigned long long off = e & PCI_CLUSTER_OFFSET_MASK;
        if ((off & 3ull) || off > bytes || bytes - off < PCI_CLUSTER_ROW_BYTES ||
            ++owned > PCI_CLUSTER_ROWS_PER_WORKER)
            return PCI_CLUSTER_REASON_ROW_GATHER_ROW;
    }
    return 0;
}

// The RTX return slot of this worker, or nullptr outside a peer session.
template <class Control>
__device__ __forceinline__ Q38CpuMailboxSlot *pci_cluster_rtx_slot(const Control *c) {
    const volatile Control *v = c;
    if (!(v->cpu_mailbox_reserved & Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN)) return nullptr;
    return reinterpret_cast<Q38CpuMailboxSlot *>(v->cpu_mailbox_device_ptr);
}
// One thread, after all payload stores: acknowledge `seq` to the RTX slot (MoE header convention).
__device__ __forceinline__ void pci_cluster_slot_ack(Q38CpuMailboxSlot *slot, unsigned seq, float gate) {
    if (!slot) return;
    slot->gate = gate; slot->worker = 0; slot->layer = 1; slot->status = 0;
    __threadfence_system();
    slot->seq = seq;
}
// One thread, on a worker error exit while a peer session is active: report it to the RTX at once
// (status 1, exit-reason bits in the `worker` word) instead of letting the RTX wait time out.
__device__ __forceinline__ void pci_cluster_slot_error(Q38CpuMailboxSlot *slot, unsigned seq, unsigned why) {
    if (!slot) return;
    slot->worker = why; slot->layer = 0; slot->status = 1;
    __threadfence_system();
    slot->seq = seq;
}

// All threads of the grid, after a successful barrier. Copies this worker's rows, compacted in
// request order, into direct->result and (peer session) the RTX return slot; returns the owned-row
// count (identical in every block).
template <class Control>
__device__ unsigned pci_cluster_row_gather(const Control *c, ResidentDirectResult *direct, const uint8_t *table) {
    __shared__ unsigned long long offs[PCI_CLUSTER_ROWS_PER_WORKER];
    __shared__ unsigned owned;
    if (!threadIdx.x) {
        const volatile Control *v = c;
        const unsigned n = v->ple_count, me = v->cpu_mailbox_worker;
        unsigned k = 0;
        for (unsigned r = 0; r < n && k < PCI_CLUSTER_ROWS_PER_WORKER; r++) {
            const unsigned long long e = v->ple_offsets[r];
            if (static_cast<unsigned>(e >> PCI_CLUSTER_OWNER_SHIFT) == me) offs[k++] = e & PCI_CLUSTER_OFFSET_MASK;
        }
        owned = k;
    }
    __syncthreads();
    const unsigned k = owned;
    auto *dst = reinterpret_cast<volatile uint32_t *>(direct->result);
    Q38CpuMailboxSlot *slot = pci_cluster_rtx_slot(c);
    auto *rtx = slot ? reinterpret_cast<volatile uint32_t *>(slot->result) : nullptr;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < k * PCI_CLUSTER_ROW_WORDS; i += gridDim.x * blockDim.x) {
        const unsigned row = i / PCI_CLUSTER_ROW_WORDS, word = i % PCI_CLUSTER_ROW_WORDS;
        const uint32_t value = reinterpret_cast<const uint32_t *>(table + offs[row])[word];
        dst[i] = value;
        if (rtx) rtx[i] = value;
    }
    __threadfence_system();
    return k;
}

// ---------------------------------------------------------------------------------------------
// Mode 7 (LM-head top-k, 2026-11-03). Every worker holds one row slice of lm_head.weight (BF16) at the
// END of its PLE allocation (PLE owner, PLE_LM_HEAD_FILE): [rows x 5120 B weights][rows x 4 B scratch].
// Request (broadcast): x = control->x (FP32, exact copy of the BF16 hidden state), ple_count = K,
//   ple_offsets[2w] = byte offset of worker w's slice, ple_offsets[2w+1] = its row count.
// Worker: logits = W_slice . x with FP32 accumulation, rounded to BF16 (as vLLM's BF16 LM head), then an
//   exact top-K (2-pass radix select on 16-bit order keys; ties at the threshold taken in arrival order)
//   and the FP32 log-sum-exp over the whole slice.
// Result (slot and direct, 32-bit words): [0] = count (= min(K, rows)), [1..count] = (bf16 bits << 16) |
//   local row index; gate = log-sum-exp. The RTX scatters the candidates into a -inf logits row at
//   row_start + index and combines the 24 log-sum-exps into the exact softmax normalizer.
// Checks before the barrier: K in [1, 2559]; slice offset 16-B aligned; rows in [1, 16384]; and the slice
//   plus scratch end EXACTLY at the table end (so the scratch writes cannot reach PLE rows).
#define PCI_CLUSTER_MODE_LM_TOPK 7u
#define PCI_CLUSTER_LM_HIDDEN 2560u
#define PCI_CLUSTER_LM_ROW_BYTES (PCI_CLUSTER_LM_HIDDEN * 2u)
#define PCI_CLUSTER_LM_ROW_VEC (PCI_CLUSTER_LM_ROW_BYTES / 16u)
#define PCI_CLUSTER_LM_MAX_ROWS 16384u
#define PCI_CLUSTER_LM_MAX_K 2559u
#define PCI_CLUSTER_REASON_LM_CONFIG 2048u   // no table / size / return flag / bad K / bad worker index
#define PCI_CLUSTER_REASON_LM_SLICE 4096u    // slice misaligned, bad row count, or not ending at the table end
// Two rows per request (2026-11-04, MTP verify): control->reserved == PCI_CLUSTER_LM_FMT_BF16X2 means x holds
// two BF16 rows (row 0 at x, row 1 at PCI_CLUSTER_LM_X1_OFF, as the multi-row MoE mode 10). The slice is read
// once for both; row 1's logits go to the worker's batch scratch; row r's result starts at word r*(K+1)
// (count, then the candidates); row 1's log-sum-exp is the float in result word PCI_CLUSTER_LM_LSE1_WORD.
// reserved == 0: one FP32 row (the original layout).
#define PCI_CLUSTER_LM_FMT_BF16X2 2u
#define PCI_CLUSTER_LM_X1_OFF 0x9470u
#define PCI_CLUSTER_LM_LSE1_WORD 2559u
// N rows (2026-11-04 evening): reserved = N in 2..4; row r's x at the multi-row MoE offsets (0x8070, 0x9470, 0xC000,
// 0xD400), logits of row r >= 1 in the batch scratch at (r-1) * rows * 4, result of row r at word r*(K+1), LSE of
// row r >= 1 in word PCI_CLUSTER_LM_LSE1_WORD - (r-1). Needs N*(K+1) <= 2560 - N.
#define PCI_CLUSTER_LM_MAX_NROWS 4u
__host__ __device__ constexpr unsigned pci_cluster_lm_x_off(unsigned r) { return r == 0 ? 0x8070u : r == 1 ? 0x9470u : r == 2 ? 0xC000u : 0xD400u; }

__device__ __forceinline__ unsigned pci_cluster_bf16_bits(float v) {        // round to nearest even
    unsigned u = __float_as_uint(v);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return 0x7fc0u; // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
__device__ __forceinline__ unsigned pci_cluster_order_key(unsigned b) {      // ascending in value
    return (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
}

template <class Control>
__device__ unsigned pci_cluster_lm_validate(const Control *c, const uint8_t *table) {
    const volatile Control *v = c;
    const unsigned k = v->ple_count, me = v->cpu_mailbox_worker;
    const unsigned long long bytes = pci_cluster_table_bytes(c);
    if (!table || !bytes || k < 1 || k > PCI_CLUSTER_LM_MAX_K || me >= 24 ||
        !(v->cpu_mailbox_reserved & Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN))
        return PCI_CLUSTER_REASON_LM_CONFIG;
    const unsigned long long off = v->ple_offsets[2 * me], rows = v->ple_offsets[2 * me + 1];
    const unsigned n = v->reserved;
    if (n == 1 || n > PCI_CLUSTER_LM_MAX_NROWS) return PCI_CLUSTER_REASON_LM_CONFIG;
    if (n >= 2 && (n * (k + 1u) > 2560u - n || !v->batch_scratch || v->batch_scratch_bytes < (n - 1ull) * rows * 4ull))
        return PCI_CLUSTER_REASON_LM_CONFIG;
    if ((off & 15ull) || rows < 1 || rows > PCI_CLUSTER_LM_MAX_ROWS || off > bytes ||
        bytes - off != rows * (PCI_CLUSTER_LM_ROW_BYTES + 4ull))
        return PCI_CLUSTER_REASON_LM_SLICE;
    return 0;
}

// All threads of the grid: one warp per row, logits rounded to BF16 into the scratch words.
// xs: PCI_CLUSTER_LM_HIDDEN floats of the caller's shared memory (the shared kernel lends its q38_xs,
// 2026-11-03, to stay within 2 blocks per SM); the 2-argument form declares its own.
template <class Control>
__device__ void pci_cluster_lm_matvec(const Control *c, const uint8_t *table, float *xs);
template <class Control>
__device__ void pci_cluster_lm_matvec(const Control *c, const uint8_t *table) {
    __shared__ float xs[PCI_CLUSTER_LM_HIDDEN];
    pci_cluster_lm_matvec(c, table, xs);
}
template <class Control>
__device__ void pci_cluster_lm_matvec(const Control *c, const uint8_t *table, float *xs) {
    const volatile Control *v = c;
    const unsigned me = v->cpu_mailbox_worker;
    const unsigned long long off = v->ple_offsets[2 * me];
    const unsigned rows = static_cast<unsigned>(v->ple_offsets[2 * me + 1]);
    for (unsigned i = threadIdx.x; i < PCI_CLUSTER_LM_HIDDEN; i += blockDim.x) xs[i] = v->x[i];
    __syncthreads();
    const uint4 *w = reinterpret_cast<const uint4 *>(table + off);
    unsigned *scratch = reinterpret_cast<unsigned *>(const_cast<uint8_t *>(table) + off + size_t(rows) * PCI_CLUSTER_LM_ROW_BYTES);
    const unsigned lane = threadIdx.x & 31u, warps_per_block = blockDim.x >> 5;
    const unsigned nwarps = gridDim.x * warps_per_block;
    for (unsigned r = blockIdx.x * warps_per_block + (threadIdx.x >> 5); r < rows; r += nwarps) {
        const uint4 *row = w + size_t(r) * PCI_CLUSTER_LM_ROW_VEC;
        float acc = 0.0f;
        for (unsigned j = lane; j < PCI_CLUSTER_LM_ROW_VEC; j += 32u) {
            const uint4 q = row[j];
            const float *xj = xs + j * 8u;
            const unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int t = 0; t < 4; t++) {
                acc = fmaf(__uint_as_float(words[t] << 16), xj[2 * t], acc);
                acc = fmaf(__uint_as_float(words[t] & 0xffff0000u), xj[2 * t + 1], acc);
            }
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
        if (!lane) scratch[r] = pci_cluster_bf16_bits(acc);
    }
    __threadfence();
}

// Block 0 only, after a grid barrier: exact top-K + log-sum-exp; writes the result to the RTX slot and
// direct; returns the log-sum-exp (valid in thread 0).
// Two-row matvec (all threads of the grid): BF16 x rows into xs0 / xs1 (exact widening), one weight read
// per table row for both; per row the same fmaf order as pci_cluster_lm_matvec.
template <class Control>
__device__ void pci_cluster_lm_matvec2(const Control *c, const uint8_t *table, float *xs0, float *xs1) {
    const volatile Control *v = c;
    const unsigned me = v->cpu_mailbox_worker;
    const unsigned long long off = v->ple_offsets[2 * me];
    const unsigned rows = static_cast<unsigned>(v->ple_offsets[2 * me + 1]);
    const volatile uint32_t *x0 = reinterpret_cast<const volatile uint32_t *>(v->x);
    const volatile uint32_t *x1 = reinterpret_cast<const volatile uint32_t *>(reinterpret_cast<const volatile char *>(c) + PCI_CLUSTER_LM_X1_OFF);
    for (unsigned i = threadIdx.x; i < PCI_CLUSTER_LM_HIDDEN / 2; i += blockDim.x) {
        const uint32_t a = x0[i], b = x1[i];
        xs0[2 * i] = __uint_as_float(a << 16); xs0[2 * i + 1] = __uint_as_float(a & 0xffff0000u);
        xs1[2 * i] = __uint_as_float(b << 16); xs1[2 * i + 1] = __uint_as_float(b & 0xffff0000u);
    }
    __syncthreads();
    const uint4 *w = reinterpret_cast<const uint4 *>(table + off);
    unsigned *scratch0 = reinterpret_cast<unsigned *>(const_cast<uint8_t *>(table) + off + size_t(rows) * PCI_CLUSTER_LM_ROW_BYTES);
    unsigned *scratch1 = reinterpret_cast<unsigned *>(v->batch_scratch);
    const unsigned lane = threadIdx.x & 31u, warps_per_block = blockDim.x >> 5;
    const unsigned nwarps = gridDim.x * warps_per_block;
    for (unsigned r = blockIdx.x * warps_per_block + (threadIdx.x >> 5); r < rows; r += nwarps) {
        const uint4 *row = w + size_t(r) * PCI_CLUSTER_LM_ROW_VEC;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (unsigned j = lane; j < PCI_CLUSTER_LM_ROW_VEC; j += 32u) {
            const uint4 q = row[j];
            const float *xa = xs0 + j * 8u, *xb = xs1 + j * 8u;
            const unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int t = 0; t < 4; t++) {
                const float wl = __uint_as_float(words[t] << 16), wh = __uint_as_float(words[t] & 0xffff0000u);
                acc0 = fmaf(wl, xa[2 * t], acc0); acc0 = fmaf(wh, xa[2 * t + 1], acc0);
                acc1 = fmaf(wl, xb[2 * t], acc1); acc1 = fmaf(wh, xb[2 * t + 1], acc1);
            }
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) { acc0 += __shfl_down_sync(0xffffffffu, acc0, s); acc1 += __shfl_down_sync(0xffffffffu, acc1, s); }
        if (!lane) { scratch0[r] = pci_cluster_bf16_bits(acc0); scratch1[r] = pci_cluster_bf16_bits(acc1); }
    }
    __threadfence();
}

// 3-4 rows: x rows as BF16 words, rows 0-1 in xsA (2 x H/2 words), rows 2-3 in xsB; exact widening in the
// inner loop, the same fmaf order per row as pci_cluster_lm_matvec.
template <int N, class Control>
__device__ void pci_cluster_lm_matvecN(const Control *c, const uint8_t *table, uint32_t *xsA, uint32_t *xsB) {
    const volatile Control *v = c;
    const unsigned me = v->cpu_mailbox_worker;
    const unsigned long long off = v->ple_offsets[2 * me];
    const unsigned rows = static_cast<unsigned>(v->ple_offsets[2 * me + 1]);
    for (int r = 0; r < N; r++) {
        const volatile uint32_t *src = reinterpret_cast<const volatile uint32_t *>(reinterpret_cast<const volatile char *>(c) + pci_cluster_lm_x_off(r));
        uint32_t *dst = (r < 2 ? xsA : xsB) + (r & 1) * (PCI_CLUSTER_LM_HIDDEN / 2);
        for (unsigned i = threadIdx.x; i < PCI_CLUSTER_LM_HIDDEN / 2; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    const uint4 *w = reinterpret_cast<const uint4 *>(table + off);
    unsigned *scratch0 = reinterpret_cast<unsigned *>(const_cast<uint8_t *>(table) + off + size_t(rows) * PCI_CLUSTER_LM_ROW_BYTES);
    unsigned *scratchb = reinterpret_cast<unsigned *>(v->batch_scratch);
    const unsigned lane = threadIdx.x & 31u, warps_per_block = blockDim.x >> 5;
    const unsigned nwarps = gridDim.x * warps_per_block;
    for (unsigned r = blockIdx.x * warps_per_block + (threadIdx.x >> 5); r < rows; r += nwarps) {
        const uint4 *row = w + size_t(r) * PCI_CLUSTER_LM_ROW_VEC;
        float acc[N];
#pragma unroll
        for (int i = 0; i < N; i++) acc[i] = 0.0f;
        for (unsigned j = lane; j < PCI_CLUSTER_LM_ROW_VEC; j += 32u) {
            const uint4 q = row[j];
            const unsigned words[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int i = 0; i < N; i++) {
                const uint32_t *xw = (i < 2 ? xsA : xsB) + (i & 1) * (PCI_CLUSTER_LM_HIDDEN / 2) + j * 4u;
#pragma unroll
                for (int t = 0; t < 4; t++) {
                    const uint32_t xv = xw[t];
                    acc[i] = fmaf(__uint_as_float(words[t] << 16), __uint_as_float(xv << 16), acc[i]);
                    acc[i] = fmaf(__uint_as_float(words[t] & 0xffff0000u), __uint_as_float(xv & 0xffff0000u), acc[i]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < N; i++) {
            for (int s2 = 16; s2 > 0; s2 >>= 1) acc[i] += __shfl_down_sync(0xffffffffu, acc[i], s2);
            if (!lane) (i ? scratchb + size_t(i - 1) * rows : scratch0)[r] = pci_cluster_bf16_bits(acc[i]);
        }
    }
    __threadfence();
}

// scratch_override / base (2-row requests): select from that logits scratch and write the result at word base.
template <class Control>
__device__ float pci_cluster_lm_select(const Control *c, const uint8_t *table, ResidentDirectResult *direct,
                                       const unsigned *scratch_override = nullptr, unsigned base = 0) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_thr, s_take_eq, s_cnt, s_eq;
    __shared__ float s_red[32];
    const volatile Control *v = c;
    const unsigned me = v->cpu_mailbox_worker;
    const unsigned long long off = v->ple_offsets[2 * me];
    const unsigned rows = static_cast<unsigned>(v->ple_offsets[2 * me + 1]);
    const unsigned k = min(v->ple_count, rows);
    const volatile unsigned *scratch = scratch_override ? scratch_override :
        reinterpret_cast<const volatile unsigned *>(table + off + size_t(rows) * PCI_CLUSTER_LM_ROW_BYTES);
    const unsigned tid = threadIdx.x, nt = blockDim.x;
    // pass 1: high byte of the order key
    for (unsigned i = tid; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    for (unsigned r = tid; r < rows; r += nt) atomicAdd(&hist[pci_cluster_order_key(scratch[r]) >> 8], 1u);
    __syncthreads();
    if (!tid) {
        unsigned acc = 0, b = 255;
        for (;; b--) { if (acc + hist[b] >= k || !b) break; acc += hist[b]; }
        s_thr = b; s_take_eq = k - acc;                     // keys with high byte > b: acc (< k)
    }
    __syncthreads();
    const unsigned hi = s_thr, need = s_take_eq;
    // pass 2: low byte among keys with that high byte
    for (unsigned i = tid; i < 256; i += nt) hist[i] = 0;
    __syncthreads();
    for (unsigned r = tid; r < rows; r += nt) { const unsigned key = pci_cluster_order_key(scratch[r]); if ((key >> 8) == hi) atomicAdd(&hist[key & 255u], 1u); }
    __syncthreads();
    if (!tid) {
        unsigned acc = 0, b = 255;
        for (;; b--) { if (acc + hist[b] >= need || !b) break; acc += hist[b]; }
        s_thr = (hi << 8) | b; s_take_eq = need - acc;      // keys > threshold: (k - need) + acc; take need-acc ties
        s_cnt = 0; s_eq = 0;
    }
    __syncthreads();
    const unsigned thr = s_thr, take_eq = s_take_eq;
    Q38CpuMailboxSlot *slot = pci_cluster_rtx_slot(c);
    volatile unsigned *out_slot = slot ? reinterpret_cast<volatile unsigned *>(slot->result) : nullptr;
    volatile unsigned *out_dir = reinterpret_cast<volatile unsigned *>(direct->result);
    float mx = -INFINITY;
    for (unsigned r = tid; r < rows; r += nt) {
        const unsigned b = scratch[r], key = pci_cluster_order_key(b);
        mx = fmaxf(mx, __uint_as_float(b << 16));
        bool take = key > thr;
        if (!take && key == thr) take = atomicAdd(&s_eq, 1u) < take_eq;
        if (take) {
            const unsigned slot_i = 1u + atomicAdd(&s_cnt, 1u);
            const unsigned e = (b << 16) | r;
            out_dir[base + slot_i] = e;
            if (out_slot) out_slot[base + slot_i] = e;
        }
    }
    // log-sum-exp (FP32) over the BF16-rounded logits
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, s));
    if (!(tid & 31u)) s_red[tid >> 5] = mx;
    __syncthreads();
    if (!tid) { float m = s_red[0]; for (unsigned w = 1; w < (nt >> 5); w++) m = fmaxf(m, s_red[w]); s_red[0] = m; }
    __syncthreads();
    const float gmax = s_red[0];
    __syncthreads();
    float sum = 0.0f;
    for (unsigned r = tid; r < rows; r += nt) sum += __expf(__uint_as_float(scratch[r] << 16) - gmax);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, s);
    if (!(tid & 31u)) s_red[tid >> 5] = sum;
    __syncthreads();
    float lse = 0.0f;
    if (!tid) {
        float t = 0.0f; for (unsigned w = 0; w < (nt >> 5); w++) t += s_red[w];
        lse = gmax + __logf(t);
        out_dir[base] = s_cnt;
        if (out_slot) out_slot[base] = s_cnt;
    }
    __threadfence_system();
    return lse;
}

// ---------------------------------------------------------------------------------------------
// Worker-side MoE timing (2026-11-03, batched-MoE research). When ple_reserved[0] bit 0 is set (per
// worker, by q38_rtx_peer_worker_timing_enable; saved/restored with the other row-gather words), block 0
// thread 0 of a mode-1 (MoE) request records %globaltimer stamps into a ring at control +
// PCI_CLUSTER_WTIMING_OFF: 256 records x 64 B = 16 KiB at 0x3C000-0x3FFFF, inside the owner-BAR exposure
// (0x40000) and outside the control struct (< 0xB000), the direct result (0x20000), the leader-aggregation
// slots (0x28000-0x32803) and the multi-row leader slots (0x33000-0x3BFFF, resident_multirow_abi.cuh;
// 2026-11-04: the ring moved from 0x38000 / 512 records to make room for them).
// M10 clock: compare durations, not absolute times, with the RTX.
#define PCI_CLUSTER_WTIMING_OFF 0x3C000u
#define PCI_CLUSTER_WTIMING_RECS 256u
struct PciClusterWorkerTiming {
    unsigned seq, layer, role, kind;       // kind 1 = routed, 2 = shared
    unsigned long long t_seen, t_checked, t_gateup, t_down, t_done, t_extra;   // t_extra: kind-specific (kind 3: see routed mode 8)
};
static_assert(sizeof(PciClusterWorkerTiming) == 64, "worker timing record layout (read by the RTX library)");
__device__ __forceinline__ unsigned long long pci_cluster_gtimer() {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
template <class Control>
__device__ __forceinline__ bool pci_cluster_wtiming_on(const Control *c) {
    return (static_cast<const volatile unsigned *>(c->ple_reserved)[0] & 1u) != 0;
}
template <class Control>
__device__ __forceinline__ void pci_cluster_wtiming_write(Control *c, unsigned seq, unsigned layer, unsigned role, unsigned kind,
        unsigned long long seen, unsigned long long checked, unsigned long long gateup, unsigned long long down,
        unsigned long long extra = 0) {
    if (!pci_cluster_wtiming_on(c)) return;
    auto *r = reinterpret_cast<volatile PciClusterWorkerTiming *>(reinterpret_cast<char *>(c) + PCI_CLUSTER_WTIMING_OFF) +
              seq % PCI_CLUSTER_WTIMING_RECS;
    r->layer = layer; r->role = role; r->kind = kind;
    r->t_seen = seen; r->t_checked = checked; r->t_gateup = gateup; r->t_down = down; r->t_extra = extra;
    r->t_done = pci_cluster_gtimer();
    __threadfence_system();
    r->seq = seq;                          // written last: a record is valid when seq matches
}
