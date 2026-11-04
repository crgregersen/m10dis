// Batched MoE (worker mode 8, 2026-11-03): host batch-area layout helpers shared by the routed and
// shared worker kernels (resident_cpu_mailbox_abi.cuh, Q38_BATCH_*).
#pragma once
#include "resident_cpu_mailbox_abi.cuh"

#define Q38_BATCH_MODE 8u
#define Q38_BATCH_PIPE_MODE 9u        // pipelined: request in host slot seq & 1, header there (resident_cpu_mailbox_abi.cuh)
#define Q38_BATCH_REASON_CONFIG 8192u   // R, batch area, return flag, worker index, scratch
#define Q38_BATCH_REASON_ROWS 16384u    // an expert id or route of the batch is invalid
#ifndef Q38_AGG_DEADLINE
#define Q38_AGG_DEADLINE 2000000000ULL
#endif

// Host batch area in this worker's VA space. Set by the services when they register the CPU mailbox
// (region base + Q38_BATCH_OFF); not derived from cpu_mailbox_device_ptr, which the RTX peer library
// replaces by the M10 alias of the RTX BAR1 return slot.
template <class Control>
__device__ __forceinline__ char *q38_batch_area(const Control *c) {
    return reinterpret_cast<char *>(static_cast<const volatile Control *>(c)->batch_area);
}
// Per-slot addresses (slot 0 = the mode-8 layout; mode 9 uses slot seq & 1). Rows of slot s start at s * PIPE_ROWS.
__host__ __device__ __forceinline__ size_t q38_batch_row0(unsigned slot) { return size_t(slot) * Q38_BATCH_PIPE_ROWS; }
__host__ __device__ __forceinline__ size_t q38_batch_req_x_off(unsigned slot) { return Q38_BATCH_REQ_X + q38_batch_row0(slot) * Q38_BATCH_ROW_BYTES; }
__host__ __device__ __forceinline__ size_t q38_batch_req_ids_off(unsigned slot) { return Q38_BATCH_REQ_IDS + q38_batch_row0(slot) * Q38_BATCH_K * 4; }
__host__ __device__ __forceinline__ size_t q38_batch_req_routes_off(unsigned slot) { return Q38_BATCH_REQ_ROUTES + q38_batch_row0(slot) * Q38_BATCH_K * 4; }
__host__ __device__ __forceinline__ size_t q38_batch_req_hdr_off(unsigned slot) { return Q38_BATCH_REQ_HDR + size_t(slot) * Q38_BATCH_HDR_BYTES; }
__host__ __device__ __forceinline__ size_t q38_batch_gate_off(unsigned slot) { return Q38_BATCH_GATE + q38_batch_row0(slot) * 4; }
__host__ __device__ __forceinline__ size_t q38_batch_part_off(unsigned w, unsigned slot) {
    return Q38_BATCH_PART0 + size_t(w) * Q38_BATCH_PART_STRIDE + q38_batch_row0(slot) * Q38_BATCH_ROW_BYTES;
}
__host__ __device__ __forceinline__ size_t q38_batch_flag_off(unsigned w, unsigned slot) {
    return Q38_BATCH_PART0 + size_t(w) * Q38_BATCH_PART_STRIDE + Q38_BATCH_PART_ROWS_BYTES + size_t(slot) * Q38_BATCH_FLAG_SLOT_WORDS * 4;
}
// header words: 0 fmt (Q38_BATCH_FMT_*), 1 rows, 2 layer, 3 seq (mode 9; mode 8 uses only word 0)
__device__ __forceinline__ volatile unsigned *q38_batch_req_hdr(const char *batch, unsigned slot) {
    return reinterpret_cast<volatile unsigned *>(const_cast<char *>(batch) + q38_batch_req_hdr_off(slot));
}
__device__ __forceinline__ volatile float *q38_batch_gate(char *batch, unsigned slot) {
    return reinterpret_cast<volatile float *>(batch + q38_batch_gate_off(slot));
}
// Stage the request's x rows into device memory as FP32 (grid-stride, all threads). BF16 requests are widened
// exactly. Returns true when the request is BF16 (every staged value is then exactly BF16). Host reads are 16-byte
// cache-volatile loads (__ldcv: the request area is rewritten every exchange), 4 in flight per thread: with one
// 4-byte volatile load per thread a worker had ~10 KB in flight and the 24-worker fan-out was latency bound.
__device__ __forceinline__ bool q38_batch_stage_x(const char *batch, unsigned R, int H, float *dst, unsigned slot = 0) {
    // one host read of the format word per block (every thread reading it cost ~60 us per request: ~1,900
    // same-word host reads from the 24 workers at once)
    __shared__ unsigned fmt;
    __syncthreads();
    if (!threadIdx.x) fmt = q38_batch_req_hdr(batch, slot)[0];
    __syncthreads();
    const bool bf16 = fmt == Q38_BATCH_FMT_BF16;
    const unsigned stride = gridDim.x * blockDim.x, t0 = blockIdx.x * blockDim.x + threadIdx.x;
    const uint4 *src = reinterpret_cast<const uint4 *>(batch + q38_batch_req_x_off(slot));
    const unsigned n = bf16 ? R * unsigned(H) / 8 : R * unsigned(H) / 4;          // 16-byte items
    for (unsigned base = t0; base < n; base += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int k = 0; k < 4; k++) { const unsigned i = base + k * stride; if (i < n) v[k] = __ldcv(src + i); }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const unsigned i = base + k * stride;
            if (i >= n) break;
            if (bf16) {
                float4 *d = reinterpret_cast<float4 *>(dst) + 2 * size_t(i);       // 8 BF16 -> 8 FP32
                d[0] = make_float4(__uint_as_float(v[k].x << 16), __uint_as_float(v[k].x & 0xFFFF0000u),
                                   __uint_as_float(v[k].y << 16), __uint_as_float(v[k].y & 0xFFFF0000u));
                d[1] = make_float4(__uint_as_float(v[k].z << 16), __uint_as_float(v[k].z & 0xFFFF0000u),
                                   __uint_as_float(v[k].w << 16), __uint_as_float(v[k].w & 0xFFFF0000u));
            } else {
                reinterpret_cast<uint4 *>(dst)[i] = v[k];
            }
        }
    }
    return bf16;
}
__device__ __forceinline__ float *q38_batch_part(char *batch, unsigned w, unsigned slot = 0) {
    return reinterpret_cast<float *>(batch + q38_batch_part_off(w, slot));
}
__device__ __forceinline__ volatile unsigned *q38_batch_flag(char *batch, unsigned w, unsigned slot = 0) {
    return reinterpret_cast<volatile unsigned *>(batch + q38_batch_flag_off(w, slot));
}
