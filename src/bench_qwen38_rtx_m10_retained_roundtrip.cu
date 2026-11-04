// Serialized RTX -> M10 GPU transform -> RTX round trip.
// Both directional UVM routes and their PTE aliases remain active together
// for the complete exchange loop; one sequence is in flight at a time.
#define FENCE_HANDOFF_EVENT_ONLY 1
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <emmintrin.h>
extern "C" {
#include <nvstatus.h>
#include <uvm_linux_ioctl.h>
}
#include "qwen38_route_journal.h"

static void rtcheck(cudaError_t e, const char *where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
#define RT_CHECK(x) rtcheck((x), #x)
#define CK(x) RT_CHECK(x)
#include "resident_multicast_mapping.cuh"

#ifdef FENCE_HANDOFF_EVENT_ONLY
#define FH_STAMP(index) do { } while (0)
#else
#define FH_STAMP(index) do { clock_stamps[index] = clock64(); } while (0)
#endif

static void set_device_context(const QMem &m);

constexpr NvU64 PAGE = 4096;
constexpr NvU64 TARGET_OFF = 0x1000;
constexpr NvU32 PAGE_WORDS = PAGE / sizeof(NvU32);
constexpr NvU32 H_ELEMENTS = 2560;
constexpr NvU32 TOP_K = 10;
constexpr NvU32 REQUEST_LOGICAL_BYTES = H_ELEMENTS * 4 + TOP_K * 4 + TOP_K * 4;
constexpr NvU32 REQUEST_BYTES = 3 * PAGE; // 10,320 payload bytes, page padded to 12 KiB
constexpr NvU32 REQUEST_WORDS = REQUEST_BYTES / sizeof(NvU32);
constexpr NvU32 RESPONSE_LOGICAL_BYTES = H_ELEMENTS * sizeof(float);
constexpr NvU32 RESPONSE_BYTES = 3 * PAGE; // 10,240 payload bytes, third leaf partly used
constexpr NvU32 RESPONSE_WORDS = RESPONSE_BYTES / sizeof(NvU32);
constexpr NvU32 RESPONSE_PAD_WORDS = (RESPONSE_BYTES - RESPONSE_LOGICAL_BYTES) / 4;
constexpr NvU32 MAILBOX_HEADER_BYTES = 16;
constexpr NvU32 PROD_REQUEST_BYTES = REQUEST_LOGICAL_BYTES; // H + top-10 IDs + weights
constexpr NvU32 PROD_RESULT_BYTES = RESPONSE_LOGICAL_BYTES; // 2560 FP32 results
constexpr NvU32 RETAINED_BODY_BYTES = REQUEST_BYTES - MAILBOX_HEADER_BYTES;
constexpr NvU32 RETAINED_BODY_WORDS = RETAINED_BODY_BYTES / sizeof(NvU32);
constexpr NvU32 RETAINED_BODY_VECTORS = RETAINED_BODY_BYTES / sizeof(uint4);
constexpr NvU32 RETAINED_RUN_ID = 0x38a10a55u;
constexpr NvU32 RETAINED_PAD = 0x3c7e91b2u;
constexpr NvU32 GUARD_BEFORE = 0x62a5b10c;
constexpr NvU32 GUARD_AFTER = 0x73b6c21d;
constexpr NvU32 RESPONSE_PAD = 0x7f1a2b3c;
constexpr NvU32 QWEN_H = 2560, QWEN_W = 32, QWEN_K = 10, QWEN_E = 512, QWEN_GROUP = 128;

__host__ __device__ static inline float request_h(NvU32 seq, NvU32 i) {
    return float((i * 7u + seq * 3u) % 97u) * 0.125f;
}
__host__ __device__ static inline NvU32 f32_bits(float v) {
    union { float f; NvU32 u; } bits;
    bits.f = v;
    return bits.u;
}
__host__ __device__ static inline NvU32 request_word(NvU32 seq, NvU32 i) {
    if (i < H_ELEMENTS) return f32_bits(request_h(seq, i));
    if (i < H_ELEMENTS + TOP_K) return 3u + ((i - H_ELEMENTS) * 17u + seq) % 128u;
    if (i < H_ELEMENTS + 2 * TOP_K) {
        const float w = float((i - H_ELEMENTS - TOP_K + 1u) * (seq + 1u)) * 0.0625f;
        return f32_bits(w);
    }
    return 0xa17e0000u ^ (seq * 0x00010101u) ^ (i * 0x9e3779b9u);
}

__host__ __device__ static inline NvU32 request_id(NvU32 seq, NvU32 i) {
    return 3u + (i * 17u + seq) % 128u;
}
__host__ __device__ static inline float request_weight(NvU32 seq, NvU32 i) {
    return float((i + 1u) * (seq + 1u)) * 0.0625f;
}
__host__ __device__ static inline float request_bias(NvU32 seq) {
    float bias = 0.0f;
    for (NvU32 k = 0; k < TOP_K; ++k)
        bias += float(request_id(seq, k)) * request_weight(seq, k);
    return bias;
}
__host__ __device__ static inline float response_value(NvU32 seq, NvU32 i) {
    return request_h(seq, i) + request_bias(seq);
}
__host__ __device__ static inline NvU32 response_word(NvU32 seq, NvU32 i) {
    return f32_bits(response_value(seq, i));
}

__global__ static void rtx_publish_request_page(volatile NvU32 *dst, NvU32 seq, NvU32 page) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const NvU32 first = page * PAGE_WORDS;
        for (NvU32 i = 0; i < PAGE_WORDS; ++i) dst[i] = request_word(seq, first + i);
        __threadfence_system();
    }
}
__global__ static void rtx_publish_request_full(volatile NvU32 *dst, NvU32 seq) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (NvU32 i = 0; i < REQUEST_WORDS; ++i) dst[i] = request_word(seq, i);
        __threadfence_system();
    }
}
__global__ static void verify_copy_words(const volatile NvU32 *src, NvU32 *dst, NvU32 count) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (NvU32 i = 0; i < count; ++i) dst[i] = src[i];
}
__global__ static void prime_host_alias(volatile NvU32 *ptr, NvU32 pages, NvU32 marker) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (NvU32 p = 0; p < pages; ++p) ptr[p * PAGE_WORDS] = marker + p;
}
__global__ static void compute_test_slice(const volatile NvU32 *request, float *result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float route_bias = 0.0f;
        for (NvU32 k = 0; k < TOP_K; ++k)
            route_bias += float(request[H_ELEMENTS + k]) *
                          __uint_as_float(request[H_ELEMENTS + TOP_K + k]);
        for (NvU32 i = 0; i < H_ELEMENTS; ++i)
            result[i] = __uint_as_float(request[i]) + route_bias;
    }
}
__global__ static void m10_publish_slice_page(const float *result, volatile NvU32 *dst, NvU32 page) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const NvU32 first = page * PAGE_WORDS;
        for (NvU32 i = 0; i < PAGE_WORDS && first + i < H_ELEMENTS; ++i)
            dst[i] = __float_as_uint(result[first + i]);
        __threadfence_system();
    }
}
__global__ static void m10_publish_slice_full(const float *result, volatile NvU32 *dst) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (NvU32 i = 0; i < H_ELEMENTS; ++i) dst[i] = __float_as_uint(result[i]);
        __threadfence_system();
    }
}

// Repeated request path. Payload stores stay distributed across 256 threads;
// the sequence word is written only after all payload stores and a system fence.
__global__ static void retained_request_send(const uint4 *src,
                                             volatile uint4 *dst_body,
                                             NvU32 vector_count,
                                             volatile NvU32 *dst_header,
                                             NvU32 sequence,
                                             NvU32 run_id)
{
    if (blockIdx.x != 0) return;
    for (NvU32 i = threadIdx.x; i < vector_count; i += blockDim.x) {
        const uint4 v = src[i];
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst_body + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        dst_header[1] = run_id;
        dst_header[2] = PROD_REQUEST_BYTES;
        dst_header[3] = 0x52514553u; // "RQES"
        dst_header[0] = sequence;
        __threadfence_system();
    }
}

// One M10 block consumes the actual BAR1-delivered H/ID/weight fields.
__global__ static void retained_m10_compute(const volatile NvU32 *header,
                                             const volatile NvU32 *body,
                                             float *result,
                                             NvU32 sequence,
                                             NvU32 run_id,
                                             NvU32 *status)
{
    __shared__ float route_bias;
    __shared__ NvU32 request_status;
    if (threadIdx.x == 0) {
        request_status = (header[0] == sequence && header[1] == run_id &&
                          header[2] == PROD_REQUEST_BYTES && header[3] == 0x52514553u) ? 0u : 1u;
        route_bias = 0.0f;
        if (request_status == 0) {
            for (NvU32 k = 0; k < TOP_K; ++k) {
                const NvU32 id = body[H_ELEMENTS + k];
                const float weight = __uint_as_float(body[H_ELEMENTS + TOP_K + k]);
                route_bias += float(id) * weight;
            }
        }
        *status = request_status;
    }
    __syncthreads();
    if (request_status != 0) return;
    for (NvU32 i = threadIdx.x; i < H_ELEMENTS; i += blockDim.x)
        result[i] = __uint_as_float(body[i]) + route_bias;
}

// The M10 returns the complete padded result through its retained requester
// alias. Completion is published after the final payload store.
__global__ static void retained_m10_return(const float *result,
                                            volatile uint4 *dst_body,
                                            NvU32 vector_count,
                                            volatile NvU32 *dst_header,
                                            NvU32 sequence,
                                            NvU32 run_id,
                                            const NvU32 *status)
{
    if (blockIdx.x != 0) return;
    for (NvU32 i = threadIdx.x; i < vector_count; i += blockDim.x) {
        const NvU32 first = i * 4;
        const uint4 v = {
            first + 0 < H_ELEMENTS ? __float_as_uint(result[first + 0]) : RETAINED_PAD,
            first + 1 < H_ELEMENTS ? __float_as_uint(result[first + 1]) : RETAINED_PAD,
            first + 2 < H_ELEMENTS ? __float_as_uint(result[first + 2]) : RETAINED_PAD,
            first + 3 < H_ELEMENTS ? __float_as_uint(result[first + 3]) : RETAINED_PAD,
        };
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst_body + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        dst_header[1] = run_id;
        dst_header[2] = status[0];
        dst_header[3] = PROD_RESULT_BYTES;
        dst_header[0] = sequence;
        __threadfence_system();
    }
}

// RTX checks every returned element on-device and retains every result epoch
// for an independent full CPU comparison after the exchange loop.
__global__ static void retained_verify_result(const volatile NvU32 *header,
                                               const volatile NvU32 *body,
                                               const uint4 *request_body,
                                               NvU32 sequence,
                                               NvU32 run_id,
                                               float *history,
                                               NvU32 *status)
{
    if (blockIdx.x != 0) return;
    __shared__ NvU32 mismatch_count;
    __shared__ float bias_shared;
    if (threadIdx.x == 0) {
        mismatch_count = (header[0] == sequence && header[1] == run_id &&
                          header[2] == 0 && header[3] == PROD_RESULT_BYTES) ? 0u : 1u;
        const NvU32 *req = reinterpret_cast<const NvU32 *>(request_body);
        float bias = 0.0f;
        for (NvU32 k = 0; k < TOP_K; ++k) {
            const NvU32 id = req[H_ELEMENTS + k];
            const float weight = __uint_as_float(req[H_ELEMENTS + TOP_K + k]);
            bias += float(id) * weight;
        }
        bias_shared = bias;
    }
    __syncthreads();
    const NvU32 *req = reinterpret_cast<const NvU32 *>(request_body);
    NvU32 local_mismatches = 0;
    for (NvU32 i = threadIdx.x; i < H_ELEMENTS; i += blockDim.x) {
        const float observed = __uint_as_float(body[i]);
        const float expected = __uint_as_float(req[i]) + bias_shared;
        history[i] = observed;
        if (__float_as_uint(observed) != __float_as_uint(expected)) ++local_mismatches;
    }
    for (NvU32 i = H_ELEMENTS + threadIdx.x; i < RETAINED_BODY_WORDS; i += blockDim.x)
        if (body[i] != RETAINED_PAD) ++local_mismatches;
    if (local_mismatches) atomicAdd(&mismatch_count, local_mismatches);
    __syncthreads();
    if (threadIdx.x == 0) *status = mismatch_count;
}

struct QwenSliceWeights { NvU32 *gate, *up, *down; float *gate_scale, *up_scale, *down_scale; };
struct QwenSliceParams {
    QwenSliceWeights w;
    const float *h, *route_weights;
    const NvS32 *expert_ids;
    float *activations, *out;
};

static std::vector<NvU8> read_blob(const std::string &path, size_t expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || size_t(f.tellg()) != expected) throw std::runtime_error("unexpected fixture file size: " + path);
    std::vector<NvU8> v(expected); f.seekg(0); f.read(reinterpret_cast<char *>(v.data()), expected);
    if (!f) throw std::runtime_error("fixture read failed: " + path);
    return v;
}
static std::vector<float> read_f32_blob(const std::string &path, size_t count) {
    auto bytes = read_blob(path, count * sizeof(float)); std::vector<float> v(count);
    std::memcpy(v.data(), bytes.data(), bytes.size()); return v;
}
template<class T> static T *device_upload(const std::vector<T> &v, int expected_device) {
    int current_device = -1;
    RT_CHECK(cudaGetDevice(&current_device));
    if (current_device != expected_device)
        throw std::runtime_error("device_upload called with the wrong current CUDA device");
    T *p = nullptr; RT_CHECK(cudaMalloc(&p, v.size() * sizeof(T)));
    cudaPointerAttributes attributes{};
    RT_CHECK(cudaPointerGetAttributes(&attributes, p));
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != expected_device) {
        cudaFree(p);
        throw std::runtime_error("device_upload allocation does not belong to the requested CUDA device");
    }
    RT_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
static std::vector<NvU32> pack_qwen_gate(const std::vector<NvU8> &raw) {
    std::vector<NvU32> p(raw.size() / 4);
    for (size_t i = 0; i < p.size(); ++i)
        p[i] = NvU32(raw[4*i]) | (NvU32(raw[4*i+1]) << 8) | (NvU32(raw[4*i+2]) << 16) | (NvU32(raw[4*i+3]) << 24);
    return p;
}
static std::vector<NvU32> pack_qwen_down(const std::vector<NvU8> &raw) {
    std::vector<NvU32> p(size_t(QWEN_E) * (QWEN_W / 4) * QWEN_H);
    for (NvU32 e = 0; e < QWEN_E; ++e) for (NvU32 g = 0; g < QWEN_W / 4; ++g)
        for (NvU32 row = 0; row < QWEN_H; ++row) {
            NvU32 v = 0;
            for (NvU32 j = 0; j < 4; ++j) v |= NvU32(raw[(size_t(e)*QWEN_W + g*4+j)*QWEN_H + row]) << (8*j);
            p[(size_t(e)*(QWEN_W/4)+g)*QWEN_H+row] = v;
        }
    return p;
}
__device__ __forceinline__ static float qwen_fp8_e4m3(NvU32 x) {
    const NvU32 b = x & 255u, a = b & 127u;
    const float v = a < 8u ? float(a) * (1.0f / 512.0f) : __uint_as_float((a << 20) + (120u << 23));
    return (b & 128u) ? -v : v;
}
__device__ static float qwen_warp_sum(float x) {
    for (int d = 16; d; d >>= 1) x += __shfl_down_sync(0xffffffffu, x, d);
    return x;
}
__global__ static void qwen_gate_slice(QwenSliceParams p, const volatile NvU32 *header,
                                       const volatile NvU32 *body, NvU32 sequence,
                                       NvU32 run_id, NvU32 *status) {
    __shared__ float gate_partials[8], up_partials[8];
    __shared__ NvU32 request_ok;
    const NvU32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    if (threadIdx.x == 0) {
        request_ok = header[0] == sequence && header[1] == run_id &&
                     header[2] == PROD_REQUEST_BYTES && header[3] == 0x52514553u;
        *status = request_ok ? 0u : 1u;
    }
    __syncthreads();
    if (!request_ok) return;
    for (NvU32 task = blockIdx.x; task < QWEN_K * QWEN_W; task += gridDim.x) {
        const NvU32 k = task / QWEN_W, r = task % QWEN_W;
        const NvS32 expert = reinterpret_cast<const volatile NvS32 *>(body + QWEN_H)[k];
        if (expert < 0 || expert >= QWEN_E) { if (threadIdx.x == 0) *status = 2u; __syncthreads(); return; }
        float ga = 0.0f, up = 0.0f;
        const size_t wb = (size_t(expert) * QWEN_W + r) * (QWEN_H / 4);
        const size_t sb = (size_t(expert) * QWEN_W + r) * (QWEN_H / QWEN_GROUP);
        const volatile float4 *h = reinterpret_cast<const volatile float4 *>(body);
        for (NvU32 word = threadIdx.x; word < QWEN_H / 4; word += blockDim.x) {
            const NvU32 gw = p.w.gate[wb + word], uw = p.w.up[wb + word];
            const float gs = p.w.gate_scale[sb + word / (QWEN_GROUP / 4)];
            const float us = p.w.up_scale[sb + word / (QWEN_GROUP / 4)];
            const volatile float4 &xv = h[word];
            ga = fmaf(qwen_fp8_e4m3(gw) * gs, xv.x, ga);
            ga = fmaf(qwen_fp8_e4m3(gw >> 8) * gs, xv.y, ga);
            ga = fmaf(qwen_fp8_e4m3(gw >> 16) * gs, xv.z, ga);
            ga = fmaf(qwen_fp8_e4m3(gw >> 24) * gs, xv.w, ga);
            up = fmaf(qwen_fp8_e4m3(uw) * us, xv.x, up);
            up = fmaf(qwen_fp8_e4m3(uw >> 8) * us, xv.y, up);
            up = fmaf(qwen_fp8_e4m3(uw >> 16) * us, xv.z, up);
            up = fmaf(qwen_fp8_e4m3(uw >> 24) * us, xv.w, up);
        }
        ga = qwen_warp_sum(ga); up = qwen_warp_sum(up);
        if (lane == 0) { gate_partials[warp] = ga; up_partials[warp] = up; }
        __syncthreads();
        if (warp == 0) {
            ga = lane < 8 ? gate_partials[lane] : 0.0f;
            up = lane < 8 ? up_partials[lane] : 0.0f;
            ga = qwen_warp_sum(ga); up = qwen_warp_sum(up);
            if (lane == 0) p.activations[task] = (ga / (1.0f + expf(-ga))) * up;
        }
        __syncthreads();
    }
}
__global__ static void qwen_down_slice(QwenSliceParams p, const volatile NvU32 *body,
                                       const NvU32 *status) {
    __shared__ float acts[QWEN_K * QWEN_W];
    if (status[0] != 0) return;
    for (NvU32 i = threadIdx.x; i < QWEN_K * QWEN_W; i += blockDim.x) acts[i] = p.activations[i];
    __syncthreads();
    const volatile NvS32 *ids = reinterpret_cast<const volatile NvS32 *>(body + QWEN_H);
    const volatile float *weights = reinterpret_cast<const volatile float *>(body + QWEN_H + QWEN_K);
    for (NvU32 row = blockIdx.x * blockDim.x + threadIdx.x; row < QWEN_H; row += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (NvU32 k = 0; k < QWEN_K; ++k) {
            float part = 0.0f;
            for (NvU32 word = 0; word < QWEN_W / 4; ++word) {
                const NvU32 q = p.w.down[(size_t(ids[k]) * (QWEN_W / 4) + word) * QWEN_H + row];
                part = fmaf(qwen_fp8_e4m3(q), acts[k*QWEN_W+word*4], part);
                part = fmaf(qwen_fp8_e4m3(q >> 8), acts[k*QWEN_W+word*4+1], part);
                part = fmaf(qwen_fp8_e4m3(q >> 16), acts[k*QWEN_W+word*4+2], part);
                part = fmaf(qwen_fp8_e4m3(q >> 24), acts[k*QWEN_W+word*4+3], part);
            }
            sum = fmaf(part * p.w.down_scale[size_t(ids[k]) * QWEN_H + row], weights[k], sum);
        }
        p.out[row] = sum;
    }
}
__global__ static void copy_peer_words(const volatile NvU32 *src, NvU32 *dst, NvU32 n) {
    for (NvU32 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

// Timed transport kernel: one CTA, coalesced 16-byte payload accesses, then
// per-thread system fences and a sequence-last publication. Clock stamps are
// from this requester GPU's clock domain; CUDA events provide the primary us
// measurement. Receiver verification is performed after the timed interval.
__global__ static void transmit_vec4_with_sequence(const uint4 *src,
                                                    volatile uint4 *dst_payload,
                                                    NvU32 vector_count,
                                                    volatile NvU32 *dst_sequence,
                                                    NvU32 sequence,
                                                    NvU64 *clock_stamps)
{
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) FH_STAMP(0);
    __syncthreads();
    for (NvU32 i = threadIdx.x; i < vector_count; i += blockDim.x) {
        const uint4 v = src[i];
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst_payload + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) FH_STAMP(1);
    __syncthreads();
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        FH_STAMP(2);
        *dst_sequence = sequence;
        __threadfence_system();
        FH_STAMP(3);
    }
}

// Test the fence-cost hypothesis without serializing the PCIe payload. All
// threads issue their coalesced remote stores; a CTA barrier completes the
// producers' local work; thread 0 then performs one system fence and publishes
// the sequence. This is an experimental ordering protocol for the custom
// BAR1 mapping and must pass full destination checks before being accepted.
__global__ static void transmit_vec4_distributed_one_fence(const uint4 *src,
                                                            volatile uint4 *dst_payload,
                                                            NvU32 vector_count,
                                                            volatile NvU32 *dst_sequence,
                                                            NvU32 sequence,
                                                            NvU64 *clock_stamps)
{
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) FH_STAMP(0);
    __syncthreads();
    for (NvU32 i = threadIdx.x; i < vector_count; i += blockDim.x) {
        const uint4 v = src[i];
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst_payload + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        FH_STAMP(1); // all CTA payload stores have been issued
        __threadfence_system();
        FH_STAMP(2); // one system fence after CTA handoff
        *dst_sequence = sequence;
        __threadfence_system();
        FH_STAMP(3);
    }
}

// Local handoff experiment: the CTA first stages the complete production
// payload into shared memory. After the block barrier, thread 0 alone issues
// all remote PCIe stores, then publishes the sequence. This preserves the
// full payload transfer while reducing the number of system-fence callers.
__global__ static void transmit_vec4_single_sender(const uint4 *src,
                                                    volatile uint4 *dst_payload,
                                                    NvU32 vector_count,
                                                    volatile NvU32 *dst_sequence,
                                                    NvU32 sequence,
                                                    NvU64 *clock_stamps)
{
    extern __shared__ uint4 staged[];
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) FH_STAMP(0);
    for (NvU32 i = threadIdx.x; i < vector_count; i += blockDim.x)
        staged[i] = src[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        FH_STAMP(1); // local staging and handoff complete
        for (NvU32 i = 0; i < vector_count; ++i) {
            const uint4 v = staged[i];
            asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(dst_payload + i), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                         : "memory");
        }
        FH_STAMP(2); // last remote store issued
        __threadfence_system();
        FH_STAMP(3); // one sender's payload fence complete
        *dst_sequence = sequence;
        __threadfence_system();
        FH_STAMP(4);
    }
}

static NvProcessorUuid uuid_for(const char *bdf, CUdevice *device = nullptr) {
    CUdevice d{}; CUKQ(cuDeviceGetByPCIBusId(&d, bdf));
    CUuuid c{}; CUKQ(cuDeviceGetUuid(&c, d));
    NvProcessorUuid u{}; static_assert(sizeof(u) == sizeof(c.bytes));
    std::memcpy(&u, c.bytes, sizeof(u));
    if (device) *device = d;
    return u;
}

static bool ioctl_log(int fd, const char *tag, UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS &p) {
    errno = 0;
    int rc = ioctl(fd, UVM_GLM53_SERIAL_BAR1_PEER_READ, &p);
    int e = errno;
    std::printf("%s,rc,%d,errno,%d,status,0x%08x,owner,%u,external,%u,dma,%u,remap,%u,map_id,0x%016" PRIx64
                ",route_id,0x%016" PRIx64 ",cpu_resource,0x%016" PRIx64
                ",bar_offset,0x%016" PRIx64 ",bar_bus,0x%016" PRIx64
                ",dma_addr,0x%016" PRIx64 ",dma_len,0x%016" PRIx64
                ",destination,0x%016" PRIx64 ",dma_direction,%u,dma_error,%u"
                ",va,0x%016" PRIx64 ",pte_phys,0x%016" PRIx64 ",pte_before,0x%016" PRIx64
                ",temporary_pte,0x%016" PRIx64 ",pte_after,0x%016" PRIx64 "\n",
                tag, rc, e, (unsigned)p.rmStatus, p.owner_active, p.owner_external,
                p.dma_active, p.remap_active, (uint64_t)p.mapping_id, (uint64_t)p.route_id,
                (uint64_t)p.cpu_resource_start, (uint64_t)p.allocation_bar_offset,
                (uint64_t)p.allocation_bar_bus_start, (uint64_t)p.requester_dma_address,
                (uint64_t)p.requester_dma_length, (uint64_t)p.destination_gpu_sys_address,
                p.dma_direction, p.dma_mapping_error, (uint64_t)p.target_va,
                (uint64_t)p.target_pte_phys, (uint64_t)p.pte_before,
                (uint64_t)p.temporary_pte, (uint64_t)p.pte_after);
    std::fflush(stdout);
    const bool ok = rc == 0 && p.rmStatus == NV_OK;
    // Route journal (no-op unless Q38_RTX_PEER_ROUTE_JOURNAL): acquisitions and releases of
    // global patched-UVM owner/DMA records, so the daemon can release them after a client crash.
    if (ok && p.op == UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_ACQUIRE_EXTERNAL_OWNER && p.owner_active)
        q38_journal_line("OWNER\t%llu\t%s\t%s", (unsigned long long)p.mapping_id,
                         q38_uuid_hex(p.requester_uuid).c_str(), q38_uuid_hex(p.owner_uuid).c_str());
    else if (ok && p.op == UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_MAP_REQUESTER && p.dma_active)
        q38_journal_line("DMA\t%llu\t%llu\t%s\t%s", (unsigned long long)p.mapping_id, (unsigned long long)p.route_id,
                         q38_uuid_hex(p.requester_uuid).c_str(), q38_uuid_hex(p.owner_uuid).c_str());
    else if (ok && p.op == UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_UNMAP_REQUESTER && !p.dma_active)
        q38_journal_line("DMA_UNMAPPED\t%llu\t%llu", (unsigned long long)p.mapping_id, (unsigned long long)p.route_id);
    else if (ok && p.op == UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_RELEASE_OWNER && !p.owner_active)
        q38_journal_line("OWNER_RELEASED\t%llu", (unsigned long long)p.mapping_id);
    return ok;
}

static UVM_GLM53_VA_SPACE_PDB_ENTRY find_pdb(int fd, const NvProcessorUuid &uuid) {
    std::vector<UVM_GLM53_VA_SPACE_PDB_ENTRY> entries(128);
    UVM_GLM53_LIST_VA_SPACE_PDBS_PARAMS p{};
    p.entry_capacity = entries.size(); p.entries = (NvU64)entries.data();
    errno = 0;
    const int list_rc = ioctl(fd, UVM_GLM53_LIST_VA_SPACE_PDBS, &p);
    if (list_rc || p.rmStatus != NV_OK) {
        std::fprintf(stderr, "PDB_LIST_FAILURE,rc,%d,errno,%d,status,0x%08x,capacity,%u,count,%u,matched,%u,total,%u,overflow,%u,pid,%u\n",
                     list_rc, errno, (unsigned)p.rmStatus, p.entry_capacity, p.entry_count,
                     p.matched_va_spaces, p.total_va_spaces, p.overflow, (unsigned)getpid());
        std::fflush(stderr);
        throw std::runtime_error("UVM PDB listing failed");
    }
    auto hex_uuid = [](const NvProcessorUuid &u) {
        for (size_t j = 0; j < sizeof(u); ++j) std::fprintf(stderr, "%02x", ((const NvU8 *)&u)[j]);
    };
    std::fprintf(stderr, "PDB_ENUM,requested_uuid,"); hex_uuid(uuid); std::fprintf(stderr, ",pid,%u,count,%u\n", (unsigned)getpid(), p.entry_count);
    for (NvU32 i = 0; i < p.entry_count; ++i) {
        const auto &e = entries[i];
        std::fprintf(stderr, "PDB_ENTRY,index,%u,uuid,", i); hex_uuid(e.gpu_uuid);
        std::fprintf(stderr, ",pid,%u,did_set,%u,va_space,0x%016" PRIx64 ",gpu_va_space,0x%016" PRIx64 ",root,0x%016" PRIx64 ",aperture,%u\n",
                     e.current_pid, e.did_set_page_directory, (uint64_t)e.va_space_ptr,
                     (uint64_t)e.gpu_va_space_ptr, (uint64_t)e.root_phys_address, e.root_phys_aperture);
        if (!std::memcmp(&e.gpu_uuid, &uuid, sizeof(uuid)) &&
            e.current_pid == (NvU32)getpid() && e.did_set_page_directory)
            return e;
    }
    throw std::runtime_error("requester PDB not found for current PID");
}

static UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS query_pte(
        int fd, const NvProcessorUuid &requester, NvU64 va, const char *tag) {
    const auto pdb = find_pdb(fd, requester);
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q{};
    q.requester_uuid = requester;
    q.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
    q.va_space_ptr = pdb.va_space_ptr; q.gpu_va_space_ptr = pdb.gpu_va_space_ptr;
    q.root_phys_address = pdb.root_phys_address; q.root_phys_aperture = pdb.root_phys_aperture;
    q.target_va = va; q.page_size = PAGE;
    const bool ok = ioctl_log(fd, tag, q);
    if (!ok || !(q.pte_after & 1) || !q.target_pte_phys)
        throw std::runtime_error(std::string(tag) + " returned no valid 4K PTE");
    return q;
}

struct Route {
    NvProcessorUuid requester{}, owner{};
    bool owner_active = false, dma_active = false, remap_active = false;
    bool requester_quiesced = true;
    NvU64 map_id = 0, route_id = 0;
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS query{}, apply{}, restore{};
};

[[noreturn]] static void preserve(const char *reason) {
    std::fprintf(stderr, "PRESERVE,%s; mappings and allocations retained\n", reason);
    std::fflush(stderr);
    for (;;) sleep(60);
}

static void route_setup(Route &r, int fd,
                        const NvProcessorUuid &requester,
                        const NvProcessorUuid &owner_uuid,
                        const QMem &owner_mem,
                        NvU64 requester_va,
                        NvU64 owner_offset,
                        const UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS *prequery = nullptr) {
    r.requester = requester; r.owner = owner_uuid;
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS ext{};
    ext.requester_uuid = requester; ext.owner_uuid = owner_uuid;
    ext.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_ACQUIRE_EXTERNAL_OWNER;
    ext.requested_owner_bar1_bus_start = owner_mem.rmmap.addr;
    ext.requested_resource_length = owner_mem.map_bytes;
    ext.requested_cpu_resource_start = 0;
    const bool ext_ok = ioctl_log(fd, "ROUTE_EXTERNAL_OWNER_RETAIN", ext);
    r.owner_active = ext.owner_active && ext.owner_external;
    r.map_id = ext.mapping_id;
    if (!ext_ok || !r.owner_active) throw std::runtime_error("external owner mapping not retained");

    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS dm{};
    dm.requester_uuid = requester; dm.owner_uuid = owner_uuid;
    dm.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_MAP_REQUESTER;
    dm.mapping_id = r.map_id;
    const bool dm_ok = ioctl_log(fd, "ROUTE_DMA_MAP", dm);
    r.dma_active = dm.dma_active; r.route_id = dm.route_id;
    if (!dm_ok || !r.dma_active || dm.dma_mapping_error || !dm.dma_aligned_for_4k_leaf ||
        owner_offset > dm.requester_dma_length || PAGE > dm.requester_dma_length - owner_offset)
        throw std::runtime_error("requester DMA mapping does not cover a 4K target leaf");

    if (prequery) {
        r.query = *prequery;
        if (r.query.target_va != requester_va ||
            std::memcmp(&r.query.requester_uuid, &requester, sizeof(requester)) ||
            !(r.query.pte_after & 1) || !r.query.target_pte_phys)
            throw std::runtime_error("saved requester PTE does not match the route target");
        r.query.owner_uuid = owner_uuid;
        std::printf("ROUTE_PTE_PREQUERY_REUSE,va,0x%016" PRIx64 ",original_pte,0x%016" PRIx64 "\n",
                    (uint64_t)requester_va, (uint64_t)r.query.pte_after);
    } else {
        r.query = query_pte(fd, requester, requester_va, "ROUTE_PTE_QUERY");
        r.query.owner_uuid = owner_uuid;
    }

    const NvU64 dst = dm.destination_gpu_sys_address + owner_offset;
    if (dst < dm.destination_gpu_sys_address || (dst & (PAGE - 1)))
        throw std::runtime_error("DMA destination page alignment failed");
    r.apply.requester_uuid = requester; r.apply.owner_uuid = owner_uuid;
    r.apply.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_APPLY;
    r.apply.mapping_id = r.map_id; r.apply.route_id = r.route_id;
    r.apply.va_space_ptr = r.query.va_space_ptr;
    r.apply.gpu_va_space_ptr = r.query.gpu_va_space_ptr;
    r.apply.root_phys_address = r.query.root_phys_address;
    r.apply.root_phys_aperture = r.query.root_phys_aperture;
    r.apply.target_va = requester_va; r.apply.page_size = PAGE;
    r.apply.target_pte_phys = r.query.target_pte_phys;
    r.apply.expected_current_pte = r.query.pte_after;
    r.apply.requested_destination_gpu_sys_address = dst;
    r.apply.requested_allow_writable = 1;
    r.restore.requester_uuid = requester; r.restore.owner_uuid = owner_uuid;
    r.restore.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_RESTORE;
    r.restore.mapping_id = r.map_id; r.restore.route_id = r.route_id;
    r.restore.va_space_ptr = r.query.va_space_ptr;
    r.restore.gpu_va_space_ptr = r.query.gpu_va_space_ptr;
    r.restore.root_phys_address = r.query.root_phys_address;
    r.restore.root_phys_aperture = r.query.root_phys_aperture;
    r.restore.target_va = requester_va; r.restore.page_size = PAGE;
    r.restore.target_pte_phys = r.query.target_pte_phys;
    const bool apply_ok = ioctl_log(fd, "ROUTE_PTE_APPLY", r.apply);
    r.remap_active = r.apply.remap_active;
    if (!apply_ok || !r.remap_active || r.apply.pte_after != r.apply.temporary_pte)
        throw std::runtime_error("temporary requester PTE did not verify");
    r.restore.expected_current_pte = r.apply.temporary_pte;
}

static void route_cleanup(Route &r, int fd) {
    if (r.remap_active) {
        if (!r.requester_quiesced) preserve("requester work may still use temporary PTE");
        const bool restore_ok = ioctl_log(fd, "ROUTE_PTE_RESTORE", r.restore);
        r.remap_active = r.restore.remap_active;
        if (!restore_ok || r.remap_active || r.restore.pte_after != r.query.pte_after)
            preserve("requester PTE exact restoration not confirmed");
    }
    if (r.dma_active) {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS un{};
        un.requester_uuid = r.requester; un.owner_uuid = r.owner;
        un.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_UNMAP_REQUESTER;
        un.mapping_id = r.map_id; un.route_id = r.route_id;
        const bool unmap_ok = ioctl_log(fd, "ROUTE_DMA_UNMAP", un);
        r.dma_active = un.dma_active;
        if (!unmap_ok || r.dma_active) preserve("requester DMA unmap not confirmed");
    }
    if (r.owner_active) {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS rel{};
        rel.requester_uuid = r.requester; rel.owner_uuid = r.owner;
        rel.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_RELEASE_OWNER;
        rel.mapping_id = r.map_id;
        const bool release_ok = ioctl_log(fd, "ROUTE_EXTERNAL_OWNER_RELEASE", rel);
        r.owner_active = rel.owner_active;
        if (!release_ok || r.owner_active) preserve("external owner route release not confirmed");
    }
}

struct MultiRoute {
    NvProcessorUuid requester{}, owner{};
    bool owner_active = false, dma_active = false, requester_quiesced = true;
    NvU64 map_id = 0, route_id = 0, owner_offset = 0;
    std::vector<UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS> queries;
    std::vector<UVM_GLM53_MULTI_PTE_REMAP_PARAMS> active;
};

static void multi_route_setup(MultiRoute &r, int fd,
                              const NvProcessorUuid &requester,
                              const NvProcessorUuid &owner_uuid,
                              const QMem &owner_mem, NvU64 requester_va,
                              NvU64 owner_offset,
                              const std::vector<UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS> *saved = nullptr,
                              NvU64 owner_window_offset = 0,
                              NvU64 owner_window_length = 0,
                              NvU32 page_count = 3,
                              const UVM_GLM53_VA_SPACE_PDB_ENTRY *requester_pdb = nullptr,
                              NvU64 leaf_page_size = PAGE,
                              bool allow_writable = true,
                              bool use_owner_bus_address = false) {
    if (!page_count || page_count > 16 ||
        (leaf_page_size != PAGE && leaf_page_size != (128u << 10)))
        throw std::runtime_error("multi route supports 1..16 4-KiB or 128-KiB leaves");
    r.requester = requester; r.owner = owner_uuid;
    const NvU64 retained_length = owner_window_length ? owner_window_length : owner_mem.map_bytes;
    if (owner_window_offset > owner_mem.map_bytes ||
        retained_length > owner_mem.map_bytes - owner_window_offset ||
        owner_mem.rmmap.addr > ~NvU64(0) - owner_window_offset)
        throw std::runtime_error("multi route owner subwindow is outside retained RM mapping");
    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS ext{};
    ext.requester_uuid = requester; ext.owner_uuid = owner_uuid;
    ext.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_ACQUIRE_EXTERNAL_OWNER;
    ext.requested_owner_bar1_bus_start = owner_mem.rmmap.addr + owner_window_offset;
    ext.requested_resource_length = retained_length;
    ioctl_log(fd, "MULTI_ROUTE_OWNER_RETAIN", ext);
    r.owner_active = ext.owner_active && ext.owner_external; r.map_id = ext.mapping_id;
    if (!r.owner_active) {
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      "multi route owner reference not retained: status=0x%08x, external=%u, bus=0x%llx, length=0x%llx",
                      (unsigned)ext.rmStatus, ext.owner_external,
                      (unsigned long long)ext.requested_owner_bar1_bus_start,
                      (unsigned long long)ext.requested_resource_length);
        throw std::runtime_error(detail);
    }

    UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS dm{};
    dm.requester_uuid = requester; dm.owner_uuid = owner_uuid;
    dm.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_MAP_REQUESTER; dm.mapping_id = r.map_id;
    ioctl_log(fd, "MULTI_ROUTE_DMA_MAP", dm);
    r.dma_active = dm.dma_active; r.route_id = dm.route_id;
    if (dm.rmStatus != NV_OK || !r.dma_active || dm.dma_mapping_error ||
        (leaf_page_size == PAGE && !dm.dma_aligned_for_4k_leaf))
        throw std::runtime_error("multi route DMA mapping does not cover requested leaves");

    // For a host-bridge/IOMMU route, encode the DMA API's requester address.
    // For a verified same-switch route, encode the owner's PCI BAR bus
    // address directly: the switch must decode that address locally instead
    // of sending an IOVA to the IOMMU. Keep the DMA mapping alive in either
    // mode because it owns the existing route lifetime and cleanup contract.
    const NvU64 address_base = use_owner_bus_address ?
                               owner_mem.rmmap.addr + owner_window_offset :
                               dm.destination_gpu_sys_address;
    if (address_base < owner_mem.rmmap.addr ||
        address_base > ~NvU64(0) - retained_length)
        throw std::runtime_error("multi-route selected destination address overflows");
    const NvU64 alignment_delta = (leaf_page_size - ((address_base + owner_offset) & (leaf_page_size - 1))) & (leaf_page_size - 1);
    if (owner_offset > ~NvU64(0) - alignment_delta)
        throw std::runtime_error("multi-route aligned owner offset overflow");
    owner_offset += alignment_delta;
    // owner_offset is relative to the retained subwindow used by DMA. Keep
    // the allocation-relative value in the route record for local pointers.
    r.owner_offset = owner_window_offset + owner_offset;
    const NvU64 route_bytes = NvU64(page_count) * leaf_page_size;
    if (route_bytes > dm.requester_dma_length || owner_offset > dm.requester_dma_length - route_bytes ||
        owner_offset > retained_length || route_bytes > retained_length - owner_offset)
        throw std::runtime_error("multi route DMA mapping does not cover aligned leaves");

    const auto pdb = requester_pdb ? *requester_pdb : find_pdb(fd, requester);
    r.queries.clear(); r.active.clear(); r.queries.reserve(page_count); r.active.reserve(page_count);
    std::vector<NvU64> candidates; candidates.reserve(page_count);
    for (NvU32 i = 0; i < page_count; ++i) {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q{};
        if (saved) {
            q = saved->at(i);
            // A big-page leaf may be a valid, queryable but currently empty
            // slot (pte_after == 0). Preserve that exact zero for rollback.
            // The 4-KiB path still requires a present original mapping.
            if (q.target_va != requester_va + NvU64(i) * leaf_page_size || !q.target_pte_phys ||
                (!(q.pte_after & 1) && (leaf_page_size == PAGE || q.pte_after != 0)))
                throw std::runtime_error("saved leaf does not match multi-route requester VA");
        } else {
            q.requester_uuid = requester; q.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
            q.va_space_ptr = pdb.va_space_ptr; q.gpu_va_space_ptr = pdb.gpu_va_space_ptr;
            q.root_phys_address = pdb.root_phys_address; q.root_phys_aperture = pdb.root_phys_aperture;
            q.target_va = requester_va + NvU64(i) * leaf_page_size; q.page_size = leaf_page_size;
            ioctl_log(fd, "MULTI_ROUTE_PREQUERY", q);
            // Keep non-present big-page entries eligible; the guarded remap
            // records and restores their exact original zero value.
            if (!q.target_pte_phys ||
                (!(q.pte_after & 1) && (leaf_page_size == PAGE || q.pte_after != 0)))
                throw std::runtime_error("multi-route requester leaf missing");
        }
        q.owner_uuid = owner_uuid;
        const NvU64 destination = address_base + owner_offset + NvU64(i) * leaf_page_size;
        if (destination < address_base || (destination & (leaf_page_size - 1)))
            throw std::runtime_error("multi-route destination alignment/overflow failure");
        if (i == 0)
            std::fprintf(stderr,
                         "MULTI_ROUTE_PTE_ADDRESS,mode,%s,owner_bus_base,0x%llx,dma_address,0x%llx,owner_offset,0x%llx,destination,0x%llx,length,0x%llx\n",
                         use_owner_bus_address ? "owner_pci_bus" : "requester_dma",
                         (unsigned long long)address_base,
                         (unsigned long long)dm.destination_gpu_sys_address,
                         (unsigned long long)r.owner_offset,
                         (unsigned long long)destination,
                         (unsigned long long)route_bytes);
        std::fflush(stderr);

        if (leaf_page_size == PAGE) {
            // For present 4-KiB aliases, ask the active HAL for the exact
            // candidate and restore it before the guarded multi-PTE install.
            UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS a{};
            a.requester_uuid = requester; a.owner_uuid = owner_uuid;
            a.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_APPLY;
            a.mapping_id = r.map_id; a.route_id = r.route_id;
            a.va_space_ptr = q.va_space_ptr; a.gpu_va_space_ptr = q.gpu_va_space_ptr;
            a.root_phys_address = q.root_phys_address; a.root_phys_aperture = q.root_phys_aperture;
            a.target_va = q.target_va; a.page_size = leaf_page_size; a.target_pte_phys = q.target_pte_phys;
            a.expected_current_pte = q.pte_after; a.requested_destination_gpu_sys_address = destination;
            a.requested_allow_writable = allow_writable ? 1 : 0;
            const bool applied = ioctl_log(fd, "MULTI_ROUTE_HAL_CANDIDATE_APPLY", a);
            if (!applied || !a.remap_active || a.pte_after != a.temporary_pte)
                throw std::runtime_error("HAL candidate apply failed");
            UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS restore{};
            restore.requester_uuid = requester; restore.owner_uuid = owner_uuid;
            restore.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_RESTORE;
            restore.mapping_id = r.map_id; restore.route_id = r.route_id;
            restore.va_space_ptr = q.va_space_ptr; restore.gpu_va_space_ptr = q.gpu_va_space_ptr;
            restore.root_phys_address = q.root_phys_address; restore.root_phys_aperture = q.root_phys_aperture;
            restore.target_va = q.target_va; restore.page_size = leaf_page_size; restore.target_pte_phys = q.target_pte_phys;
            restore.expected_current_pte = a.temporary_pte;
            const bool restored = ioctl_log(fd, "MULTI_ROUTE_HAL_CANDIDATE_RESTORE", restore);
            if (!restored || restore.remap_active || restore.pte_after != q.pte_after)
                preserve("HAL candidate PTE did not return exactly to its original value");
            candidates.push_back(a.temporary_pte);
        }
        else {
            // The serial HAL helper refuses to synthesize a large-page PTE
            // when that leaf is currently non-present. Ask the same active
            // GM107 HAL to build the candidate for the first 4-KiB subpage,
            // restore that control PTE exactly, then let the guarded multi-PTE
            // transaction validate/install the candidate at the 128-KiB leaf.
            UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS q4 = q;
            q4.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE;
            q4.target_va = requester_va;
            q4.page_size = PAGE;
            q4.target_pte_phys = 0;
            if (!ioctl_log(fd, "MULTI_ROUTE_4K_CANDIDATE_PREQUERY", q4) ||
                !(q4.pte_after & 1ULL))
                throw std::runtime_error("4-KiB HAL candidate control leaf is not present");

            UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS a{};
            a.requester_uuid = requester; a.owner_uuid = owner_uuid;
            a.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_APPLY;
            a.mapping_id = r.map_id; a.route_id = r.route_id;
            a.va_space_ptr = q4.va_space_ptr; a.gpu_va_space_ptr = q4.gpu_va_space_ptr;
            a.root_phys_address = q4.root_phys_address; a.root_phys_aperture = q4.root_phys_aperture;
            a.target_va = q4.target_va; a.page_size = PAGE; a.target_pte_phys = q4.target_pte_phys;
            a.expected_current_pte = q4.pte_after;
            a.requested_destination_gpu_sys_address = destination;
            a.requested_allow_writable = allow_writable ? 1 : 0;
            if (!ioctl_log(fd, "MULTI_ROUTE_4K_HAL_CANDIDATE_APPLY", a) ||
                !a.remap_active || a.pte_after != a.temporary_pte)
                throw std::runtime_error("4-KiB HAL candidate generation failed");

            UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS restore{};
            restore.requester_uuid = requester; restore.owner_uuid = owner_uuid;
            restore.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_RESTORE;
            restore.mapping_id = r.map_id; restore.route_id = r.route_id;
            restore.va_space_ptr = q4.va_space_ptr; restore.gpu_va_space_ptr = q4.gpu_va_space_ptr;
            restore.root_phys_address = q4.root_phys_address; restore.root_phys_aperture = q4.root_phys_aperture;
            restore.target_va = q4.target_va; restore.page_size = PAGE;
            restore.target_pte_phys = q4.target_pte_phys;
            restore.expected_current_pte = a.temporary_pte;
            if (!ioctl_log(fd, "MULTI_ROUTE_4K_HAL_CANDIDATE_RESTORE", restore) ||
                restore.remap_active || restore.pte_after != q4.pte_after)
                preserve("4-KiB candidate PTE did not return exactly to its original value");
            candidates.push_back(a.temporary_pte);
        }
        r.queries.push_back(q);
    }

    for (NvU32 i = 0; i < page_count; ++i) {
        const auto &q = r.queries[i];
        UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
        p.gpu_uuid = requester; p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_APPLY;
        p.root_phys_aperture = q.root_phys_aperture; p.va_space_ptr = q.va_space_ptr;
        p.gpu_va_space_ptr = q.gpu_va_space_ptr; p.root_phys_address = q.root_phys_address;
        p.target_va = q.target_va; p.page_size = leaf_page_size; p.target_pte_phys = q.target_pte_phys;
        p.expected_current_pte = q.pte_after; p.requested_temporary_pte = candidates[i];
        p.requested_destination_gpu_sys_address = address_base + owner_offset + NvU64(i) * leaf_page_size;
        p.requested_allow_writable = allow_writable ? 1 : 0;
        errno = 0; const int rc = ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &p);
        std::printf("MULTI_ROUTE_APPLY,page,%u,rc,%d,errno,%d,status,0x%08x,id,0x%016" PRIx64
                    ",va,0x%016" PRIx64 ",original,0x%016" PRIx64 ",temporary,0x%016" PRIx64
                    ",after,0x%016" PRIx64 ",active,%u\n", i, rc, errno,
                    (unsigned)p.rmStatus, (uint64_t)p.mapping_id, (uint64_t)p.target_va,
                    (uint64_t)p.pte_before, (uint64_t)p.temporary_pte, (uint64_t)p.pte_after, p.active_count);
        std::fflush(stdout);
        if (rc == 0 && p.rmStatus == NV_OK) {
            r.active.push_back(p);
            q38_journal_line("PTE\t%llu\t%s\t%llu\t%llu\t%llu\t%llu\t%llu", (unsigned long long)p.mapping_id,
                             q38_uuid_hex(requester).c_str(), (unsigned long long)p.target_va,
                             (unsigned long long)p.target_pte_phys, (unsigned long long)p.temporary_pte,
                             (unsigned long long)p.pte_before, (unsigned long long)r.map_id);
        }
        if (rc || p.rmStatus != NV_OK || p.pte_after != candidates[i])
            throw std::runtime_error("guarded multi-PTE apply failed exact candidate check");
    }

    // In same-switch mode the installed SYS PTE contains the owner's PCI bus
    // address, not the DMA API's IOVA. The RTX BAR allocation remains owned by
    // QMem for the entire transaction, so release this per-requester DMA map
    // and temporary UVM owner reference now. This bounds RM route-state use
    // while the independent guarded PTE records stay installed until teardown.
    if (use_owner_bus_address) {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS un{};
        un.requester_uuid = r.requester; un.owner_uuid = r.owner;
        un.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_UNMAP_REQUESTER;
        un.mapping_id = r.map_id; un.route_id = r.route_id;
        if (!ioctl_log(fd, "MULTI_ROUTE_DMA_UNMAP_AFTER_PTE_INSTALL", un) || un.dma_active)
            throw std::runtime_error("same-switch DMA map release failed after PTE install");
        r.dma_active = false;

        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS rel{};
        rel.requester_uuid = r.requester; rel.owner_uuid = r.owner;
        rel.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_RELEASE_OWNER;
        rel.mapping_id = r.map_id;
        if (!ioctl_log(fd, "MULTI_ROUTE_OWNER_RELEASE_AFTER_PTE_INSTALL", rel) || rel.owner_active)
            throw std::runtime_error("same-switch owner reference release failed after PTE install");
        r.owner_active = false;
    }
}

static void multi_route_cleanup(MultiRoute &r, int fd) {
    if (!r.active.empty()) {
        if (!r.requester_quiesced) preserve("multi-route GPU work may remain outstanding");
        for (size_t i = r.active.size(); i > 0; --i) {
            const auto &entry = r.active[i - 1];
            UVM_GLM53_MULTI_PTE_REMAP_PARAMS p{};
            p.op = UVM_GLM53_MULTI_PTE_REMAP_OP_RESTORE;
            p.mapping_id = entry.mapping_id; p.target_va = entry.target_va; p.target_pte_phys = entry.target_pte_phys;
            errno = 0; const int rc = ioctl(fd, UVM_GLM53_MULTI_PTE_REMAP, &p);
            const auto &q = r.queries[i - 1];
            std::printf("MULTI_ROUTE_RESTORE,page,%zu,rc,%d,errno,%d,status,0x%08x,id,0x%016" PRIx64
                        ",before,0x%016" PRIx64 ",expected,0x%016" PRIx64 ",after,0x%016" PRIx64
                        ",restored,%u,active,%u\n", i - 1, rc, errno, (unsigned)p.rmStatus,
                        (uint64_t)p.mapping_id, (uint64_t)p.pte_before, (uint64_t)q.pte_after,
                        (uint64_t)p.pte_after, p.pte_restored, p.active_count);
            std::fflush(stdout);
            if (rc || p.rmStatus != NV_OK || !p.pte_restored || p.pte_after != q.pte_after)
                preserve("multi-route exact PTE restoration failed");
            q38_journal_line("PTE_RESTORED\t%llu", (unsigned long long)entry.mapping_id);
        }
        r.active.clear();
    }
    if (r.dma_active) {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS un{};
        un.requester_uuid = r.requester; un.owner_uuid = r.owner;
        un.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_UNMAP_REQUESTER;
        un.mapping_id = r.map_id; un.route_id = r.route_id;
        if (!ioctl_log(fd, "MULTI_ROUTE_DMA_UNMAP", un) || un.dma_active)
            preserve("multi-route DMA unmap failed");
        r.dma_active = false;
    }
    if (r.owner_active) {
        UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS rel{};
        rel.requester_uuid = r.requester; rel.owner_uuid = r.owner;
        rel.op = UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_RELEASE_OWNER; rel.mapping_id = r.map_id;
        if (!ioctl_log(fd, "MULTI_ROUTE_OWNER_RELEASE", rel) || rel.owner_active)
            preserve("multi-route owner release failed");
        r.owner_active = false;
    }
}

static std::vector<NvU32> production_payload(bool request, NvU32 seq, NvU32 bytes) {
    if ((bytes & 15) || bytes > (request ? PROD_REQUEST_BYTES : PROD_RESULT_BYTES))
        throw std::runtime_error("production payload size must be 16-byte aligned and in range");
    std::vector<NvU32> words(bytes / sizeof(NvU32));
    for (NvU32 i = 0; i < words.size(); ++i)
        words[i] = request ? request_word(seq, i) : response_word(seq, i);
    return words;
}

static void verify_owner_mailbox(QMem &owner, bool request, NvU32 seq,
                                 const std::vector<NvU32> &payload,
                                 const char *phase, NvU32 sample) {
    const size_t bytes = MAILBOX_HEADER_BYTES + payload.size() * sizeof(NvU32);
    std::vector<NvU32> actual(bytes / sizeof(NvU32));
    set_device_context(owner);
    RT_CHECK(cudaMemcpy(actual.data(), reinterpret_cast<const void *>(owner.va + TARGET_OFF),
                        bytes, cudaMemcpyDeviceToHost));
    if (actual[0] != seq)
        throw std::runtime_error(std::string(phase) + " sequence mismatch at sample " + std::to_string(sample));
    if (std::memcmp(actual.data() + MAILBOX_HEADER_BYTES / 4, payload.data(),
                    payload.size() * sizeof(NvU32)) != 0)
        throw std::runtime_error(std::string(phase) + " payload mismatch at sample " + std::to_string(sample));
    const NvU32 guard0 = *reinterpret_cast<volatile NvU32 *>(
        reinterpret_cast<char *>(owner.cpu) + TARGET_OFF - sizeof(NvU32));
    // Payload padding is intentionally initialized inside the mapped window.
    // The trailing guard therefore belongs at the mapping boundary, not at
    // the end of the shorter logical payload.
    const NvU32 mapped_payload_bytes = request ? REQUEST_BYTES : RESPONSE_BYTES;
    const NvU32 guard1 = *reinterpret_cast<volatile NvU32 *>(
        reinterpret_cast<char *>(owner.cpu) + TARGET_OFF + mapped_payload_bytes);
    if (guard0 != GUARD_BEFORE || guard1 != GUARD_AFTER) {
        std::fprintf(stderr,
                     "BAR1_GUARD_MISMATCH,phase,%s,sample,%u,guard0_addr,%p,guard0_expected,0x%08x,guard0_observed,0x%08x,guard1_addr,%p,guard1_expected,0x%08x,guard1_observed,0x%08x,seq_observed,0x%08x,payload_bytes,%zu\n",
                     phase, sample,
                     reinterpret_cast<void *>(reinterpret_cast<char *>(owner.cpu) + TARGET_OFF - 4),
                     GUARD_BEFORE, guard0,
                     reinterpret_cast<void *>(reinterpret_cast<char *>(owner.cpu) + TARGET_OFF + mapped_payload_bytes),
                     GUARD_AFTER, guard1, actual[0], payload.size() * sizeof(NvU32));
        std::fflush(stderr);
        throw std::runtime_error(std::string(phase) + " BAR1 guard changed at sample " + std::to_string(sample));
    }
}

static void write_cpu_bar1_mailbox(QMem &owner, NvU32 seq,
                                  const std::vector<NvU32> &payload) {
    auto *base = reinterpret_cast<volatile NvU32 *>(
        reinterpret_cast<char *>(owner.cpu) + TARGET_OFF);
    auto *dst = reinterpret_cast<__m128i *>(const_cast<NvU32 *>(base + MAILBOX_HEADER_BYTES / 4));
    const auto *src = reinterpret_cast<const char *>(payload.data());
    const size_t vectors = payload.size() * sizeof(NvU32) / 16;
    for (size_t i = 0; i < vectors; ++i) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i * 16));
        _mm_stream_si128(dst + i, v);
    }
    _mm_sfence();
    base[0] = seq; // publish completion/sequence only after payload stores drain
    _mm_sfence();
}

static void run_cpu_bar1_baseline(QMem &owner, const char *label, bool request,
                                  NvU32 bytes, NvU32 measured) {
    const NvU32 warmups = 5;
    for (NvU32 i = 0; i < warmups + measured; ++i) {
        const NvU32 seq = 0x41000000u + i;
        const auto payload = production_payload(request, seq, bytes);
        const auto t0 = std::chrono::steady_clock::now();
        write_cpu_bar1_mailbox(owner, seq, payload);
        const double elapsed_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - t0).count();
        verify_owner_mailbox(owner, request, seq, payload, label, i);
        std::printf("TX_SAMPLE,transport,cpu_bar1,phase,%s,direction,%s,seq,%u,payload_bytes,%u,control_bytes,4,store_sfence_us,%.3f,receiver_check,PASS\n",
                    i < warmups ? "warmup" : "measured", label, seq, bytes, elapsed_us);
    }
}

static void wait_bench_event(cudaEvent_t event, MultiRoute &route, const char *label) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        const cudaError_t status = cudaEventQuery(event);
        if (status == cudaSuccess) { route.requester_quiesced = true; return; }
        if (status != cudaErrorNotReady) {
            std::fprintf(stderr, "TX_EVENT_ERROR,%s,%s\n", label, cudaGetErrorString(status));
            preserve("transmission benchmark event reported an execution error; resources retained");
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            std::fprintf(stderr, "TX_WATCHDOG,%s,seconds,10; retaining PTE and backing resources\n", label);
            preserve("transmission benchmark completion watchdog expired");
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

static void run_gpu_direction_compare(int fd, MultiRoute &route, bool &route_active,
                                      QMem &requester, QMem &owner,
                                      const NvProcessorUuid &requester_uuid,
                                      const NvProcessorUuid &owner_uuid,
                                      NvU32 *requester_alias, const char *label,
                                      bool request, NvU32 bytes, NvU32 measured) {
    if (bytes + MAILBOX_HEADER_BYTES + TARGET_OFF > QWEN_MAP || (bytes & 15))
        throw std::runtime_error("production transfer exceeds retained mapping");
    set_device_context(requester);
    NvU32 *source = nullptr;
    NvU64 *clock_stamps = nullptr;
    cudaEvent_t start{}, stop{};
    RT_CHECK(cudaMalloc(&source, bytes));
    RT_CHECK(cudaMalloc(&clock_stamps, 5 * sizeof(NvU64)));
    RT_CHECK(cudaEventCreate(&start));
    RT_CHECK(cudaEventCreate(&stop));

    route_active = true; // cleanup handles partial acquisition too
    multi_route_setup(route, fd, requester_uuid, owner_uuid, owner,
                      (NvU64)requester_alias, TARGET_OFF);
    constexpr NvU32 warmups = 10;
    for (NvU32 i = 0; i < warmups + measured; ++i) {
        set_device_context(requester);
        const NvU32 seq = 0x52000000u + i;
        const auto payload = production_payload(request, seq, bytes);
        RT_CHECK(cudaMemcpyAsync(source, payload.data(), bytes, cudaMemcpyHostToDevice, requester.stream));
        RT_CHECK(cudaStreamSynchronize(requester.stream)); // source preparation excluded
        const auto run_variant = [&](int variant_id, const char *variant) {
            set_device_context(requester);
            const auto host_t0 = std::chrono::steady_clock::now();
            RT_CHECK(cudaEventRecord(start, requester.stream));
            route.requester_quiesced = false;
            if (variant_id == 1) {
                transmit_vec4_single_sender<<<1, 256, bytes, requester.stream>>>(
                    reinterpret_cast<const uint4 *>(source),
                    reinterpret_cast<volatile uint4 *>(reinterpret_cast<char *>(requester_alias) + MAILBOX_HEADER_BYTES),
                    bytes / sizeof(uint4), reinterpret_cast<volatile NvU32 *>(requester_alias),
                    seq, clock_stamps);
            } else if (variant_id == 2) {
                transmit_vec4_distributed_one_fence<<<1, 256, 0, requester.stream>>>(
                    reinterpret_cast<const uint4 *>(source),
                    reinterpret_cast<volatile uint4 *>(reinterpret_cast<char *>(requester_alias) + MAILBOX_HEADER_BYTES),
                    bytes / sizeof(uint4), reinterpret_cast<volatile NvU32 *>(requester_alias),
                    seq, clock_stamps);
            } else {
                transmit_vec4_with_sequence<<<1, 256, 0, requester.stream>>>(
                    reinterpret_cast<const uint4 *>(source),
                    reinterpret_cast<volatile uint4 *>(reinterpret_cast<char *>(requester_alias) + MAILBOX_HEADER_BYTES),
                    bytes / sizeof(uint4), reinterpret_cast<volatile NvU32 *>(requester_alias),
                    seq, clock_stamps);
            }
            RT_CHECK(cudaGetLastError());
            RT_CHECK(cudaEventRecord(stop, requester.stream));
            wait_bench_event(stop, route, label);
            const double host_submit_complete_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - host_t0).count();
            float event_ms = 0.0f;
            RT_CHECK(cudaEventElapsedTime(&event_ms, start, stop));
            NvU64 stamps[5]{};
#ifndef FENCE_HANDOFF_EVENT_ONLY
            RT_CHECK(cudaMemcpy(stamps, clock_stamps, sizeof(stamps), cudaMemcpyDeviceToHost));
#endif

            // Verify the complete destination payload after every variant.
            verify_owner_mailbox(owner, request, seq, payload, label, i);
            const double event_us = double(event_ms) * 1000.0;
            const double mb_s = event_us > 0.0 ? (double(bytes + 4) / event_us) : 0.0;
            const NvU64 local_cycles = variant_id == 1 ? stamps[1] - stamps[0] : 0;
            const NvU64 remote_store_cycles = variant_id == 1 ? stamps[2] - stamps[1] : stamps[1] - stamps[0];
            const NvU64 first_fence_cycles = variant_id == 1 ? stamps[3] - stamps[2] : stamps[2] - stamps[1];
            const NvU64 sequence_cycles = variant_id == 1 ? stamps[4] - stamps[3] : stamps[3] - stamps[2];
            const NvU64 total_cycles = variant_id == 1 ? stamps[4] - stamps[0] : stamps[3] - stamps[0];
            std::printf("TX_COMPARE_SAMPLE,phase,%s,direction,%s,variant,%s,seq,%u,payload_bytes,%u,control_bytes,4,event_us,%.3f,local_handoff_cycles,%llu,remote_store_cycles,%llu,first_fence_rendezvous_cycles,%llu,sequence_store_fence_cycles,%llu,total_publish_cycles,%llu,submit_to_completion_us,%.3f,payload_plus_control_MBps,%.3f,receiver_check,PASS\n",
                        i < warmups ? "warmup" : "measured", label, variant, seq, bytes, event_us,
                        (unsigned long long)local_cycles, (unsigned long long)remote_store_cycles,
                        (unsigned long long)first_fence_cycles, (unsigned long long)sequence_cycles,
                        (unsigned long long)total_cycles, host_submit_complete_us, mb_s);
            std::fflush(stdout);
        };

        // Every variant receives the same payload and sequence. Rotate order
        // to reduce systematic clock-state and warm-cache bias.
        switch (i % 3u) {
            case 0:
                run_variant(0, "distributed_256_all_threads_fence");
                run_variant(2, "distributed_256_one_thread_fence");
                run_variant(1, "staged_single_sender");
                break;
            case 1:
                run_variant(2, "distributed_256_one_thread_fence");
                run_variant(1, "staged_single_sender");
                run_variant(0, "distributed_256_all_threads_fence");
                break;
            default:
                run_variant(1, "staged_single_sender");
                run_variant(0, "distributed_256_all_threads_fence");
                run_variant(2, "distributed_256_one_thread_fence");
                break;
        }
    }
    multi_route_cleanup(route, fd);
    route_active = false;
    set_device_context(requester);
    RT_CHECK(cudaEventDestroy(start));
    RT_CHECK(cudaEventDestroy(stop));
    RT_CHECK(cudaFree(clock_stamps));
    RT_CHECK(cudaFree(source));
}

static void wait_event(cudaEvent_t event, Route &route, const char *name) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        cudaError_t q = cudaEventQuery(event);
        if (q == cudaSuccess) { route.requester_quiesced = true; return; }
        if (q != cudaErrorNotReady)
            throw std::runtime_error(std::string(name) + " event failed: " + cudaGetErrorString(q));
        if (std::chrono::steady_clock::now() >= deadline)
            preserve((std::string(name) + " completion watchdog expired").c_str());
        usleep(1000);
    }
}

static void wait_event(cudaEvent_t event, MultiRoute &route, const char *name) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    const auto spin_deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(150);
    for (;;) {
        const cudaError_t q = cudaEventQuery(event);
        if (q == cudaSuccess) { route.requester_quiesced = true; return; }
        if (q != cudaErrorNotReady)
            throw std::runtime_error(std::string(name) + " event failed: " + cudaGetErrorString(q));
        if (std::chrono::steady_clock::now() >= deadline)
            preserve((std::string(name) + " completion watchdog expired").c_str());
        // Avoid imposing a 1 ms scheduling quantum on stages that usually
        // finish in tens of microseconds. Back off briefly if delayed.
        if (std::chrono::steady_clock::now() >= spin_deadline)
            std::this_thread::sleep_for(std::chrono::microseconds(25));
    }
}

static void set_device_context(const QMem &m) {
    CUKQ(cuCtxSetCurrent(m.ctx)); RT_CHECK(cudaSetDevice(m.ordinal));
}

struct RetainedRoundTiming {
    NvU32 sequence;
    double host_total_us;
    double host_rtx_send_stage_us;
    double host_m10_compute_return_stage_us;
    double host_rtx_verify_stage_us;
    double rtx_send_us;
    double m10_compute_us;
    double m10_return_us;
    double rtx_verify_us;
};

static void run_retained_modelshape_roundtrip(
        int fd, MultiRoute &request_route, MultiRoute &response_route,
        bool &request_route_active, bool &response_route_active,
        QMem &rtx, QMem &m10, const NvProcessorUuid &rtx_uuid,
        const NvProcessorUuid &m10_uuid, NvU32 *rtx_host_base,
        NvU32 *m10_host_base,
        const std::vector<UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS> &m10_saved_ptes,
        NvU32 iterations)
{
    if (iterations == 0 || iterations > 10000)
        throw std::runtime_error("retained round-trip iterations must be in 1..10000");
    if ((RETAINED_BODY_BYTES % sizeof(uint4)) != 0 ||
        RETAINED_BODY_VECTORS != 767 || RETAINED_BODY_WORDS != 3068)
        throw std::runtime_error("retained production payload layout is not 16-byte aligned");

    const size_t request_bytes = size_t(iterations) * RETAINED_BODY_BYTES;
    const size_t history_bytes = size_t(iterations) * H_ELEMENTS * sizeof(float);
    std::vector<NvU32> request_host(size_t(iterations) * RETAINED_BODY_WORDS, RETAINED_PAD);
    for (NvU32 seq = 1; seq <= iterations; ++seq) {
        NvU32 *row = request_host.data() + size_t(seq - 1) * RETAINED_BODY_WORDS;
        for (NvU32 i = 0; i < PROD_REQUEST_BYTES / sizeof(NvU32); ++i)
            row[i] = request_word(seq, i);
    }

    uint4 *request_source = nullptr;
    float *result_history = nullptr;
    float *m10_result_local = nullptr;
    NvU32 *round_status = nullptr;
    NvU32 *m10_status = nullptr;
    cudaEvent_t send_start{}, send_stop{}, compute_start{}, compute_stop{};
    cudaEvent_t return_start{}, return_stop{}, verify_start{}, verify_stop{};
    std::vector<RetainedRoundTiming> timings;
    timings.reserve(iterations);
    bool routes_cleaned = false;

    // Both owner/BAR1 routes and both three-leaf requester mappings are
    // acquired before the first exchange and retained for the whole run.
    request_route_active = true;
    set_device_context(rtx);
    multi_route_setup(request_route, fd, rtx_uuid, m10_uuid, m10,
                      (NvU64)rtx_host_base, TARGET_OFF);
    response_route_active = true;
    set_device_context(m10);
    multi_route_setup(response_route, fd, m10_uuid, rtx_uuid, rtx,
                      (NvU64)m10_host_base, TARGET_OFF, &m10_saved_ptes);

    set_device_context(rtx);
    RT_CHECK(cudaMalloc(&request_source, request_bytes));
    RT_CHECK(cudaMalloc(&result_history, history_bytes));
    RT_CHECK(cudaMalloc(&round_status, sizeof(NvU32)));
    RT_CHECK(cudaMemcpy(request_source, request_host.data(), request_bytes,
                        cudaMemcpyHostToDevice));
    RT_CHECK(cudaMemset(result_history, 0, history_bytes));
    RT_CHECK(cudaMemset(round_status, 0xff, sizeof(NvU32)));
    RT_CHECK(cudaEventCreate(&send_start)); RT_CHECK(cudaEventCreate(&send_stop));
    RT_CHECK(cudaEventCreate(&verify_start)); RT_CHECK(cudaEventCreate(&verify_stop));
    set_device_context(m10);
    RT_CHECK(cudaMalloc(&m10_result_local, H_ELEMENTS * sizeof(float)));
    RT_CHECK(cudaMalloc(&m10_status, sizeof(NvU32)));
    RT_CHECK(cudaEventCreate(&compute_start)); RT_CHECK(cudaEventCreate(&compute_stop));
    RT_CHECK(cudaEventCreate(&return_start)); RT_CHECK(cudaEventCreate(&return_stop));

    auto *m10_header = reinterpret_cast<const volatile NvU32 *>(m10.va + TARGET_OFF);
    auto *m10_body = reinterpret_cast<const volatile NvU32 *>(m10.va + TARGET_OFF + MAILBOX_HEADER_BYTES);
    auto *request_destination = reinterpret_cast<volatile uint4 *>(
        reinterpret_cast<char *>(rtx_host_base) + MAILBOX_HEADER_BYTES);
    auto *request_header_destination = reinterpret_cast<volatile NvU32 *>(rtx_host_base);
    auto *response_header_destination = reinterpret_cast<volatile NvU32 *>(m10_host_base);
    auto *response_destination = reinterpret_cast<volatile uint4 *>(
        reinterpret_cast<char *>(m10_host_base) + MAILBOX_HEADER_BYTES);
    auto *rtx_response_header = reinterpret_cast<const volatile NvU32 *>(rtx.va + TARGET_OFF);
    auto *rtx_response_body = reinterpret_cast<const volatile NvU32 *>(
        rtx.va + TARGET_OFF + MAILBOX_HEADER_BYTES);

    std::puts("RETAINED_ROUNDTRIP_SETUP,request_route,RTX_to_M10,response_route,M10_to_RTX,route_rebuilds_in_loop,0,request_logical_bytes,10320,request_body_bytes,12272,result_logical_bytes,10240,result_body_bytes,12272");
    std::fflush(stdout);
    for (NvU32 seq = 1; seq <= iterations; ++seq) {
        const auto host_start = std::chrono::steady_clock::now();
        const uint4 *src = request_source + size_t(seq - 1) * RETAINED_BODY_VECTORS;

        const auto host_send_start = std::chrono::steady_clock::now();
        set_device_context(rtx);
        request_route.requester_quiesced = false;
        RT_CHECK(cudaEventRecord(send_start, rtx.stream));
        retained_request_send<<<1, 256, 0, rtx.stream>>>(
            src, request_destination, RETAINED_BODY_VECTORS,
            request_header_destination, seq, RETAINED_RUN_ID);
        RT_CHECK(cudaGetLastError());
        RT_CHECK(cudaEventRecord(send_stop, rtx.stream));
        wait_event(send_stop, request_route, "retained RTX request publication");
        float send_ms = 0.0f;
        RT_CHECK(cudaEventElapsedTime(&send_ms, send_start, send_stop));
        const double host_send_stage_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - host_send_start).count();

        const auto host_m10_start = std::chrono::steady_clock::now();
        set_device_context(m10);
        RT_CHECK(cudaEventRecord(compute_start, m10.stream));
        retained_m10_compute<<<1, 256, 0, m10.stream>>>(
            m10_header, m10_body, m10_result_local, seq, RETAINED_RUN_ID, m10_status);
        RT_CHECK(cudaGetLastError());
        RT_CHECK(cudaEventRecord(compute_stop, m10.stream));

        response_route.requester_quiesced = false;
        RT_CHECK(cudaEventRecord(return_start, m10.stream));
        retained_m10_return<<<1, 256, 0, m10.stream>>>(
            m10_result_local, response_destination, RETAINED_BODY_VECTORS,
            response_header_destination, seq, RETAINED_RUN_ID, m10_status);
        RT_CHECK(cudaGetLastError());
        RT_CHECK(cudaEventRecord(return_stop, m10.stream));
        wait_event(return_stop, response_route, "retained M10 result publication");
        float compute_ms = 0.0f, return_ms = 0.0f;
        RT_CHECK(cudaEventElapsedTime(&compute_ms, compute_start, compute_stop));
        RT_CHECK(cudaEventElapsedTime(&return_ms, return_start, return_stop));
        const double host_m10_stage_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - host_m10_start).count();

        const auto host_verify_start = std::chrono::steady_clock::now();
        set_device_context(rtx);
        RT_CHECK(cudaEventRecord(verify_start, rtx.stream));
        retained_verify_result<<<1, 256, 0, rtx.stream>>>(
            rtx_response_header, rtx_response_body,
            request_source + size_t(seq - 1) * RETAINED_BODY_VECTORS,
            seq, RETAINED_RUN_ID,
            result_history + size_t(seq - 1) * H_ELEMENTS, round_status);
        RT_CHECK(cudaGetLastError());
        RT_CHECK(cudaEventRecord(verify_stop, rtx.stream));
        wait_event(verify_stop, request_route, "retained RTX result verification");
        float verify_ms = 0.0f;
        RT_CHECK(cudaEventElapsedTime(&verify_ms, verify_start, verify_stop));
        NvU32 device_mismatches = 0xffffffffu;
        RT_CHECK(cudaMemcpy(&device_mismatches, round_status, sizeof(device_mismatches),
                            cudaMemcpyDeviceToHost));
        if (device_mismatches != 0)
            throw std::runtime_error("GPU-side result/header/padding mismatch at sequence " +
                                     std::to_string(seq) + ": " + std::to_string(device_mismatches));
        const double host_verify_stage_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - host_verify_start).count();

        const auto host_end = std::chrono::steady_clock::now();
        const double host_total_us = std::chrono::duration<double, std::micro>(host_end - host_start).count();
        timings.push_back({seq, host_total_us, host_send_stage_us, host_m10_stage_us,
                           host_verify_stage_us, double(send_ms) * 1000.0,
                           double(compute_ms) * 1000.0, double(return_ms) * 1000.0,
                           double(verify_ms) * 1000.0});
        if (seq <= 3 || seq == iterations || seq % 1000 == 0) {
            std::printf("RETAINED_ROUND,seq,%u,device_mismatches,0,host_total_us,%.3f,host_send_stage_us,%.3f,host_m10_stage_us,%.3f,host_verify_stage_us,%.3f,rtx_send_kernel_us,%.3f,m10_compute_kernel_us,%.3f,m10_return_kernel_us,%.3f,rtx_verify_kernel_us,%.3f\n",
                        seq, host_total_us, host_send_stage_us, host_m10_stage_us,
                        host_verify_stage_us, double(send_ms) * 1000.0,
                        double(compute_ms) * 1000.0, double(return_ms) * 1000.0,
                        double(verify_ms) * 1000.0);
        }
    }

    // Independent full CPU verification of every returned element and both
    // owner-side BAR mappings after all exchanges have completed.
    std::vector<float> observed_history(size_t(iterations) * H_ELEMENTS);
    set_device_context(rtx);
    RT_CHECK(cudaMemcpy(observed_history.data(), result_history, history_bytes,
                        cudaMemcpyDeviceToHost));
    NvU64 cpu_mismatches = 0;
    NvU32 first_bad_seq = 0, first_bad_index = 0;
    float max_abs_error = 0.0f;
    for (NvU32 seq = 1; seq <= iterations; ++seq) {
        const float bias = request_bias(seq);
        for (NvU32 i = 0; i < H_ELEMENTS; ++i) {
            const float expected = request_h(seq, i) + bias;
            const float observed = observed_history[size_t(seq - 1) * H_ELEMENTS + i];
            const float error = std::abs(observed - expected);
            max_abs_error = std::max(max_abs_error, error);
            if (f32_bits(observed) != f32_bits(expected)) {
                if (cpu_mismatches == 0) { first_bad_seq = seq; first_bad_index = i; }
                ++cpu_mismatches;
            }
        }
    }
    if (cpu_mismatches != 0)
        throw std::runtime_error("independent CPU full-result comparison failed, mismatches=" +
            std::to_string(cpu_mismatches) + ",first_seq=" + std::to_string(first_bad_seq) +
            ",first_index=" + std::to_string(first_bad_index));

    const NvU32 last_seq = iterations;
    auto *owner_request = reinterpret_cast<volatile NvU32 *>(
        reinterpret_cast<char *>(m10.cpu) + TARGET_OFF);
    auto *owner_response = reinterpret_cast<volatile NvU32 *>(
        reinterpret_cast<char *>(rtx.cpu) + TARGET_OFF);
    const NvU32 *last_request = request_host.data() + size_t(iterations - 1) * RETAINED_BODY_WORDS;
    NvU32 bar_request_bad = (owner_request[0] != last_seq ||
                              owner_request[1] != RETAINED_RUN_ID ||
                              owner_request[2] != PROD_REQUEST_BYTES ||
                              owner_request[3] != 0x52514553u) ? 1u : 0u;
    for (NvU32 i = 0; i < RETAINED_BODY_WORDS; ++i)
        if (owner_request[4 + i] != last_request[i]) ++bar_request_bad;
    NvU32 bar_response_bad = (owner_response[0] != last_seq ||
                               owner_response[1] != RETAINED_RUN_ID ||
                               owner_response[2] != 0 ||
                               owner_response[3] != PROD_RESULT_BYTES) ? 1u : 0u;
    for (NvU32 i = 0; i < H_ELEMENTS; ++i)
        if (owner_response[4 + i] != f32_bits(request_h(last_seq, i) + request_bias(last_seq)))
            ++bar_response_bad;
    for (NvU32 i = H_ELEMENTS; i < RETAINED_BODY_WORDS; ++i)
        if (owner_response[4 + i] != RETAINED_PAD) ++bar_response_bad;
    if (bar_request_bad || bar_response_bad)
        throw std::runtime_error("CPU BAR1 owner verification mismatch: request=" +
            std::to_string(bar_request_bad) + ",response=" + std::to_string(bar_response_bad));

    // A normal path reaches this only after all requester work has completed.
    multi_route_cleanup(request_route, fd); request_route_active = false;
    multi_route_cleanup(response_route, fd); response_route_active = false;
    routes_cleaned = true;
    std::vector<double> host_values, host_send_values, host_m10_values, host_verify_values;
    std::vector<double> send_values, compute_values, return_values, verify_values;
    for (const auto &t : timings) {
        host_values.push_back(t.host_total_us);
        host_send_values.push_back(t.host_rtx_send_stage_us);
        host_m10_values.push_back(t.host_m10_compute_return_stage_us);
        host_verify_values.push_back(t.host_rtx_verify_stage_us);
        send_values.push_back(t.rtx_send_us);
        compute_values.push_back(t.m10_compute_us); return_values.push_back(t.m10_return_us);
        verify_values.push_back(t.rtx_verify_us);
    }
    auto mean = [](const std::vector<double> &v) {
        double sum = 0.0; for (double x : v) sum += x; return v.empty() ? 0.0 : sum / v.size();
    };
    auto percentile = [](std::vector<double> v, double p) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[std::min(v.size() - 1, size_t(std::ceil(p * v.size())) - 1)];
    };
    std::ofstream csv("retained-roundtrip-timing.csv", std::ios::trunc);
    if (!csv) throw std::runtime_error("cannot create retained-roundtrip-timing.csv");
    csv << "sequence,host_total_us,host_rtx_send_stage_us,host_m10_compute_return_stage_us,host_rtx_verify_stage_us,host_unassigned_us,rtx_send_kernel_us,m10_compute_kernel_us,m10_return_kernel_us,rtx_verify_kernel_us\n";
    for (const auto &t : timings) {
        const double assigned = t.host_rtx_send_stage_us +
            t.host_m10_compute_return_stage_us + t.host_rtx_verify_stage_us;
        csv << t.sequence << ',' << t.host_total_us << ',' << t.host_rtx_send_stage_us << ','
            << t.host_m10_compute_return_stage_us << ',' << t.host_rtx_verify_stage_us << ','
            << (t.host_total_us - assigned) << ',' << t.rtx_send_us << ','
            << t.m10_compute_us << ',' << t.m10_return_us << ',' << t.rtx_verify_us << '\n';
    }
    csv.close();
    if (!csv) throw std::runtime_error("failed writing retained-roundtrip-timing.csv");
    std::printf("RETAINED_ROUNDTRIP_RESULT,status,PASS,rounds,%u,logical_request_bytes,%u,request_written_bytes,%u,logical_result_bytes,%u,result_written_bytes,%u,full_cpu_mismatches,%llu,max_abs_error,%.9g,cpu_bar_request_mismatches,%u,cpu_bar_response_mismatches,%u,route_rebuilds,0\n",
                iterations, PROD_REQUEST_BYTES, RETAINED_BODY_BYTES + MAILBOX_HEADER_BYTES,
                PROD_RESULT_BYTES, RETAINED_BODY_BYTES + MAILBOX_HEADER_BYTES,
                (unsigned long long)cpu_mismatches, max_abs_error, bar_request_bad, bar_response_bad);
    std::printf("RETAINED_TIMING,phase,host_checked_round,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(host_values), percentile(host_values, .50), percentile(host_values, .95),
                percentile(host_values, .99), host_values.empty() ? 0.0 : *std::max_element(host_values.begin(), host_values.end()));
    std::printf("RETAINED_TIMING,phase,host_RTX_send_stage,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(host_send_values), percentile(host_send_values, .50), percentile(host_send_values, .95),
                percentile(host_send_values, .99), host_send_values.empty() ? 0.0 : *std::max_element(host_send_values.begin(), host_send_values.end()));
    std::printf("RETAINED_TIMING,phase,host_M10_compute_return_stage,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(host_m10_values), percentile(host_m10_values, .50), percentile(host_m10_values, .95),
                percentile(host_m10_values, .99), host_m10_values.empty() ? 0.0 : *std::max_element(host_m10_values.begin(), host_m10_values.end()));
    std::printf("RETAINED_TIMING,phase,host_RTX_verify_stage,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(host_verify_values), percentile(host_verify_values, .50), percentile(host_verify_values, .95),
                percentile(host_verify_values, .99), host_verify_values.empty() ? 0.0 : *std::max_element(host_verify_values.begin(), host_verify_values.end()));
    std::printf("RETAINED_TIMING,phase,host_unassigned_remainder,mean_us,%.3f\n",
                mean(host_values) - mean(host_send_values) - mean(host_m10_values) - mean(host_verify_values));
    std::printf("RETAINED_TIMING,phase,RTX_send_kernel,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(send_values), percentile(send_values, .50), percentile(send_values, .95),
                percentile(send_values, .99), send_values.empty() ? 0.0 : *std::max_element(send_values.begin(), send_values.end()));
    std::printf("RETAINED_TIMING,phase,M10_compute_kernel,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(compute_values), percentile(compute_values, .50), percentile(compute_values, .95),
                percentile(compute_values, .99), compute_values.empty() ? 0.0 : *std::max_element(compute_values.begin(), compute_values.end()));
    std::printf("RETAINED_TIMING,phase,M10_return_kernel,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(return_values), percentile(return_values, .50), percentile(return_values, .95),
                percentile(return_values, .99), return_values.empty() ? 0.0 : *std::max_element(return_values.begin(), return_values.end()));
    std::printf("RETAINED_TIMING,phase,RTX_verify_kernel,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                mean(verify_values), percentile(verify_values, .50), percentile(verify_values, .95),
                percentile(verify_values, .99), verify_values.empty() ? 0.0 : *std::max_element(verify_values.begin(), verify_values.end()));
    std::fflush(stdout);

    set_device_context(rtx);
    RT_CHECK(cudaEventDestroy(send_start)); RT_CHECK(cudaEventDestroy(send_stop));
    RT_CHECK(cudaEventDestroy(verify_start)); RT_CHECK(cudaEventDestroy(verify_stop));
    RT_CHECK(cudaFree(request_source)); RT_CHECK(cudaFree(result_history)); RT_CHECK(cudaFree(round_status));
    set_device_context(m10);
    RT_CHECK(cudaEventDestroy(compute_start)); RT_CHECK(cudaEventDestroy(compute_stop));
    RT_CHECK(cudaEventDestroy(return_start)); RT_CHECK(cudaEventDestroy(return_stop));
    RT_CHECK(cudaFree(m10_result_local)); RT_CHECK(cudaFree(m10_status));
    (void)routes_cleaned;
}

struct QwenFixtureRows {
    NvU32 rows = 0;
    std::vector<float> h, weights, expected;
    std::vector<NvS32> ids;
};
static QwenFixtureRows load_qwen_fixture(const std::string &request_path, const std::string &partial_path) {
    std::ifstream f(request_path, std::ios::binary); QwenFixtureRows r;
    f.read(reinterpret_cast<char *>(&r.rows), sizeof(r.rows));
    if (!f || !r.rows || r.rows > 1000) throw std::runtime_error("invalid Qwen request fixture row count");
    r.h.resize(size_t(r.rows) * QWEN_H); r.ids.resize(size_t(r.rows) * QWEN_K); r.weights.resize(size_t(r.rows) * QWEN_K);
    for (NvU32 i = 0; i < r.rows; ++i) {
        f.read(reinterpret_cast<char *>(r.h.data() + size_t(i) * QWEN_H), QWEN_H * sizeof(float));
        f.read(reinterpret_cast<char *>(r.ids.data() + size_t(i) * QWEN_K), QWEN_K * sizeof(NvS32));
        f.read(reinterpret_cast<char *>(r.weights.data() + size_t(i) * QWEN_K), QWEN_K * sizeof(float));
    }
    if (!f || f.peek() != EOF) throw std::runtime_error("truncated or trailing Qwen request fixture data");
    std::ifstream e(partial_path, std::ios::binary | std::ios::ate);
    const size_t expected_count = size_t(r.rows) * 20 * QWEN_H;
    if (!e || size_t(e.tellg()) != expected_count * sizeof(float))
        throw std::runtime_error("Qwen partial oracle size does not match [rows,20,2560]");
    r.expected.resize(expected_count); e.seekg(0); e.read(reinterpret_cast<char *>(r.expected.data()), expected_count * sizeof(float));
    if (!e) throw std::runtime_error("Qwen partial oracle read failed");
    return r;
}

static void run_qwen_local_compute_check(QMem &m10, const std::string &bundle,
                                         const std::string &request_file,
                                         const std::string &partial_file) {
    const auto fixture = load_qwen_fixture(request_file, partial_file);
    const size_t expert_bytes = size_t(QWEN_E) * QWEN_W * QWEN_H;
    auto gate = pack_qwen_gate(read_blob(bundle + "/gate.u8", expert_bytes));
    auto up = pack_qwen_gate(read_blob(bundle + "/up.u8", expert_bytes));
    auto down = pack_qwen_down(read_blob(bundle + "/down.u8", expert_bytes));
    auto gate_scale = read_f32_blob(bundle + "/gate_scale.f32", size_t(QWEN_E) * QWEN_W * (QWEN_H / QWEN_GROUP));
    auto up_scale = read_f32_blob(bundle + "/up_scale.f32", size_t(QWEN_E) * QWEN_W * (QWEN_H / QWEN_GROUP));
    auto down_scale = read_f32_blob(bundle + "/down_scale.f32", size_t(QWEN_E) * QWEN_H);

    set_device_context(m10);
    QwenSliceWeights dw{};
    dw.gate = device_upload(gate, m10.ordinal); dw.up = device_upload(up, m10.ordinal); dw.down = device_upload(down, m10.ordinal);
    dw.gate_scale = device_upload(gate_scale, m10.ordinal); dw.up_scale = device_upload(up_scale, m10.ordinal); dw.down_scale = device_upload(down_scale, m10.ordinal);

    std::vector<NvU32> body(PROD_REQUEST_BYTES / sizeof(NvU32), RETAINED_PAD);
    std::memcpy(body.data(), fixture.h.data(), QWEN_H * sizeof(float));
    std::memcpy(body.data() + QWEN_H, fixture.ids.data(), QWEN_K * sizeof(NvS32));
    std::memcpy(body.data() + QWEN_H + QWEN_K, fixture.weights.data(), QWEN_K * sizeof(float));
    const std::vector<NvU32> header{1u, RETAINED_RUN_ID, PROD_REQUEST_BYTES, 0x52514553u};
    NvU32 *d_header = device_upload(header, m10.ordinal);
    NvU32 *d_body = device_upload(body, m10.ordinal);
    float *d_act = nullptr, *d_out = nullptr;
    NvU32 *d_status = nullptr;
    RT_CHECK(cudaMalloc(&d_act, QWEN_K * QWEN_W * sizeof(float)));
    RT_CHECK(cudaMalloc(&d_out, QWEN_H * sizeof(float)));
    RT_CHECK(cudaMalloc(&d_status, sizeof(NvU32)));
    QwenSliceParams params{dw, nullptr, nullptr, nullptr, d_act, d_out};

    std::printf("QWEN_LOCAL_COMPUTE,cuda_ordinal,%d,weights_bytes,%zu,row,0\n", m10.ordinal,
                (gate.size() + up.size() + down.size()) * sizeof(NvU32) +
                (gate_scale.size() + up_scale.size() + down_scale.size()) * sizeof(float));
    qwen_gate_slice<<<25, 256, 0, m10.stream>>>(params,
        reinterpret_cast<const volatile NvU32 *>(d_header),
        reinterpret_cast<const volatile NvU32 *>(d_body), 1u, RETAINED_RUN_ID, d_status);
    RT_CHECK(cudaGetLastError());
    RT_CHECK(cudaStreamSynchronize(m10.stream));
    NvU32 status = ~0u;
    RT_CHECK(cudaMemcpy(&status, d_status, sizeof(status), cudaMemcpyDeviceToHost));
    std::printf("QWEN_LOCAL_STAGE,gate,PASS,status,%u\n", status);
    if (status != 0) throw std::runtime_error("local Qwen gate stage returned nonzero status");

    qwen_down_slice<<<25, 256, 0, m10.stream>>>(params,
        reinterpret_cast<const volatile NvU32 *>(d_body), d_status);
    RT_CHECK(cudaGetLastError());
    RT_CHECK(cudaStreamSynchronize(m10.stream));
    std::vector<float> observed(QWEN_H);
    RT_CHECK(cudaMemcpy(observed.data(), d_out, QWEN_H * sizeof(float), cudaMemcpyDeviceToHost));
    const float *expected = fixture.expected.data();
    NvU64 mismatches = 0; float max_abs_error = 0.0f;
    for (NvU32 i = 0; i < QWEN_H; ++i) {
        const float err = std::abs(observed[i] - expected[i]);
        max_abs_error = std::max(max_abs_error, err);
        if (!(err <= 2.0e-6f + 2.0e-4f * std::abs(expected[i]))) ++mismatches;
    }
    std::printf("QWEN_LOCAL_STAGE,down,%s,mismatches,%llu,max_abs_error,%.9g\n",
                mismatches ? "FAIL" : "PASS", (unsigned long long)mismatches, max_abs_error);
    if (mismatches) throw std::runtime_error("local Qwen slice differs from CPU oracle");

    for (void *p : {(void *)dw.gate, (void *)dw.up, (void *)dw.down,
                    (void *)dw.gate_scale, (void *)dw.up_scale, (void *)dw.down_scale,
                    (void *)d_header, (void *)d_body, (void *)d_act, (void *)d_out, (void *)d_status})
        RT_CHECK(cudaFree(p));
}

static void run_retained_qwen_expert_roundtrip(
        int fd, MultiRoute &request_route, MultiRoute &response_route,
        bool &request_active, bool &response_active, QMem &rtx, QMem &m10,
        const NvProcessorUuid &rtx_uuid, const NvProcessorUuid &m10_uuid,
        NvU32 *rtx_host_base, NvU32 *m10_host_base,
        const std::vector<UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS> &m10_saved_ptes,
        NvU32 iterations, NvU32 warmups, const std::string &bundle, const std::string &request_file,
        const std::string &partial_file) {
    if (!iterations || iterations + warmups > 10000)
        throw std::runtime_error("expert measured rounds must be nonzero and warmup+measured rounds <= 10000");
    const auto fixture = load_qwen_fixture(request_file, partial_file);
    std::printf("QWEN_EXPERT_FIXTURE,layer,0,rows,%u,worker_index,0,warmups,%u,measured_iterations,%u,request_logical_bytes,%u,output_bytes,%u\n",
                fixture.rows, warmups, iterations, REQUEST_LOGICAL_BYTES, QWEN_H * 4);

    const size_t expert_bytes = size_t(QWEN_E) * QWEN_W * QWEN_H;
    auto gate_raw = read_blob(bundle + "/gate.u8", expert_bytes);
    auto up_raw = read_blob(bundle + "/up.u8", expert_bytes);
    auto down_raw = read_blob(bundle + "/down.u8", expert_bytes);
    auto gate = pack_qwen_gate(gate_raw), up = pack_qwen_gate(up_raw), down = pack_qwen_down(down_raw);
    auto gate_scale = read_f32_blob(bundle + "/gate_scale.f32", size_t(QWEN_E) * QWEN_W * (QWEN_H / QWEN_GROUP));
    auto up_scale = read_f32_blob(bundle + "/up_scale.f32", size_t(QWEN_E) * QWEN_W * (QWEN_H / QWEN_GROUP));
    auto down_scale = read_f32_blob(bundle + "/down_scale.f32", size_t(QWEN_E) * QWEN_H);
    set_device_context(m10);
    QwenSliceWeights dw{};
    dw.gate = device_upload(gate, m10.ordinal); dw.up = device_upload(up, m10.ordinal); dw.down = device_upload(down, m10.ordinal);
    dw.gate_scale = device_upload(gate_scale, m10.ordinal); dw.up_scale = device_upload(up_scale, m10.ordinal); dw.down_scale = device_upload(down_scale, m10.ordinal);
    const size_t resident_weight_bytes = (gate.size() + up.size() + down.size()) * sizeof(NvU32) +
        (gate_scale.size() + up_scale.size() + down_scale.size()) * sizeof(float);
    std::printf("QWEN_EXPERT_WEIGHTS_READY,ordinal,%d,expert_count,512,slice_width,32,bytes,%zu,load_once,YES\n", m10.ordinal, resident_weight_bytes);

    set_device_context(m10);
    float *d_h = nullptr, *d_routes = nullptr, *d_act = nullptr, *d_result = nullptr;
    NvU32 *d_status = nullptr;
    RT_CHECK(cudaMalloc(&d_h, QWEN_H * sizeof(float)));
    RT_CHECK(cudaMalloc(&d_routes, QWEN_K * sizeof(float)));
    RT_CHECK(cudaMalloc(&d_act, QWEN_K * QWEN_W * sizeof(float)));
    RT_CHECK(cudaMalloc(&d_result, QWEN_H * sizeof(float)));
    RT_CHECK(cudaMalloc(&d_status, sizeof(NvU32)));
    QwenSliceParams params{dw, d_h, d_routes, nullptr, d_act, d_result};

    const NvU32 request_vectors = PROD_REQUEST_BYTES / sizeof(uint4);
    const size_t per_request_bytes = request_vectors * sizeof(uint4);
    std::vector<NvU32> packed_requests(size_t(fixture.rows) * (PROD_REQUEST_BYTES / sizeof(NvU32)), RETAINED_PAD);
    for (NvU32 row = 0; row < fixture.rows; ++row) {
        NvU32 *dst = packed_requests.data() + size_t(row) * (PROD_REQUEST_BYTES / sizeof(NvU32));
        std::memcpy(dst, fixture.h.data() + size_t(row) * QWEN_H, QWEN_H * sizeof(float));
        std::memcpy(dst + QWEN_H, fixture.ids.data() + size_t(row) * QWEN_K, QWEN_K * sizeof(NvS32));
        std::memcpy(dst + QWEN_H + QWEN_K, fixture.weights.data() + size_t(row) * QWEN_K, QWEN_K * sizeof(float));
    }
    set_device_context(rtx);
    uint4 *d_requests = nullptr; NvU32 *d_return_copy = nullptr;
    float *d_return_values = nullptr;
    RT_CHECK(cudaMalloc(&d_requests, size_t(fixture.rows) * per_request_bytes));
    RT_CHECK(cudaMemcpy(d_requests, packed_requests.data(), size_t(fixture.rows) * per_request_bytes, cudaMemcpyHostToDevice));
    RT_CHECK(cudaMalloc(&d_return_copy, (RETAINED_BODY_WORDS + 4) * sizeof(NvU32)));
    RT_CHECK(cudaMalloc(&d_return_values, QWEN_H * sizeof(float)));

    request_active = true;
    multi_route_setup(request_route, fd, rtx_uuid, m10_uuid, m10,
                      (NvU64)rtx_host_base, TARGET_OFF);
    response_active = true;
    set_device_context(m10);
    multi_route_setup(response_route, fd, m10_uuid, rtx_uuid, rtx,
                      (NvU64)m10_host_base, TARGET_OFF, &m10_saved_ptes);

    auto *request_dest = reinterpret_cast<volatile uint4 *>(reinterpret_cast<char *>(rtx_host_base) + MAILBOX_HEADER_BYTES);
    auto *request_header = reinterpret_cast<volatile NvU32 *>(rtx_host_base);
    auto *m10_header = reinterpret_cast<const volatile NvU32 *>(m10.va + TARGET_OFF);
    auto *m10_body = reinterpret_cast<const volatile NvU32 *>(m10.va + TARGET_OFF + MAILBOX_HEADER_BYTES);
    auto *response_dest = reinterpret_cast<volatile uint4 *>(reinterpret_cast<char *>(m10_host_base) + MAILBOX_HEADER_BYTES);
    auto *response_header = reinterpret_cast<volatile NvU32 *>(m10_host_base);
    auto *rtx_response = reinterpret_cast<const volatile NvU32 *>(rtx.va + TARGET_OFF);
    auto *rtx_response_body = rtx_response + MAILBOX_HEADER_BYTES / 4;

    cudaEvent_t request_event{}, response_event{};
    set_device_context(rtx); RT_CHECK(cudaEventCreate(&request_event));
    set_device_context(m10); RT_CHECK(cudaEventCreate(&response_event));
    struct PhaseTiming { NvU32 seq, fixture_row; double request_us, compute_return_us, receive_validate_us, total_us; };
    std::vector<PhaseTiming> timings; timings.reserve(iterations);
    std::vector<float> observed(QWEN_H);
    NvU64 total_mismatches = 0; float max_abs_error = 0.0f;
    for (NvU32 seq = 1; seq <= warmups + iterations; ++seq) {
        const NvU32 row = (seq - 1) % fixture.rows;
        const auto t0 = std::chrono::steady_clock::now();
        const auto request_t0 = std::chrono::steady_clock::now();
        set_device_context(rtx); request_route.requester_quiesced = false;
        retained_request_send<<<1, 256, 0, rtx.stream>>>(d_requests + size_t(row) * (per_request_bytes / sizeof(uint4)),
            request_dest, request_vectors, request_header, seq, RETAINED_RUN_ID);
        RT_CHECK(cudaGetLastError()); RT_CHECK(cudaEventRecord(request_event, rtx.stream));
        wait_event(request_event, request_route, "RTX Qwen request publication");
        const double request_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - request_t0).count();

        const auto compute_t0 = std::chrono::steady_clock::now();
        set_device_context(m10);
        // The request body resides in the retained M10 BAR1 allocation; the gate kernel reads it directly.
        qwen_gate_slice<<<25, 256, 0, m10.stream>>>(params, m10_header, m10_body, seq, RETAINED_RUN_ID, d_status);
        RT_CHECK(cudaGetLastError());
        qwen_down_slice<<<25, 256, 0, m10.stream>>>(params, m10_body, d_status);
        RT_CHECK(cudaGetLastError());
        response_route.requester_quiesced = false;
        retained_m10_return<<<1, 256, 0, m10.stream>>>(d_result, response_dest,
            RETAINED_BODY_VECTORS, response_header, seq, RETAINED_RUN_ID, d_status);
        RT_CHECK(cudaGetLastError()); RT_CHECK(cudaEventRecord(response_event, m10.stream));
        wait_event(response_event, response_route, "M10 Qwen slice return");
        const double compute_return_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - compute_t0).count();

        const auto verify_t0 = std::chrono::steady_clock::now();
        set_device_context(rtx);
        copy_peer_words<<<12, 256, 0, rtx.stream>>>(rtx_response, d_return_copy, RETAINED_BODY_WORDS + 4);
        RT_CHECK(cudaGetLastError()); RT_CHECK(cudaStreamSynchronize(rtx.stream));
        NvU32 hdr[4]{};
        RT_CHECK(cudaMemcpy(hdr, d_return_copy, sizeof(hdr), cudaMemcpyDeviceToHost));
        RT_CHECK(cudaMemcpy(d_return_values, d_return_copy + 4, QWEN_H * sizeof(float), cudaMemcpyDeviceToDevice));
        RT_CHECK(cudaMemcpy(observed.data(), d_return_values, QWEN_H * sizeof(float), cudaMemcpyDeviceToHost));
        if (hdr[0] != seq || hdr[1] != RETAINED_RUN_ID || hdr[2] != 0 || hdr[3] != PROD_RESULT_BYTES)
            throw std::runtime_error("Qwen return header/status mismatch at seq " + std::to_string(seq));
        NvU64 mismatches = 0; float round_max = 0.0f;
        const float *expected = fixture.expected.data() + (size_t(row) * 20) * QWEN_H;
        for (NvU32 i = 0; i < QWEN_H; ++i) {
            const float err = std::abs(observed[i] - expected[i]); round_max = std::max(round_max, err);
            if (!(err <= 2.0e-6f + 2.0e-4f * std::abs(expected[i]))) ++mismatches;
        }
        total_mismatches += mismatches; max_abs_error = std::max(max_abs_error, round_max);
        if (mismatches) throw std::runtime_error("Qwen partial mismatch at seq " + std::to_string(seq) +
                                                 ", row " + std::to_string(row) + ", count " + std::to_string(mismatches));
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        const double verify_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - verify_t0).count();
        const NvU32 measured_round = seq > warmups ? seq - warmups : 0;
        if (measured_round) timings.push_back({seq, row, request_us, compute_return_us, verify_us, elapsed});
        if (seq <= warmups && (seq == 1 || seq == warmups))
            std::printf("QWEN_EXPERT_WARMUP,seq,%u,row,%u,status,PASS,mismatches,%llu,max_abs,%.9g\n",
                        seq, row, (unsigned long long)mismatches, round_max);
        if (measured_round && (measured_round <= 3 || measured_round == iterations || measured_round % 100 == 0))
            std::printf("QWEN_EXPERT_ROUND,seq,%u,row,%u,status,PASS,mismatches,%llu,max_abs,%.9g,request_publish_us,%.3f,compute_return_us,%.3f,receive_validate_us,%.3f,end_to_end_us,%.3f\n",
                        seq, row, (unsigned long long)mismatches, round_max, request_us, compute_return_us, verify_us, elapsed);
    }
    auto phase_stats = [&](auto value, const char *name) {
        std::vector<double> v; v.reserve(timings.size()); for (const auto &t : timings) v.push_back(value(t));
        std::sort(v.begin(), v.end()); const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        std::printf("QWEN_EXPERT_TIMING,phase,%s,mean_us,%.3f,median_us,%.3f,p95_us,%.3f,p99_us,%.3f,max_us,%.3f\n",
                    name, mean, v[v.size()/2], v[(v.size()*95)/100], v[(v.size()*99)/100], v.back());
    };
    phase_stats([](const PhaseTiming &t){return t.request_us;}, "RTX_request_publish");
    phase_stats([](const PhaseTiming &t){return t.compute_return_us;}, "M10_expert_compute_and_return");
    phase_stats([](const PhaseTiming &t){return t.receive_validate_us;}, "RTX_receive_and_full_CPU_compare");
    phase_stats([](const PhaseTiming &t){return t.total_us;}, "end_to_end_checked_round");
    std::ofstream csv("qwen-expert-roundtrip.csv", std::ios::trunc);
    if (!csv) throw std::runtime_error("cannot create qwen-expert-roundtrip.csv");
    csv << "seq,fixture_row,request_publish_us,compute_return_us,receive_validate_us,end_to_end_us\n";
    for (const auto &t : timings) csv << t.seq << ',' << t.fixture_row << ',' << t.request_us << ',' << t.compute_return_us << ',' << t.receive_validate_us << ',' << t.total_us << '\n';
    if (!csv) throw std::runtime_error("failed writing qwen-expert-roundtrip.csv");
    std::printf("QWEN_EXPERT_RESULT,PASS,warmups,%u,measured_rounds,%u,fixture_rows,%u,total_mismatches,%llu,max_abs_error,%.9g\n",
                warmups, iterations, fixture.rows, (unsigned long long)total_mismatches, max_abs_error);
    std::fflush(stdout);
    request_route.requester_quiesced = true; response_route.requester_quiesced = true;
    multi_route_cleanup(response_route, fd); response_active = false;
    multi_route_cleanup(request_route, fd); request_active = false;

    set_device_context(rtx); RT_CHECK(cudaEventDestroy(request_event));
    set_device_context(m10);
    RT_CHECK(cudaEventDestroy(response_event));
    for (void *p : {(void *)dw.gate, (void *)dw.up, (void *)dw.down, (void *)dw.gate_scale,
                    (void *)dw.up_scale, (void *)dw.down_scale, (void *)d_h, (void *)d_routes,
                    (void *)d_act, (void *)d_result, (void *)d_status}) if (p) RT_CHECK(cudaFree(p));
    set_device_context(rtx);
    for (void *p : {(void *)d_requests, (void *)d_return_copy, (void *)d_return_values}) if (p) RT_CHECK(cudaFree(p));
}

#ifndef Q38_RTX_TRANSPORT_LIBRARY_ONLY
int main(int argc, char **argv) {
    // PCI addresses are never defaulted: pass them from qwen38_pci_discover.py.
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s M10_BDF RTX_BDF [mode ...]  (discover with qwen38_pci_discover.py)\n", argv[0]);
        return 2;
    }
    const std::string mbdf = argv[1];
    const std::string rbdf = argv[2];
    const bool transmission_mode = argc > 3 && std::string(argv[3]) == "--transmission";
    const bool sync_compare_mode = argc > 3 && std::string(argv[3]) == "--sync-compare";
    const bool retained_roundtrip_mode = argc > 3 && std::string(argv[3]) == "--retained-roundtrip";
    const bool retained_expert_mode = argc > 3 && std::string(argv[3]) == "--retained-expert-roundtrip";
    const bool qwen_local_check_mode = argc > 3 && std::string(argv[3]) == "--qwen-local-check";
    const NvU32 retained_iterations = argc > 4 ? NvU32(std::strtoul(argv[4], nullptr, 10)) : 1000;
    const NvU32 retained_warmups = argc > 8 ? NvU32(std::strtoul(argv[8], nullptr, 10)) : 0;
    QMem m10{}, rtx{};
    bool m10_started = false, rtx_started = false;
    CUdevice rdev{};
    void *rtx_request_host = nullptr, *m10_response_host = nullptr;
    NvU32 *rtx_host_base = nullptr, *m10_host_base = nullptr;
    NvU32 *m10_verify = nullptr, *rtx_verify = nullptr;
    float *m10_result_local = nullptr;
    cudaStream_t rtx_stream{};
    cudaEvent_t rtx_event{}, m10_event{};
    int fd = -1;
    Route active_route{};
    bool route_in_use = false;
    MultiRoute active_multi_route{};
    bool multi_route_in_use = false;
    MultiRoute retained_request_route{}, retained_response_route{};
    bool retained_request_route_active = false, retained_response_route_active = false;
    NvProcessorUuid mu{}, ru{};
    try {
        CUKQ(cuInit(0));
        auto get_export = reinterpret_cast<ExportGetterQ>(dlsym(RTLD_DEFAULT, "rm_trace_get_last_export"));
        if (!get_export) throw std::runtime_error("RM export capture LD_PRELOAD is required");
        m10_started = true; qcreate_mapping(m10, mbdf, -1, -1, get_export);
        rtx_started = true; qcreate_mapping(rtx, rbdf, -1, -1, get_export);
        if (m10.rmmap.addr % QWEN_MAP || rtx.rmmap.addr % QWEN_MAP ||
            TARGET_OFF + REQUEST_BYTES + sizeof(NvU32) > QWEN_MAP ||
            TARGET_OFF + RESPONSE_BYTES + sizeof(NvU32) > QWEN_MAP)
            throw std::runtime_error("owner BAR1 mapping alignment/coverage invalid");
        mu = uuid_for(mbdf.c_str()); ru = uuid_for(rbdf.c_str(), &rdev);

        set_device_context(m10);
        RT_CHECK(cudaMalloc(&m10_verify, (REQUEST_WORDS + 2) * sizeof(NvU32)));
        RT_CHECK(cudaMalloc(&m10_result_local, H_ELEMENTS * sizeof(float)));
        RT_CHECK(cudaEventCreateWithFlags(&m10_event, cudaEventDisableTiming));

        set_device_context(rtx);
        RT_CHECK(cudaMalloc(&rtx_verify, (RESPONSE_WORDS + 2) * sizeof(NvU32)));
        RT_CHECK(cudaStreamCreateWithFlags(&rtx_stream, cudaStreamNonBlocking));
        RT_CHECK(cudaEventCreateWithFlags(&rtx_event, cudaEventDisableTiming));

        fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error(std::string("open UVM: ") + strerror(errno));
        UVM_INITIALIZE_PARAMS init{};
        if (ioctl(fd, UVM_INITIALIZE, &init) || init.rmStatus != NV_OK)
            throw std::runtime_error("UVM initialize failed");
        for (const NvProcessorUuid *u : {&mu, &ru}) {
            UVM_REGISTER_GPU_PARAMS reg{}; reg.gpu_uuid = *u;
            if (ioctl(fd, UVM_REGISTER_GPU, &reg) || reg.rmStatus != NV_OK)
                throw std::runtime_error("UVM GPU registration failed");
        }

        // Create and touch the M10 mapped-host response alias first. Query all
        // three exact original 4 KiB PTEs before RTX alias construction.
        set_device_context(m10);
        RT_CHECK(cudaHostAlloc(&m10_response_host, RESPONSE_BYTES, cudaHostAllocMapped));
        std::memset(m10_response_host, 0x48, RESPONSE_BYTES);
        RT_CHECK(cudaHostGetDevicePointer((void **)&m10_host_base, m10_response_host, 0));
        if ((NvU64)m10_host_base & (PAGE - 1))
            throw std::runtime_error("M10 response alias is not 4K aligned");
        prime_host_alias<<<1, 1, 0, m10.stream>>>(
            reinterpret_cast<volatile NvU32 *>(m10_host_base), 3, 0x284a6c8eu);
        RT_CHECK(cudaGetLastError()); RT_CHECK(cudaStreamSynchronize(m10.stream));
        std::vector<UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS> m10_saved_ptes;
        m10_saved_ptes.reserve(3);
        for (NvU32 page = 0; page < 3; ++page)
            m10_saved_ptes.push_back(query_pte(fd, mu, (NvU64)m10_host_base + page * PAGE,
                                               "M10_RESPONSE_PTE_PREQUERY"));

        // RTX mapped-host requester VA likewise has three touched 4 KiB leaves.
        set_device_context(rtx);
        RT_CHECK(cudaHostAlloc(&rtx_request_host, REQUEST_BYTES, cudaHostAllocMapped));
        std::memset(rtx_request_host, 0x39, REQUEST_BYTES);
        RT_CHECK(cudaHostGetDevicePointer((void **)&rtx_host_base, rtx_request_host, 0));
        if ((NvU64)rtx_host_base & (PAGE - 1))
            throw std::runtime_error("RTX request alias is not 4K aligned");
        prime_host_alias<<<1, 1, 0, rtx.stream>>>(
            reinterpret_cast<volatile NvU32 *>(rtx_host_base), 3, 0x17395b7du);
        RT_CHECK(cudaGetLastError()); RT_CHECK(cudaStreamSynchronize(rtx.stream));

        auto *mbar = reinterpret_cast<volatile NvU32 *>(m10.cpu);
        volatile NvU32 *m_page = mbar + TARGET_OFF / 4;
        volatile NvU32 *m_guard0 = m_page - 1;
        volatile NvU32 *m_guard1 = m_page + REQUEST_WORDS;
        auto *rbar = reinterpret_cast<volatile NvU32 *>(rtx.cpu);
        volatile NvU32 *r_page = rbar + TARGET_OFF / 4;
        volatile NvU32 *r_guard0 = r_page - 1;
        volatile NvU32 *r_guard1 = r_page + RESPONSE_WORDS;
        *m_guard0 = GUARD_BEFORE; *m_guard1 = GUARD_AFTER;
        *r_guard0 = GUARD_BEFORE; *r_guard1 = GUARD_AFTER;
        *reinterpret_cast<volatile NvU32 *>(reinterpret_cast<char *>(m10.cpu) + TARGET_OFF - 4) = GUARD_BEFORE;
        *reinterpret_cast<volatile NvU32 *>(reinterpret_cast<char *>(rtx.cpu) + TARGET_OFF - 4) = GUARD_BEFORE;
        for (NvU32 i = 0; i < REQUEST_WORDS; ++i) m_page[i] = 0xdead5100u ^ i;
        for (NvU32 i = 0; i < RESPONSE_WORDS; ++i) r_page[i] = RESPONSE_PAD;
        asm volatile("sfence" ::: "memory");

        if (qwen_local_check_mode) {
            if (argc < 7) throw std::runtime_error("usage: binary M10_BDF RTX_BDF --qwen-local-check BUNDLE REQUESTS PARTIALS");
            run_qwen_local_compute_check(m10, argv[4], argv[5], argv[6]);
        } else if (retained_roundtrip_mode) {
            run_retained_modelshape_roundtrip(fd, retained_request_route, retained_response_route,
                retained_request_route_active, retained_response_route_active,
                rtx, m10, ru, mu, rtx_host_base, m10_host_base, m10_saved_ptes,
                retained_iterations);
        } else if (retained_expert_mode) {
            if (argc < 8) throw std::runtime_error("usage: binary M10_BDF RTX_BDF --retained-expert-roundtrip ITER BUNDLE REQUESTS PARTIALS");
            run_retained_qwen_expert_roundtrip(fd, retained_request_route, retained_response_route,
                retained_request_route_active, retained_response_route_active,
                rtx, m10, ru, mu, rtx_host_base, m10_host_base, m10_saved_ptes,
                retained_iterations, retained_warmups, argv[5], argv[6], argv[7]);
        } else if (sync_compare_mode) {
            std::puts("SYNC_HANDOFF_COMPARE,variants,distributed_256_all_threads_fence|distributed_256_one_thread_fence|staged_single_sender,warmups_per_variant,10,measured_samples_per_variant_direction,500");
            std::printf("TX_PRODUCTION_SHAPE,request_H_bytes,%u,expert_id_bytes,%u,route_weight_bytes,%u,request_payload_bytes,%u,request_sequence_bytes,4,result_fp32_bytes,%u,result_sequence_bytes,4,mailbox_header_bytes,%u\n",
                        H_ELEMENTS * 4, TOP_K * 4, TOP_K * 4, PROD_REQUEST_BYTES,
                        PROD_RESULT_BYTES, MAILBOX_HEADER_BYTES);
            run_gpu_direction_compare(fd, active_multi_route, multi_route_in_use,
                                      rtx, m10, ru, mu, rtx_host_base, "RTX_TO_M10",
                                      true, PROD_REQUEST_BYTES, 500);
            run_gpu_direction_compare(fd, active_multi_route, multi_route_in_use,
                                      m10, rtx, mu, ru, m10_host_base, "M10_TO_RTX",
                                      false, PROD_RESULT_BYTES, 500);
            std::puts("SYNC_HANDOFF_COMPARE_RESULT,PASS,all_variant_payloads_verified_on_owner_GPU");
        } else if (transmission_mode) {
            std::puts("TX_BENCH,scope,RTX-M10-direct-BAR1,setup_timed,NO,warmups_per_direction,10,measured_samples_per_direction,100");
            std::printf("TX_PRODUCTION_SHAPE,request_H_bytes,%u,expert_id_bytes,%u,route_weight_bytes,%u,request_payload_bytes,%u,request_sequence_bytes,4,result_fp32_bytes,%u,result_sequence_bytes,4,mailbox_header_bytes,%u\n",
                        H_ELEMENTS * 4, TOP_K * 4, TOP_K * 4, PROD_REQUEST_BYTES,
                        PROD_RESULT_BYTES, MAILBOX_HEADER_BYTES);
            // CPU control and GPU transport target the same retained owner
            // allocation/mailbox layout. CPU path measures NT stores + SFENCE;
            // GPU path measures a coalesced SYS write + system-fence sequence.
            run_cpu_bar1_baseline(m10, "CPU_TO_M10", true, PROD_REQUEST_BYTES, 100);
            run_cpu_bar1_baseline(rtx, "CPU_TO_RTX", false, PROD_RESULT_BYTES, 100);
            run_gpu_direction_compare(fd, active_multi_route, multi_route_in_use,
                              rtx, m10, ru, mu, rtx_host_base, "RTX_TO_M10",
                              true, PROD_REQUEST_BYTES, 100);
            run_gpu_direction_compare(fd, active_multi_route, multi_route_in_use,
                              m10, rtx, mu, ru, m10_host_base, "M10_TO_RTX",
                              false, PROD_RESULT_BYTES, 100);
            std::puts("TX_BENCH_RESULT,PASS,all_production_payloads_verified_on_owner_GPU");
        } else {

        std::printf("PAIR,requester_m10,%s,owner_rtx,%s,request_logical_bytes,%u,request_mapped_bytes,%u,response_logical_bytes,%u,response_mapped_bytes,%u,leaves_per_direction,3\n",
                    mbdf.c_str(), rbdf.c_str(), REQUEST_LOGICAL_BYTES, REQUEST_BYTES,
                    RESPONSE_LOGICAL_BYTES, RESPONSE_BYTES);
        std::vector<NvU32> m_seen(REQUEST_WORDS + 2), r_seen(RESPONSE_WORDS + 2);
        for (NvU32 seq = 1; seq <= 2; ++seq) {
            const auto round_start = std::chrono::steady_clock::now();
            for (NvU32 i = 0; i < REQUEST_WORDS; ++i) m_page[i] = 0xdead5100u ^ i;
            for (NvU32 i = 0; i < RESPONSE_WORDS; ++i) r_page[i] = RESPONSE_PAD;
            *m_guard0 = GUARD_BEFORE; *m_guard1 = GUARD_AFTER;
            *r_guard0 = GUARD_BEFORE; *r_guard1 = GUARD_AFTER;
            asm volatile("sfence" ::: "memory");

            auto req_t0 = std::chrono::steady_clock::now();
            set_device_context(rtx); multi_route_in_use = true;
            multi_route_setup(active_multi_route, fd, ru, mu, m10,
                              (NvU64)rtx_host_base, TARGET_OFF);
            const double req_setup_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - req_t0).count();
            set_device_context(rtx); active_multi_route.requester_quiesced = false;
            auto req_gpu_t0 = std::chrono::steady_clock::now();
            rtx_publish_request_full<<<1, 1, 0, rtx_stream>>>(
                reinterpret_cast<volatile NvU32 *>(rtx_host_base), seq);
            RT_CHECK(cudaGetLastError()); RT_CHECK(cudaEventRecord(rtx_event, rtx_stream));
            wait_event(rtx_event, active_multi_route, "RTX request full-window");
            const double req_gpu_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - req_gpu_t0).count();
            auto req_teardown_t0 = std::chrono::steady_clock::now();
            multi_route_cleanup(active_multi_route, fd); multi_route_in_use = false;
            const double req_teardown_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - req_teardown_t0).count();

            // Independent M10 CUDA-path and CPU BAR1 checks of the complete
            // padded request. No host copy is used to feed the response kernel.
            set_device_context(m10);
            verify_copy_words<<<1, 1, 0, m10.stream>>>(
                reinterpret_cast<const volatile NvU32 *>(m10.va + TARGET_OFF - 4),
                m10_verify, REQUEST_WORDS + 2);
            RT_CHECK(cudaGetLastError()); RT_CHECK(cudaStreamSynchronize(m10.stream));
            RT_CHECK(cudaMemcpy(m_seen.data(), m10_verify, (REQUEST_WORDS + 2) * 4,
                                cudaMemcpyDeviceToHost));
            NvU32 req_cpu_bad = 0, req_gpu_bad = 0;
            for (NvU32 i = 0; i < REQUEST_WORDS; ++i) {
                const NvU32 expected = request_word(seq, i);
                if (m_page[i] != expected) ++req_cpu_bad;
                if (m_seen[i + 1] != expected) ++req_gpu_bad;
            }
            if (m_seen[0] != GUARD_BEFORE || m_seen[REQUEST_WORDS + 1] != GUARD_AFTER ||
                *m_guard0 != GUARD_BEFORE || *m_guard1 != GUARD_AFTER || req_cpu_bad || req_gpu_bad)
                throw std::runtime_error("M10 request comparison or guard check failed");
            std::printf("ROUND_REQUEST,seq,%u,logical_bytes,%u,padded_transferred_bytes,%u,h_bytes,%u,expert_id_bytes,%u,route_weight_bytes,%u,m10_cuda_mismatches,%u,cpu_bar_mismatches,%u,guards,PASS\n",
                        seq, REQUEST_LOGICAL_BYTES, REQUEST_BYTES, H_ELEMENTS * 4,
                        TOP_K * 4, TOP_K * 4, req_gpu_bad, req_cpu_bad);
            std::printf("REQUEST_FIELDS,seq,%u,h0,%.3f,h_last,%.3f,id0,%u,id9,%u,w0,%.3f,w9,%.3f,all10_route_bias,%.3f\n",
                        seq, request_h(seq, 0), request_h(seq, H_ELEMENTS - 1),
                        request_id(seq, 0), request_id(seq, TOP_K - 1),
                        request_weight(seq, 0), request_weight(seq, TOP_K - 1), request_bias(seq));

            // M10 consumes H and all ten ID/weight pairs, then computes
            // one communication-test slice into its own local CUDA buffer.
            auto compute_start = std::chrono::steady_clock::now();
            compute_test_slice<<<1, 1, 0, m10.stream>>>(
                reinterpret_cast<const volatile NvU32 *>(m10.va + TARGET_OFF),
                m10_result_local);
            RT_CHECK(cudaGetLastError()); RT_CHECK(cudaStreamSynchronize(m10.stream));
            const double m10_compute_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - compute_start).count();

            auto resp_t0 = std::chrono::steady_clock::now();
            set_device_context(m10); multi_route_in_use = true;
            multi_route_setup(active_multi_route, fd, mu, ru, rtx,
                              (NvU64)m10_host_base, TARGET_OFF, &m10_saved_ptes);
            const double resp_setup_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - resp_t0).count();
            set_device_context(m10); active_multi_route.requester_quiesced = false;
            auto resp_gpu_t0 = std::chrono::steady_clock::now();
            m10_publish_slice_full<<<1, 1, 0, m10.stream>>>(m10_result_local,
                reinterpret_cast<volatile NvU32 *>(m10_host_base));
            RT_CHECK(cudaGetLastError()); RT_CHECK(cudaEventRecord(m10_event, m10.stream));
            wait_event(m10_event, active_multi_route, "M10 response full-window");
            const double resp_gpu_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - resp_gpu_t0).count();
            auto resp_teardown_t0 = std::chrono::steady_clock::now();
            multi_route_cleanup(active_multi_route, fd); multi_route_in_use = false;
            const double resp_teardown_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - resp_teardown_t0).count();

            set_device_context(rtx);
            verify_copy_words<<<1, 1, 0, rtx_stream>>>(
                reinterpret_cast<const volatile NvU32 *>(rtx.va + TARGET_OFF - 4),
                rtx_verify, RESPONSE_WORDS + 2);
            RT_CHECK(cudaGetLastError()); RT_CHECK(cudaStreamSynchronize(rtx_stream));
            RT_CHECK(cudaMemcpy(r_seen.data(), rtx_verify, (RESPONSE_WORDS + 2) * 4,
                                cudaMemcpyDeviceToHost));
            NvU32 resp_cpu_bad = 0, resp_gpu_bad = 0;
            for (NvU32 i = 0; i < H_ELEMENTS; ++i) {
                const NvU32 expected = response_word(seq, i);
                if (r_page[i] != expected) ++resp_cpu_bad;
                if (r_seen[i + 1] != expected) ++resp_gpu_bad;
            }
            for (NvU32 i = H_ELEMENTS; i < RESPONSE_WORDS; ++i) {
                if (r_page[i] != RESPONSE_PAD) ++resp_cpu_bad;
                if (r_seen[i + 1] != RESPONSE_PAD) ++resp_gpu_bad;
            }
            if (r_seen[0] != GUARD_BEFORE || r_seen[RESPONSE_WORDS + 1] != GUARD_AFTER ||
                *r_guard0 != GUARD_BEFORE || *r_guard1 != GUARD_AFTER ||
                resp_gpu_bad || resp_cpu_bad)
                throw std::runtime_error("RTX response comparison or guard check failed");
            const double total_us = std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - round_start).count();
            std::printf("ROUND_RESPONSE,seq,%u,logical_bytes,%u,peer_written_bytes,%u,page_coverage_bytes,%u,rtx_cuda_mismatches,%u,cpu_bar_mismatches,%u,padding_checked,%u,guards,PASS\n",
                        seq, RESPONSE_LOGICAL_BYTES, RESPONSE_LOGICAL_BYTES, RESPONSE_BYTES, resp_gpu_bad,
                        resp_cpu_bad, RESPONSE_PAD_WORDS * 4);
            std::printf("ROUND_TIMING,seq,%u,request_setup_us,%.3f,request_submit_complete_us,%.3f,request_teardown_us,%.3f,m10_compute_launch_sync_us,%.3f,response_setup_us,%.3f,response_submit_complete_us,%.3f,response_teardown_us,%.3f,total_checked_us,%.3f\n",
                        seq, req_setup_us, req_gpu_us, req_teardown_us,
                        m10_compute_us, resp_setup_us, resp_gpu_us, resp_teardown_us, total_us);
            std::fflush(stdout);
        }
        std::puts("RTX_M10_MODEL_SHAPE_ROUNDTRIP,rounds,2,h_fp32,2560,expert_ids,10,route_weights,10,result_fp32,2560,result,PASS");
        }

        RT_CHECK(cudaEventDestroy(rtx_event)); rtx_event = nullptr;
        RT_CHECK(cudaEventDestroy(m10_event)); m10_event = nullptr;
        set_device_context(rtx);
        RT_CHECK(cudaFree(rtx_verify)); rtx_verify = nullptr;
        RT_CHECK(cudaStreamDestroy(rtx_stream)); rtx_stream = nullptr;
        set_device_context(m10);
        RT_CHECK(cudaFree(m10_verify)); m10_verify = nullptr;
        RT_CHECK(cudaFree(m10_result_local)); m10_result_local = nullptr;
        RT_CHECK(cudaFreeHost(rtx_request_host)); rtx_request_host = nullptr;
        RT_CHECK(cudaFreeHost(m10_response_host)); m10_response_host = nullptr;
        close(fd); fd = -1;
        qdestroy_mem(rtx); rtx_started = false;
        qdestroy_mem(m10); m10_started = false;
        return 0;
    }
    catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL,%s\n", e.what());
        if (route_in_use) {
            if (active_route.remap_active && !active_route.requester_quiesced)
                preserve("GPU operation may still reference active route");
            route_cleanup(active_route, fd);
            route_in_use = false;
        }
        if (multi_route_in_use) {
            if (!active_multi_route.requester_quiesced && !active_multi_route.active.empty())
                preserve("GPU operation may still reference multi-PTE route");
            multi_route_cleanup(active_multi_route, fd);
            multi_route_in_use = false;
        }
        if (retained_request_route_active) {
            if (!retained_request_route.requester_quiesced && !retained_request_route.active.empty())
                preserve("RTX request route may still be referenced by outstanding work");
            multi_route_cleanup(retained_request_route, fd);
            retained_request_route_active = false;
        }
        if (retained_response_route_active) {
            if (!retained_response_route.requester_quiesced && !retained_response_route.active.empty())
                preserve("M10 response route may still be referenced by outstanding work");
            multi_route_cleanup(retained_response_route, fd);
            retained_response_route_active = false;
        }
        if (rtx_event) cudaEventDestroy(rtx_event);
        if (m10_event) cudaEventDestroy(m10_event);
        if (rtx_verify) { set_device_context(rtx); cudaFree(rtx_verify); }
        if (m10_verify) { set_device_context(m10); cudaFree(m10_verify); }
        if (m10_result_local) { set_device_context(m10); cudaFree(m10_result_local); }
        if (rtx_request_host) cudaFreeHost(rtx_request_host);
        if (m10_response_host) cudaFreeHost(m10_response_host);
        if (rtx_stream) { set_device_context(rtx); cudaStreamDestroy(rtx_stream); }
        if (fd >= 0) close(fd);
        if (rtx_started) qdestroy_mem(rtx);
        if (m10_started) qdestroy_mem(m10);
        return 1;
    }
}
#endif
