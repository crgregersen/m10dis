#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

constexpr int HIDDEN = 2560;
constexpr int SLICE = 160;
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

static void check(cudaError_t e, const char *where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}
#define CK(x) check((x), #x)

static std::vector<float> read_f32(const std::string &path, size_t count) {
    std::vector<float> v(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()), count * sizeof(float));
    if (!f || f.peek() != EOF) {
        std::fprintf(stderr, "bad float file %s\n", path.c_str());
        std::exit(1);
    }
    return v;
}

static float *upload(const std::vector<float> &v) {
    float *p = nullptr;
    CK(cudaMalloc(&p, v.size() * sizeof(float)));
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return p;
}

static uint16_t *upload_bf16_file(const std::string &path, size_t count) {
    std::vector<uint16_t> bits(count);
    std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char *>(bits.data()), count * sizeof(uint16_t));
    if (!f || f.peek() != EOF) {
        std::fprintf(stderr, "bad half file %s\n", path.c_str());
        std::exit(1);
    }
    uint16_t *p = nullptr;
    CK(cudaMalloc(&p, count * sizeof(uint16_t)));
    CK(cudaMemcpy(p, bits.data(), count * sizeof(uint16_t), cudaMemcpyHostToDevice));
    return p;
}

__device__ __forceinline__ float bf16_to_float(uint16_t value) {
    return __uint_as_float(uint32_t(value) << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ static void make_shared_hidden(const uint16_t *gate_w, const uint16_t *up_w,
                                          const float *x, float *hidden) {
    __shared__ float gate_warp[WARPS], up_warp[WARPS];
    const int row = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float g = 0.0f, u = 0.0f;
    const uint16_t *gw = gate_w + size_t(row) * HIDDEN;
    const uint16_t *uw = up_w + size_t(row) * HIDDEN;
    for (int col = threadIdx.x; col < HIDDEN; col += blockDim.x) {
        g = fmaf(bf16_to_float(gw[col]), x[col], g);
        u = fmaf(bf16_to_float(uw[col]), x[col], u);
    }
    g = warp_sum(g);
    u = warp_sum(u);
    if (lane == 0) {
        gate_warp[warp] = g;
        up_warp[warp] = u;
    }
    __syncthreads();
    if (warp == 0) {
        g = lane < WARPS ? gate_warp[lane] : 0.0f;
        u = lane < WARPS ? up_warp[lane] : 0.0f;
        g = warp_sum(g);
        u = warp_sum(u);
        if (lane == 0) {
            float silu = g / (1.0f + expf(-g));
            hidden[row] = silu * u;
        }
    }
}

__global__ static void shared_down_partial(const uint16_t *down_w, const float *hidden,
                                           float *partial) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * WARPS + warp;
    if (row >= HIDDEN) return;
    float sum = 0.0f;
    const uint16_t *w = down_w + size_t(row) * SLICE;
    for (int col = lane; col < SLICE; col += 32)
        sum = fmaf(bf16_to_float(w[col]), hidden[col], sum);
    sum = warp_sum(sum);
    if (lane == 0) partial[row] = sum;
}

__global__ static void shared_gate_scale(const uint16_t *weight, const float *x, float *scale) {
    __shared__ float partial[THREADS];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
        sum = fmaf(bf16_to_float(weight[i]), x[i], sum);
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) scale[0] = 1.0f / (1.0f + expf(-partial[0]));
}

struct DeviceWork {
    uint16_t *gate = nullptr, *up = nullptr, *down = nullptr;
    float *x = nullptr;
    float *hidden = nullptr, *partial = nullptr;
    uint16_t *shared_gate_weight = nullptr;
    float *shared_gate_scale = nullptr;
    cudaStream_t stream = nullptr;
    struct IterTiming {
        cudaEvent_t begin = nullptr, gate_begin = nullptr, gate_end = nullptr;
        cudaEvent_t down_begin = nullptr, down_end = nullptr;
        cudaEvent_t scalar_begin = nullptr, scalar_end = nullptr, end = nullptr;
    };
    std::vector<IterTiming> timings;
};

static double percentile(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q * (v.size() - 1)))];
}

static double mean(const std::vector<float> &v) {
    double s = 0.0;
    for (float x : v) s += x;
    return s / v.size();
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s PREPARED_DIR DEVICES ROUNDS CSV OUTPUT_PARTIALS_F32\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const int devices = std::atoi(argv[2]);
    const int rounds = std::atoi(argv[3]);
    constexpr int warmup = 100;
    if (devices != 4) {
        std::fprintf(stderr, "shared expert requires exactly four GM107 slices on one M10\n");
        return 2;
    }
    if (rounds < 1 || rounds > 10000) {
        std::fprintf(stderr, "rounds must be in 1..10000\n");
        return 2;
    }
    int available = 0;
    CK(cudaGetDeviceCount(&available));
    if (available < devices) return 2;
    auto x_host = read_f32(dir + "/input.bin", HIDDEN);
    std::vector<DeviceWork> work(devices);
    std::vector<std::vector<float>> references(devices);
    for (int dev = 0; dev < devices; ++dev) {
        CK(cudaSetDevice(dev));
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, dev));
        if (prop.major != 5 || prop.minor != 0) {
            std::fprintf(stderr, "device %d is not GM107/sm_50\n", dev);
            return 2;
        }
        char stem[32];
        std::snprintf(stem, sizeof(stem), "/shared.%d.", dev);
        std::string base = dir + stem;
        references[dev] = read_f32(base + "ref.bin", HIDDEN);
        work[dev].gate = upload_bf16_file(base + "gate.bin", size_t(SLICE) * HIDDEN);
        work[dev].up = upload_bf16_file(base + "up.bin", size_t(SLICE) * HIDDEN);
        work[dev].down = upload_bf16_file(base + "down.bin", size_t(HIDDEN) * SLICE);
        work[dev].x = upload(x_host);
        CK(cudaMalloc(&work[dev].hidden, SLICE * sizeof(float)));
        CK(cudaMalloc(&work[dev].partial, HIDDEN * sizeof(float)));
        if (dev == 0) {
            work[dev].shared_gate_weight = upload_bf16_file(dir + "/shared_gate_weight.bin", HIDDEN);
            CK(cudaMalloc(&work[dev].shared_gate_scale, sizeof(float)));
        }
        CK(cudaStreamCreate(&work[dev].stream));
        work[dev].timings.resize(rounds);
        for (auto &t : work[dev].timings) {
            CK(cudaEventCreate(&t.begin)); CK(cudaEventCreate(&t.gate_begin));
            CK(cudaEventCreate(&t.gate_end)); CK(cudaEventCreate(&t.down_begin));
            CK(cudaEventCreate(&t.down_end)); CK(cudaEventCreate(&t.end));
            if (dev == 0) {
                CK(cudaEventCreate(&t.scalar_begin)); CK(cudaEventCreate(&t.scalar_end));
            }
        }
        std::printf("SHARED_DEVICE,%d,bdf_ordered_visible_device,sm_%d%d,slice_start,%d,slice_width,%d\n",
                    dev, prop.major, prop.minor, dev * SLICE, SLICE);
    }

    for (int dev = 0; dev < devices; ++dev) {
        CK(cudaSetDevice(dev));
        auto &w = work[dev];
        for (int iteration = -warmup; iteration < rounds; ++iteration) {
            DeviceWork::IterTiming *t = iteration >= 0 ? &w.timings[iteration] : nullptr;
            if (t) {
                CK(cudaEventRecord(t->begin, w.stream));
                CK(cudaEventRecord(t->gate_begin, w.stream));
            }
            make_shared_hidden<<<SLICE, THREADS, 0, w.stream>>>(w.gate, w.up, w.x, w.hidden);
            CK(cudaGetLastError());
            if (t) CK(cudaEventRecord(t->gate_end, w.stream));
            if (t) CK(cudaEventRecord(t->down_begin, w.stream));
            shared_down_partial<<<(HIDDEN + WARPS - 1) / WARPS, THREADS, 0, w.stream>>>(
                w.down, w.hidden, w.partial);
            CK(cudaGetLastError());
            if (t) CK(cudaEventRecord(t->down_end, w.stream));
            if (dev == 0) {
                if (t) CK(cudaEventRecord(t->scalar_begin, w.stream));
                shared_gate_scale<<<1, THREADS, 0, w.stream>>>(
                    w.shared_gate_weight, w.x, w.shared_gate_scale);
                CK(cudaGetLastError());
                if (t) CK(cudaEventRecord(t->scalar_end, w.stream));
            }
            if (t) CK(cudaEventRecord(t->end, w.stream));
        }
    }

    std::ofstream csv(argv[4]);
    csv << "device,iteration,gate_up_us,down_us,gate_scalar_us,total_us\n";
    std::vector<float> all_partials(size_t(devices) * HIDDEN + 1);
    bool ok = true;
    for (int dev = 0; dev < devices; ++dev) {
        CK(cudaSetDevice(dev));
        auto &w = work[dev];
        CK(cudaStreamSynchronize(w.stream));
        std::vector<float> got(HIDDEN);
        CK(cudaMemcpy(got.data(), w.partial, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
        int bad = 0;
        float max_abs = 0.0f;
        for (int i = 0; i < HIDDEN; ++i) {
            float e = std::abs(got[i] - references[dev][i]);
            max_abs = std::max(max_abs, e);
            if (!std::isfinite(got[i]) || e > 2e-5f + 1e-3f * std::abs(references[dev][i])) ++bad;
        }
        std::vector<float> gate_us, down_us, scalar_us, total_us;
        gate_us.reserve(rounds); down_us.reserve(rounds); scalar_us.reserve(rounds); total_us.reserve(rounds);
        for (int i = 0; i < rounds; ++i) {
            auto &t = w.timings[i];
            float g=0.0f, d=0.0f, s=0.0f, total=0.0f;
            CK(cudaEventElapsedTime(&g, t.gate_begin, t.gate_end));
            CK(cudaEventElapsedTime(&d, t.down_begin, t.down_end));
            CK(cudaEventElapsedTime(&total, t.begin, t.end));
            if (dev == 0) CK(cudaEventElapsedTime(&s, t.scalar_begin, t.scalar_end));
            g *= 1000.0f; d *= 1000.0f; s *= 1000.0f; total *= 1000.0f;
            gate_us.push_back(g); down_us.push_back(d); scalar_us.push_back(s); total_us.push_back(total);
            csv << dev << ',' << i << ',' << g << ',' << d << ',' << s << ',' << total << '\n';
        }
        std::copy(got.begin(), got.end(), all_partials.begin() + size_t(dev) * HIDDEN);
        std::printf("SHARED_RESULT,device,%d,bad,%d,max_abs,%.8g,rounds,%d,"
                    "gate_up_mean_us,%.3f,down_mean_us,%.3f,scalar_mean_us,%.3f,total_mean_us,%.3f,"
                    "total_median_us,%.3f,total_p95_us,%.3f,total_max_us,%.3f\n",
                    dev, bad, max_abs, rounds, mean(gate_us), mean(down_us), mean(scalar_us),
                    mean(total_us), percentile(total_us, 0.50), percentile(total_us, 0.95),
                    *std::max_element(total_us.begin(), total_us.end()));
        ok &= bad == 0;
    }
    float gate_scale = 0.0f;
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(&gate_scale, work[0].shared_gate_scale, sizeof(float), cudaMemcpyDeviceToHost));
    all_partials[size_t(devices) * HIDDEN] = gate_scale;
    std::printf("SHARED_GATE_SCALE_GPU0,%.9g\n", gate_scale);
    std::ofstream out(argv[5], std::ios::binary);
    out.write(reinterpret_cast<const char *>(all_partials.data()),
              all_partials.size() * sizeof(float));
    if (!out) ok = false;

    for (int dev = 0; dev < devices; ++dev) {
        CK(cudaSetDevice(dev));
        auto &w = work[dev];
        for (uint16_t *p : {w.gate, w.up, w.down, w.shared_gate_weight})
            if (p) CK(cudaFree(p));
        for (float *p : {w.x, w.hidden, w.partial, w.shared_gate_scale})
            if (p) CK(cudaFree(p));
        for (auto &t : w.timings) {
            CK(cudaEventDestroy(t.begin)); CK(cudaEventDestroy(t.gate_begin));
            CK(cudaEventDestroy(t.gate_end)); CK(cudaEventDestroy(t.down_begin));
            CK(cudaEventDestroy(t.down_end)); CK(cudaEventDestroy(t.end));
            if (dev == 0) { CK(cudaEventDestroy(t.scalar_begin)); CK(cudaEventDestroy(t.scalar_end)); }
        }
        CK(cudaStreamDestroy(w.stream));
    }
    std::printf("SHARED_STATUS,%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
