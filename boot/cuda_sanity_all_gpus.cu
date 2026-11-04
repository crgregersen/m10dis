#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void write_word(unsigned int *dst, unsigned int value)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        dst[0] = value;
}

static void check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main()
{
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    std::printf("CUDA_DEVICE_COUNT,%d\n", count);

    for (int dev = 0; dev < count; ++dev) {
        check(cudaSetDevice(dev), "cudaSetDevice");
        unsigned int *device_word = nullptr;
        check(cudaMalloc(&device_word, sizeof(*device_word)), "cudaMalloc");
        const unsigned int expected = 0x53000000u | static_cast<unsigned int>(dev);
        write_word<<<1, 1>>>(device_word, expected);
        check(cudaGetLastError(), "kernel launch");
        check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        unsigned int observed = 0;
        check(cudaMemcpy(&observed,
                         device_word,
                         sizeof(observed),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy");
        check(cudaFree(device_word), "cudaFree");

        std::printf("CUDA_SANITY,gpu,%d,expected,0x%08x,observed,0x%08x,match,%s\n",
                    dev,
                    expected,
                    observed,
                    expected == observed ? "YES" : "NO");
        if (expected != observed)
            return 2;
    }

    return 0;
}

