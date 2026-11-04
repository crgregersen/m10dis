// Build as a cubin with -arch=sm_50. Loaded into each shared worker's existing
// CUDA context by RELOAD_KERNELS; expert allocations remain owned by that worker.
#define QWEN_M10_SHARED_LIBRARY_ONLY
#include "bench_resident_multilayer_shared.cu"
#include "resident_dynamic_shared_v2.cuh"
