// Build as a cubin with -arch=sm_50. Loaded into each routed worker's existing
// CUDA context by RELOAD_KERNELS; expert allocations remain owned by that worker.
#define QWEN_M10_LIBRARY_ONLY
#include "bench_resident_multilayer_fp8.cu"
#include "resident_dynamic_routed_v3.cuh"
