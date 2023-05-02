#ifndef _BENCHMARK_CUH
#define _BENCHMARK_CUH

#include "util.cuh"

// Kernel for the benchmark
__global__ void elementwise_add(const int *x, const int *y,
                                int *z, unsigned int stride,
                                unsigned int size) {
    // TODO: elementwise_add should compute
    // z[i * stride] = x[i * stride] + y[i * stride]
    // where i goes from 0 to size-1.
    // Distribute the work across all CUDA threads allocated by
    // elementwise_add<<<72, 1024>>>(x, y, z, stride, N);
    // Use the CUDA variables gridDim, blockDim, blockIdx, and threadIdx.
}

#endif
