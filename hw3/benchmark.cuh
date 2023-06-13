#ifndef _BENCHMARK_CUH
#define _BENCHMARK_CUH

#include "util.cuh"

// Kernel for the benchmark
__global__ void elementwise_add(const int *x, const int *y,
                                int *z, unsigned int stride,
                                unsigned int size) {
    
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    while(id < size){
        z[id * stride] = x[id * stride] + y[id * stride];
        id += blockDim.x * gridDim.x;
    }

    // z[i * stride] = x[i * stride] + y[i * stride]
    // where i goes from 0 to size-1.
    // Distribute the work across all CUDA threads allocated by
    // elementwise_add<<<72, 1024>>>(x, y, z, stride, N);
    // Use the CUDA variables gridDim, blockDim, blockIdx, and threadIdx.
}

#endif
