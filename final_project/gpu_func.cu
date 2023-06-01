#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/

int myGEMM(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{   
    // Allocate device memory
    nn_real *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M*K*sizeof(nn_real));
    cudaMalloc((void **)&d_B, K*N*sizeof(nn_real));
    cudaMalloc((void **)&d_C, M*N*sizeof(nn_real));

    // Copy data to device
    cudaMemcpy(d_A, A, M*K*sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N*sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*N*sizeof(nn_real), cudaMemcpyHostToDevice);

    // Create cublas handle
    dim3 threads(32, 32);
    dim3 blocks(ceil(M / 32.f), ceil(N / 32.f));

    myGEMM_kernel<<<blocks, threads, 2 * 32 * K * sizeof(nn_real)>>>(d_A, d_B, d_C, alpha, beta, M, N, K);

    // Copy data back to host
    cudaMemcpy(C, d_C, M*N*sizeof(nn_real), cudaMemcpyDeviceToHost);

    return 0;
}

/* Helper functions for neural networks */
__global__ void myGEMM_kernel(nn_real *__restrict__ A, nn_real *__restrict__ B,
           nn_real *__restrict__ C,
           nn_real alpha, nn_real beta,
           int M, int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_loc = threadIdx.x;
    int j_loc = threadIdx.y;

    extern __shared__ float s[];
    
    while(i < K){
        s[j_loc * K + i_loc] = A[j * K + i];
        i += blockDim.x * gridDim.x;
    }

    i = blockIdx.x * blockDim.x + threadIdx.x;

    while(j < K){
        s[32 * K + j_loc * 32 + i_loc] = B[j * K + i];
        j += blockDim.y * gridDim.y;
    }

    j = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();

    if(i < M){
        if(j < N){
            C[j * M + i] = 0.0;
            for(int k = 0; k < K; k++){
                C[j * N + i] += alpha * s[j_loc * K + k] * s[32 * K + 32 * k + i_loc];
            }
            C[j * N + i] += beta * C[j * M + i];
        }
    }
}
