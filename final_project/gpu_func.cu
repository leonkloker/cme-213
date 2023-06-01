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

    // Call kernel
    myGEMM_kernel<<<blocks, threads, 2 * 32 * K * sizeof(nn_real)>>>(d_A, d_B, d_C, alpha, beta, M, N, K);

    // Copy data back to host
    cudaMemcpy(C, d_C, M*N*sizeof(nn_real), cudaMemcpyDeviceToHost);

    return 0;
}

// Kernel to perform GEMM
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

    while(i < N){
        while(j < M){
            C[j * N + i] = 0.0;
            for(int k = 0; k < K; k++){
                C[j * N + i] += alpha * s[j_loc * K + k] * s[32 * K + 32 * k + i_loc];
            }
            C[j * N + i] += beta * C[j * N + i];
            j += blockDim.y * gridDim.y;
        }
        i += blockDim.x * gridDim.x;
    }
}

void sigmoid_gpu(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
    mat2.set_size(mat.n_rows, mat.n_cols);
    int N = mat.n_rows * mat.n_cols;
    ASSERT_MAT_SAME_SIZE(mat, mat2);

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, N*sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, N*sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat.memptr(), N*sizeof(nn_real), cudaMemcpyHostToDevice);

    sigmoid_kernel<<<ceil(N / 256.f), 256>>>(mat.memptr(), mat2.memptr(), N);

    // copy data back to host
    cudaMemcpy(mat2.memptr(), d_mat2, N*sizeof(nn_real), cudaMemcpyDeviceToHost);
  
}

__global__ void sigmoid_kernel(const nn_real *mat, nn_real *mat2, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  while(idx < N){
    mat2[idx] = 1 / (1 + exp(-mat[idx]));
    idx += blockDim.x * gridDim.x;  
  }
}

void softmax_gpu(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
    mat2.set_size(mat.n_rows, mat.n_cols);
    ASSERT_MAT_SAME_SIZE(mat, mat2);

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, mat.n_rows * mat.n_cols * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, mat.n_rows * mat.n_cols * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat.memptr(), mat.n_rows * mat.n_cols * sizeof(nn_real), cudaMemcpyHostToDevice);

    softmax_kernel<<<ceil(mat.n_cols / 256.f), 256>>>(mat.memptr(), mat2.memptr(), mat.n_rows, mat.n_cols);

    // copy data back to host
    cudaMemcpy(mat2.memptr(), d_mat2, mat.n_rows * mat.n_cols * sizeof(nn_real), cudaMemcpyDeviceToHost);
}

__global__ void softmax_kernel(nn_real *mat, nn_real *mat2, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while(idx < N){
        nn_real max_val = -FLT_MAX;
        nn_real sum = 0.0f;

        for (int k = 0; k < M; k++)
        {
            nn_real val = matrix[k * N + idx];
            if (val > max_val)
            {
                max_val = val;
            }
        }

        for (int k = 0; k < M; k++)
        {
            sum += exp(matrix[k * N + idx] - max_val);
        }

        for (int k = 0; k < M; k++)
        {
            mat2[k * N + idx] = exp(matrix[k * N + idx] - max_val) / sum;
        }
        idx += blockDim.x * gridDim.x;
    }
}
