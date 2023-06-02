#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(const nn_real *__restrict__ A, const nn_real *__restrict__ B,
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

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

/*
Kernel called to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
__global__ void myGEMM_kernel(const nn_real *__restrict__ A, const nn_real *__restrict__ B,
           nn_real *__restrict__ C, nn_real alpha, nn_real beta, int M, int N, int K)
{
    // Get thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_loc = threadIdx.x;
    int j_loc = threadIdx.y;

    // Shared memory
    extern __shared__ nn_real s[];
    
    // Copy A to shared memory
    while(i < K){
        s[j_loc * K + i_loc] = A[j * K + i];
        i += blockDim.x * gridDim.x;
    }

    i = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy B to shared memory
    while(j < K){
        s[32 * K + j_loc * 32 + i_loc] = B[j * K + i];
        j += blockDim.y * gridDim.y;
    }

    j = blockIdx.y * blockDim.y + threadIdx.y;
    __syncthreads();

    // Perform GEMM
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

/*
Routine to calculate sigmoid elementwise on a matrix
*/
void sigmoid_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{   
    // Set size of output matrix
    //mat2.set_size(mat.n_rows, mat.n_cols);
    //int N = mat.n_rows * mat.n_cols;
    //ASSERT_MAT_SAME_SIZE(mat, mat2);

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, M * N * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, M * N * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);

    // call kernel
    sigmoid_kernel<<<ceil(N / 256.f), 256>>>(d_mat, d_mat2, M * N);

    // copy data back to host
    cudaMemcpy(mat2, d_mat2, M * N * sizeof(nn_real), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_mat);
    cudaFree(d_mat2);
}

/*
Kernel called to calculate sigmoid elementwise on a matrix
*/
__global__ void sigmoid_kernel(const nn_real *mat, nn_real *mat2, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate sigmoid
    while(idx < size){
        mat2[idx] = 1 / (1 + exp(-mat[idx]));
        idx += blockDim.x * gridDim.x;  
    }
}

/*
Routine to calculate softmax along first dimension of a matrix
*/
void softmax_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{
    // Set size of output matrix
    //mat2.set_size(mat.n_rows, mat.n_cols);
    //ASSERT_MAT_SAME_SIZE(mat, mat2);

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, M * N * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, M * N * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);

    // call kernel
    softmax_kernel<<<ceil(N / 256.f), 256>>>(d_mat, d_mat2, M, N);

    // copy data back to host
    cudaMemcpy(mat2, d_mat2, M * N * sizeof(nn_real), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_mat);
    cudaFree(d_mat2);
}

/*
Kernel called to calculate softmax along first dimension of a matrix
*/
__global__ void softmax_kernel(const nn_real *mat, nn_real *mat2, int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    while(idx < N){
        // Find max value in each column
        nn_real max_val = - 1<<30;
        nn_real sum = 0.0f;

        for (int k = 0; k < M; k++)
        {
            nn_real val = mat[k * N + idx];
            if (val > max_val)
            {
                max_val = val;
            }
        }

        // Calculate denominator
        for (int k = 0; k < M; k++)
        {
            sum += exp(mat[k * N + idx] - max_val);
        }

        // Calculate softmax
        for (int k = 0; k < M; k++)
        {
            mat2[k * N + idx] = exp(mat[k * N + idx] - max_val) / sum;
        }

        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to repeat a matrix N, M times in each dimension, respectively
*/
void repmat_gpu(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N)
{
    // Set size of output matrix
    //int M = M * mat.n_rows;
    //int N = N * mat.n_cols;
    //mat2.set_size(M, N);

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, K * L * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, K * L * M * N * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat, K * L * sizeof(nn_real), cudaMemcpyHostToDevice);

    // call kernel
    dim3 threads(32, 32);
    dim3 blocks(ceil(M / 32.f), ceil(N / 32.f)); 
    repmat_kernel<<<blocks, threads>>>(d_mat, d_mat2, K, L, M, N);

    // copy data back to host
    cudaMemcpy(mat2, d_mat2, K * L * M * N * sizeof(nn_real), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_mat);
    cudaFree(d_mat2);
}

/*
Kernel called to repeat a matrix N, M times in each dimension, respectively
*/
__global__ void repmat_kernel(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N) {

    uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
    uint j = (blockIdx.y * blockDim.y) + threadIdx.y; 

    // repeat matrix
    while(i < N){
        while(j < M){
            mat2[j * N + i] = mat[i%L + (j%K) * L]; 
            j += blockDim.y * gridDim.y;
        }
        i += blockDim.x * gridDim.x;
    }
}

/*
Routine to add two matrices mat3 = alpha * mat + beta * mat2
*/
void addmat_gpu(const nn_real* mat, const nn_real* mat2, nn_real* mat3, 
                nn_real alpha, nn_real beta, int M, int N)
{
    // Set size of output matrix
    //ASSERT_MAT_SAME_SIZE(mat, mat2);
    //int M = mat.n_rows;
    //int N = mat.n_cols;
    //mat3.set_size(M, N);

    // allocate device memory
    nn_real *d_mat, *d_mat2, *d_mat3;
    cudaMalloc((void **)&d_mat, M * N * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, M * N * sizeof(nn_real));
    cudaMalloc((void **)&d_mat3, M * N * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);

    // call kernel
    addmat_kernel<<<ceil(M * N / 256.f), 256>>>(d_mat, d_mat2, d_mat3, alpha, beta, M * N);

    // copy data back to host
    cudaMemcpy(mat3, d_mat3, M * N * sizeof(nn_real), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_mat);
    cudaFree(d_mat2);
    cudaFree(d_mat3);
}

/*
Kernel called to add two matrices mat3 = alpha * mat + beta * mat2
*/
__global__ void addmat_kernel(const nn_real* mat, const nn_real* mat2, nn_real* mat3, nn_real alpha, 
                            nn_real beta, int size) {
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // add matrices
    while(idx < size){
        mat3[idx] = alpha * mat[idx] + beta * mat2[idx];
        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to multiply two matrices element-wise mat2 = alpha * mat .* mat2
*/
void elemmultmat_gpu(const nn_real* mat, nn_real* mat2, nn_real alpha, int M, int N)
{
    // Set size of output matrix
    //ASSERT_MAT_SAME_SIZE(mat, mat2);
    //int M = mat.n_rows;
    //int N = mat.n_cols;

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, M * N * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, M * N * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, mat2, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);

    // call kernel
    elemmultmat_kernel<<<ceil(M * N / 256.f), 256>>>(d_mat, d_mat2, alpha, M * N);

    // copy data back to host
    cudaMemcpy(mat2, d_mat2, M * N * sizeof(nn_real), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_mat);
    cudaFree(d_mat2);
}

/*
Kernel called to multiply two matrices element-wise mat2 = alpha * mat .* mat2
*/
__global__ void elemmultmat_kernel(const nn_real* mat, nn_real* mat2, nn_real alpha, 
                                    int size) {
    uint idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // multiply matrices
    while(idx < size){
        mat2[idx] = alpha * mat[idx] * mat2[idx];
        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to transpose a matrix
*/
void transpose_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{
    // Set size of output matrix
    //int M = mat.n_rows;
    //int N = mat.n_cols;
    //mat2.set_size(N, M);

    // allocate device memory
    nn_real *d_mat, *d_mat2;
    cudaMalloc((void **)&d_mat, M * N * sizeof(nn_real));
    cudaMalloc((void **)&d_mat2, M * N * sizeof(nn_real));

    // copy data to device
    cudaMemcpy(d_mat, mat, M * N * sizeof(nn_real), cudaMemcpyHostToDevice);

    // call kernel
    dim3 threads(32, 32);
    dim3 blocks(ceil(M / 32.f), ceil(N / 32.f));
    transpose_kernel<<<blocks, threads, 32 * 32 * sizeof(nn_real)>>>(d_mat, d_mat2, M, N);

    // copy data back to host
    cudaMemcpy(mat2, d_mat2, M * N * sizeof(nn_real), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_mat);
    cudaFree(d_mat2);
}

/*
Routine called to transpose a matrix
*/
__global__ void transpose_kernel(const nn_real* mat, nn_real* mat2, int M, int N)
{   
    // shared memory
    __shared__ float s[32][33];

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    int width = gridDim.x * 32;

    // load block into shared memory
    for (int j = 0; j < 32; j += blockDim.y){
        s[threadIdx.y+j][threadIdx.x] = mat[(y+j)*width + x];
    }

    __syncthreads();

    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // transpose and write block from shared memory
    for (int j = 0; j < 32; j += blockDim.y){
        mat2[(y+j)*width + x] = s[threadIdx.x][threadIdx.y + j];
    }
}
