#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

/*
Routine to calculate the general matrix multiplication C = alpha * A * B + beta * C
*/
int myGEMM(const nn_real *__restrict__ A, const nn_real *__restrict__ B, 
        nn_real *__restrict__ C, nn_real alpha, nn_real beta, int M, int N, int K)
{
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ceil(N / 32.f), ceil(M / 32.f));
    myGEMM_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

/*
Kernel called to calculate the general matrix multiplication C = alpha * A * B + beta * C
*/
__global__ void myGEMM_kernel(const nn_real *A, const nn_real *B, nn_real *C, 
                            nn_real alpha, nn_real beta, int M, int N, int K)
{
    __shared__ nn_real As[32][32];
    __shared__ nn_real Bs[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + ty;
    int j = blockIdx.x * blockDim.x + tx;

    nn_real Cvalue = 0.0;

    int num_iters = (K + 31) / 32;

    for (int m = 0; m < num_iters; ++m)
    {
        if (i < M && m * 32 + tx < K)
            As[ty][tx] = A[(m * 32 + tx) * M + i];
        else
            As[ty][tx] = 0.0;

        if (j < N && m * 32 + ty < K)
            Bs[ty][tx] = B[j * K + m * 32 + ty];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        int calc_size = (m == num_iters - 1 && K % 32 != 0) ? K % 32 : 32;
        
        for (int e = 0; e < calc_size; ++e)
            Cvalue += As[ty][e] * Bs[e][tx];

        __syncthreads();
    }

    if (i < M && j < N)
        C[j * M + i] = alpha * Cvalue + beta * C[j * M + i];
}


/*
Routine to calculate sigmoid elementwise on a matrix
*/
void sigmoid_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{
    sigmoid_kernel<<<ceil(M * N / 256.f), 256>>>(mat, mat2, M * N);
}

/*
Kernel called to calculate sigmoid elementwise on a matrix
*/
__global__ void sigmoid_kernel(const nn_real *mat, nn_real *mat2, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
    softmax_kernel<<<ceil(N / 256.f), 256>>>(mat, mat2, M, N);
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
            nn_real val = mat[idx * M + k];
            if (val > max_val)
            {
                max_val = val;
            }
        }

        // Calculate denominator
        for (int k = 0; k < M; k++)
        {
            sum += exp(mat[idx * M + k] - max_val);
        }

        // Calculate softmax
        for (int k = 0; k < M; k++)
        {
            mat2[idx * M + k] = exp(mat[idx * M + k] - max_val) / sum;
        }

        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to repeat a matrix N, M times in each dimension, respectively
*/
void repmat_gpu(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N)
{
    dim3 threads(32, 32);
    dim3 blocks(ceil(N * L / 32.f), ceil(K * M / 32.f)); 
    repmat_kernel<<<blocks, threads>>>(mat, mat2, K, L, M, N);
}

/*
Kernel called to repeat a matrix N, M times in each dimension, respectively
*/
__global__ void repmat_kernel(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N)
{
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    int i = (blockIdx.y * blockDim.y) + threadIdx.y; 

    while(i < M * K){
        while(j < N * L){
            mat2[j * (K * M) + i] = mat[i%K + (j%L) * K]; 
            j += blockDim.x * gridDim.x;
        }
        i += blockDim.y * gridDim.y;
    }
}

/*
Routine to add two matrices mat3 = alpha * mat + beta * mat2
*/
void addmat_gpu(const nn_real* mat, nn_real* mat2, nn_real* mat3, 
                nn_real alpha, nn_real beta, int M, int N)
{
    addmat_kernel<<<ceil(M * N / 256.f), 256>>>(mat, mat2, mat3, alpha, beta, M * N);
}

/*
Kernel called to add two matrices mat3 = alpha * mat + beta * mat2
*/
__global__ void addmat_kernel(const nn_real* mat, nn_real* mat2, nn_real* mat3, nn_real alpha, 
                            nn_real beta, int size)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

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
    elemmultmat_kernel<<<ceil(M * N / 256.f), 256>>>(mat, mat2, alpha, M * N);
}

/*
Kernel called to multiply two matrices element-wise mat2 = alpha * mat .* mat2
*/
__global__ void elemmultmat_kernel(const nn_real* mat, nn_real* mat2, nn_real alpha, 
                                    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while(idx < size){
        mat2[idx] = alpha * mat[idx] * mat2[idx];
        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to calculate the transpose mat2 = mat1^T
*/
void transpose_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{
    dim3 threads(32, 8);
    dim3 blocks(ceil(M / 32.f), ceil(N / 32.f));
    transpose_kernel<<<blocks, threads>>>(mat, mat2, M, N);
}

/*
Kernel called to calculate the transpose mat2 = mat1^T
*/
__global__ void transpose_kernel(const nn_real* mat, nn_real* mat2, int M, int N)
{
    __shared__ nn_real tile[32][33];

    int xIndex = blockIdx.x * 32 + threadIdx.x;
    int yIndex = blockIdx.y * 32 + threadIdx.y;

    for (int m = 0; m < 32; m += 8) {
        if(xIndex < M && (yIndex + m) < N)
            tile[threadIdx.y + m][threadIdx.x] = mat[(yIndex + m) * M + xIndex];
    }

    __syncthreads();

    xIndex = blockIdx.y * 32 + threadIdx.x;
    yIndex = blockIdx.x * 32 + threadIdx.y;

    for (int m = 0; m < 32; m += 8) {
        if(xIndex < N && (yIndex + m) < M)
            mat2[(yIndex + m) * N + xIndex] = tile[threadIdx.x][threadIdx.y + m];
    }
}

void equal_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{
    equal_kernel<<<ceil(M * N / 256.f), 256>>>(mat, mat2, M, N);
}

__global__ void equal_kernel(const nn_real* mat, nn_real* mat2, int M, int N)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    while(idx < M * N){
        mat2[idx] = mat[idx];
        idx += blockDim.x * gridDim.x;
    }
}

void sum_axis1_gpu(const nn_real* mat, nn_real* mat2, int M, int N)
{
    sum_axis1_kernel<<<ceil(M / 256.f), 256>>>(mat, mat2, M, N);
}

__global__ void sum_axis1_kernel(const nn_real* mat, nn_real* mat2, int M, int N)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    while(idx < M){
        mat2[idx] = 0.0;
        for (int k = 0; k < N; k++){
            mat2[idx] += mat[idx + k * M];
        }
        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to add a scalar to a matrix mat2 = alpha + beta * mat
*/
void scalaradd_gpu(const nn_real* mat, nn_real* mat2, int M, int N, nn_real alpha, nn_real beta)
{
    scalaradd_kernel<<<ceil(M * N / 256.f), 256>>>(mat, mat2, M, N, alpha, beta);
}

/*
Kernel called to add a scalar to a matrix mat2 = alpha + beta * mat
*/
__global__ void scalaradd_kernel(const nn_real* mat, nn_real* mat2, int M, int N, 
                            nn_real alpha, nn_real beta)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    while(idx < M * N){
        mat2[idx] = alpha + beta * mat[idx];
        idx += blockDim.x * gridDim.x;
    }
}

void setzero_gpu(nn_real* mat, int M, int N)
{
    setzero_kernel<<<ceil(M * N / 256.f), 256>>>(mat, M, N);
}

__global__ void setzero_kernel(nn_real* mat, int M, int N)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    while(idx < M * N){
        mat[idx] = 0.0;
        idx += blockDim.x * gridDim.x;
    }
}

void softmax_der_gpu(const nn_real* mat, const nn_real* mat2, nn_real* mat3, int M, int N)
{
    softmax_der_kernel<<<ceil(M * N / 256.f), 256>>>(mat, mat2, mat3, M, N);
}

__global__ void softmax_der_kernel(const nn_real* mat, const nn_real* mat2, nn_real* mat3, int M, int N)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    while(idx < M * N){
        mat3[idx] = mat[idx] * mat2[idx] * (1 - mat2[idx]);
        idx += blockDim.x * gridDim.x;
    }
}

/*
Routine to do the forward pass through the neural network nn
*/
void feedforward_gpu(int n0, int n1, int n2, int nbatch, nn_real* d_X,
    nn_real* d_W1, nn_real* d_W2, nn_real* d_b1, nn_real* d_b2, nn_real* d_a1, 
    nn_real* d_a2, nn_real* d_z1, nn_real* d_z2)
{
    // Forward propagate through first dense layer
    repmat_gpu(d_b1, d_z1, n1, 1, 1, nbatch);
    myGEMM(d_W1, d_X, d_z1, 1.0, 1.0, n1, nbatch, n0);

    // Apply sigmoid
    sigmoid_gpu(d_z1, d_a1, n1, nbatch);

    // Forward propagate through second dense layer
    repmat_gpu(d_b2, d_z2, n2, 1, 1, nbatch);
    myGEMM(d_W2, d_a1, d_z2, 1.0, 1.0, n2, nbatch, n1);

    // Apply softmax
    softmax_gpu(d_z2, d_a2, n2, nbatch);
}

/*
Routine to do the backward pass through the neural network nn
*/
void backprop_gpu(int n0, int n1, int n2, int nbatch, nn_real* d_a1, nn_real* d_a2,
                nn_real* d_z1, nn_real* d_z2, nn_real* d_W1, nn_real* d_W2, nn_real* d_X,
                nn_real* d_b1, nn_real* d_b2, nn_real* d_y, nn_real* d_db1, nn_real* d_db2,
                nn_real* d_dW1, nn_real* d_dW2, nn_real* d_h1, nn_real* d_h2, nn_real* d_h3,
                nn_real* d_h4, nn_real* d_h5, nn_real* d_h6, nn_real reg)
{
    // Calculate gradients in last layer
    addmat_gpu(d_a2, d_y, d_h1, 1.f / nbatch, -1.f / nbatch, n2, nbatch);
    transpose_gpu(d_a1, d_h2, n1, nbatch);
    equal_gpu(d_W2, d_dW2, n2, n1);

    myGEMM(d_h1, d_h2, d_dW2, 1.0, reg, n2, n1, nbatch);
    sum_axis1_gpu(d_h1, d_db2, n2, nbatch);
    
    // Calculate gradients in first layer
    transpose_gpu(d_W2, d_h3, n2, n1);
    myGEMM(d_h3, d_h1, d_h4, 1.0, 0.0, n1, nbatch, n2);
    softmax_der_gpu(d_h4, d_a1, d_h5, n1, nbatch);
    transpose_gpu(d_X, d_h6, n0, nbatch);
    equal_gpu(d_W1, d_dW1, n1, n0);

    myGEMM(d_h5, d_h6, d_dW1, 1.0, reg, n1, n0, nbatch);
    sum_axis1_gpu(d_h5, d_db1, n1, nbatch);
}

/*
Routine to do the backward pass through the neural network nn
*/
void gradient_descent_gpu(int n0, int n1, int n2, 
                        nn_real* d_W1, nn_real* d_W2, nn_real* d_b1, nn_real* d_b2,
                        nn_real* d_db1, nn_real* d_db2, nn_real* d_dW1, nn_real* d_dW2, 
                        nn_real learning_rate)
{
    // Update weights
    addmat_gpu(d_dW1, d_W1, d_W1, -learning_rate, 1.0, n1, n0);
    addmat_gpu(d_dW2, d_W2, d_W2, -learning_rate, 1.0, n2, n1);

    // Update biases
    addmat_gpu(d_db1, d_b1, d_b1, -learning_rate, 1.0, n1, 1);
    addmat_gpu(d_db2, d_b2, d_b2, -learning_rate, 1.0, n2, 1);
}

/*
Routine to do a prediction using a neural network nn
*/
void predict_gpu(nn_real* label, int n0, int n1, int n2, int nbatch, nn_real* d_X,
    nn_real* d_W0, nn_real* d_W1, nn_real* d_b0, nn_real* d_b1, nn_real* d_a1, 
    nn_real* d_yc, nn_real* d_z1, nn_real* d_z2)
{ 
    feedforward_gpu(n0, n1, n2, nbatch, d_X, d_W0, d_W1, d_b0, d_b1, d_a1, d_yc, d_z1, d_z2);
    argmax<<<ceil(nbatch / 256.f), 256>>>(d_yc, label, n2, nbatch);
}

__global__ void argmax(const nn_real* values, nn_real* labels, int M, int N)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    while (idx < N){
        nn_real max = values[idx * M];
        for (int k = 0; k < M; k++){
            if (values[idx * M + k] > max){
                max = values[idx * M + k];
                labels[idx] = k;
            }
        }
        idx += blockDim.x * gridDim.x;
    }
}
