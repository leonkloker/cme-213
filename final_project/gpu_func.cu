#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define BLOCK_SIZE 32

/*
Routine to calculate the general matrix multiplication C = alpha * A * B + beta * C
*/
int myGEMM(const nn_real *__restrict__ A, const nn_real *__restrict__ B, 
        nn_real *__restrict__ C, nn_real alpha, nn_real beta, int M, int N, int K)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1)/ BLOCK_SIZE, (M + BLOCK_SIZE - 1)/ BLOCK_SIZE);
    myGEMM_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, alpha, beta, M, N, K);
    return 0;
}

/*
Kernel called to calculate the general matrix multiplication C = alpha * A * B + beta * C
*/
__global__ void myGEMM_kernel(const nn_real *A, const nn_real *B, nn_real *C, 
                            nn_real alpha, nn_real beta, int M, int N, int K)
{
    __shared__ nn_real s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ nn_real s_B[BLOCK_SIZE][BLOCK_SIZE];

    int j_loc = threadIdx.x;
    int i_loc = threadIdx.y;
    int i = blockIdx.y * blockDim.y + i_loc;
    int j = blockIdx.x * blockDim.x + j_loc;

    nn_real sum = 0.0;

    int niters = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < niters; ++m){
        if (i < M && m * BLOCK_SIZE + j_loc < K){
            s_A[i_loc][j_loc] = A[(m * BLOCK_SIZE + j_loc) * M + i];
        }else{
            s_A[i_loc][j_loc] = 0.0;
        }

        if (j < N && m * BLOCK_SIZE + i_loc < K){
            s_B[i_loc][j_loc] = B[j * K + m * BLOCK_SIZE + i_loc];
        }else{
            s_B[i_loc][j_loc] = 0.0;
        }
        __syncthreads();

        int size = (m == niters - 1 && K % BLOCK_SIZE != 0) ? 
        (K % BLOCK_SIZE) : BLOCK_SIZE;
        
        for (int k = 0; k < size; k++){
            sum += s_A[i_loc][k] * s_B[k][j_loc];
        }
        __syncthreads();
    }

    if (i < M && j < N){
        C[j * M + i] = alpha * sum + beta * C[j * M + i];
    }
}

/*
Routine to calculate the matrix multiplication D = A * B + repmat(C, 1, N)
*/
int GEMM_forward(const nn_real *__restrict__ A, const nn_real *__restrict__ B, 
        const nn_real *__restrict__ C, nn_real *__restrict__ D, int M, int N, int K)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1)/ BLOCK_SIZE, (M + BLOCK_SIZE - 1)/ BLOCK_SIZE);
    GEMM_forward_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, D, M, N, K);
    return 0;
}

/*
Routine to calculate the matrix multiplication D = A * B + alpha * C
*/
int GEMM_backward(const nn_real *__restrict__ A, const nn_real *__restrict__ B, 
        const nn_real *__restrict__ C, nn_real *__restrict__ D, nn_real alpha, int M, int N, int K)
{
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1)/ BLOCK_SIZE, (M + BLOCK_SIZE - 1)/ BLOCK_SIZE);
    GEMM_backward_kernel<<<numBlocks, threadsPerBlock>>>(A, B, C, D, alpha, M, N, K);
    return 0;
}

/*
Kernel called to calculate the matrix multiplication D = A * B^T + alpha * C
*/
__global__ void GEMM_backward_kernel(const nn_real *A, const nn_real *B, const nn_real* C,
                                    nn_real* D, nn_real alpha, int M, int N, int K)
{
    __shared__ nn_real s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ nn_real s_B[BLOCK_SIZE][BLOCK_SIZE];

    int j_loc = threadIdx.x;
    int i_loc = threadIdx.y;
    int i = blockIdx.y * blockDim.y + i_loc;
    int j = blockIdx.x * blockDim.x + j_loc;

    nn_real sum = 0.0;

    int niters = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < niters; ++m){
        if (i < M && m * BLOCK_SIZE + j_loc < K){
            s_A[i_loc][j_loc] = A[(m * BLOCK_SIZE + j_loc) * M + i];
        }else{
            s_A[i_loc][j_loc] = 0.0;
        }

        if (j < N && m * BLOCK_SIZE + i_loc < K){
            s_B[i_loc][j_loc] = B[j * K + m * BLOCK_SIZE + i_loc];
        }else{
            s_B[i_loc][j_loc] = 0.0;
        }
        __syncthreads();

        int size = (m == niters - 1 && K % BLOCK_SIZE != 0) ? (K % BLOCK_SIZE) : BLOCK_SIZE;
        
        for (int k = 0; k < size; k++){
            sum += s_A[i_loc][k] * s_B[k][j_loc];
        }
        __syncthreads();
    }

    if (i < M && j < N){
        D[j * M + i] = sum + alpha * C[j * M + i];
    }
}


/*
Kernel called to calculate the matrix multiplication D = A * B + repmat(C, 1, N)
*/
__global__ void GEMM_forward_kernel(const nn_real *A, const nn_real *B, const nn_real *C, 
                                    nn_real* D, int M, int N, int K)
{
    __shared__ nn_real s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ nn_real s_B[BLOCK_SIZE][BLOCK_SIZE];

    int j_loc = threadIdx.x;
    int i_loc = threadIdx.y;
    int i = blockIdx.y * blockDim.y + i_loc;
    int j = blockIdx.x * blockDim.x + j_loc;

    nn_real sum = 0.0;

    int niters = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < niters; ++m){
        if (i < M && m * BLOCK_SIZE + j_loc < K){
            s_A[i_loc][j_loc] = A[(m * BLOCK_SIZE + j_loc) * M + i];
        }else{
            s_A[i_loc][j_loc] = 0.0;
        }

        if (j < N && m * BLOCK_SIZE + i_loc < K){
            s_B[i_loc][j_loc] = B[j * K + m * BLOCK_SIZE + i_loc];
        }else{
            s_B[i_loc][j_loc] = 0.0;
        }
        __syncthreads();

        int size = (m == niters - 1 && K % BLOCK_SIZE != 0) ? (K % BLOCK_SIZE) : BLOCK_SIZE;
        
        for (int k = 0; k < size; k++){
            sum += s_A[i_loc][k] * s_B[k][j_loc];
        }
        __syncthreads();
    }

    if (i < M && j < N){
        D[j * M + i] = sum + C[i];
    }
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

        for (int k = 0; k < M; k++)
        {
            sum += exp(mat[idx * M + k] - max_val);
        }

        for (int k = 0; k < M; k++)
        {
            mat2[idx * M + k] = exp(mat[idx * M + k] - max_val) / sum;
        }

        idx += blockDim.x * gridDim.x;
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

void scalarmult_gpu(nn_real* mat, nn_real alpha, int M, int N)
{
    scalarmult_kernel<<<ceil(M * N / 256.f), 256>>>(mat, alpha, M * N);
}

__global__ void scalarmult_kernel(nn_real* mat, nn_real alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while(idx < size){
        mat[idx] = alpha * mat[idx];
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
    //myGEMM(d_W1, d_X, d_z1, (nn_real) 1.0, (nn_real) 1.0, n1, nbatch, n0);
    GEMM_forward(d_W1, d_X, d_b1, d_z1, n1, nbatch, n0);

    // Apply sigmoid
    sigmoid_gpu(d_z1, d_a1, n1, nbatch);

    // Forward propagate through second dense layer
    GEMM_forward(d_W2, d_a1, d_b2, d_z2, n2, nbatch, n1);

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
                nn_real reg, nn_real weight)
{
    // Calculate gradients in last layer
    addmat_gpu(d_a2, d_y, d_h1, (nn_real) 1 / nbatch, (nn_real) -1 / nbatch, n2, nbatch);
    transpose_gpu(d_a1, d_h2, n1, nbatch);
    GEMM_backward(d_h1, d_h2, d_W2, d_dW2, reg, n2, n1, nbatch);
    sum_axis1_gpu(d_h1, d_db2, n2, nbatch);
    
    // Calculate gradients in first layer
    transpose_gpu(d_W2, d_h2, n2, n1);
    myGEMM(d_h2, d_h1, d_h3, (nn_real) 1.0, (nn_real) 0.0, n1, nbatch, n2);
    softmax_der_gpu(d_h3, d_a1, d_h1, n1, nbatch);
    transpose_gpu(d_X, d_h3, n0, nbatch);
    GEMM_backward(d_h1, d_h3, d_W1, d_dW1, reg, n1, n0, nbatch);
    sum_axis1_gpu(d_h1, d_db1, n1, nbatch);
    
    // Rescale gradients according to batch size weight
    scalarmult_gpu(d_dW1, weight, n1, n0);
    scalarmult_gpu(d_dW2, weight, n2, n1);
    scalarmult_gpu(d_db1, weight, n1, 1);
    scalarmult_gpu(d_db2, weight, n2, 1);
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
    addmat_gpu(d_dW1, d_W1, d_W1, -learning_rate, (nn_real) 1.0, n1, n0);
    addmat_gpu(d_dW2, d_W2, d_W2, -learning_rate, (nn_real) 1.0, n2, n1);

    // Update biases
    addmat_gpu(d_db1, d_b1, d_b1, -learning_rate, (nn_real) 1.0, n1, 1);
    addmat_gpu(d_db2, d_b2, d_b2, -learning_rate, (nn_real) 1.0, n2, 1);
}
