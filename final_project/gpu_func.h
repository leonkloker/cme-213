#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(const nn_real* A, const nn_real* B, nn_real* C, nn_real alpha,
            nn_real beta, int M, int N, int K);

int GEMM_forward(const nn_real *__restrict__ A, const nn_real *__restrict__ B, 
        const nn_real *__restrict__ C, nn_real *__restrict__ D, int M, int N, int K);

int GEMM_backward(const nn_real *__restrict__ A, const nn_real *__restrict__ B, 
        const nn_real *__restrict__ C, nn_real *__restrict__ D, nn_real alpha, int M, int N, int K);

void sigmoid_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void softmax_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void addmat_gpu(const nn_real* mat, nn_real* mat2, nn_real* mat3, 
                nn_real alpha, nn_real beta, int M, int N);

void elemmultmat_gpu(const nn_real* mat, nn_real* mat2, nn_real alpha, int M, int N);

void transpose_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void sum_axis1_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void scalaradd_gpu(const nn_real* mat, nn_real* mat2, int M, int N, nn_real alpha, nn_real beta);

void softmax_der_gpu(const nn_real* mat, const nn_real* mat2, nn_real* mat3, int M, int N);

void scalarmult_gpu(nn_real* mat, nn_real alpha, int M, int N);

__global__ void myGEMM_kernel(const nn_real* A, const nn_real* B, nn_real* C, nn_real alpha, 
nn_real beta, int M, int N, int K);

__global__ void GEMM_forward_kernel(const nn_real *A, const nn_real *B, const nn_real *C, 
                                    nn_real* D, int M, int N, int K);

__global__ void GEMM_backward_kernel(const nn_real *A, const nn_real *B, const nn_real* C,
                                    nn_real* D, nn_real alpha, int M, int N, int K);

__global__ void sigmoid_kernel(const nn_real *mat, nn_real *mat2, int N);

__global__ void softmax_kernel(const nn_real *mat, nn_real *mat2, int M, int N);

__global__ void addmat_kernel(const nn_real* mat, nn_real* mat2, nn_real* mat3,
                                nn_real alpha, nn_real beta, int size);
                            
__global__ void elemmultmat_kernel(const nn_real* mat, nn_real* mat2, nn_real alpha, int size);

__global__ void transpose_kernel(const nn_real* mat, nn_real* mat2, int M, int N);

__global__ void sum_axis1_kernel(const nn_real* mat, nn_real* mat2, int M, int N);

__global__ void scalaradd_kernel(const nn_real* mat, nn_real* mat2, int M, int N, 
                            nn_real alpha, nn_real beta);

__global__ void softmax_der_kernel(const nn_real* mat, const nn_real* mat2, nn_real* mat3, int M, int N);

__global__ void scalarmult_kernel(nn_real* mat, nn_real alpha, int size);

void feedforward_gpu(int n0, int n1, int n2, int nbatch, nn_real* d_X,
    nn_real* d_W1, nn_real* d_W2, nn_real* d_b1, nn_real* d_b2, nn_real* d_a1, 
    nn_real* d_a2, nn_real* d_z1, nn_real* d_z2);

void backprop_gpu(int n0, int n1, int n2, int nbatch, nn_real* d_a2, nn_real* d_a1,
                    nn_real* d_z2, nn_real* d_z1, nn_real* d_W1, nn_real* d_W2, nn_real* d_X,
                    nn_real* d_b2, nn_real* d_b1, nn_real* d_y, nn_real* d_db1, nn_real* d_db2,
                    nn_real* d_dW1, nn_real* d_dW2, nn_real* d_h1, nn_real* d_h2, nn_real* d_h3, 
                    nn_real reg, nn_real weight);

void gradient_descent_gpu(int n0, int n1, int n2, 
                        nn_real* d_W1, nn_real* d_W2, nn_real* d_b1, nn_real* d_b2,
                        nn_real* d_db1, nn_real* d_db2, nn_real* d_dW1, nn_real* d_dW2,
                        nn_real learning_rate);

#endif
