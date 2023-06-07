#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(const nn_real* A, const nn_real* B, nn_real* C, nn_real alpha,
            nn_real beta, int M, int N, int K);

void sigmoid_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void softmax_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void repmat_gpu(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N);

void addmat_gpu(const nn_real* mat, nn_real* mat2, nn_real* mat3, 
                nn_real alpha, nn_real beta, int M, int N);

void elemmultmat_gpu(const nn_real* mat, nn_real* mat2, nn_real alpha, int M, int N);

void transpose_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void equal_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void sum_axis1_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

void scalaradd_gpu(const nn_real* mat, nn_real* mat2, int M, int N, nn_real alpha, nn_real beta);

void setzero_gpu(nn_real* mat, int M, int N);

void softmax_der_gpu(const nn_real* mat, const nn_real* mat2, nn_real* mat3, int M, int N);

__global__ void myGEMM_kernel(const nn_real* A, const nn_real* B, nn_real* C, nn_real alpha, 
nn_real beta, int M, int N, int K);

__global__ void sigmoid_kernel(const nn_real *mat, nn_real *mat2, int N);

__global__ void softmax_kernel(const nn_real *mat, nn_real *mat2, int M, int N);

__global__ void repmat_kernel(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N);

__global__ void addmat_kernel(const nn_real* mat, nn_real* mat2, nn_real* mat3,
                                nn_real alpha, nn_real beta, int size);
                            
__global__ void elemmultmat_kernel(const nn_real* mat, nn_real* mat2, nn_real alpha, int size);

__global__ void transpose_kernel(const nn_real* mat, nn_real* mat2, int M, int N);

__global__ void equal_kernel(const nn_real* mat, nn_real* mat2, int M, int N);

__global__ void sum_axis1_kernel(const nn_real* mat, nn_real* mat2, int M, int N);

__global__ void scalaradd_kernel(const nn_real* mat, nn_real* mat2, int M, int N, 
                            nn_real alpha, nn_real beta);

__global__ void setzero_kernel(nn_real* mat, int M, int N);

__global__ void softmax_der_kernel(const nn_real* mat, const nn_real* mat2, nn_real* mat3, int M, int N);

__global__ void argmax(const nn_real* values, nn_real* labels, int M, int N);

void feedforward_gpu(int n0, int n1, int n2, int nbatch, nn_real* d_X,
    nn_real* d_W0, nn_real* d_W1, nn_real* d_b0, nn_real* d_b1, nn_real* d_a1, 
    nn_real* d_yc, nn_real* d_z1, nn_real* d_z2);

void predict_gpu(nn_real* label, int n0, int n1, int n2, int nbatch, nn_real* d_X,
    nn_real* d_W0, nn_real* d_W1, nn_real* d_b0, nn_real* d_b1, nn_real* d_a1, 
    nn_real* d_yc, nn_real* d_z1, nn_real* d_z2);

void backprop_gpu(int n0, int n1, int n2, int nbatch, nn_real* d_a2, nn_real* d_a1,
                    nn_real* d_z2, nn_real* d_z1, nn_real* d_W1, nn_real* d_W2, nn_real* d_X,
                    nn_real* d_b2, nn_real* d_b1, nn_real* d_y, nn_real* d_db1, nn_real* d_db2,
                    nn_real* d_dW1, nn_real* d_dW2, nn_real* d_h1, nn_real* d_h2, nn_real* d_h3, 
                    nn_real* d_h4, nn_real* d_h5, nn_real* d_h6, nn_real reg);

void gradient_descent_gpu(int n0, int n1, int n2, 
                        nn_real* d_W1, nn_real* d_W2, nn_real* d_b1, nn_real* d_b2,
                        nn_real* d_db1, nn_real* d_db2, nn_real* d_dW1, nn_real* d_dW2,
                        nn_real learning_rate);

#endif
