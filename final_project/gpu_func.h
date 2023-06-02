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

void addmat_gpu(const nn_real* mat, const nn_real* mat2, nn_real* mat3, 
                nn_real alpha, nn_real beta, int M, int N);

void elemmultmat_gpu(const nn_real* mat, nn_real* mat2, nn_real alpha, int M, int N);

void transpose_gpu(const nn_real* mat, nn_real* mat2, int M, int N);

__global__ void myGEMM_kernel(const nn_real* A, const nn_real* B, nn_real* C, nn_real alpha, 
nn_real beta, int M, int N, int K);

__global__ void sigmoid_kernel(const nn_real *mat, nn_real *mat2, int N);

__global__ void softmax_kernel(const nn_real *mat, nn_real *mat2, int M, int N);

__global__ void repmat_kernel(const nn_real* mat, nn_real* mat2, int K, int L, int M, int N);

__global__ void addmat_kernel(const nn_real* mat, const nn_real* mat2, nn_real* mat3,
                                nn_real alpha, nn_real beta, int size);
                            
__global__ void elemmultmat_kernel(const nn_real* mat, nn_real* mat2, nn_real alpha, int size);

__global__ void transpose_kernel(const nn_real* mat, nn_real* mat2, int M, int N);

#endif
