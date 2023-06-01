#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utils/common.h"
#include "utils/gpu_util.h"

int myGEMM(nn_real *A, nn_real *B, nn_real *C, nn_real alpha, nn_real beta,
           int M, int N, int K);

__global__ void myGEMM_kernel(nn_real *A, nn_real *B, nn_real *C, nn_real alpha,
                              nn_real beta, int M, int N, int K);

#endif
