#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

// Define struct for facilitating computations in gpuGEMM4d1
// - idea for Matrix, GetElement(), and GetSubMatrix() came from CUDA C++ Programming Guide
// - Note here matrices are stored in column-major order:
// - I.e. M(row, col) = *(M.elements + col * M.stride + row)
typedef struct
{
    int width;
    int height;
    int stride;
    double *elements;
} Matrix;

__device__ 
double GetElement(const Matrix A, int row, int col);

__device__ 
Matrix GetSubMatrix(Matrix B, int row, int col, int height, int width);

__global__
void gpuGEMM4d1(const Matrix A, const Matrix B, Matrix C, double alpha, double beta,
           int M, int N, int K);

__global__
void gpuGEMM4d2(const Matrix A, const Matrix B, Matrix C, double alpha, double beta,
           int M, int N, int K);

__global__
void gpuGEMM(double* A, double* B, double* C, double alpha, double beta, int M,
           int N, int K);

__global__
void naiveGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double alpha, double beta,
           int M, int N, int K);

__global__
void repmatKernel(double* mat1, double* mat2, int M, int N);

int myNaiveGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

void GPUrepmat(double* mat, double* mat2, int M, int N);

void GPUsigmoid(double* mat, double* mat2, int M, int N);

void GPUsoftmax(double* mat, double* mat2, int M, int N); 

void GPUaddition(double* mat, double* mat2, double* output_mat, double alpha, double beta, int M, int N);

void GPUsum(double* mat, double* output_vec, int M, int N, int dim);

void GPUtranspose(double* mat, double* output_mat, int M, int N);

void GPUelemwise(double* mat1, double* mat2, double* output_mat, int M, int N);

__global__
void sigmoidKernel(double* mat1, double* mat2, int M, int N);

__global__
void exponentialKernel(double* mat1, double* mat2, int M, int N);

__global__
void softmaxKernel(double* mat1, double* mat2, int M, int N);

__global__
void sum(double* mat1, double* mat2, int M, int N, int dim);

__global__
void repmat(double* mat1, double* mat2, int M, int N);

__global__
void addmat(double* mat1, double* mat2, double* output_mat, int M, int N);

__global__
void transpose(double* mat, double* output_mat, int M, int N);

__global__ 
void elemwise(double *mat1, double *mat2, double *output_mat, int M, int N);

#endif
