#include "tests.h"

#include <chrono>
#include <fstream>
#include <iomanip>

#include "../gpu_func.h"
#include "common.h"
#include "cublas_v2.h"
#include "mpi.h"
using namespace std;

#define NUM_ITERS 4 // Number of GEMMs run for timing purposes

#ifdef USE_DOUBLE
#define SCALE 4   // Factor to SCALE the GEMM problem size by
#define TOL 1e-14 // Tolerance for tests
#else
#define SCALE 10 // Factor to SCALE the GEMM problem size by
#define TOL 2e-6 // Tolerance for tests
#endif

// check whether the matrix from Seq is the same as from Par.
// write out mismatches to a file.
int checkErrors(const arma::Mat<nn_real> &Seq, const arma::Mat<nn_real> &Par,
                std::ofstream &ofs, std::vector<nn_real> &errors)
{
  int error = 0;

  for (int i = 0; i < Seq.n_rows; ++i)
  {
    for (int j = 0; j < Seq.n_cols; ++j)
    {
      if (abs(Seq(i, j) - Par(i, j)) > TOL)
      {
        ofs << "Mismatch at pos (" << i << ", " << j
            << ") diff: " << Seq(i, j) - Par(i, j) << " seq: " << Seq(i, j)
            << " par: " << Par(i, j) << endl;
        ++error;
      }
    }
  }

  if (error)
  {
    ofs << "There were " << error
        << " total locations where there was a difference between the seq and "
           "par"
        << endl;
  }
  else
  {
    ofs << "No errors were found" << endl;
  }

  nn_real err_max = arma::norm(Seq - Par, "inf") / arma::norm(Seq, "inf");
  nn_real err_l2 = arma::norm(Seq - Par, 2) / arma::norm(Seq, 2);

  if (err_max > TOL * 1e2)
  {
    cout << "Correctness test failed" << endl;
  }

  errors.push_back(err_max);
  errors.push_back(err_l2);

  return error;
}

int checkNNErrors(NeuralNetwork &seq_nn, NeuralNetwork &par_nn,
                  std::string filename)
{
  std::vector<nn_real> errors_w, errors_b;
  int error = 0;
  std::ofstream ofs(filename.c_str());
  if (!ofs.good())
  {
    std::cerr << "Unable to open the file " << filename << std::endl;
    exit(1);
  }

  for (int i = 0; i < seq_nn.num_layers; i++)
  {
    ofs << "Mismatches for W[" << i << "]" << endl;
    error += checkErrors(seq_nn.W[i], par_nn.W[i], ofs, errors_w);
    ofs << "Mismatches for b[" << i << "]" << endl;
    error += checkErrors(seq_nn.b[i], par_nn.b[i], ofs, errors_b);

    // Writing to file
    ofs << "Max norm of diff b/w seq and par: W[" << i
        << "]: " << setprecision(6) << errors_w[2 * i] << ", b[" << i
        << "]: " << errors_b[2 * i] << endl;
    ofs << "l2  norm of diff b/w seq and par: W[" << i
        << "]: " << setprecision(6) << errors_w[2 * i + 1] << ", b[" << i
        << "]: " << errors_b[2 * i + 1] << endl;

    // Writing to standard output
    cout << "Max norm of diff b/w seq and par: W[" << i
         << "]: " << setprecision(6) << errors_w[2 * i] << ", b[" << i
         << "]: " << errors_b[2 * i] << endl;
    cout << "l2  norm of diff b/w seq and par: W[" << i
         << "]: " << setprecision(6) << errors_w[2 * i + 1] << ", b[" << i
         << "]: " << errors_b[2 * i + 1] << endl;
  }

  ofs.close();
  return error;
}

void createMATS(nn_real *A, nn_real *B, nn_real *C1, nn_real *C2, int NI,
                int NJ, int NK)
{
  int i, j;

  for (j = 0; j < NK; j++)
  {
    for (i = 0; i < NI; i++)
    {
      A[i + j * NI] = ((nn_real)i * j) / NI;
    }
  }

  for (j = 0; j < NJ; j++)
  {
    for (i = 0; i < NK; i++)
    {
      B[i + j * NK] = ((nn_real)i * j + 1) / NJ;
    }
  }

  for (j = 0; j < NJ; j++)
  {
    for (i = 0; i < NI; i++)
    {
      C1[i + j * NI] = 0;
      C2[i + j * NI] = ((nn_real)i * j + 2) / NJ;
    }
  }
}

int compareGEMMResults(nn_real *myC, nn_real *refC, int NI, int NJ)
{
  int i, j;
  int fail = 0;

  arma::Mat<nn_real> mysol = arma::Mat<nn_real>(myC, NI, NJ, false);
  arma::Mat<nn_real> refsol = arma::Mat<nn_real>(refC, NI, NJ, false);

  nn_real reldiff =
      arma::norm(mysol - refsol, "inf") / arma::norm(refsol, "inf");

  if (reldiff > TOL)
  {
    fail = 1;
  }

  // Print results
  if (fail)
  {
    std::cout << "My GEMM output not matching with reference. Rel diff = "
              << reldiff << std::endl;
  }
  else
  {
    std::cout << "GEMM matched with reference successfully! Rel diff = "
              << reldiff << std::endl;
  }

  return fail;
}

void TestGEMM(int M, int N, int K)
{
  nn_real *A;
  nn_real *B;
  nn_real *C1;
  nn_real *C2;

  nn_real *dA;
  nn_real *dB;
  nn_real *dC1;
  nn_real *dC2;
  nn_real *dummy;

  nn_real alpha = 2.0;
  nn_real beta = 5.0;

  int num_iters = 100;

  A = (nn_real *)malloc(M * K * sizeof(nn_real));
  B = (nn_real *)malloc(K * N * sizeof(nn_real));
  C1 = (nn_real *)malloc(M * N * sizeof(nn_real));
  C2 = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * K);
  cudaMalloc((void **)&dB, sizeof(nn_real) * K * N);
  cudaMalloc((void **)&dC1, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dC2, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dummy, sizeof(nn_real) * M * N);

  // C1 and C2 are same. We just have two copies to compare results
  createMATS(A, B, C1, C2, M, N, K);

  cudaMemcpy(dA, A, sizeof(nn_real) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(nn_real) * K * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC1, C2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dC2, C2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dummy, C2, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  /* Warm up GPU before we run. We run one extra CuBlas */
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  stat = cublasCreate(&handle);

  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS initialization failed!" << std::endl;
    return;
  }

  stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M,
                     dB, K, &beta, dummy, M);

  /* Compute reference solution and time cuBLAS */
  using namespace std::chrono;
  high_resolution_clock::time_point ref_t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++)
  {
    stat = cublas_gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M,
                       dB, K, &beta, dC2, M);
  }

  check_launch("Reference GEMM");
  high_resolution_clock::time_point ref_t2 = high_resolution_clock::now();
  duration<double> ref_time_span =
      duration_cast<duration<double>>(ref_t2 - ref_t1);

  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    std::cerr << "CUBLAS gemm error at " << __FILE__ << ":" << __LINE__
              << std::endl;
  }

  cudaMemcpy(C2, dC2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  /* We are calling your GEMM function here */
  /* We will make one dummy call and check_launch here */
  int err;
  err = myGEMM(dA, dB, dummy, alpha, beta, M, N, K);
  check_launch("myGEMM dummy");

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++)
  {
    err = myGEMM(dA, dB, dC1, alpha, beta, M, N, K);
  }

  check_launch("myGEMM");
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> my_time_span = duration_cast<duration<double>>(t2 - t1);

  /* This error code is for your own debugging, it does not catch
     illegal memory accesses or bad kernel launches */
  if (err != 0)
  {
    std::cout << "Error in my GEMM. Error code: " << err << std::endl;
  }

  cudaMemcpy(C1, dC1, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareGEMMResults(C1, C2, M, N);

  if (fail == 0)
  {
    std::cout << "Time for reference GEMM implementation: "
              << ref_time_span.count() << " seconds" << std::endl;
    std::cout << "Time for my GEMM implementation: " << my_time_span.count()
              << " seconds" << std::endl;
  }

  free(A);
  free(B);
  free(C1);
  free(C2);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC1);
  cudaFree(dC2);
  cudaFree(dummy);
}

void BenchmarkGEMM()
{
  std::cout << std::endl
            << "Entering GEMM Benchmarking mode! Stand by." << std::endl;

  /* First GEMM problem size */
  int M = 800 * SCALE, N = 1000 * SCALE, K = 784 * SCALE;

  std::cout << std::endl
            << "Starting GEMM 1: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 1" << std::endl;

  /* Second GEMM problem size */
  M = 800 * SCALE, N = 100 * SCALE, K = 1000 * SCALE;
  std::cout << std::endl
            << "Starting GEMM 2: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 2" << std::endl;

  /* Third GEMM problem size */
  M = 800 * SCALE, N = 10 * SCALE, K = 1000 * SCALE;
  std::cout << std::endl
            << "Starting GEMM 3: "
            << "M = " << M << "; N = " << N << "; K = " << K << std::endl;
  TestGEMM(M, N, K);
  std::cout << "Completed GEMM 3" << std::endl;
}
