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

void TestSigmoid(int M, int N) {

    nn_real* mat;
    nn_real* mat2;
    nn_real* ref;
    nn_real* dMat;
    nn_real* dMat2;

    mat = (nn_real*)malloc(M*N*sizeof(nn_real));
    mat2 = (nn_real*)malloc(M*N*sizeof(nn_real));
    ref = (nn_real*)malloc(M*N*sizeof(nn_real));

    cudaMalloc((void**)&dMat, sizeof(nn_real) * M * N);
    cudaMalloc((void**)&dMat2, sizeof(nn_real) * M * N);

    // Initialization and calculation of reference output
    for(int i = 0; i < M*N; i++) {
        mat[i] = (nn_real)(rand() % 10);  // Replace with your own initialization if necessary
        ref[i] = 1.0 / (1.0 + exp(-mat[i]));  // Calculate the sigmoid of the initialized value
    }

    cudaMemcpy(dMat, mat, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

    /* Warm up GPU before we run. */
    sigmoid_gpu(dMat, dMat2, M, N);
    check_launch("sigmoid_gpu warm up");

    double refstart = MPI_Wtime();

    for(int i = 0; i < NUM_ITERS; i++) {
        sigmoid_gpu(dMat, dMat2, M, N);
    }

    check_launch("sigmoid_gpu");
    double refend = MPI_Wtime();

    cudaMemcpy(mat2, dMat2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

    int fail = compareGEMMResults(mat2, ref, M, N);

    if(fail == 0) {
        std::cout << "Time for reference sigmoid implementation: "
                  << refend - refstart << std::endl;
    }

    free(mat);
    free(mat2);
    free(ref);
    cudaFree(dMat);
    cudaFree(dMat2);
}

void TestSoftmax(int M, int N)
{
  nn_real *A;
  arma::Mat<nn_real> C1(M, N);
  arma::Mat<nn_real> C2(M, N);

  nn_real *dA;
  nn_real *dC1;
  nn_real *dC2;

  int num_iters = 100;

  A = (nn_real *)malloc(M * N * sizeof(nn_real));

  cudaMalloc((void **)&dA, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dC1, sizeof(nn_real) * M * N);
  cudaMalloc((void **)&dC2, sizeof(nn_real) * M * N);

  /* Initialization */
  for (int i = 0; i < M * N; i++)
  {
    A[i] = (nn_real)(rand() % 10); /* Replace with your own initialization if
                                       necessary */
  }

  cudaMemcpy(dA, A, sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  using namespace std::chrono;
  high_resolution_clock::time_point ref_t1 = high_resolution_clock::now();

  softmax_gpu(dA, dC1, M, N);

  for (int i = 0; i < NUM_ITERS; i++)
  {
    softmax_gpu(dA, dC1, M, N);
  }

  high_resolution_clock::time_point ref_t2 = high_resolution_clock::now();
  duration<double> ref_time_span =
      duration_cast<duration<double>>(ref_t2 - ref_t1);

  cudaMemcpy(C1.memptr(), dC1, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERS; i++)
  {
    softmax_gpu(dA, dC2, M, N);
  }

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> my_time_span = duration_cast<duration<double>>(t2 - t1);

  cudaMemcpy(C2.memptr(), dC2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  int fail = compareGEMMResults(C1.memptr(), C2.memptr(), M, N);


  std::cout << "Time for reference GEMM implementation: "
              << ref_time_span.count() << " seconds" << std::endl;
  std::cout << "Time for my GEMM implementation: " << my_time_span.count()
              << " seconds" << std::endl;

  free(A);
  cudaFree(dA);
  cudaFree(dC1);
  cudaFree(dC2);
}

void TestMatAdd(int M, int N) {
  arma::Mat<nn_real> mat1(M,N), mat2(M,N), res(M,N), res_gpu(M,N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++){
      mat1(i,j) = rand() / (nn_real) RAND_MAX;
      mat2(i,j) = rand() / (nn_real) RAND_MAX;
    }
  }

  res = 3* mat1 +  2.* mat2;
  nn_real* dmat1, *dmat2, *dres;
  cudaMalloc((void**)&dmat1, sizeof(nn_real) * M * N);
  cudaMalloc((void**)&dmat2, sizeof(nn_real) * M * N);
  cudaMalloc((void**)&dres, sizeof(nn_real) * M * N);
  cudaMemcpy(dmat1, mat1.memptr(), sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dmat2, mat2.memptr(), sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);

  addmat_gpu(dmat1, dmat2, dres, 3., 2., M, N);
  cudaMemcpy(res_gpu.memptr(), dres, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  compareGEMMResults(res.memptr(), res_gpu.memptr(), M, N);

}

void TestAddMat2(int M, int N){
  arma::Mat<nn_real> mat1(M, N), mat2(M,N), res(M,N), res_gpu(M,N);
  nn_real alpha = 1.f / 800;
  nn_real beta = -1.f / 800;
  
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++){
      mat1(i,j) = rand() / (nn_real) RAND_MAX;
      mat2(i,j) = rand() / (nn_real) RAND_MAX;
    }
  }
  res = alpha * mat1 + beta * mat2;

  nn_real* d_mat1, *d_mat2;
  cudaMalloc((void**)&d_mat1, sizeof(nn_real) * M * N);
  cudaMalloc((void**)&d_mat2, sizeof(nn_real) * M * N);
  cudaMemcpy(d_mat1, mat1.memptr(), sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat2, mat2.memptr(), sizeof(nn_real) * M * N, cudaMemcpyHostToDevice); 
  addmat_gpu(d_mat1, d_mat2, d_mat2, alpha, beta, M, N);
  cudaMemcpy(res_gpu.memptr(), d_mat2, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  compareGEMMResults(res.memptr(), res_gpu.memptr(), M, N);
}

void TestTranspose(int M, int N) {
  arma::Mat<nn_real> mat(M,N), res(N,M), res_gpu(N,M);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++){
      mat(i,j) = rand() / (nn_real) RAND_MAX;
    }
  }
  res = mat.t();

  nn_real* dmat, *dres;
  cudaMalloc((void**)&dmat, sizeof(nn_real) * M * N);
  cudaMalloc((void**)&dres, sizeof(nn_real) * M * N);
  cudaMemcpy(dmat, mat.memptr(), sizeof(nn_real) * M * N, cudaMemcpyHostToDevice);
  transpose_gpu(dmat, dres, M, N);
  cudaMemcpy(res_gpu.memptr(), dres, sizeof(nn_real) * M * N, cudaMemcpyDeviceToHost);

  compareGEMMResults(res.memptr(), res_gpu.memptr(), M, N);
}

void TestFeedforward(int n0, int n1, int n2, int nbatch) {

    arma::Mat<nn_real> X, W0, W1, b0, b1;
    X.set_size(n0, nbatch);
    W0.set_size(n1, n0);
    W1.set_size(n2, n1);
    b0.set_size(n1, 1);
    b1.set_size(n2, 1);

    for(int i = 0; i < nbatch; i++) {
        for(int j = 0; j < n0; j++) {
            X(j, i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
        }
    }
    for(int i = 0; i < n0; i++) {
        for(int j = 0; j < n1; j++) {
            W0(j, i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
        }
    }
    for(int i = 0; i < n1; i++) {
        for(int j = 0; j < n2; j++) {
            W1(j, i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
        }
    }
    b0.zeros();
    b1.zeros();
    /*for(int i = 0; i < n1; i++) {
        b0(i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
    }
    for(int i = 0; i < n2; i++) {
        b1(i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
    }*/

    std::vector<int> H = {n0,n1,n2};

    NeuralNetwork nn(H);
    nn.W = {W0, W1};
    nn.b = {b0, b1};

    nn_real reg = 0.0;
    cache bpcache;
    bpcache.X = X;

    feedforward(nn, X, bpcache);

    // Allocate device memory
    nn_real *d_X, *d_W0, *d_W1, *d_b0, *d_b1, *d_a1, *d_a2, *d_z1, *d_z2;
    cudaMalloc((void**) &d_X, nbatch * n0 * sizeof(nn_real));
    cudaMalloc((void**) &d_W0, n0 * n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_W1, n1 * n2 * sizeof(nn_real));
    cudaMalloc((void**) &d_b0, n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_b1, n2 * sizeof(nn_real));
    cudaMalloc((void**) &d_a1, nbatch * n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_a2, nbatch * n2 * sizeof(nn_real));
    cudaMalloc((void**) &d_z1, nbatch * n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_z2, nbatch * n2 * sizeof(nn_real));

    // Transfer data from host to device
    cudaMemcpy(d_X, X.memptr(), nbatch * n0 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W0, W0.memptr(), n0 * n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1.memptr(), n1 * n2 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b0, b0.memptr(), n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.memptr(), n2 * sizeof(nn_real), cudaMemcpyHostToDevice);
    
    // Perform feedforward on the GPU
    feedforward_gpu(n0, n1, n2, nbatch, d_X, d_W0, d_W1, d_b0, d_b1, d_a1, d_a2, d_z1, d_z2);
    
    // We don't compare results here, since the output of the feedforward pass is hard to compute without a non-GPU version of the entire network
    // Instead, let's just check that the output has the correct size (n_images x n_2)
    std::cout << "Output size is correct if no CUDA errors are thrown!" << std::endl;

    arma::Mat<nn_real> a2(n2, nbatch), z2(n2, nbatch), a1(n1, nbatch), z1(n1, nbatch);
    cudaMemcpy(a2.memptr(), d_a2, sizeof(nn_real) * n2 * nbatch, cudaMemcpyDeviceToHost);
    cudaMemcpy(z2.memptr(), d_z2, sizeof(nn_real) * n2 * nbatch, cudaMemcpyDeviceToHost);
    cudaMemcpy(a1.memptr(), d_a1, sizeof(nn_real) * n1 * nbatch, cudaMemcpyDeviceToHost);
    cudaMemcpy(z1.memptr(), d_z1, sizeof(nn_real) * n1 * nbatch, cudaMemcpyDeviceToHost);

    std::cout << "A2: " << std::endl;
    compareGEMMResults(a2.memptr(), bpcache.a[1].memptr(), n2, nbatch);
    std::cout << "Z2: " << std::endl;
    compareGEMMResults(z2.memptr(), bpcache.z[1].memptr(), n2, nbatch);
    std::cout << "A1: " << std::endl;
    compareGEMMResults(a1.memptr(), bpcache.a[0].memptr(), n1, nbatch);
    std::cout << "Z1: " << std::endl;
    compareGEMMResults(z1.memptr(), bpcache.z[0].memptr(), n1, nbatch);

    // Cleanup memory
    cudaFree(d_X);
    cudaFree(d_W0);
    cudaFree(d_W1);
    cudaFree(d_b0);
    cudaFree(d_b1);
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_z1);
    cudaFree(d_z2);
}

void TestBackward(int n0, int n1, int n2, int nbatch) {

    nn_real* d_a1;
    nn_real* d_a2;
    nn_real* d_z1;
    nn_real* d_z2;

    nn_real* d_W1;
    nn_real* d_W2;
    nn_real* d_b1;
    nn_real* d_b2;

    nn_real* d_X;
    nn_real* d_y;

    nn_real* d_db1;
    nn_real* d_db2;
    nn_real* d_dW1;
    nn_real* d_dW2;

    nn_real* d_h1;
    nn_real* d_h2;
    nn_real* d_h3;

    nn_real reg;
    
    cudaMalloc((void**) &d_a1, n1 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_a2, n2 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_z2, n2 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_z1, n1 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_W1, n0 * n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_W2, n2 * n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_X, n0 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_b2, n2 * sizeof(nn_real));
    cudaMalloc((void**) &d_b1, n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_y, n2 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_db1, n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_db2, n2 * sizeof(nn_real));
    cudaMalloc((void**) &d_dW1, n1 * n0 * sizeof(nn_real));
    cudaMalloc((void**) &d_dW2, n2 * n1 * sizeof(nn_real));

    cudaMalloc((void**) &d_h1, n2 * nbatch * sizeof(nn_real));
    cudaMalloc((void**) &d_h2, nbatch * n1 * sizeof(nn_real));
    cudaMalloc((void**) &d_h3, n1 * n2 * sizeof(nn_real));


    // TODO: Allocate memory for the variables and fill with test data.
    arma::Mat<nn_real> a1(n1, nbatch), a2(n2, nbatch), z1(n1, nbatch), z2(n2, nbatch), 
        w1(n1, n0), w2(n2, n1), X(n0, nbatch), 
        y(n2, nbatch), dw1(n1, n0), dw2(n2, n1), h1(n2, nbatch), h2(nbatch, n1),
        h3(n1, n2);

    for (int i = 0; i < nbatch; i++) {
        y(i % n2, i) = 1.0;
    }
    for (int i = 0; i < n1; i++) {
      for (int j = 0; j < n0; j++){
        w1(i,j) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
      }
    }
    for (int i = 0; i < n2; i++) {
      for (int j = 0; j < n1; j++){
        w2(i,j) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
      }
    }
    for (int i = 0; i < n0; i++) {
      for (int j = 0; j < nbatch; j++){
        X(i,j) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
      }
    }

    arma::Col<nn_real> b1(n1), b2(n2), db1(n1), db2(n2);

    for (int i = 0; i < n1; i++) {
      b1(i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
    }
    for (int i = 0; i < n2; i++) {
      b2(i) = (rand() - 0.5 * (nn_real) RAND_MAX)/ (nn_real) RAND_MAX;
    }

    arma::Mat<nn_real> dw1_gpu(n1,n0), dw2_gpu(n2,n1);
    arma::Col<nn_real> db1_gpu(n1), db2_gpu(n2);
    
    std::vector<int> H = {n0,n1,n2};

    NeuralNetwork nn(H);
    nn.W = {w1, w2};
    nn.b = {b1, b2};

    reg = 0.01 ;
    cache bpcache;
    bpcache.X = X;

    grads bpgrads;

    cudaMemcpy(d_X, X.memptr(), n0 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.memptr(), n2 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, w1.memptr(), n1 * n0 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, w2.memptr(), n2 * n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.memptr(), n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.memptr(), n2 * sizeof(nn_real), cudaMemcpyHostToDevice);

    feedforward(nn, X, bpcache);

    feedforward_gpu(n0, n1, n2, nbatch, d_X,
    d_W1, d_W2, d_b1, d_b2, d_a1, 
    d_a2, d_z1, d_z2);

    cudaMemcpy(a2.memptr(), d_a2, n2 * nbatch * sizeof(nn_real), cudaMemcpyDeviceToHost);
    cudaMemcpy(z1.memptr(), d_z1, n1 * nbatch * sizeof(nn_real), cudaMemcpyDeviceToHost);
    cudaMemcpy(z2.memptr(), d_z2, n2 * nbatch * sizeof(nn_real), cudaMemcpyDeviceToHost);
    cudaMemcpy(a1.memptr(), d_a1, n1 * nbatch * sizeof(nn_real), cudaMemcpyDeviceToHost);

    std::cout << "Forward: " << std::endl;
    std::cout << "A2: " << std::endl;
    compareGEMMResults(a2.memptr(), bpcache.a[1].memptr(), n2, nbatch);
    
    std::cout << "Z2: " << std::endl;
    compareGEMMResults(z2.memptr(), bpcache.z[1].memptr(), n2, nbatch);

    std::cout << "A1: " << std::endl;
    compareGEMMResults(a1.memptr(), bpcache.a[0].memptr(), n1, nbatch);

    std::cout << "Z1: " << std::endl;
    compareGEMMResults(z1.memptr(), bpcache.z[0].memptr(), n1, nbatch);

    std::cout << " Loss: " << loss(nn, a2, y, reg) << std::endl;

    backprop(nn, y, reg, bpcache, bpgrads);

    // Make a dummy call to warm up the GPU and check for errors.
    backprop_gpu(n0, n1, n2, nbatch, d_a1, d_a2,
                    d_z1, d_z2, d_W1, d_W2, d_X,
                    d_b1, d_b2, d_y, d_db1, d_db2,
                    d_dW1, d_dW2, d_h1, d_h2, d_h3,
                    reg, 1.);
    
    //check_launch("backprop_gpu dummy");

    // Now call the function with test data and measure execution time.
    double start = MPI_Wtime();
    //backprop_gpu(n0, n1, n2, nbatch, d_a2, d_a1, d_z2, d_z1, d_W1, d_W2, d_X, d_b2, 
    //d_b1, d_y, d_db1, d_db2, d_dW1, d_dW2, d_h1, d_h2, d_h3, d_h4, d_h5, d_h6, reg);
    double stop = MPI_Wtime();
    printf("Elapsed time: %f seconds\n", stop-start);

    // TODO: Copy the results back to the host memory and verify correctness.
    cudaMemcpy(dw1_gpu.memptr(), d_dW1, sizeof(nn_real) * n1 * n0, cudaMemcpyDeviceToHost);
    cudaMemcpy(dw2_gpu.memptr(), d_dW2, sizeof(nn_real) * n2 * n1, cudaMemcpyDeviceToHost);
    cudaMemcpy(db1_gpu.memptr(), d_db1, sizeof(nn_real) * n1, cudaMemcpyDeviceToHost);
    cudaMemcpy(db2_gpu.memptr(), d_db2, sizeof(nn_real) * n2, cudaMemcpyDeviceToHost);

    std::cout << std::endl << "Backward: " << std::endl;
    // Check correctness of the results.
    compareGEMMResults(dw1_gpu.memptr(), bpgrads.dW[0].memptr(), n1, n0);
    compareGEMMResults(dw2_gpu.memptr(), bpgrads.dW[1].memptr(), n2, n1);
    compareGEMMResults(db1_gpu.memptr(), bpgrads.db[0].memptr(), n1, 1);
    compareGEMMResults(db2_gpu.memptr(), bpgrads.db[1].memptr(), n2, 1);

    std::cout << "db_gpu = " << db2_gpu << std::endl;
    std::cout << "db_cpu = " << bpgrads.db[1] << std::endl;

    // TODO: Cleanup.
    cudaFree(d_a1);
    cudaFree(d_a2);
    cudaFree(d_z1);
    cudaFree(d_z2);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_X);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_y);
    cudaFree(d_db1);
    cudaFree(d_db2);
    cudaFree(d_dW1);
    cudaFree(d_dW2);
    cudaFree(d_h1);
    cudaFree(d_h2);
    cudaFree(d_h3);
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

void BenchmarkSigmoid() {

    std::cout << std::endl << "Entering Sigmoid Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int M = 800*SCALE, N = 1000*SCALE, K = 784*SCALE;

    std::cout << std::endl << "Starting Sigmoid 1: " << "M = " << M << "; N = "
              << N << "; K = " << K << std::endl;
    TestSigmoid(M, N);
    std::cout << "Completed Sigmoid 1" << std::endl;

    /* Secong GEMM Problem Size */
    M = 800*SCALE, N = 10*SCALE, K = 1000*SCALE;
    std::cout << std::endl << "Starting Sigmoid 2: " << "M = " << M << "; N = "
              << N << "; K = " << K << std::endl;
    TestSigmoid(M, N);
    std::cout << "Completed Sigmoid 2" << std::endl;
}

void BenchmarkSoftmax()
{
  std::cout << std::endl
            << "Entering Softmax Benchmarking mode! Stand by." << std::endl;

  /* First GEMM problem size */
  int M = 800 * SCALE, N = 1000 * SCALE;

  std::cout << std::endl
            << "Starting Softmax 1: "
            << "M = " << M << "; N = " << N << std::endl;
  TestSoftmax(M, N);
  std::cout << "Completed Softmax 1" << std::endl;

  /* Second GEMM problem size */
  M = 800 * SCALE, N = 100 * SCALE;
  std::cout << std::endl
            << "Starting Softmax 2: "
            << "M = " << M << "; N = " << N << std::endl;
  TestSoftmax(M, N);
  std::cout << "Completed Softmax 2" << std::endl;
}

void BenchmarkMatAdd() {

    std::cout << std::endl << "Entering MatAdd Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int M = 800*SCALE, N = 1000*SCALE, K = 10, L = 6;

    std::cout << std::endl << "Starting MatAdd 1: " << "M = " << M << "; N = "
              << N << "; K = " << K << std::endl;
    TestMatAdd(M, N);
    std::cout << "Completed MatAdd 1" << std::endl;

    /* Secong GEMM Problem Size */
    M = 4102, N = 739, K = 1000*SCALE;
    std::cout << std::endl << "Starting MatAdd 2: " << "M = " << M << "; N = "
              << N << "; K = " << K << std::endl;
    TestMatAdd(M, N);
    std::cout << "Completed MatAdd 2" << std::endl;
}

void BenchmarkTranspose() {

    std::cout << std::endl << "Entering Transpose Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int M = 8345, N = 800;

    std::cout << std::endl << "Starting Transpose 1: " << "M = " << M << "; N = "
              << N << std::endl;
    TestTranspose(M, N);
    std::cout << "Completed Transpose 1" << std::endl;

    /* Secong GEMM Problem Size */
    M = 17, N = 245;
    std::cout << std::endl << "Starting Transpose 2: " << "M = " << M << "; N = "
              << N << std::endl;
    TestTranspose(M, N);
    std::cout << "Completed Transpose 2" << std::endl;
}

void BenchmarkForward() {

    std::cout << std::endl << "Entering Forward Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int n0 = 3000, n1 = 1000, n2 = 10, nbatch = 800;

    std::cout << std::endl << "Starting Forward 1: " << std::endl;
    TestFeedforward(n0, n1, n2, nbatch);
    std::cout << "Completed Forward 1" << std::endl;

    /* Secong GEMM Problem Size */
    n0 = 10, n1 = 10, n2 = 10, nbatch = 16;
    std::cout << std::endl << "Starting Forward 2: " << std::endl;
    TestFeedforward(n0, n1, n2, nbatch);
    std::cout << "Completed Forward 2" << std::endl;
}

void BenchmarkBackward() {

    std::cout << std::endl << "Entering Backward Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int n0 = 6, n1 = 4, n2 = 8, nbatch = 12;

    std::cout << std::endl << "Starting Backward 1: " << std::endl;
    TestBackward(n0, n1, n2, nbatch);
    std::cout << "Completed Forward 1" << std::endl;

    /* Secong GEMM Problem Size */
    n0 = 784, n1 = 1000, n2 = 10, nbatch = 800;
    std::cout << std::endl << "Starting Backward 2: " << std::endl;
    TestBackward(n0, n1, n2, nbatch);
    std::cout << "Completed Forward 2" << std::endl;
}

void BenchmarkMatAdd2() {

    std::cout << std::endl << "Entering Backward Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int n0 = 6, n1 = 4, n2 = 8, nbatch = 12;

    std::cout << std::endl << "Starting Backward 1: " << std::endl;
    TestAddMat2(n2, nbatch);
    std::cout << "Completed Forward 1" << std::endl;

    /* Secong GEMM Problem Size */
    n0 = 784, n1 = 1000, n2 = 10, nbatch = 800;
    std::cout << std::endl << "Starting Backward 2: " << std::endl;
    TestAddMat2(n0, n1);
    std::cout << "Completed Forward 2" << std::endl;
}