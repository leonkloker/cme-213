#ifndef COMMON_H_
#define COMMON_H_

#include <cassert>
#include <string>

using std::string;

// #define USE_DOUBLE

extern string file_train_images;
extern string file_train_labels;
extern string file_test_images;
extern string output_dir;
extern string cpu_save_dir;
extern string cpu_load_dir;
extern string grade_tag;
extern string mpi_tag;
extern string file_test_dir;

#define IMAGE_SIZE 784 // 28 x 28
#define NUM_CLASSES 10
#define NUM_TRAIN 60000
#define NUM_TEST 10000

#ifndef USE_DOUBLE

typedef float nn_real;
#define MPI_FP MPI_FLOAT
#define cublas_gemm cublasSgemm

#else

typedef double nn_real;
#define MPI_FP MPI_DOUBLE
#define cublas_gemm cublasDgemm

#endif

#define _MSG(msg)                                                      \
  do                                                                   \
  {                                                                    \
    std::cerr << __FILE__ << "(@" << __LINE__ << "): " << msg << '\n'; \
  } while (false)

#endif