/* This is machine problem 1, part 1, recurrence problem
 *
 * The problem is to take in the number of iterations and a vector of constants,
 * and perform the recurrence on each constant to determine whether it lies in
 * the (modified) Mandelbrot Set.
 *
 */

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "recurrence.cuh"
#include "test_recurrence.h"
#include "util.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::fabs;
using std::vector;

typedef float elem_type;
typedef std::vector<elem_type> vec;

constexpr const size_t MAX_ARR_SIZE = (1 << 30);
// NOTE: change this to 100 for debugging

extern const size_t ITER_MAX_CHECK = 10;
/* Maximum number of iterations for which error is checked;
   This is to avoid having to consider the accumulation of roundoff errors.
*/

// TODO: initialize an array of size arr_size in input_array with random floats
// between -1 and 1
void initialize_array(vec &input_array, size_t arr_size) {
  input_array.resize(arr_size);
  for (unsigned i = 0; i < arr_size; i++){
    input_array[i] = std::rand() / (0.5 * RAND_MAX) - 1;
  }
}

void host_recurrence(vec &input_array, vec &output_array, size_t num_iter,
                     size_t array_size) {
  std::transform(input_array.begin(), input_array.begin() + array_size,
                 output_array.begin(), [&num_iter](elem_type &constant) {
                   elem_type z = 0;
                   for (size_t it = 0; it < num_iter; it++) {
                     z = z * z + constant;
                   }
                   return z;
                 });
}

class RecurrenceTestFixture : public ::testing::Test {
 protected:
  RecurrenceTestFixture() {
    cudaFree(0);
    // initialize cuda context to avoid including cost in timings later

    // Warm-up each of the kernels to avoid including overhead in timing.
    // If the kernels are written correctly, then they should
    // never make a bad memory access, even though we are passing in NULL
    // pointers since we are also passing in a size of 0
    recurrence<<<1, 1>>>(nullptr, nullptr, 0, 0);

    // allocate host arrays
    arr_gpu.resize(MAX_ARR_SIZE);
    arr_host.resize(MAX_ARR_SIZE);

    // pointers to device arrays
    device_input_array = nullptr;
    device_output_array = nullptr;

    // TODO: allocate num_bytes of memory to the device arrays.
    // Hint: use cudaMalloc

  }

  // TODO: deallocate memory from both device arrays
  ~RecurrenceTestFixture() {

  }

  void initialize() {
    initialize_array(init_arr, MAX_ARR_SIZE);
    check_initialization(init_arr, MAX_ARR_SIZE);

    // copy input to GPU
    cudaMemcpy(device_input_array, &init_arr[0], num_bytes,
               cudaMemcpyHostToDevice);
    check_launch("copy to gpu");
  }

  // Compute the size of the arrays in bytes for memory allocation.
  const size_t num_bytes = MAX_ARR_SIZE * sizeof(elem_type);

  vec init_arr, arr_gpu, arr_host;
  elem_type *device_input_array;
  elem_type *device_output_array;

  // You can make the graph for Q4,Q5,Q6 more easily by saving this array as a
  // csv (or something else)
  std::vector<double> performance_array;
};

// TODO: allocate num_bytes of memory to the device arrays in the TEST FIXTURE
// CONSTRUCTOR and deallocate the memory in the TEST FIXTURE DESTRUCTOR.
// Hint: use cudaMalloc and cudaFree
TEST_F(RecurrenceTestFixture, GPUAllocationTest_1) {
  // if either memory allocation failed in the fixture constructor, report an
  // error message
  if (!device_input_array || !device_output_array) {
    FAIL() << "Couldn't allocate memory!" << endl;
  }
}

// TODO: Implement initialize_array function
TEST_F(RecurrenceTestFixture, InitalizeArrayTest_2) {
  initialize_array(init_arr, MAX_ARR_SIZE);
  check_initialization(init_arr, MAX_ARR_SIZE);
}

// TODO: Implement Recurrence Kernel Fuctions in Recurrence.cuh
TEST_F(RecurrenceTestFixture, RecurrenceKernelTest_3) {
  // init array
  initialize();

  // Testing accuracy of Recurrence Kernel
  size_t num_iter = 2;
  size_t array_size = 16;
  size_t cuda_block_size = 4;
  size_t cuda_grid_size = 4;
  host_recurrence(init_arr, arr_host, num_iter, array_size);
  recurAndCheck(device_input_array, device_output_array, num_iter, array_size,
                cuda_block_size, cuda_grid_size, arr_host);

  /* Further testing with more iterations */
  array_size = 1e6;
  cuda_block_size = 1024;
  cuda_grid_size = 576;
  for (num_iter = 1; num_iter <= ITER_MAX_CHECK; ++num_iter) {
    host_recurrence(init_arr, arr_host, num_iter, array_size);
    recurAndCheck(device_input_array, device_output_array, num_iter, array_size,
                  cuda_block_size, cuda_grid_size, arr_host);
  }

  cout << "\nQuestions 1.1-1.3: your code passed all the tests!\n\n";
}

// No code changes necessary
TEST_F(RecurrenceTestFixture, RecurrenceThreadsTest_4) {
  /*
   * ––––––––––-------------------------------------------------------
   * Question 1.4: vary number of threads for a small number of blocks
   * ––––––––––-------------------------------------------------------
   */
  // init array
  initialize();

  cout << std::setw(23) << "Q1.4" << endl;
  cout << std::setw(43) << std::setfill('-') << " " << endl;
  cout << std::setw(15) << std::setfill(' ') << "Number of Threads";
  cout << std::setw(25) << "Performance TFlops/sec" << endl;
  size_t cuda_grid_size = 72;
  size_t num_iter = 4e4;
  size_t array_size = 1e6;
  double flops = 2 * num_iter * array_size;
  host_recurrence(init_arr, arr_host, num_iter, array_size);
  for (size_t cuda_block_size = 32; cuda_block_size <= 1024;
       cuda_block_size += 32) {
    double elapsed_time =
        recurAndCheck(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size, arr_host);
    double performance = flops / (elapsed_time / 1000.) / 1E12;
    performance_array.push_back(performance);
    cout << std::setw(17) << cuda_block_size;
    cout << std::setw(25) << performance << endl;
  }
  cout << endl;
}

// No code changes necessary
TEST_F(RecurrenceTestFixture, RecurrenceBlocksTest_5) {
  /*
   * ––––––––––-------------------------------------------------------
   * Question 1.5: vary number of blocks for a small number of threads
   * ––––––––––-------------------------------------------------------
   */
  // init array
  initialize();

  cout << std::setw(23) << "Q1.5" << endl;
  cout << std::setw(43) << std::setfill('-') << " " << endl;
  cout << std::setw(15) << std::setfill(' ') << "Number of Blocks";
  cout << std::setw(25) << "Performance TFlops/sec" << endl;
  size_t cuda_block_size = 128;
  size_t num_iter = 4e4;
  size_t array_size = 1e6;
  double flops = 2 * num_iter * array_size;
  host_recurrence(init_arr, arr_host, num_iter, array_size);
  for (size_t cuda_grid_size = 36; cuda_grid_size <= 1152;
       cuda_grid_size += 36) {
    double elapsed_time =
        recurAndCheck(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size, arr_host);
    double performance = flops / (elapsed_time / 1000.) / 1E12;
    performance_array.push_back(performance);
    cout << std::setw(16) << cuda_grid_size;
    cout << std::setw(25) << performance << endl;
  }
  cout << endl;
}

// No code changes necessary
TEST_F(RecurrenceTestFixture, RecurrenceItersTest_6) {
  /*
   * ––––––––––-----------------------------
   * Question 1.6: vary number of iterations
   * ––––––––––-----------------------------
   */
  // init array
  initialize();

  cout << std::setw(23) << "Q1.6" << endl;
  cout << std::setw(43) << std::setfill('-') << " " << endl;
  cout << std::setw(15) << std::setfill(' ') << "Number of Iters";
  cout << std::setw(25) << "Performance TFlops/sec" << endl;
  size_t cuda_block_size = 256;
  size_t cuda_grid_size = 576;
  size_t array_size = 1e6;
  std::vector<size_t> num_iters = {20,   40,   60,   80,   100,  120,  140,
                                   160,  180,  200,  300,  400,  500,  600,
                                   700,  800,  900,  1000, 1200, 1400, 1600,
                                   1800, 2000, 2200, 2400, 2600, 2800, 3000};
  for (size_t num_iter : num_iters) {
    double flops = 2 * num_iter * array_size;
    host_recurrence(init_arr, arr_host, num_iter, array_size);
    double elapsed_time =
        recurAndCheck(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size, arr_host);
    double performance = flops / (elapsed_time / 1000.) / 1E12;
    performance_array.push_back(performance);
    cout << std::setw(15) << num_iter;
    cout << std::setw(25) << performance << endl;
  }
  cout << endl;
}
