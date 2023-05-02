#ifndef TEST_RECURRENCE_H_
#define TEST_RECURRENCE_H_

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
#include "util.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::fabs;
using std::vector;

typedef float elem_type;
typedef std::vector<elem_type> vec;

extern const size_t ITER_MAX_CHECK;

void check_initialization(vec &input_array, size_t arr_size) {
  EXPECT_EQ(input_array.size(), arr_size)
      << "Initialization Error: Array size isn't correct." << endl;

  int count = 0;
  for (size_t i = 0; i < arr_size; i++) {
    elem_type entry = input_array[i];
    if (entry < -1.0 || entry > 1.0) {
      ADD_FAILURE() << "Initialization Error: Entry " << i
                    << " isn't between -2 and 2." << endl;
      count++;
    }

    if (count > 10) {
      FAIL() << "\nToo many (>10) errors in initialization, quitting..."
             << endl;
    }
  }
}

void checkResults(vec &array_host, elem_type *device_output_array,
                  size_t num_entries) {
  // allocate space on host for gpu results
  vec array_from_gpu(num_entries);

  // download and inspect the result on the host:
  cudaMemcpy(&array_from_gpu[0], device_output_array,
             num_entries * sizeof(elem_type), cudaMemcpyDeviceToHost);
  check_launch("copy from gpu");

  // check CUDA output versus reference output
  int error = 0;
  float max_error = 0.;
  int pos = -1;
  double inf = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < num_entries; i++) {
    double err =
        fabs(array_host[i]) <= 1
            ? fabs(array_host[i] - array_from_gpu[i])
            : fabs((array_host[i] - array_from_gpu[i]) / array_host[i]);
    if (max_error < err) {
      max_error = err;
      pos = i;
    }
    if (fabs(array_host[i]) == inf && fabs(array_from_gpu[i]) == inf) continue;
    if (fabs(array_host[i]) <= 2 &&
        fabs(array_host[i] - array_from_gpu[i]) < 1e-4)
      continue;
    if (fabs(array_host[i]) > 2 &&
        fabs((array_host[i] - array_from_gpu[i]) / array_host[i]) < 1e-4)
      continue;

    ++error;
    ADD_FAILURE() << "** Critical error at pos: " << i << " error "
                  << fabs((array_host[i] - array_from_gpu[i]) / array_host[i])
                  << " expected " << array_host[i] << " and got "
                  << array_from_gpu[i] << endl;

    if (error > 10) {
      FAIL() << "\nToo many critical errors, quitting..." << endl;
    }
  }

  if (pos >= 0) {
    cout << "Largest error found at pos: " << pos << " error " << max_error
         << " expected " << array_host[pos] << " and got "
         << array_from_gpu[pos] << endl;
  }

  if (error) {
    FAIL() << "\nCritical error(s) in recurrence kernel! Exiting..." << endl;
  }
}

double recurAndCheck(const elem_type *device_input_array,
                     elem_type *device_output_array, size_t num_iter,
                     size_t array_size, size_t cuda_block_size,
                     size_t cuda_grid_size, vec &arr_host) {
  // generate GPU output
  double elapsed_time =
      doGPURecurrence(device_input_array, device_output_array, num_iter,
                      array_size, cuda_block_size, cuda_grid_size);

  if (num_iter <= ITER_MAX_CHECK)
    checkResults(arr_host, device_output_array, array_size);

  // make sure we don't falsely say the next kernel is correct because
  // we've left the correct answer sitting in memory
  cudaMemset(device_output_array, 0, array_size * sizeof(elem_type));
  return elapsed_time;
}

#endif