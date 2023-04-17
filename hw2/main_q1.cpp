#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "test_macros.h"
#include "test_util.h"
#include "sum.h"

#include "gtest/gtest.h"

const uint kMaxInt = 100;
const uint kSize = 30000000;

std::vector<uint> initializeRandomly(uint size, uint max_int) {
  std::vector<uint> res(size);
  std::default_random_engine generator;
  std::uniform_int_distribution<uint> distribution(0, max_int);

  for (uint i = 0; i < size; ++i) res[i] = distribution(generator);

  return res;
}

TEST(testQ1, test) {
  using namespace std::chrono;

  high_resolution_clock::time_point start, end;
  duration<double> delta;

  // You can uncomment the line below to make your own simple tests
  // std::vector<uint> v = ReadVectorFromFile("vec");
  std::vector<uint> v = initializeRandomly(kSize, kMaxInt);

  std::cout << "Parallel" << std::endl;
  start = high_resolution_clock::now();
  std::vector<uint> sums = parallelSum(v);
  end = high_resolution_clock::now();
  std::cout << "Sum Even: " << sums[0] << std::endl;
  std::cout << "Sum Odd: " << sums[1] << std::endl;
  delta = duration_cast<duration<double>>(end - start);
  std::cout << "Time: " << delta.count() << std::endl;

  std::cout << "Serial" << std::endl;
  start = high_resolution_clock::now();
  std::vector<uint> sumsSer = serialSum(v);
  end = high_resolution_clock::now();
  std::cout << "Sum Even: " << sumsSer[0] << std::endl;
  std::cout << "Sum Odd: " << sumsSer[1] << std::endl;
  delta = duration_cast<duration<double>>(end - start);
  std::cout << "Time: " << delta.count() << std::endl;

  bool success = true;
  EXPECT_VECTOR_EQ(sums, sumsSer, &success);
}
