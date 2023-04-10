#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix.hpp"

TEST(testMatrix, symmetricTest) {

  MatrixSymmetric<float> m1;
  EXPECT_EQ(m1.size(), 0);
  EXPECT_EQ(m1.NormL0(), 0);

  MatrixSymmetric<float> m2(0);
  EXPECT_EQ(m2.size(), 0);
  EXPECT_EQ(m2.NormL0(), 0);

  EXPECT_THROW(MatrixSymmetric<float> m3(-1), std::invalid_argument);

  MatrixSymmetric<float> m3(5);
  for (int i = 0; i < 5; i++) m3(i,i) = 1;
  EXPECT_EQ(m3.size(), 5);
  EXPECT_EQ(m3.NormL0(), 5);

  MatrixSymmetric<float> m4(5);
  for (int i = 0; i < 5; i++){
    m4(i,0) = i;
  } 
  EXPECT_EQ(m4.NormL0(), 8);
  EXPECT_EQ(m4(0,0), 0);
  EXPECT_EQ(m4(3,0), 3);
  EXPECT_EQ(m4(1,2), 0);
  EXPECT_EQ(m4(0,4), 4);
  EXPECT_THROW(m4(5,0), std::out_of_range);
  EXPECT_THROW(m4(0,-1), std::out_of_range);

  std::stringstream s;
  s << m4;
}

TEST(testMatrix, diagonalTest) {

  MatrixDiagonal<float> m1;
  EXPECT_EQ(m1.size(), 0);
  EXPECT_EQ(m1.NormL0(), 0);

  MatrixDiagonal<float> m2(0);
  EXPECT_EQ(m2.size(), 0);
  EXPECT_EQ(m2.NormL0(), 0);

  EXPECT_THROW(MatrixDiagonal<float> m3(-1), std::invalid_argument);

  MatrixDiagonal<float> m3(5);
  for (int i = 0; i < 5; i++) m3(i,i) = 1;
  EXPECT_EQ(m3.size(), 5);
  EXPECT_EQ(m3.NormL0(), 5);

  MatrixDiagonal<float> m4(5);
  for (int i = 0; i < 5; i++){
    m4(i,i) = i;
  } 
  EXPECT_EQ(m4.NormL0(), 4);
  EXPECT_EQ(m4(0,0), 0);
  EXPECT_EQ(m4(3,3), 3);
  EXPECT_EQ(m4(1,2), 0);
  EXPECT_EQ(m4(0,4), 0);
  EXPECT_THROW(m4(5,0), std::out_of_range);
  EXPECT_THROW(m4(0,-1), std::out_of_range);

  std::stringstream s;
  s << m4;
}

/*
TODO:

For both the MatrixDiagonal and the MatrixSymmetric classes, do the following:

Write at least the following tests to get full credit here:
1. Declare an empty matrix with the default constructor for MatrixSymmetric.
Assert that the NormL0 and size functions return appropriate values for these.
2. Using the second constructor that takes size as argument, create a matrix of
size zero. Repeat the assertions from (1).
3. Provide a negative argument to the second constructor and assert that the
constructor throws an exception.
4. Create and initialize a matrix of some size, and verify that the NormL0
function returns the correct value.
5. Create a matrix, initialize some or all of its elements, then retrieve and
check that they are what you initialized them to.
6. Create a matrix of some size. Make an out-of-bounds access into it and check
that an exception is thrown.
7. Test the stream operator using std::stringstream and using the "<<" operator.

*/
