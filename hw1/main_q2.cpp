#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
#include "matrix_rect.hpp"

TEST(testMatrix, mnxmn) {
  Matrix2D<float> A(5,10);
  Matrix2D<float> B(5,10);
  for (int i = 0; i < 5; i++){
    for (int j = 0; j < 10; j++){
      A(i,j) = 1;
      B(i,j) = 2;
    }
  }
  EXPECT_EQ(A.dot(B)(0,0), 2);
  EXPECT_EQ(A.dot(B)(1,5), 2);

}

TEST(testMatrix, 1nxmn) {
  Matrix2D<float> A(1,5);
  Matrix2D<float> B(10,5);
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 5; j++){
      A(0,j) = 1;
      B(i,j) = 2;
    }
  }
  EXPECT_EQ(A.dot(B)(0,0), 2);
  EXPECT_EQ(A.dot(B)(7,3), 2);
}

TEST(testMatrix, mnxm1) {
  Matrix2D<float> A(10,5);
  Matrix2D<float> B(10,1);
  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 5; j++){
      A(i,j) = 1;
      B(i,0) = i;
    }
  }
  EXPECT_EQ(A.dot(B)(0,0), 0);
  EXPECT_EQ(A.dot(B)(9,4), 9);
}

TEST(testMatrix, 11xmn) {
  Matrix2D<float> A(1,1);
  Matrix2D<float> B(3,4);
  A(0,0) = 4;
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 4; j++){
      B(i,j) = 3;
    }
  }
  EXPECT_EQ(A.dot(B)(0,0), 12);
  EXPECT_EQ(A.dot(B)(2,3), 12);
}

TEST(testMatrix, mnx11) {
  Matrix2D<float> A(3,4);
  Matrix2D<float> B(1,1);
  B(0,0) = 4;
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 4; j++){
      A(i,j) = i;
    }
  }
  EXPECT_EQ(A.dot(B)(0,0), 0);
  EXPECT_EQ(A.dot(B)(2,3), 8);
}

TEST(testMatrix, print){
  Matrix2D<float> A(3,4);
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 4; j++){
      A(i,j) = i*j;
    }
  }
  std::cout << A;
}

/*
TODO:
Test your implementation by writing tests that cover most scenarios of 2D matrix
broadcasting. Say you are testing the result C = A * B, test with:
1. A of shape (m != 1, n != 1), B of shape (m != 1, n != 1)
2. A of shape (1, n != 1), B of shape (m != 1, n != 1)
3. A of shape (m != 1, n != 1), B of shape (m != 1, 1)
4. A of shape (1, 1), B of shape (m != 1, n != 1)
Please test any more cases that you can think of.
*/
