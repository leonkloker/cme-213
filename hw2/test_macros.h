#ifndef TEST_MACROS_H_
#define TEST_MACROS_H_

#include "gtest/gtest.h"

#define EXPECT_VECTOR_EQ_EPS(ref_vec, test_vec, eps, success) { \
  if (ref_vec.size() != test_vec.size()) { \
    std::cerr << __FILE__ << ":" << __LINE__ << "\t" \
              << "\n\tERROR:" << "Dimension Mismatch (" \
              << ref_vec.size() << " != " << test_vec.size() << ")" \
              << std::endl; \
    *success = false; \
  } else { \
    for (unsigned int i = 0; i < ref_vec.size(); ++i) { \
      if (ref_vec[i] - test_vec[i] > eps or test_vec[i] - ref_vec[i] > eps) { \
        std::cerr << __FILE__ << ":" << __LINE__ << "\t" \
                  << "\n\tERROR: " << "Value Mismatch (" \
                  << ref_vec[i] << " != " << test_vec[i] << ", index = " << i \
                  << ")" << std::endl; \
        *success = false; \
        break; \
      } \
    } \
  } \
  EXPECT_TRUE(*success); \
}

#define EXPECT_VECTOR_EQ(ref_vec, test_vec, success) { \
  EXPECT_VECTOR_EQ_EPS(ref_vec, test_vec, 0, success); \
}

#endif /* TEST_MACROS_H_ */
