#ifndef _UTILS_MNIST_H_
#define _UTILS_MNIST_H_

#include <armadillo>
#include <cstring>

#include "common.h"

int reverse_int(int i);
void read_mnist(std::string filename, arma::Mat<nn_real> &mat);
void read_mnist_label(std::string filename, arma::Row<nn_real> &vec);

#endif