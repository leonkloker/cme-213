#ifndef TESTS_H_
#define TESTS_H_

#include "common.h"
#include "neural_network.h"

int checkErrors(const arma::Mat<nn_real>& Seq, const arma::Mat<nn_real>& Par,
                std::ofstream& ofs, std::vector<nn_real>& errors);

int checkNNErrors(NeuralNetwork& seq_nn, NeuralNetwork& par_nn,
                  std::string filename);

void BenchmarkGEMM();

#endif
