#ifndef UTILS_TWO_LAYER_NET_H_
#define UTILS_TWO_LAYER_NET_H_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <armadillo>
#include <cassert>

#include "common.h"

bool file_exists(const std::string &name);

template <class T>
void load_cpu_data_test(T &arma_mat, std::stringstream &s)
{
  if (!file_exists(s.str()))
  {
    std::cerr << "Failed to load CPU data from file " << s.str() << std::endl;
    std::cerr << "Make sure the file exists\n";
    exit(1);
  }
  arma_mat.load(s.str(), arma::raw_ascii);
}

template <class T>
void save_cpu_data_test(T &arma_mat, std::stringstream &s)
{
  if (!arma_mat.save(s.str(), arma::raw_ascii))
  {
    std::cerr << "Failed to save CPU data to file " << s.str() << std::endl;
    exit(1);
  }
}

#define ASSERT_MAT_SAME_SIZE(mat1, mat12) \
  assert(mat1.n_rows == mat2.n_rows && mat1.n_cols == mat2.n_cols)

class NeuralNetwork
{
public:
  const int num_layers = 2;
  // H[i] is the number of neurons in layer i (where i=0 implies input layer)
  std::vector<int> H;
  // Weights of the neural network
  // W[i] are the weights of the i^th layer
  std::vector<arma::Mat<nn_real>> W;
  // Biases of the neural network
  // b[i] is the row vector biases of the i^th layer
  std::vector<arma::Col<nn_real>> b;

  NeuralNetwork(std::vector<int> &_H)
  {
    W.resize(num_layers);
    b.resize(num_layers);
    H = _H;

    assert(H.size() == 3);

    for (int i = 0; i < num_layers; i++)
    {
      arma::arma_rng::set_seed(arma::arma_rng::seed_type(i));
      assert(i + 1 < 3);
      W[i] = 0.01 * arma::randn<arma::Mat<nn_real>>(H[i + 1], H[i]);
      b[i].zeros(H[i + 1]);
    }
  }
};

// Parallel train function
void parallel_train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                    const arma::Mat<nn_real> &y, nn_real learning_rate,
                    std::ofstream &error_file, nn_real reg = 0.0,
                    const int epochs = 15, const int batch_size = 800,
                    int print_every = -1, int debug = 0);

struct grads
{
  std::vector<arma::Mat<nn_real>> dW;
  std::vector<arma::Col<nn_real>> db;
};

struct cache
{
  arma::Mat<nn_real> X;
  std::vector<arma::Mat<nn_real>> z;
  std::vector<arma::Mat<nn_real>> a;
  arma::Mat<nn_real> yc;
};

// Feedforward pass
void feedforward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                 struct cache &bpcache);

// Loss computation
nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg);

// Sequential training on the CPU
void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, nn_real learning_rate,
           nn_real reg = 0.0, const int epochs = 15, const int batch_size = 800,
           bool grad_check = false, int print_every = -1, int debug = 0);

// Predict the labels using a trained model
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label);

/*
 * Performs gradient check by comparing numerical and analytical gradients.
 */
bool gradcheck(struct grads &grads1, struct grads &grads2);

/*
 * Compares the two label vectors to compute precision.
 */
nn_real precision(arma::Row<nn_real> vec1, arma::Row<nn_real> vec2);

/*
 * Converts label vector into a matrix of one-hot label vectors
 * @params label : label vector
 * @params C : Number of classes
 * @params [out] y : The y matrix.
 */
void label_to_y(arma::Row<nn_real> label, int C, arma::Mat<nn_real> &y);

void save_label(std::string filename, arma::Row<nn_real> &label);

/* Computes the gradient using finite-difference.
 * This is used for debugging.
 */
void numgrad(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             const arma::Mat<nn_real> &y, nn_real reg, struct grads &numgrads);

/* Functions to save data to file for debugging purposes */
void save_cpu_data(NeuralNetwork &nn, int iter);
void save_gpu_error(NeuralNetwork &nn, int iter, std::ofstream &error_file);

#endif
