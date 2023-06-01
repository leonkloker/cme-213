#include "neural_network.h"

#define MAX_REL_ERROR_THRESHOLD 1000

// Compute the precision of the prediction
nn_real precision(arma::Row<nn_real> vec1, arma::Row<nn_real> vec2)
{
  return arma::accu(vec1 == vec2) / (nn_real)vec1.size();
}

/*
 * Converts label vector into a matrix of one-hot label vectors.
 */
void label_to_y(arma::Row<nn_real> label, int C, arma::Mat<nn_real> &y)
{
  y.set_size(C, label.size());
  y.fill(0);

  for (int i = 0; i < label.size(); ++i)
  {
    assert(label(i) >= 0);
    assert(label(i) < C);
    y(label(i), i) = 1;
  }
}

// Save labels to a file
void save_label(std::string dir_name, arma::Row<nn_real> &label)
{
  std::string filename = dir_name + "/Pred_testset.txt";
  std::ofstream file(filename);

  if (file.is_open())
  {
    for (int i = 0; i < label.size(); ++i)
    {
      file << label(i);
    }

    file.close();
  }
  else
  {
    std::cerr << "Save label to file " << filename << " failed!" << std::endl;
  }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             const arma::Mat<nn_real> &y, nn_real reg, struct grads &numgrads)
{
  nn_real h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i)
  {
    numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

    for (int j = 0; j < nn.W[i].n_rows; ++j)
    {
      for (int k = 0; k < nn.W[i].n_cols; ++k)
      {
        nn_real oldval = nn.W[i](j, k);
        nn.W[i](j, k) = oldval + h;
        feedforward(nn, X, numcache);
        nn_real fxph = loss(nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward(nn, X, numcache);
        nn_real fxnh = loss(nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2 * h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

  for (int i = 0; i < nn.num_layers; ++i)
  {
    numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

    for (int j = 0; j < nn.b[i].size(); ++j)
    {
      nn_real oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward(nn, X, numcache);
      nn_real fxph = loss(nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward(nn, X, numcache);
      nn_real fxnh = loss(nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2 * h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Returns the relative error between two matrices.
 */
nn_real rel_error(arma::Mat<nn_real> &mat1, arma::Mat<nn_real> &mat2)
{
  ASSERT_MAT_SAME_SIZE(mat1, mat2);
  arma::Mat<nn_real> threshold =
      arma::Mat<nn_real>(mat1.n_rows, mat1.n_cols, arma::fill::ones);
  return arma::max(arma::max(
      arma::abs(mat1 - mat2) /
      arma::max(threshold, arma::max(arma::abs(mat1), arma::abs(mat2)))));
}

/*
 * Performs gradient check
 */
bool gradcheck(struct grads &grads1, struct grads &grads2)
{
  assert(grads1.dW.size() == grads2.dW.size());
  assert(grads1.db.size() == grads2.db.size());

  for (int i = grads1.dW.size() - 1; i >= 0; --i)
  {
    nn_real error = rel_error(grads1.dW[i], grads2.dW[i]);
    std::cout << "dW[" << i << "] rel error: " << error << "\n";

    if (error > MAX_REL_ERROR_THRESHOLD)
    {
      return false;
    }
  }

  for (int i = grads1.db.size() - 1; i >= 0; --i)
  {
    nn_real error = rel_error(grads1.db[i], grads2.db[i]);
    std::cout << "db[" << i << "] rel error: " << error << "\n";

    if (error > MAX_REL_ERROR_THRESHOLD)
    {
      return false;
    }
  }

  return true;
}

void save_cpu_data(NeuralNetwork &nn, int iter)
{
  std::stringstream s;
  s << cpu_save_dir + "/seq_epoch-W0-" << iter << grade_tag << ".mat";
  printf("Saving W0 CPU data to file %s\n", s.str().c_str());
  save_cpu_data_test(nn.W[0], s);
  std::stringstream t;
  t << cpu_save_dir + "/seq_epoch-W1-" << iter << grade_tag << ".mat";
  printf("Saving W1 CPU data to file %s\n", t.str().c_str());
  save_cpu_data_test(nn.W[1], t);
  std::stringstream u;
  u << cpu_save_dir + "/seq_epoch-b0-" << iter << grade_tag << ".mat";
  printf("Saving b0 CPU data to file %s\n", u.str().c_str());
  save_cpu_data_test(nn.b[0], u);
  std::stringstream v;
  v << cpu_save_dir + "/seq_epoch-b1-" << iter << grade_tag << ".mat";
  printf("Saving b1 CPU data to file %s\n", v.str().c_str());
  save_cpu_data_test(nn.b[1], v);
}

void save_gpu_error(NeuralNetwork &nn, int iter, std::ofstream &error_file)
{
  arma::Mat<nn_real> A, B, C, D;

  std::stringstream s;
  s << cpu_load_dir + "/seq_epoch-W0-" << iter << grade_tag << ".mat";
  printf("Loading CPU data W0 from file %s\n", s.str().c_str());
  load_cpu_data_test(A, s);
  nn_real max_errW0 = arma::norm(nn.W[0] - A, "inf") / arma::norm(A, "inf");
  nn_real L2_errW0 = arma::norm(nn.W[0] - A, 2) / arma::norm(A, 2);

  std::stringstream t;
  t << cpu_load_dir + "/seq_epoch-W1-" << iter << grade_tag << ".mat";
  printf("Loading CPU data W1 from file %s\n", t.str().c_str());
  load_cpu_data_test(B, t);
  nn_real max_errW1 = arma::norm(nn.W[1] - B, "inf") / arma::norm(B, "inf");
  nn_real L2_errW1 = arma::norm(nn.W[1] - B, 2) / arma::norm(B, 2);

  std::stringstream u;
  u << cpu_load_dir + "/seq_epoch-b0-" << iter << grade_tag << ".mat";
  printf("Loading CPU data b0 from file %s\n", u.str().c_str());
  load_cpu_data_test(C, u);
  nn_real max_errb0 = arma::norm(nn.b[0] - C, "inf") / arma::norm(C, "inf");
  nn_real L2_errb0 = arma::norm(nn.b[0] - C, 2) / arma::norm(C, 2);

  std::stringstream v;
  v << cpu_load_dir + "/seq_epoch-b1-" << iter << grade_tag << ".mat";
  printf("Loading CPU data b1 from file %s\n", v.str().c_str());
  load_cpu_data_test(D, v);
  nn_real max_errb1 = arma::norm(nn.b[1] - D, "inf") / arma::norm(D, "inf");
  nn_real L2_errb1 = arma::norm(nn.b[1] - D, 2) / arma::norm(D, 2);

  int ow = 15;

  if (iter == 0)
  {
    error_file << std::left << std::setw(ow) << "Iteration" << std::left
               << std::setw(ow) << "Max Err W0" << std::left << std::setw(ow)
               << "Max Err W1" << std::left << std::setw(ow) << "Max Err b0"
               << std::left << std::setw(ow) << "Max Err b1" << std::left
               << std::setw(ow) << "L2 Err W0" << std::left << std::setw(ow)
               << "L2 Err W1" << std::left << std::setw(ow) << "L2 Err b0"
               << std::left << std::setw(ow) << "L2 Err b1"
               << "\n";
  }

  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow)
             << max_errW0 << std::left << std::setw(ow) << max_errW1
             << std::left << std::setw(ow) << max_errb0 << std::left
             << std::setw(ow) << max_errb1 << std::left << std::setw(ow)
             << L2_errW0 << std::left << std::setw(ow) << L2_errW1 << std::left
             << std::setw(ow) << L2_errb0 << std::left << std::setw(ow)
             << L2_errb1 << "\n";
}