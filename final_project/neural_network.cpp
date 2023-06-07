#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>
#include <iomanip>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

#define MPI_SAFE_CALL(call)                                                  \
  do                                                                         \
  {                                                                          \
    int err = call;                                                          \
    if (err != MPI_SUCCESS)                                                  \
    {                                                                        \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

int get_num_batches(int N, int batch_size)
{
  return (N + batch_size - 1) / batch_size;
}

int get_batch_size(int N, int batch_size, int batch)
{
  int num_batches = get_num_batches(N, batch_size);
  return (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
}

int get_mini_batch_size(int batch_size, int num_procs, int rank)
{
  int mini_batch_size = batch_size / num_procs;
  return rank < batch_size % num_procs ? mini_batch_size + 1 : mini_batch_size;
}

nn_real norms(NeuralNetwork &nn)
{
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i)
  {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real> &mat, arma::Mat<nn_real> &mat2)
{
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

// feedforward pass
void feedforward(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                 struct cache &cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork &nn, const arma::Mat<nn_real> &y, nn_real reg,
              const struct cache &bpcache, struct grads &bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork &nn, const arma::Mat<nn_real> &yc,
             const arma::Mat<nn_real> &y, nn_real reg)
{
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
             arma::Row<nn_real> &label)
{
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i)
  {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
           const arma::Mat<nn_real> &y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug)
{
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  assert(X.n_cols == y.n_cols);

  int num_batches = get_num_batches(N, batch_size);

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    int batch_start = 0;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      int last_col = batch_start + get_batch_size(N, batch_size, batch);
      assert(last_col <= X.n_cols);
      assert(last_col <= y.n_cols);
      assert(last_col > batch_start);
      if (batch == num_batches - 1)
      {
        assert(last_col == X.n_cols);
      }
      arma::Mat<nn_real> X_batch = X.cols(batch_start, last_col - 1);
      arma::Mat<nn_real> y_batch = y.cols(batch_start, last_col - 1);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0)
      {
        if (grad_check)
        {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i)
      {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i)
      {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to the "cpu_save_dir" folder.
         In the later runs (with same parameters), you can use just the debug
         flag to output diff b/w CPU and GPU without running the CPU version
         version. */
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag)
      {
        save_cpu_data(nn, iter);
      }

      batch_start = last_col;
      iter++;
    }
  }
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork &nn, const arma::Mat<nn_real> &X,
                    const arma::Mat<nn_real> &y, nn_real learning_rate,
                    std::ofstream &error_file, nn_real reg, const int epochs,
                    const int batch_size, int print_every, int debug)
{
  assert(learning_rate > 0);
  assert(reg >= 0);
  assert(epochs >= 0);
  assert(batch_size > 0);

  int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank == 0)
  {
    assert(X.n_cols > 0);
    assert(X.n_rows == IMAGE_SIZE);
    assert(y.n_cols == X.n_cols);
    assert(y.n_rows == NUM_CLASSES);
    assert(nn.H[0] == IMAGE_SIZE);
    assert(nn.H[2] == NUM_CLASSES);
  }

  int N = (rank == 0) ? X.n_cols : 0;

  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  assert(N > 0);

  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way using memptr().
     Or you can allocate your own array memory space and store the elements in a
     row major way. Remember to update the Armadillo matrices in NeuralNetwork
     &nn of rank 0 before returning from the function. */

  /* allocate memory before the iterations */
  // Data sets
  const int num_batches = get_num_batches(N, batch_size);
  int mini_batch_size_alloc;
  {
    const int max_batch_size = batch_size;
    mini_batch_size_alloc = max_batch_size / num_procs + 1;
  }

  // Network dimensions
  int n0 = nn.H[0];
  int n1 = nn.H[1];
  int n2 = nn.H[2];
  int nbatch = batch_size;

  // Network parameters
  nn_real* d_W1, *d_W2, *d_b1, *d_b2;
  nn_real* d_a1, *d_a2, *d_z1, *d_z2;
  nn_real* d_dW1, *d_dW2, *d_db1, *d_db2;
  nn_real* d_h1, *d_h2, *d_h3, *d_h4, *d_h5, *d_h6;
  nn_real* d_X, *d_y;

  // Allocate memory
  cudaMalloc(&d_X, n0 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_y, n2 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_W1, n1 * n0 * sizeof(nn_real));
  cudaMalloc(&d_W2, n2 * n1 * sizeof(nn_real));
  cudaMalloc(&d_b1, n1 * sizeof(nn_real));
  cudaMalloc(&d_b2, n2 * sizeof(nn_real));
  cudaMalloc(&d_a1, n1 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_a2, n2 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_z1, n1 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_z2, n2 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_dW1, n1 * n0 * sizeof(nn_real));
  cudaMalloc(&d_dW2, n2 * n1 * sizeof(nn_real));
  cudaMalloc(&d_db1, n1 * sizeof(nn_real));
  cudaMalloc(&d_db2, n2 * sizeof(nn_real));
  cudaMalloc(&d_h1, n2 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_h2, nbatch * n1 * sizeof(nn_real));
  cudaMalloc(&d_h3, n1 * n2 * sizeof(nn_real));
  cudaMalloc(&d_h4, n1 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_h5, n1 * nbatch * sizeof(nn_real));
  cudaMalloc(&d_h6, nbatch * n0 * sizeof(nn_real));

  // Copy data to GPU
  cudaMemcpy(d_W1, nn.W[0].memptr(), n1 * n0 * sizeof(nn_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W2, nn.W[1].memptr(), n2 * n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, nn.b[0].memptr(), n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, nn.b[1].memptr(), n2 * sizeof(nn_real), cudaMemcpyHostToDevice);

  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;
  int last_col = 0;

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    nbatch = batch_size;
    for (int batch = 0; batch < num_batches; ++batch)
    {
      /*
       * Possible implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */

      last_col = (batch + 1) * batch_size - 1;
 
      if ((batch + 1) * batch_size >= X.n_cols)
      {
        last_col = X.n_cols - 1;
        nbatch = last_col - batch * batch_size + 1;
      }

      arma::Mat<nn_real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<nn_real> y_batch = y.cols(batch * batch_size, last_col);

      cudaMemcpy(d_X, X_batch.memptr(), n0 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y_batch.memptr(), n2 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                          FEED FORWARD                            //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      
      feedforward_gpu(n0, n1, n2, nbatch, d_X, d_W1, d_W2, d_b1, d_b2, d_a1, 
                      d_a2, d_z1, d_z2);
      
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         BACK PROPAGATE                           //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      
      backprop_gpu(n0, n1, n2, nbatch, d_a1, d_a2, d_z1, d_z2, d_W1, d_W2,
                  d_X, d_b1, d_b2, d_y, d_db1, d_db2, d_dW1, d_dW2, d_h1, d_h2,
                  d_h3, d_h4, d_h5, d_h6, reg);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      gradient_descent_gpu(n0, n1, n2, d_W1, d_W2, d_b1, d_b2,
                          d_db1, d_db2, d_dW1, d_dW2, learning_rate);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      if (print_every > 0 && iter % print_every == 0)
      {
        cudaMemcpy(nn.W[0].memptr(), d_W1, n1 * n0 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), d_W2, n2 * n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), d_b1, n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), d_b2, n2 * sizeof(nn_real), cudaMemcpyDeviceToHost);

        arma::Mat<nn_real> y_hat_batch(n2, nbatch);
        cudaMemcpy(y_hat_batch.memptr(), d_a2, n2 * nbatch * sizeof(nn_real), cudaMemcpyDeviceToHost);

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, y_hat_batch, y_batch, reg) << "\n";
      }
      
      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }

      /* Following debug routine assumes that you have already updated the arma
         matrices in the NeuralNetwork nn.  */
      if (debug && rank == 0 && print_flag)
      {
        // TODO
        // Copy data back to the CPU

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */

        save_gpu_error(nn, iter, error_file);
      }

      iter++;
    }
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  cudaMemcpy(nn.W[0].memptr(), d_W1, n1 * n0 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.W[1].memptr(), d_W2, n2 * n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[0].memptr(), d_b1, n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  cudaMemcpy(nn.b[1].memptr(), d_b2, n2 * sizeof(nn_real), cudaMemcpyDeviceToHost);

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  cudaFree(d_W1);
  cudaFree(d_W2);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_db1);
  cudaFree(d_db2);
  cudaFree(d_dW1);
  cudaFree(d_dW2);
  cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_a1);
  cudaFree(d_a2);
  cudaFree(d_z1);
  cudaFree(d_z2);
  cudaFree(d_h1);
  cudaFree(d_h2);
  cudaFree(d_h3);
  cudaFree(d_h4);
  cudaFree(d_h5);
  cudaFree(d_h6);
}
