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

  int rank, num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int print_flag = 0;

  int N = (rank == 0) ? X.n_cols : 0;
  
  if (num_procs > 1)
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                         MEMORY ALLOCATION                        //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  // Data sets
  const int num_batches = get_num_batches(N, batch_size);
  const int max_batch_size = batch_size;
  const int mini_batch_size_alloc = max_batch_size / num_procs + 1;

  // Network dimensions
  int n0 = nn.H[0];
  int n1 = nn.H[1];
  int n2 = nn.H[2];
  int nbatch = batch_size;
  int nminibatch = mini_batch_size_alloc;

  // Network parameters and data pointers
  nn_real* d_W1, *d_W2, *d_b1, *d_b2;
  nn_real* d_a1, *d_a2, *d_z1, *d_z2;
  nn_real* d_dW1, *d_dW2, *d_db1, *d_db2;
  nn_real* d_h1, *d_h2, *d_h3;
  nn_real* d_X, *d_y;

  // Allocate memory on GPU
  cudaMalloc(&d_X, n0 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_y, n2 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_W1, n1 * n0 * sizeof(nn_real));
  cudaMalloc(&d_W2, n2 * n1 * sizeof(nn_real));
  cudaMalloc(&d_b1, n1 * sizeof(nn_real));
  cudaMalloc(&d_b2, n2 * sizeof(nn_real));
  cudaMalloc(&d_a1, n1 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_a2, n2 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_z1, n1 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_z2, n2 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_dW1, n1 * n0 * sizeof(nn_real));
  cudaMalloc(&d_dW2, n2 * n1 * sizeof(nn_real));
  cudaMalloc(&d_db1, n1 * sizeof(nn_real));
  cudaMalloc(&d_db2, n2 * sizeof(nn_real));
  cudaMalloc(&d_h1, n1 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_h2, n1 * mini_batch_size_alloc * sizeof(nn_real));
  cudaMalloc(&d_h3, std::max(n0, n1) * mini_batch_size_alloc * sizeof(nn_real));

  // Copy data to GPU
  cudaMemcpy(d_W1, nn.W[0].memptr(), n1 * n0 * sizeof(nn_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W2, nn.W[1].memptr(), n2 * n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, nn.b[0].memptr(), n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, nn.b[1].memptr(), n2 * sizeof(nn_real), cudaMemcpyHostToDevice);

  // Buffers for MPI communication
  nn_real *X_recvbuf = (nn_real*) malloc(n0 * mini_batch_size_alloc * sizeof(nn_real));
  nn_real *y_recvbuf = (nn_real*) malloc(n2 * mini_batch_size_alloc * sizeof(nn_real));
  nn_real *dW1 = (nn_real*) malloc(n1 * n0 * sizeof(nn_real));
  nn_real *dW2 = (nn_real*) malloc(n2 * n1 * sizeof(nn_real));
  nn_real *db1 = (nn_real*) malloc(n1 * sizeof(nn_real));
  nn_real *db2 = (nn_real*) malloc(n2 * sizeof(nn_real));

  // Parameters for MPI splitting
  int *X_sendcounts = (int*) malloc(num_procs * sizeof(int));
  int *X_displs = (int*) malloc(num_procs * sizeof(int));
  int *y_sendcounts = (int*) malloc(num_procs * sizeof(int));
  int *y_displs = (int*) malloc(num_procs * sizeof(int));
  int X_recvcount;
  int y_recvcount;
  nn_real weight;

  arma::Mat<nn_real> X_batch, y_batch;

  int iter = 0;
  int last_col = 0;

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                         TRAINING LOOP                            //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    
    // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
    //                      MINIBATCH HANDLING                          //
    // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

    if (rank == 0 && num_procs > 1){
      for (int k = 0; k < num_procs; k++){
        nminibatch = get_mini_batch_size(batch_size, num_procs, k);
        X_sendcounts[k] = n0 * nminibatch;
        X_displs[k] = k == 0 ? 0 : X_displs[k - 1] + X_sendcounts[k - 1];
        y_sendcounts[k] = n2 * nminibatch;
        y_displs[k] = k == 0 ? 0 : y_displs[k - 1] + y_sendcounts[k - 1];
      }
    }

    nbatch = get_mini_batch_size(batch_size, num_procs, rank);
    weight = (nn_real) nbatch / batch_size;
    X_recvcount = n0 * nbatch;
    y_recvcount = n2 * nbatch;

    for (int batch = 0; batch < num_batches; ++batch)
    {

      last_col = (batch + 1) * batch_size - 1;

      if (last_col >= N)
      {
        last_col = N - 1;
        nbatch = N - batch * batch_size;
        weight = (nn_real) get_mini_batch_size(nbatch, num_procs, rank) / nbatch;

        if (rank == 0 && num_procs > 1){
          for (int k = 0; k < num_procs; k++){
            nminibatch = get_mini_batch_size(nbatch, num_procs, k);
            X_sendcounts[k] = n0 * nminibatch;
            X_displs[k] = k == 0 ? 0 : X_displs[k - 1] + X_sendcounts[k - 1];
            y_sendcounts[k] = n2 * nminibatch;
            y_displs[k] = k == 0 ? 0 : y_displs[k - 1] + y_sendcounts[k - 1];
          }
        }

        nbatch = get_mini_batch_size(nbatch, num_procs, rank);
        X_recvcount = n0 * nbatch;
        y_recvcount = n2 * nbatch;
      }

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                     MINIBATCH DISTRIBUTION                       //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      if (rank == 0){
        X_batch = X.cols(batch * batch_size, last_col);
        y_batch = y.cols(batch * batch_size, last_col);
      }

      if (num_procs > 1){
        MPI_Scatterv(X_batch.memptr(), X_sendcounts, X_displs, MPI_FP, X_recvbuf, X_recvcount,
                  MPI_FP, 0, MPI_COMM_WORLD);
        MPI_Scatterv(y_batch.memptr(), y_sendcounts, y_displs, MPI_FP, y_recvbuf, y_recvcount,
                  MPI_FP, 0, MPI_COMM_WORLD);
        
        cudaMemcpy(d_X, X_recvbuf, n0 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y_recvbuf, n2 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
      } else {
        cudaMemcpy(d_X, X_batch.memptr(), n0 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y_batch.memptr(), n2 * nbatch * sizeof(nn_real), cudaMemcpyHostToDevice);
      }

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
                  d_h3, reg, weight);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         EXCHANGE GRADIENTS                       //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      
      if (num_procs > 1){
        cudaMemcpy(dW1, d_dW1, n1 * n0 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(dW2, d_dW2, n2 * n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(db1, d_db1, n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(db2, d_db2, n2 * sizeof(nn_real), cudaMemcpyDeviceToHost);

        MPI_Allreduce(MPI_IN_PLACE, dW1, n1 * n0, MPI_FP, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, dW2, n2 * n1, MPI_FP, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, db1, n1, MPI_FP, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, db2, n2, MPI_FP, MPI_SUM, MPI_COMM_WORLD);

        cudaMemcpy(d_dW1, dW1, n1 * n0 * sizeof(nn_real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dW2, dW2, n2 * n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_db1, db1, n1 * sizeof(nn_real), cudaMemcpyHostToDevice);
        cudaMemcpy(d_db2, db2, n2 * sizeof(nn_real), cudaMemcpyHostToDevice);
      }

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    GRADIENT DESCENT STEP                         //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      gradient_descent_gpu(n0, n1, n2, d_W1, d_W2, d_b1, d_b2,
                          d_db1, d_db2, d_dW1, d_dW2, learning_rate);

      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                         POST PROCESSING                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

      if (print_every <= 0)
      {
        print_flag = batch == 0;
      }
      else
      {
        print_flag = iter % print_every == 0;
      }
      if (debug && rank == 0 && print_flag)
      {
        /*cudaMemcpy(nn.W[0].memptr(), d_W1, n1 * n0 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.W[1].memptr(), d_W2, n2 * n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[0].memptr(), d_b1, n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
        cudaMemcpy(nn.b[1].memptr(), d_b2, n2 * sizeof(nn_real), cudaMemcpyDeviceToHost);

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */
        //save_gpu_error(nn, iter, error_file);
      }

      iter++;
    }
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                  Update Neural Network on CPU                    //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  if (rank == 0){
    cudaMemcpy(nn.W[0].memptr(), d_W1, n1 * n0 * sizeof(nn_real), cudaMemcpyDeviceToHost);
    cudaMemcpy(nn.W[1].memptr(), d_W2, n2 * n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
    cudaMemcpy(nn.b[0].memptr(), d_b1, n1 * sizeof(nn_real), cudaMemcpyDeviceToHost);
    cudaMemcpy(nn.b[1].memptr(), d_b2, n2 * sizeof(nn_real), cudaMemcpyDeviceToHost);
  }

  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
  //                    Free memory allocations                       //
  // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //

  /*cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_W1);
  cudaFree(d_W2);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_z1);
  cudaFree(d_z2);
  cudaFree(d_a1);
  cudaFree(d_a2);
  cudaFree(d_dW1);
  cudaFree(d_dW2);
  cudaFree(d_db1);
  cudaFree(d_db2);
  cudaFree(d_h1);
  cudaFree(d_h2);
  cudaFree(d_h3);*/
}
