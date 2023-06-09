#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <mpi.h>
#include <unistd.h>

#include <armadillo>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#include "utils/common.h"
#include "utils/mnist.h"
#include "utils/neural_network.h"
#include "utils/tests.h"

// TODO: edit the following to choose which directory you wish
// to store your CPU results in.
string cpu_save_dir = "./cpu_output";
// TODO: directory where the CPU results should be loaded from.
string cpu_load_dir = "./cpu_output";
// TODO: path to the MNIST file location.
string file_train_images = "./MNIST_DATA/train-images.idx3-ubyte";
string file_train_labels = "./MNIST_DATA/train-labels.idx1-ubyte";
string file_test_images = "./MNIST_DATA/t10k-images.idx3-ubyte";
string file_test_labels = "./MNIST_DATA/t10k-labels.idx1-ubyte";

string output_dir = "Outputs";

string grade_tag;
string mpi_tag;

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

int main(int argc, char *argv[])
{
  // Initialize MPI
  int num_procs = 0, rank = 0;
  MPI_Init(&argc, &argv);
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  mpi_tag = std::string("-") + std::to_string(num_procs);

  // Assign a GPU device to each MPI proc
  int nDevices;
  cudaGetDeviceCount(&nDevices);

  if (nDevices < num_procs && rank == 0)
  {
    std::cerr << "Please allocate at least as many GPUs as the number of MPI procs.\n";
    std::cerr << "Number of processes: " << num_procs << std::endl;
    std::cerr << "Number of GPUs: " << nDevices << std::endl;
    exit(1);
  }

  checkCudaErrors(cudaSetDevice(rank));

  if (rank == 0)
  {
    printf("Number of MPI processes         %1d\n", num_procs);
    printf("Number of CUDA devices          %1d\n", nDevices);
  }

  // Read in command line arguments
  std::vector<int> H(3);
  nn_real reg = 1e-4;
  nn_real learning_rate = 0.001;
  int num_epochs = 1;
  int batch_size = 32;
  int num_neuron = 32;
  int run_seq = 0;
  int debug = 0;
  int grade = 0;
  int print_every = 0;

  int option = 0;

  while ((option = getopt(argc, argv, "n:r:l:e:b:g:p:sd")) != -1)
  {
    switch (option)
    {
    case 'n': // number of neurons
      num_neuron = atoi(optarg);
      break;

    case 'r': // regularization factor
      reg = atof(optarg);
      break;

    case 'l': // learning rate
      learning_rate = atof(optarg);
      break;

    case 'e': // number of epochs
      num_epochs = atoi(optarg);
      break;

    case 'b': // batch size
      batch_size = atoi(optarg);
      break;

    case 'g':
      // grading mode
      // We write to files NNErrors*.txt and calculate the error
      // only if g in {1,2,3}.
      grade = atoi(optarg);
      break;

    case 'p': // frequency for saving NN coefficients
      print_every = atoi(optarg);
      break;

    case 's': // run the sequential code
      run_seq = 1;
      break;

    case 'd':
      // calculate the GPU calculation error.
      // this option assumes that the result from the CPU run
      // has been saved in a file.
      debug = 1;
      break;
    }
  }

  /* This option is going to be used to test correctness.
     DO NOT change the following parameters */
  switch (grade)
  {
  case 0: // No grading
    break;

  case 1: // Low lr, high iters
    learning_rate = 0.0005;
    num_epochs = 40;
    reg = 1e-4;
    batch_size = 800;
    num_neuron = 1000;
    break;

  case 2: // Medium lr, medium iters
    learning_rate = 0.001;
    num_epochs = 10;
    reg = 1e-4;
    batch_size = 800;
    num_neuron = 1000;
    break;

  case 3: // High lr, very few iters
    learning_rate = 0.002;
    num_epochs = 1;
    reg = 1e-4;
    batch_size = 800;
    num_neuron = 1000;
    break;

  case 4:
    break;
  }

  // GEMM tests
  if (grade == 4)
  {
    if (rank == 0)
    {
      BenchmarkGEMM();
    }

    MPI_Finalize();
    return 0;
  }

  // DNN learning tests
  if (grade == 1)
  {
    print_every = 600;
  }
  else if (grade == 2)
  {
    print_every = 150;
  }
  else if (grade == 3)
  {
    print_every = 15;
  }

  // DNN learning tests
  if (grade > 0)
  {
    debug = 1; // always run in debug mode
    grade_tag = std::string("-" + std::to_string(grade));
  }
  else
    grade_tag = std::string();
  // grade_tag is added to the file name when saving the CPU results.
  // It is sequal to the grading mode or is left empty if grade = 0.

  H[0] = IMAGE_SIZE;
  H[1] = num_neuron;
  H[2] = NUM_CLASSES;

  arma::Mat<nn_real> x_train, y_train, label_train;
  arma::Mat<nn_real> x_dev, y_dev, label_dev;

  arma::Mat<nn_real> x_test(IMAGE_SIZE, NUM_TEST);
  arma::Row<nn_real> label_test = arma::zeros<arma::Row<nn_real>>(NUM_TEST);
  arma::Mat<nn_real> y_test =
      arma::zeros<arma::Mat<nn_real>>(NUM_CLASSES, NUM_TEST);

  NeuralNetwork seq_nn(H);
  NeuralNetwork nn(H);

  if (rank == 0)
  {
    if (grade > 0)
    {
      printf("Grading mode on; grading mode   %d\n", grade);
    }
    else
    {
      printf("Grading mode off\n");
    }
    printf("Number of neurons           %5d\n", num_neuron);
    printf("Number of epochs              %3d\n", num_epochs);
    printf("Batch size                   %4d\n", batch_size);
    printf("Regularization    %15.8g\n", reg);
    printf("Learning rate     %15.8g\n", learning_rate);
    if (run_seq == 1)
    {
      printf("The sequential code is run\n");
    }
    else
    {
      printf("The sequential code is not run\n");
    }

    if (debug == 1)
    {
      printf("The debug option is on\n");
      printf("The output directory is %s\n", output_dir.c_str());
      if (run_seq == 1)
      {
        printf("The CPU results are saved to %s\n", cpu_save_dir.c_str());
      }
      printf("The CPU results are loaded from %s\n", cpu_load_dir.c_str());
    }
    else
    {
      printf("The debug option is off\n");
    }

    // Read MNIST images into Armadillo mat vector
    arma::Mat<nn_real> x(IMAGE_SIZE, NUM_TRAIN);
    // label contains the prediction for each
    arma::Row<nn_real> label = arma::zeros<arma::Row<nn_real>>(NUM_TRAIN);
    // y is the matrix of one-hot label vectors where only y[c] = 1,
    // where c is the right class.
    arma::Mat<nn_real> y =
        arma::zeros<arma::Mat<nn_real>>(NUM_CLASSES, NUM_TRAIN);

    std::cout << "Loading training data" << std::endl;
    read_mnist(file_train_images, x);
    read_mnist_label(file_train_labels, label);
    label_to_y(label, NUM_CLASSES, y);

    /* Print stats of training data */
    std::cout << "Training data information:" << std::endl;
    std::cout << "Size of x_train, N =  " << x.n_cols << std::endl;
    std::cout << "Size of label_train = " << label.size() << std::endl;

    assert(x.n_cols == NUM_TRAIN && x.n_rows == IMAGE_SIZE);
    assert(label.size() == NUM_TRAIN);

    /* Split into train set and dev set, you should use train set to train your
       neural network and dev set to evaluate its precision */
    int dev_size = (int)(0.1 * NUM_TRAIN);

    assert(dev_size > 0);

    x_train = x.cols(0, NUM_TRAIN - dev_size - 1);
    y_train = y.cols(0, NUM_TRAIN - dev_size - 1);
    label_train = label.cols(0, NUM_TRAIN - dev_size - 1);

    assert(x_train.n_cols > 0);
    assert(x_train.n_rows == IMAGE_SIZE);
    assert(y_train.n_cols == x_train.n_cols);
    assert(y_train.n_rows == NUM_CLASSES);

    x_dev = x.cols(NUM_TRAIN - dev_size, NUM_TRAIN - 1);
    y_dev = y.cols(NUM_TRAIN - dev_size, NUM_TRAIN - 1);
    label_dev = label.cols(NUM_TRAIN - dev_size, NUM_TRAIN - 1);

    std::cout << "Loading testing data" << std::endl;
    read_mnist(file_test_images, x_test);
    read_mnist_label(file_test_labels, label_test);
    label_to_y(label_test, NUM_CLASSES, y_test);
  }

  // For grading mode 1, 2, or 3 we need to check whether the sequential code
  // needs to be run or not.
  if (grade > 0 && run_seq == 0)
  {
    for (int i = 0; i < seq_nn.num_layers; i++)
    {
      std::stringstream s;
      s << cpu_save_dir + "/seq_nn-W" << i << grade_tag << ".mat";
      if (!file_exists(s.str()))
      {
        run_seq = 1;
      }
      std::stringstream u;
      u << cpu_save_dir + "/seq_nn-b" << i << grade_tag << ".mat";
      if (!file_exists(u.str()))
      {
        run_seq = 1;
      }
    }
    if (run_seq == 1)
    {
      printf("Running the sequential code in order to have reference data for grading mode %d\n", grade);
    }
  }

  /* Run the sequential code if the serial flag is set */
  using namespace std::chrono;
  if ((rank == 0) && (run_seq))
  {
    std::cout << "Start Sequential Training" << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    train(seq_nn, x_train, y_train, learning_rate, reg, num_epochs, batch_size,
          false, print_every, debug);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "Time for Sequential Training: " << time_span.count()
              << " seconds" << std::endl;

    // Saving data to file
    if (grade > 0 || debug == 1)
    {
      for (int i = 0; i < seq_nn.num_layers; i++)
      {
        std::stringstream s;
        s << cpu_save_dir + "/seq_nn-W" << i << grade_tag << ".mat";
        printf("Saving W%1d CPU data to file %s\n", i, s.str().c_str());
        save_cpu_data_test(seq_nn.W[i], s);
        std::stringstream u;
        u << cpu_save_dir + "/seq_nn-b" << i << grade_tag << ".mat";
        printf("Saving b%1d CPU data to file %s\n", i, u.str().c_str());
        save_cpu_data_test(seq_nn.b[i], u);
      }
    }

    {
      arma::Row<nn_real> label_pred;
      predict(seq_nn, x_dev, label_pred);
      nn_real prec = precision(label_pred, label_dev);
      printf("Precision on validation set for sequential training = %20.16f\n", prec);
    }

    {
      arma::Row<nn_real> label_pred;
      predict(seq_nn, x_test, label_pred);
      nn_real prec = precision(label_pred, label_test);
      printf("Precision on testing set for sequential training = %20.16f\n", prec);
    }
  }

  /* Train the neural network in parallel*/
  if (rank == 0)
    std::cout << std::endl
              << "Start Parallel Training" << std::endl;

  std::ofstream error_file;
  if (debug)
  {
    string error_filename = output_dir + "/CpuGpuDiff" + mpi_tag + grade_tag + ".txt";
    error_file.open(error_filename);
    if (!error_file.good())
    {
      std::cerr << "Unable to open the file " << error_filename << std::endl;
      std::cerr << "Make sure the directory " << output_dir << " exists" << std::endl;
      exit(1);
    }

    if (rank == 0)
      printf("The error in the GPU calculation during training is saved to %s\n", error_filename.c_str());
  }

  MPI_Barrier(MPI_COMM_WORLD);

  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  /* ---- Parallel Training ---- */
  if (rank == 0)
  {
    assert(x_train.n_cols > 0);
    assert(x_train.n_rows == IMAGE_SIZE);
    assert(y_train.n_cols == x_train.n_cols);
    assert(y_train.n_rows == NUM_CLASSES);
    assert(nn.H[0] == IMAGE_SIZE);
    assert(nn.H[2] == NUM_CLASSES);
  }

  parallel_train(nn, x_train, y_train, learning_rate, error_file, reg,
                 num_epochs, batch_size, print_every, debug);

  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  error_file.close();

  if (rank == 0)
    std::cout << "Time for Parallel Training: " << time_span.count()
              << " seconds" << std::endl;

  /* Note: Make sure after parallel training, rank 0's neural network is up to
   * date */

  /* Run predictions for the parallel NN */
  if (rank == 0)
  {
    {
      arma::Row<nn_real> label_pred;
      predict(nn, x_dev, label_pred);
      nn_real prec = precision(label_pred, label_dev);
      printf("Precision on validation set for parallel training = %20.16f\n", prec);
    }

    {
      arma::Row<nn_real> label_pred;
      predict(nn, x_test, label_pred);
      nn_real prec = precision(label_pred, label_test);
      printf("Precision on testing set for parallel training = %20.16f\n", prec);
    }
  }

  /* If grading mode is on, compare CPU and GPU results and check for
   * correctness */
  if (rank == 0 && (grade > 0 || debug == 1))
  {
    std::cout << std::endl
              << "Checking for correctness..."
              << std::endl;
    // Reading data from file
    for (int i = 0; i < seq_nn.num_layers; i++)
    {
      std::stringstream s;
      s << cpu_load_dir + "/seq_nn-W" << i << grade_tag << ".mat";
      printf("Loading from file %s\n", s.str().c_str());
      load_cpu_data_test(seq_nn.W[i], s);
      std::stringstream u;
      u << cpu_load_dir + "/seq_nn-b" << i << grade_tag << ".mat";
      printf("Loading from file %s\n", u.str().c_str());
      load_cpu_data_test(seq_nn.b[i], u);
    }
    {
      string error_filename = output_dir + "/NNErrors" + mpi_tag + grade_tag + ".txt";
      printf("\nThe error in the GPU DNN at the completion of training is saved to %s\n", error_filename.c_str());
    }
    checkNNErrors(seq_nn, nn,
                  output_dir + "/NNErrors" + mpi_tag + grade_tag + ".txt");
  }

  MPI_Finalize();
  return 0;
}
