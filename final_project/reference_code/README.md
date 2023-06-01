# Parallel Neural Network Training using Multiple GPUs (CME 213 Final Project)
This repository contains the code (written in C++) used for my final project for CME 213 at Stanford University (Winter 2020 quarter). The final project for this course consisted of adapting a serial implementation of a neural network built to recognize handwritten digits (using the MNIST dataset) to run on multiple GPUs in parallel (i.e. utilizing both MPI and CUDA) so as to greatly decrease training time. A full account of the work that I completed may be found in my [final project report](https://github.com/jbinagia/cme213-final-project/blob/master/CME_213_Final_Report.pdf). Additionally, a description as provided to students of the course may be found on the [course website](https://stanford-cme213.github.io/Homework/FinalProjectPart1.pdf). 

## Usage
To run the compiled program on a single processor/GPU, the syntax is: 
`./main [args]` 
where `[args]` include but are not limited to: 
- `-n num`: to change the number of neurons in the hidden layer
- `-r num`: to change the degree of regularization
- `-l num`: to change the learning rate
- `e num`: to change the number of epochs
- `b num`: to change the batch size

Similarly, to run on `N` processes/GPUs, the syntax is: `mpirun -mca btl ^openib -n [N] ./main [args]`. 

Additional installation and usage information may be found on the [CME 213 course website](https://stanford-cme213.github.io/Homework/FinalProjectPart2.pdf). 
