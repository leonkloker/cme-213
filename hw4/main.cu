/*
 * 2D Heat Diffusion
 *
 * In this homework you will be implementing a finite difference 2D-Heat Diffusion Solver
 * in three different ways, in particular with and without using shared memory.
 * You will implement stencils of orders 2, 4 and 8.  A reference CPU implementation
 * has been provided.  You should keep all existing classes, method names, function names,
 * and variables as is.
 *
 * The simParams and Grid classes are provided for convenience. The simParams class will
 * load a file containing all the information needed for the simulation and calculate the
 * maximum stable CFL number.  The Grid will set up a grid with the appropriate boundary and
 * initial conditions.
 *
 * Some general notes about declaring N-dimensional arrays.
 * You may have seen / been taught to do this in the past:
 * int **A = (int **)malloc(numRows * sizeof(int *));
 * for (int r = 0; r < numRows; ++r)
 *     A[r] = (int *)malloc(numCols * sizeof(int));
 *
 * so that you can then access elements of A with the notation A[row][col], which involves dereferencing
 * two pointers.  This is a *really bad* way to represent 2D arrays for a couple of reasons.
 *
 * 1) For a NxN array, it does N+1 mallocs which is slow.  And on the gpu setting up this data
 *    structure is inconvenient.  But you should know how to do it.
 * 2) There is absolutely no guarantee that different rows are even remotely close in memory;
 *    subsequent rows could allocated on complete opposite sides of the address space
 *    which leads to terrible cache behavior.
 * 3) The double indirection leads to really high memory latency.  To access location A[i][j],
 *    first we have to make a trip to memory to fetch A[i], and once we get that pointer, we have to make another
 *    trip to memory to fetch (A[i])[j].  It would be far better if we only had to make one trip to
 *    memory.  This is especially important on the gpu.
 *
 * The *better way* - just allocate one 1-D array of size N*N.  Then just calculate the correct offset -
 * A[i][j] = *(A + i * numCols + j).  There is only one allocation, adjacent rows are as close as they can be
 * and we only make one trip to memory to fetch a value.  The grid implements this storage scheme
 * "under the hood" and overloads the () operator to allow the more familiar (x, y) notation.
 *
 * For the GPU code in this exercise you don't need to worry about trying to be fancy and overload an operator
 * or use some #define macro magic to mimic the same behavior - you can just do the raw addressing calculations.
 *
 * For the first part of the homework where you will implement the kernels without using shared memory
 * each thread should compute exactly one output.
 *
 * For the second part with shared memory - it is recommended that you use 1D blocks since the ideal
 * implementation will have each thread outputting more than 1 value and the addressing arithmetic
 * is actually easier with 1D blocks.
 */

#include <ostream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>
#include <string>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <unistd.h>

#include "gtest/gtest.h"
#include "simParams.h"
#include "Grid.h"
#include "CPUComputation.h"
#include "Errors.h"

#include "gpuStencil.cu"

// Declare some global variables so different google tests can access the same data and only run once
simParams params("params.in");
Grid grid(params.gx(), params.gy());

void initGrid(Grid &grid, const simParams &params)
{

    const int gx = params.gx();
    const int gy = params.gy();
    const double dx = params.dx();
    const double dy = params.dy();
    const double a = 0.06 / sqrt(dx * dy);
    for (int i = 0; i < gx; ++i)
    {
        for (int j = 0; j < gy; ++j)
        {
            grid.hGrid_.at(i + gx * j) = sin(i * a * dx) * sin(j * a * dy);
        }
    }

    grid.toGPU();
}

TEST(DiffusionTest, GlobalTest)
{
    Grid gpuGrid(grid);
    std::vector<double> errorsg;
    double elapsed = gpuComputationGlobal(gpuGrid, params); // Calculation on the GPU

    cout << "Order: " << params.order() << ", "
         << params.nx() << "x" << params.ny() << ", "
         << params.iters() << " iterations" << endl;
    cout << setw(15) << " " << setw(15) << "time (ms)" << setw(15) << "GBytes/sec" << endl;
    cout << setw(15) << "Global" << setw(15) << setprecision(6) << elapsed
         << setw(15) << (params.calcBytes() / (elapsed / 1E3)) / 1E9 << endl;

    // Copy back the solution
    gpuGrid.fromGPU();
    // Check for errors
    int error = checkErrors(grid, gpuGrid, params, "globalErrors.txt", errorsg);
    PrintErrors(errorsg);

    // Tell Gtest failure occurs
    if (error)
    {
        FAIL() << "There was an error in the computation, quitting..." << endl;
    }
    // for debugging, save data to file
    // gpuGrid.saveStateToFile("final_gpu_global.csv");
}

TEST(DiffusionTest, BlockTest)
{
    Grid gpuGrid(grid);
    std::vector<double> errorsb;
    double elapsed = gpuComputationBlock(gpuGrid, params);

    cout << "Order: " << params.order() << ", "
         << params.nx() << "x" << params.ny() << ", "
         << params.iters() << " iterations" << endl;
    cout << setw(15) << " " << setw(15) << "time (ms)" << setw(15) << "GBytes/sec" << endl;
    cout << setw(15) << "Block" << setw(15) << setprecision(6) << elapsed
         << setw(15) << (params.calcBytes() / (elapsed / 1E3)) / 1E9 << endl;

    gpuGrid.fromGPU();
    int error = checkErrors(grid, gpuGrid, params, "globalErrors.txt", errorsb);
    PrintErrors(errorsb);

    // Tell Gtest failure occurs
    if (error)
    {
        FAIL() << "There was an error in the computation, quitting..." << endl;
    }
    // gpuGrid.saveStateToFile("final_gpu_block.csv");
}

TEST(DiffusionTest, SharedTest)
{
    Grid gpuGrid(grid);
    std::vector<double> errorss;
    double elapsed = 0;
    if (params.order() == 2)
    {
        elapsed = gpuComputationShared<2>(gpuGrid, params);
    }
    else if (params.order() == 4)
    {
        elapsed = gpuComputationShared<4>(gpuGrid, params);
    }
    else if (params.order() == 8)
    {
        elapsed = gpuComputationShared<8>(gpuGrid, params);
    }

    cout << "Order: " << params.order() << ", "
         << params.nx() << "x" << params.ny() << ", "
         << params.iters() << " iterations" << endl;
    cout << setw(15) << " " << setw(15) << "time (ms)" << setw(15) << "GBytes/sec" << endl;
    cout << setw(15) << "Shared" << setw(15) << setprecision(6) << elapsed
         << setw(15) << (params.calcBytes() / (elapsed / 1E3)) / 1E9 << endl;
    gpuGrid.fromGPU();
    int error = checkErrors(grid, gpuGrid, params, "sharedErrors.txt", errorss);
    PrintErrors(errorss);

    // Tell Gtest failure occurs
    if (error)
    {
        FAIL() << "There was an error in the computation, quitting..." << endl;
    }
    // gpuGrid.saveStateToFile("final_gpu_shared.csv");
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    // Let's initialize our grid and params
    initGrid(grid, params);

    // for debugging, you may want to uncomment this line
    // grid.saveStateToFile("init");
    // save our initial state, useful for making sure we got setup and BCs right

    cout << "Order: " << params.order() << ", "
         << params.nx() << "x" << params.ny() << ", "
         << params.iters() << " iterations" << endl;
    cout << setw(15) << " " << setw(15) << "time (ms)" << setw(15) << "GBytes/sec" << endl;

    // compute our reference solution
    double elapsed = cpuComputation(grid, params);

    // for debugging, you may want to uncomment the following line
    grid.saveStateToFile("final_cpu");

    // Print statistics for CPU calculation
    cout << setw(15) << "CPU" << setw(15) << setprecision(6) << elapsed
         << setw(15) << params.calcBytes() / (elapsed / 1E3) / 1E9 << endl;

    return RUN_ALL_TESTS();
}