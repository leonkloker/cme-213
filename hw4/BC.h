#ifndef BC_H__
#define BC_H__

#include "simParams.h"
#include "Grid.h"

// Kernel to update the values on the boundary
__global__ void gpuUpdateBCsOnly(float *curr, float *prev, int gx, int gy, int b,
                                 float scaling_factor)
{
    const int tidx = blockIdx.x * blockDim.x + threadIdx.x;

    const int ny = gy - 2 * b;
    const int n_xblock = gx * b;
    const int n_yblock = ny * b;

    int idx;

    // Block 1
    if (tidx < n_xblock)
    {
        idx = tidx;
        curr[idx] = prev[idx] * scaling_factor;
    }
    else if (tidx < 2 * n_xblock)
    { // Block 2
        idx = tidx - n_xblock + gx * (gy - b);
        curr[idx] = prev[idx] * scaling_factor;
    }
    else if (tidx < 2 * n_xblock + n_yblock)
    { // Block 3
        idx = tidx - 2 * n_xblock;
        const int i = idx % b;
        const int j = idx / b;
        idx = i + gx * (b + j);
        curr[idx] = prev[idx] * scaling_factor;
    }
    else if (tidx < 2 * (n_xblock + n_yblock))
    { // Block 4
        idx = tidx - 2 * n_xblock - n_yblock;
        const int i = idx % b;
        const int j = idx / b;
        idx = i + (gx - b) + gx * (b + j);
        curr[idx] = prev[idx] * scaling_factor;
    }
}

// Function to update the values on the boundary for the CPU implementation
void updateBCsOnly(Grid &grid, Grid &prev, const simParams &params)
{
    const int borderSize = params.order() / 2;

    const int gx = params.gx();
    const int gy = params.gy();

    const float dt = params.dt();
    const double dx = params.dx();
    const double dy = params.dy();
    const double a = 0.06 / sqrt(dx * dy);
    const float scaling_factor = exp(-2 * a * a * dt);
    assert(scaling_factor > 0);

    const int upper_border_x = gx - borderSize;
    const int upper_border_y = gy - borderSize;

    for (int i = 0; i < gx; ++i)
    {
        for (int j = 0; j < borderSize; ++j)
        {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }

        for (int j = upper_border_y; j < gy; ++j)
        {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }
    }

    for (int j = borderSize; j < upper_border_y; ++j)
    {
        for (int i = 0; i < borderSize; ++i)
        {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }

        for (int i = upper_border_x; i < gx; ++i)
        {
            grid.hGrid_[i + gx * j] = prev.hGrid_[i + gx * j] * scaling_factor;
        }
    }

    /*
    // Testing that the boundary conditions were correctly applied
    for (int i = 0; i < gx; ++i)
      for (int j = 0; j < gy; ++j)
        if (i<borderSize || i >= upper_border_x || j<borderSize || j >= upper_border_y)
          assert(grid.hGrid_[i + gx * j] == prev.hGrid_[i + gx * j] * scaling_factor);
    */
}

class boundary_conditions
{
private:
    int gx, gy, b;
    float scaling_factor;
    int blocks_BC, threads_BC;

public:
    boundary_conditions(const simParams &params)
    {

        // Total number of threads needed for the BCs
        b = params.borderSize();

        gx = params.gx();
        gy = params.gy();

        const int ny = params.ny();
        const int nthreads_BC = 2 * (gx * b + ny * b);

        // We use a thread block with 192 threads
        threads_BC = 192;

        // We assume the number of grid blocks is less than 65535
        const int nblocks_BC = (nthreads_BC + threads_BC - 1) / threads_BC;
        blocks_BC = nblocks_BC;
        const float dt = params.dt();
        const double dx = params.dx();
        const double dy = params.dy();
        const double a = 0.06 / sqrt(dx * dy);
        scaling_factor = exp(-2 * a * a * dt);
    }

    void updateBC(float *prev, float *curr)
    {
        gpuUpdateBCsOnly<<<blocks_BC, threads_BC>>>(curr, prev, gx, gy, b,
                                                    scaling_factor);
    }
};

#endif