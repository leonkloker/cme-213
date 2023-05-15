#ifndef CPUCOMPUTATION_H_
#define CPUCOMPUTATION_H_

#include <cmath>

#include "mp1-util.h"
#include "simParams.h"
#include "Grid.h"
#include "BC.h"

template <int order>
inline float stencil(float *curr_grid, int gx, int x, int y, float xcfl,
                     float ycfl)
{
    if (order == 2)
    {
        return curr_grid[x + gx * y] +
               xcfl * (curr_grid[x + 1 + gx * y] + curr_grid[x - 1 + gx * y] -
                       2 * curr_grid[x + gx * y]) +
               ycfl * (curr_grid[x + gx * (y + 1)] + curr_grid[x + gx * (y - 1)] -
                       2 * curr_grid[x + gx * y]);
    }
    else if (order == 4)
    {
        return curr_grid[x + gx * y] +
               xcfl * (-curr_grid[x + 2 + gx * y] + 16 * curr_grid[x + 1 + gx * y] -
                       30 * curr_grid[x + gx * y] + 16 * curr_grid[x - 1 + gx * y] -
                       curr_grid[x - 2 + gx * y]) +
               ycfl * (-curr_grid[x + gx * (y + 2)] + 16 * curr_grid[x + gx * (y + 1)] -
                       30 * curr_grid[x + gx * y] + 16 * curr_grid[x + gx * (y - 1)] -
                       curr_grid[x + gx * (y - 2)]);
    }
    else if (order == 8)
    {
        return curr_grid[x + gx * y] +
               xcfl * (-9 * curr_grid[x + 4 + gx * y] + 128 * curr_grid[x + 3 + gx * y] -
                       1008 * curr_grid[x + 2 + gx * y] + 8064 * curr_grid[x + 1 + gx * y] -
                       14350 * curr_grid[x + gx * y] + 8064 * curr_grid[x - 1 + gx * y] -
                       1008 * curr_grid[x - 2 + gx * y] + 128 * curr_grid[x - 3 + gx * y] -
                       9 * curr_grid[x - 4 + gx * y]) +
               ycfl * (-9 * curr_grid[x + gx * (y + 4)] + 128 * curr_grid[x + gx * (y + 3)] -
                       1008 * curr_grid[x + gx * (y + 2)] + 8064 * curr_grid[x + gx * (y + 1)] -
                       14350 * curr_grid[x + gx * y] + 8064 * curr_grid[x + gx * (y - 1)] -
                       1008 * curr_grid[x + gx * (y - 2)] + 128 * curr_grid[x + gx * (y - 3)] -
                       9 * curr_grid[x + gx * (y - 4)]);
    }
    else
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
}

double cpuComputation(Grid &curr_grid, const simParams &params)
{
    Grid next_grid(curr_grid);

    event_pair timer;
    start_timer(&timer);

    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    int nx = params.nx();
    int ny = params.ny();

    int gx = params.gx();
    int borderSize = params.borderSize();

    for (int i = 0; i < params.iters(); ++i)
    {
        // update the values on the boundary only
        updateBCsOnly(curr_grid, next_grid, params);

        // apply stencil
        if (params.order() == 2)
        {
            for (int y = borderSize; y < ny + borderSize; ++y)
            {
                for (int x = borderSize; x < nx + borderSize; ++x)
                {
                    next_grid.hGrid_[x + gx * y] = stencil<2>(curr_grid.hGrid_.data(), gx, x, y,
                                                              xcfl, ycfl);
                }
            }
        }
        else if (params.order() == 4)
        {
            for (int y = borderSize; y < ny + borderSize; ++y)
            {
                for (int x = borderSize; x < nx + borderSize; ++x)
                {
                    next_grid.hGrid_[x + gx * y] = stencil<4>(curr_grid.hGrid_.data(), gx, x, y,
                                                              xcfl, ycfl);
                }
            }
        }
        else if (params.order() == 8)
        {
            for (int y = borderSize; y < ny + borderSize; ++y)
            {
                for (int x = borderSize; x < nx + borderSize; ++x)
                {
                    next_grid.hGrid_[x + gx * y] = stencil<8>(curr_grid.hGrid_.data(), gx, x, y,
                                                              xcfl, ycfl);
                }
            }
        }

        Grid::swap(curr_grid, next_grid);
    }
    return stop_timer(&timer);
}

#endif
