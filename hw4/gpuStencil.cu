#include <math_constants.h>

#include "BC.h"

/**
 * Calculates the next finite difference step given a
 * grid point and step lengths.
 *
 * @param curr Pointer to the grid point that should be updated.
 * @param width Number of grid points in the x dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 * @returns Grid value of next timestep.
 */
template<int order>
__device__
float Stencil(const float* curr, int width, float xcfl, float ycfl) {
    switch(order) {
        case 2:
            return curr[0] + xcfl * (curr[-1] + curr[1] - 2.f * curr[0]) +
                   ycfl * (curr[width] + curr[-width] - 2.f * curr[0]);

        case 4:
            return curr[0] + xcfl * (-curr[2] + 16.f * curr[1] - 30.f * curr[0]
                                     + 16.f * curr[-1] - curr[-2])
                           + ycfl * (- curr[2 * width] + 16.f * curr[width]
                                     - 30.f * curr[0] + 16.f * curr[-width]
                                     - curr[-2 * width]);

        case 8:
            return curr[0] + xcfl * (-9.f * curr[4] + 128.f * curr[3]
                                     - 1008.f * curr[2] + 8064.f * curr[1]
                                     - 14350.f * curr[0] + 8064.f * curr[-1]
                                     - 1008.f * curr[-2] + 128.f * curr[-3]
                                     - 9.f * curr[-4])
                           + ycfl * (-9.f * curr[4 * width]
                                     + 128.f * curr[3 * width]
                                     - 1008.f * curr[2 * width]
                                     + 8064.f * curr[width]
                                     - 14350.f * curr[0]
                                     + 8064.f * curr[-width]
                                     - 1008.f * curr[-2 * width]
                                     + 128.f * curr[-3 * width]
                                     - 9.f * curr[-4 * width]);

        default:
            printf("ERROR: Order %d not supported", order);
            return CUDART_NAN_F;
    }
}

/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be very simple and only use global memory
 * and 1d threads and blocks.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order>
__global__
void gpuStencilGlobal(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                float xcfl, float ycfl) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = (id % nx) + order/2;
    unsigned int y = (id / nx) + order/2;
    unsigned int gridpoint = x + y * gx;
    next[gridpoint] = Stencil<order>(curr + gridpoint, gx, xcfl, ycfl);
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilGlobal kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationGlobal(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int gx = curr_grid.gx();
    int gy = curr_grid.gy();
    int order = params.order();
    int nx = gx - order;
    int ny = gy - order;
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    float threads = 1 << 8;

    event_pair timer;
    start_timer(&timer);

    curr_grid.saveStateToFile("gpu_global_init.csv");

    for(int i = 0; i < params.iters(); ++i) {
        
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch (order){
            case 2:
                gpuStencilGlobal<2><<<ceil(nx * ny / threads), threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            case 4:
                gpuStencilGlobal<4><<<ceil(nx * ny / threads), threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            case 8:
                gpuStencilGlobal<8><<<ceil(nx * ny / threads), threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            default:
                printf("ERROR: Order %d not supported", order);
                return 0;
        }

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilGlobal");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size (blockDim.y * numYPerStep) * blockDim.x. Each thread
 * should calculate at most numYPerStep updates. It should still only use
 * global memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param nx Number of grid points in the x dimension to which the full
 *           stencil can be applied (ie the number of points that are at least
 *           order/2 grid points away from the boundary).
 * @param ny Number of grid points in the y dimension to which th full
 *           stencil can be applied.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int order, int numYPerStep>
__global__
void gpuStencilBlock(float* next, const float* __restrict__ curr, int gx, int nx, int ny,
                    float xcfl, float ycfl) {
    unsigned int x = threadIdx.x + blockDim.x * blockIdx.x + order/2;
    unsigned int y = (threadIdx.y + blockDim.y * blockIdx.y) * numYPerStep + order/2;
    unsigned int gridpoint = x + y * gx;
    unsigned int i = 0;
    while (i < numYPerStep && y + i < ny + order/2 && x < nx + order/2){
        next[gridpoint + i * gx] = Stencil<order>(curr + gridpoint + i * gx, gx, xcfl, ycfl);
        i++;
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilBlock kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
double gpuComputationBlock(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int gx = curr_grid.gx();
    int gy = curr_grid.gy();
    int order = params.order();
    int nx = gx - order;
    int ny = gy - order;
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    dim3 threads(64, 8);
    dim3 blocks(ceil(nx / 64.f), ceil(ny / 64.f));

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch (order){
            case 2:
                gpuStencilBlock<2, 8><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            case 4:
                gpuStencilBlock<4, 8><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            case 8:
                gpuStencilBlock<8, 8><<<blocks, threads>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, nx, ny, xcfl, ycfl);
                break;
            default:
                printf("ERROR: Order %d not supported", order);
                return 0;
        }
        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilBlock");
    return stop_timer(&timer);
}


/**
 * Kernel to propagate finite difference grid from the current
 * time point to the next.
 *
 * This kernel should be optimized to compute finite difference updates
 * in blocks of size side * side using shared memory.
 *
 * @param next[out] Next grid state.
 * @param curr Current grid state.
 * @param gx Number of grid points in the x dimension.
 * @param gy Number of grid points in the y dimension.
 * @param xcfl Courant number for x dimension.
 * @param ycfl Courant number for y dimension.
 */
template<int side, int order>
__global__
void gpuStencilShared(float* next, const float* __restrict__ curr, int gx, int gy,
               float xcfl, float ycfl) {
    // TODO
    extern __shared__ float s[];

    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x + order/2;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y + order/2;
    unsigned int gridpoint = x + y * gx;

    unsigned int sx = threadIdx.x + order/2;
    unsigned int sy = threadIdx.y + order/2;
    unsigned int sgridpoint = sx + sy * (blockDim.x + order);

    if (x < gx && y < gy){
        s[sgridpoint] = curr[gridpoint];
    }
    if (threadIdx.x < order/2){
        s[sgridpoint - order/2] = curr[gridpoint - order/2];
    }
    if (threadIdx.x >= blockDim.x - order/2){
        s[sgridpoint + order/2] = curr[gridpoint + order/2];
    }
    if (threadIdx.y < order/2){
        s[sgridpoint - order/2 * (blockDim.x + order)] = curr[gridpoint - order/2 * gx];
    }
    if (threadIdx.y >= blockDim.y - order/2){
        s[sgridpoint + order/2 * (blockDim.x + order)] = curr[gridpoint + order/2 * gx];
    }
    __syncthreads();

    if (x < gx - order/2 && y < gy - order/2){
        next[gridpoint] = Stencil<order>(s + sgridpoint, blockDim.x + order, xcfl, ycfl);
    }
}

/**
 * Propagates the finite difference 2D heat diffusion solver
 * using the gpuStencilShared kernel.
 *
 * Use this function to do necessary setup and propagate params.iters()
 * number of times.
 *
 * @param curr_grid The current state of the grid.
 * @param params Parameters for the finite difference computation.
 * @returns Time required for computation.
 */
template<int order>
double gpuComputationShared(Grid& curr_grid, const simParams& params) {

    boundary_conditions BC(params);

    Grid next_grid(curr_grid);

    // TODO: Declare variables/Compute parameters.
    int gx = curr_grid.gx();
    int gy = curr_grid.gy();
    float xcfl = params.xcfl();
    float ycfl = params.ycfl();

    dim3 threads(32, 32);
    dim3 blocks(ceil(gx / 32.f), ceil(gy / 32.f));

    event_pair timer;
    start_timer(&timer);

    for(int i = 0; i < params.iters(); ++i) {
        // update the values on the boundary only
        BC.updateBC(next_grid.dGrid_, curr_grid.dGrid_);

        // TODO: Apply stencil.
        switch (order){
            case 2:
                gpuStencilShared<32, order><<<blocks, threads, (32 + order) * (32 + order) * sizeof(float)>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
                break;
            case 4:
                gpuStencilShared<32, order><<<blocks, threads, (32 + order) * (32 + order) * sizeof(float)>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
                break;
            case 8:
                gpuStencilShared<32, order><<<blocks, threads, (32 + order) * (32 + order) * sizeof(float)>>>(
                next_grid.dGrid_, curr_grid.dGrid_, gx, gy, xcfl, ycfl);
                break;
            default:
                printf("ERROR: Order %d not supported", order);
                return 0;
        }

        Grid::swap(curr_grid, next_grid);
    }

    check_launch("gpuStencilShared");
    return stop_timer(&timer);
}
