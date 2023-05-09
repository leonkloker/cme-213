#ifndef _PAGERANK_CUH
#define _PAGERANK_CUH

#include "util.cuh"

/* 
 * Each kernel handles the update of one pagerank score. In other
 * words, each kernel handles one row of the update:
 *
 *      pi(t+1) = (1/2)* A pi(t) + (1 / (2N))
 *
 */
__global__ void device_graph_propagate(const uint *graph_indices,
    const uint *graph_edges,
    const float *graph_nodes_in,
    float *graph_nodes_out,
    const float *inv_edges_per_node,
    int num_nodes) {

    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    while (id < num_nodes) {
        graph_nodes_out[id] = 1 / (float) (2 * num_nodes);
        for (int i = graph_indices[id]; i < graph_indices[id+1]; i++) {
            graph_nodes_out[id] += 0.5 * graph_nodes_in[graph_edges[i]] * inv_edges_per_node[graph_edges[i]];
        }
        id += blockDim.x * gridDim.x;
    }
}

/* 
 * This function executes a specified number of iterations of the
 * pagerank algorithm. The variables are:
 *
 * h_graph_indices, h_graph_edges:
 *     These arrays describe the indices of the neighbors of node i.
 *     Specifically, node i is adjacent to all nodes in the range
 *     h_graph_edges[h_graph_indices[i] ... h_graph_indices[i+1]].
 *
 * h_node_values_input:
 *     An initial guess of pi(0).
 *
 * h_gpu_node_values_output:
 *     Output array for the pagerank vector.
 *
 * h_inv_edges_per_node:
 *     The i'th element in this array is the reciprocal of the
 *     out degree of the i'th node.
 *
 * nr_iterations:
 *     The number of iterations to run the pagerank algorithm for.
 *
 * num_nodes:
 *     The number of nodes in the whole graph (ie N).
 *
 * avg_edges:
 *     The average number of edges in the graph. You are guaranteed
 *     that the whole graph has num_nodes * avg_edges edges.
 */
double device_graph_iterate(
    const uint *h_graph_indices,
    const uint *h_graph_edges,
    const float *h_node_values_input,
    float *h_gpu_node_values_output,
    const float *h_inv_edges_per_node,
    int nr_iterations,
    int num_nodes,
    int avg_edges
) {
    // TODO: allocate GPU memory
    uint *d_graph_indices = nullptr;
    uint *d_graph_edges = nullptr;
    float *d_node_values_input = nullptr;
    float *d_gpu_node_values_output = nullptr;
    float *d_inv_edges_per_node = nullptr;

    cudaMalloc(&d_graph_indices, (num_nodes+1) * sizeof(uint));
    cudaMalloc(&d_graph_edges, num_nodes * avg_edges * sizeof(uint));
    cudaMalloc(&d_node_values_input, num_nodes * sizeof(float));
    cudaMalloc(&d_gpu_node_values_output, num_nodes * sizeof(float));
    cudaMalloc(&d_inv_edges_per_node, num_nodes * sizeof(float));

    // TODO: check for allocation failure
    if (cudaPeekAtLastError() != 0){
        std::cout << cudaGetLastError() << std::endl << std::flush;
    }

    // TODO: copy data to the GPU
    cudaMemcpy(d_graph_indices, h_graph_indices, (num_nodes+1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph_edges, h_graph_edges, num_nodes * avg_edges * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_node_values_input, h_node_values_input, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gpu_node_values_output, h_gpu_node_values_output, num_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_edges_per_node, h_inv_edges_per_node, num_nodes * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernels
    event_pair timer;
    start_timer(&timer);

    const int block_size = 192;

    // TODO: launch your kernels the appropriate number of iterations
    for (unsigned int i = 0; i < nr_iterations; i++) {
        device_graph_propagate<<<ceil(num_nodes / block_size), block_size>>>(
        d_graph_indices, d_graph_edges, d_node_values_input, 
        d_gpu_node_values_output, d_inv_edges_per_node, num_nodes);
        cudaDeviceSynchronize();
        std::swap(d_node_values_input, d_gpu_node_values_output);
    }
    std::swap(d_node_values_input, d_gpu_node_values_output);

    check_launch("gpu graph propagate");
    double gpu_elapsed_time = stop_timer(&timer);

    // TODO: copy final data back to the host for correctness checking
    cudaMemcpy(h_gpu_node_values_output, d_gpu_node_values_output, 
        num_nodes * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: free the memory you allocated!
    cudaFree(d_graph_indices);
    cudaFree(d_graph_edges);
    cudaFree(d_node_values_input);
    cudaFree(d_gpu_node_values_output);
    cudaFree(d_inv_edges_per_node);

    return gpu_elapsed_time;
}

/**
 * This function computes the number of bytes read from and written to
 * global memory by the pagerank algorithm.
 * 
 * nodes:
 *      The number of nodes in the graph
 *
 * edges: 
 *      The average number of edges in the graph
 *
 * iterations:
 *      The number of iterations the pagerank algorithm was run
 */
uint get_total_bytes(uint nodes, uint edges, uint iterations)
{
    uint bytes = iterations * nodes * 4 * (2 + 6*edges);
    return bytes;
}

#endif
