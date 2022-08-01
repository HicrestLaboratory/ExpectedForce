//#include "stdafx.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <cmath> // check for nanwhen printing results
#include <atomic> // std::atomic for correct duration summing in streams
#include <algorithm> // max_element, std::min
#include <utility> // pair
#include <math.h> // ceil

// for debugging only
#define OBSERVED_NODE -1
#define DEBUG 0

typedef std::vector<size_t> size_t_vector;
typedef std::vector<int> int_vector;
typedef std::vector<std::pair<int, int>> pair_vector;
typedef std::vector<std::vector<size_t>> interval_tree_struct;
typedef std::vector<std::set<int>> set_vector;

// common graph data
struct graph_data_t {
    int biggest_chunk;
    int longest_neighbor_seq;
    size_t cluster_count;
} graph_summary;

/**
 * Kernel for generating all paths of length 2 in the graph and calculating the cluster sizes.
 */
__global__ void GeneratePairs(size_t* indexes, int* neighbors, int* vertex_start, int* pairs, int* vertex_length, int* cluster_sizes, int* cluster_starts, int vertex_count, size_t maxNum, int vertexOffset, int neighborOffset) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // if (i == 5 && vertexOffset == 99) {
    //     for (int j = 0; j< vertex_count; j++) {
    //         printf("%d: %llu\n", j, indexes[j]);
    //     }
    //     printf("Maxnum: %llu\n", maxNum);
    // }


    if (i >= maxNum) {
        return;
    }

    // if (i == 7346592 && vertexOffset == 99 ) {
    //     printf("Here I am \n");
    // }

    // search in an 'interval tree' to map onto correct vertex's combinations
    int index = 0;
    int array_start = 0;
    int array_size = vertex_count - 1;
    while (array_start <= array_size) {
        index = (array_start + array_size) / 2;
        if (indexes[index] == i || (indexes[index] < i && indexes[index + 1] > i) || (indexes[index] < i && index == vertex_count - 1)) {
            for (int j = index; j < vertex_count - 2; j++){
                if (indexes[j] == indexes[j + 1]) {
                    index = j + 1;
                } else {
                    break;
                }
            }
            break;
        }
        if (indexes[index] < i) {
            array_start = index + 1;
        } else {
            array_size = index - 1;
        }
    }
    // if (i == 7346592 && vertexOffset == 99) {
    //     printf("Index: %d\n", index);
    // }
    int pair_combination = i - indexes[index];

    // get neighbor indexes
    int edge_one = pairs[2 * pair_combination];
    int edge_two = pairs[2 * pair_combination + 1];
    // if (i == 5) {
    //     printf("Neighbors: %d and %d\n", edge_one, edge_two);
    // }
    // get neighbors
    int neighbor_one = neighbors[vertex_start[index] - neighborOffset + edge_one];
    int neighbor_two = neighbors[vertex_start[index] - neighborOffset + edge_two];

    int cluster_index = 3 * i;
    int cluster_size = vertex_length[neighbor_one] + vertex_length[neighbor_two] + vertex_length[vertexOffset + index] - 4;
    cluster_sizes[i] = cluster_size;
    // if (i == 5) {
    //     printf("Cluster size: %d\n", cluster_size);
    // }
    // middle vertex and two neighbors
    cluster_starts[cluster_index] = vertexOffset + index;
    cluster_starts[cluster_index + 1] = neighbor_one;
    cluster_starts[cluster_index + 2] = neighbor_two;
}

/**
 * Calculates the expected force for the cluster.
 */
__global__ void CountClusterExpectedForce(int* cluster_size, int* cluster_start, size_t* total_vertex_size, float* output, size_t maxNum) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= maxNum) {
        return;
    }

    int vertex = cluster_start[i];
    
    int size = cluster_size[i];
    int vertex_total = total_vertex_size[vertex];
    float normalized = (float) size/vertex_total;
    output[i] = -(__logf(normalized) * (normalized));
}

/**
 * Reads the graph.
 */
void read_graph(int_vector &v_start, int_vector &vertex_length, int_vector &neighbors_vector, set_vector &neighbor_sets, std::string infilename, char delimiter = ' ', int ignore_weights = 0) 
{
	v_start.clear(); 
    vertex_length.clear();
    neighbors_vector.clear();

	std::ifstream infile;
	infile.open(infilename);
	std::string temp;
	
	int last_node = -1;
	int node_count = 0;
    int vertex_count = 0;
    
    // start of vertex 0
    v_start.push_back(0);
    std::set<int> neighbor_set;

	while (getline(infile, temp, delimiter)) {
        int node = std::stoi(temp);
        node_count++;

		if (node != last_node) {
            if (vertex_count != 0) {
                vertex_length.push_back(vertex_count);
                v_start.push_back(node_count-1);
                neighbor_sets.push_back(neighbor_set);
            }
        	last_node = node;
            vertex_count = 0;
            neighbor_set = std::set<int>();
		}
		
        vertex_count++;

		getline(infile, temp);
        int neighbor = std::stoi(temp);
        if (neighbor == last_node) {
            // edge to itself, e.g. 1->1
            vertex_count--;
            node_count--;
        } else {
            // normal case
            neighbors_vector.push_back(neighbor); 
            neighbor_set.insert(neighbor);
        }
	}

    // length of the last vertex negihbors 
    vertex_length.push_back(vertex_count);
    neighbor_sets.push_back(neighbor_set);
}

/**
 * Generates pair combinations in 2 values i-j, where i is i-th neighbor of source vertex and j is the source vertex's j-th neighbor.
 */
void generate_pairs(int_vector &pairs, int_vector &pair_count, int highest_degree) {
    pairs.clear();
    pair_count.clear();

    // vertex degree 0
    pair_count.push_back(0);
    // vertex degree 1
    pair_count.push_back(0);

    int count = 0; 

    for (int i = 1; i < highest_degree; i++) {
        for (int j = 0; j < i; j++) {
            pairs.push_back(j);
            pairs.push_back(i);
            count++;
        }
        pair_count.push_back(count);
    }
}

/**
 * Splits all vertices in graph into chunks that can be computed at once. Also calculates several other properties of the graph.
 */
void split_into_generating_chunks(int_vector &generating_chunks, int_vector &vertex_length, int_vector &pair_count, int_vector &chunk_size, int_vector &cluster_start, interval_tree_struct &intervals, int blocks, int threads) {
    size_t cluster_count = 0;
    int size = 0;
    int chunk_start = 0;
    int current_vertices = 0;
    int limit = blocks * threads;
    int neighbors_in_chunk = 0;
    int biggest_chunk = 0;
    int longest_neighbor_seq = 0;
    size_t_vector indexes;

    for (int i = 0; i < vertex_length.size(); i++) {
        cluster_start.push_back(cluster_count);

        int vertex_combinations = pair_count[vertex_length[i]];
        cluster_count += vertex_combinations;
        if (size + vertex_combinations > limit || vertex_combinations > limit) {
            if (current_vertices == 0) {
                std::invalid_argument("Insufficient size of blocks and threads for generating chunks. Required combined size at least " + std::to_string(vertex_combinations) + ".");
            }
            intervals.push_back(indexes);
            chunk_size.push_back(size);
            biggest_chunk = std::max(biggest_chunk, (int) indexes.size());
            longest_neighbor_seq = std::max(longest_neighbor_seq, neighbors_in_chunk);
            generating_chunks.push_back(chunk_start);
            chunk_start += current_vertices;
            generating_chunks.push_back(chunk_start - 1);
            current_vertices = 1;
            size = vertex_combinations;
            indexes = std::vector<size_t>(1, 0);
            neighbors_in_chunk = vertex_length[i];
        } else {
            indexes.push_back((size_t) size);
            size += vertex_combinations;
            current_vertices++;
            neighbors_in_chunk += vertex_length[i];
        }
    }

    if (current_vertices != 0) {
        generating_chunks.push_back(chunk_start);
        generating_chunks.push_back(chunk_start + current_vertices - 1);
        intervals.push_back(indexes);
        chunk_size.push_back(size);
        biggest_chunk = std::max(biggest_chunk, (int) indexes.size());
        longest_neighbor_seq = std::max(longest_neighbor_seq, neighbors_in_chunk);
    }

    graph_summary.biggest_chunk = biggest_chunk;
    graph_summary.longest_neighbor_seq = longest_neighbor_seq;
    graph_summary.cluster_count = cluster_count;
}

/**
 * Checks for cuda errors, logs if there is any error and finishes execution.
 */
void check_error() {
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
        printf("Error!\nLast Error: %s\nDetails: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
        exit(error);
    }
}

/**
 * Computes the time interval between two cuda events in microseconds.
 */
int microseconds(cudaEvent_t* start, cudaEvent_t* stop) {
    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, *start, *stop);
    return static_cast<int>(miliseconds * 1000);
}

/**
 * Entry function, parameters: 
 * - filename: name fo the file whence the graph shall be read
 * - blocks - nubmer of blocks to be used
 * - threads - number of threads per block to be used, max. 1024 (GPU limitation)
 * - streams - 1..n
 */
int main(int argc, char* argv[]) { //takes a filename (es: fb_full) as input; print its ExF in result.txt 

	// cout << "This program determines the Expected Force of every node for each graph.\n Stores the results in 'FILENAME_results.txt'" << endl;
	
    if(argc < 5) {
        std::cout << "Insufficient number of arguments: " << argc << std::endl;
        exit(3);
    }

    std::string filename = argv[1];
    int blocks = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int streamCount = atoi(argv[4]);
    bool print_results = false;
    if (argc > 6) {
        print_results = true;
    }

    int_vector v_start, vertex_length, neighbors_vector;
    set_vector neighbor_sets;

    interval_tree_struct intervals;
    int_vector pairs, pair_count, generating_chunks, chunk_size, path_vertex_one, cluster_start;

    // set up time-measuring objects
    cudaEvent_t repetition_start, repetition_stop, graph_read_start, graph_read_stop;
    cudaEvent_t copy1_start[streamCount], copy1_stop[streamCount], 
                kernel1_start[streamCount], kernel1_stop[streamCount], 
                copy2_start[streamCount], copy2_stop[streamCount],
                copy3_start[streamCount], copy3_stop[streamCount],
                kernel2_start[streamCount], kernel2_stop[streamCount],
                copy4_start[streamCount], copy4_stop[streamCount];
    std::int64_t repetition_duration = 0, graph_read_duration = 0, copy1_duration = 0, kernel1_duration = 0, copy2_duration = 0, mid_process_duration = 0, copy3_duration = 0, kernel2_duration = 0, copy4_duration = 0, end_process_duration = 0;
    size_t total_gpu_memory, free_beginning, free_stage_1, free_stage_2;

    cudaMemGetInfo(&free_beginning, &total_gpu_memory);

    // initialize events
    cudaEventCreate(&repetition_start);
    cudaEventCreate(&repetition_stop);
    cudaEventCreate(&graph_read_start);
    cudaEventCreate(&graph_read_stop);
    for (size_t i = 0; i < streamCount; i++) {
        cudaEventCreate(&copy1_start[i]);
        cudaEventCreate(&copy1_stop[i]);
        cudaEventCreate(&kernel1_start[i]);
        cudaEventCreate(&kernel1_stop[i]);
        cudaEventCreate(&copy2_start[i]);
        cudaEventCreate(&copy2_stop[i]);
        cudaEventCreate(&copy3_start[i]);
        cudaEventCreate(&copy3_stop[i]);
        cudaEventCreate(&kernel2_start[i]);
        cudaEventCreate(&kernel2_stop[i]);
        cudaEventCreate(&copy4_start[i]);
        cudaEventCreate(&copy4_stop[i]);
    }

    // used for computation of average cuda times in second parallel block
    int parallel_block2_repeats = 0;


    std::cout << "Evaluating file " << filename << std::endl;

    int repetitions = 1;

    cudaEventRecord(graph_read_start);

    //reads graph
    read_graph(v_start, vertex_length, neighbors_vector, neighbor_sets, filename, ' ', 1); //converts graph to a v-graph-like structure

    int highest_degree = *std::max_element(vertex_length.begin(), vertex_length.end());

    generate_pairs(pairs, pair_count, highest_degree);

    split_into_generating_chunks(generating_chunks, vertex_length, pair_count, chunk_size, cluster_start, intervals, blocks, threads);

    int biggest_chunk = graph_summary.biggest_chunk;
    int most_neighbors = graph_summary.longest_neighbor_seq;

    cudaEventRecord(graph_read_stop);

    // transform from C++ vector to arrays that can be copied in async by CUDA
    int* vertex_start;
    cudaMallocHost((void**) &vertex_start, sizeof(int) * v_start.size());
    std::copy(v_start.begin(), v_start.end(), vertex_start);
    v_start = std::vector<int>(); // empties the v_start, move constructor works like swap()

    int* neighbors;
    cudaMallocHost((void**) &neighbors, sizeof(int) * neighbors_vector.size());
    std::copy(neighbors_vector.begin(), neighbors_vector.end(), neighbors);
    neighbors_vector = std::vector<int>(); // empties the neighbors_vector, move constructor works like swap()

    // transform interval tree (std::vector<std::vector<int>>) into cuda pinned-memory arrays
    size_t* interval_tree;
    int* interval_tree_node_start;
    int* interval_tree_node_length;

    int interval_tree_elements = 0;
    for (size_t tree_node = 0; tree_node < intervals.size(); tree_node++) {
        interval_tree_elements += intervals[tree_node].size();
    }

    cudaMallocHost((void**) &interval_tree, sizeof(size_t) * interval_tree_elements);
    cudaMallocHost((void**) &interval_tree_node_start, sizeof(int) * intervals.size());
    cudaMallocHost((void**) &interval_tree_node_length, sizeof(int) * intervals.size());

    size_t interval_offset = 0;
    for (size_t tree_node = 0; tree_node < intervals.size(); tree_node++) {
        std::copy(intervals[tree_node].begin(), intervals[tree_node].end(), interval_tree + interval_offset);
        interval_tree_node_start[tree_node] = interval_offset;
        interval_tree_node_length[tree_node] = intervals[tree_node].size(); 
        interval_offset += intervals[tree_node].size();
    }

    //cleanup
    intervals.clear(); // should destroy or 'inner' vectors?

    for (int rep = 0; rep < repetitions; rep++) {

        cudaEventRecord(repetition_start);

        // std::cout << "Allocating first pointers" << std::endl;

        cudaStream_t streams[streamCount];
        std::vector<size_t*> index_pointers;
        std::vector<int*> vertex_start_pointers;
        std::vector<int*> neighbor_pointers;
        std::vector<int*> cluster_size_pointers;
        std::vector<int*> host_cluster_size_pointers;
        std::vector<int*> cluster_start_pointers;
        std::vector<int*> host_cluster_start_pointers;
        for (int i = 0; i < streamCount; i++) {
            cudaStreamCreate(&streams[i]);
            check_error();

            size_t* index_ptr; 
            cudaMalloc((void**)&index_ptr, sizeof(size_t) * biggest_chunk);
            index_pointers.push_back(index_ptr);
            
            int* vertex_start_ptr;
            cudaMalloc((void**)&vertex_start_ptr, sizeof(int) * biggest_chunk);
            vertex_start_pointers.push_back(vertex_start_ptr);

            int* neighbor_ptr;
            cudaMalloc((void**)&neighbor_ptr, sizeof(int) * most_neighbors);
            neighbor_pointers.push_back(neighbor_ptr);

            int* cluster_size_ptr;
            cudaMalloc((void**)&cluster_size_ptr, sizeof(int) * graph_summary.cluster_count);
            cluster_size_pointers.push_back(cluster_size_ptr);

            int* host_cluster_size_ptr;
            cudaMallocHost((void**)&host_cluster_size_ptr, sizeof(int) * graph_summary.cluster_count);
            host_cluster_size_pointers.push_back(host_cluster_size_ptr);

            int* cluster_start_ptr;
            cudaMalloc((void**)&cluster_start_ptr, sizeof(int) * graph_summary.cluster_count * 3);
            cluster_start_pointers.push_back(cluster_start_ptr);

            int* host_cluster_start_ptr;
            cudaMallocHost((void**)&host_cluster_start_ptr, sizeof(int) * graph_summary.cluster_count * 3);
            host_cluster_start_pointers.push_back(host_cluster_start_ptr);
        }
        check_error();

        std::cout << "pairs_ptr size: " << sizeof(int) * pairs.size() << std::endl;
        std::cout << "length_ptr size: " << sizeof(int) * vertex_length.size() << std::endl;
        std::cout << "cluster_sizes size: " << sizeof(int) * 4 * graph_summary.cluster_count << std::endl;
        std::cout << "start_vertex size: " << sizeof(int) * 4 * graph_summary.cluster_count << std::endl;
        std::cout << "index_ptr size xstreams: " << sizeof(size_t) * biggest_chunk << std::endl;
        std::cout << "vertex_start_ptr size xstreams: " << sizeof(int) * biggest_chunk << std::endl;
        std::cout << "neighbor_ptr size xstreams: " << sizeof(int) * graph_summary.cluster_count << std::endl;
        std::cout << "cluster_size_ptr size xstreams: " << sizeof(int) * graph_summary.cluster_count * 3 << std::endl;

        // std::cout << "Allocating common pointers" << std::endl;

        int* pairs_ptr;
        cudaMalloc((void**)&pairs_ptr, sizeof(int) * pairs.size());
        cudaMemcpy(pairs_ptr, pairs.data(), sizeof(int) * pairs.size(), cudaMemcpyHostToDevice);
        check_error();

        int* length_ptr;
        cudaMalloc((void**)&length_ptr, sizeof(int) * vertex_length.size());
        cudaMemcpy(length_ptr, vertex_length.data(), sizeof(int) * vertex_length.size(), cudaMemcpyHostToDevice);
        check_error();

        // measures all generated clusters (one 'computed' cluster can be formed in 4 different configurations -> 4 'actual' clusters)
        size_t formed_clusters_count = 4 * graph_summary.cluster_count;

        int* cluster_sizes;
        cudaMallocHost((void**) &cluster_sizes, sizeof(int) * formed_clusters_count); // cluster sizes * 4 -> each cluster has 4 initiation variants
        int* start_vertex;
        cudaMallocHost((void**) &start_vertex, sizeof(int) * formed_clusters_count); // cluster sizes * 4 -> each cluster has 4 starting points

        // offsets in cluster_sizes and start_vertex to the first unwritten slot
        size_t data_offset = 0;
        
        std::vector<size_t> total_cluster_size(vertex_length.size(), 0);

        // std::cout << "Allocated" << std::endl;
        cudaMemGetInfo(&free_stage_1, &total_gpu_memory);
        for (size_t index = 0; index < generating_chunks.size(); index += 2 * streamCount) {
            int streamsUsed = std::min((int) ((generating_chunks.size() / 2) - index/2), (int) streamCount);
            for (int i = 0; i < streamsUsed; i++) {
                size_t chunk_index = index + 2 * i;
                size_t chunk_start = generating_chunks[chunk_index];
                size_t chunk_end = generating_chunks[chunk_index + 1];
                size_t number_of_clusters = cluster_start[chunk_end] + pair_count[vertex_length[chunk_end]] - cluster_start[chunk_start];
                size_t neighbor_size = vertex_start[chunk_end] + vertex_length[chunk_end] - vertex_start[chunk_start];
                size_t interval_tree_offset = index/2 + i;

                cudaEventRecord(copy1_start[i], streams[i]);
                // std::cout << "Copying index pointers: node " << interval_tree_offset << ", from offset " << (*(interval_tree_node_start + interval_tree_offset)) << ", size " << (*(interval_tree_node_length + interval_tree_offset)) << std::endl;
                cudaMemcpyAsync(index_pointers[i], interval_tree + (*(interval_tree_node_start + interval_tree_offset)), sizeof(size_t) * (*(interval_tree_node_length + interval_tree_offset)), cudaMemcpyHostToDevice);
                check_error();

                // std::cout << "Copying vertex start pointers" << std::endl;
                cudaMemcpy(vertex_start_pointers[i], vertex_start + chunk_start, sizeof(int) * (chunk_end - chunk_start + 1), cudaMemcpyHostToDevice);
                check_error();

                // std::cout << "Copying neighbors" << std::endl;
                cudaMemcpy(neighbor_pointers[i], neighbors + vertex_start[chunk_start], sizeof(int) * neighbor_size, cudaMemcpyHostToDevice);
                check_error();

                cudaEventRecord(copy1_stop[i], streams[i]);

                // std::cout << "Blocks " << blocks << ", threads " << threads << ", i " << i << ", vertex count " << chunk_end - chunk_start + 1 << ", cluster count " << number_of_clusters << ", vertex offset " << chunk_start << ", neighbor offset " << vertex_start[chunk_start] << std::endl;
                cudaEventRecord(kernel1_start[i], streams[i]);
                GeneratePairs<<<blocks, threads, 0, streams[i]>>>(index_pointers[i], neighbor_pointers[i], vertex_start_pointers[i], pairs_ptr, length_ptr, cluster_size_pointers[i], cluster_start_pointers[i], chunk_end - chunk_start + 1, number_of_clusters, chunk_start, vertex_start[chunk_start]);
                cudaEventRecord(kernel1_stop[i], streams[i]);
                // GeneratePairs<<<blocks, threads, 0, streams[i]>>>(index_pointers[i], neighbor_pointers[i], vertex_start_pointers[i], pairs.data(), vertex_length.data(), cluster_size_pointers[i], cluster_start_pointers[i], chunk_end - chunk_start + 1, number_of_clusters, chunk_start, vertex_start[chunk_start]);
                cudaDeviceSynchronize();
                check_error();

                cudaEventRecord(copy2_start[i], streams[i]);

                // std::cout << "Copy back 1" << std::endl;
                cudaMemcpy(host_cluster_size_pointers[i], cluster_size_pointers[i], sizeof(int) * number_of_clusters, cudaMemcpyDeviceToHost);
                check_error();

                // std::cout << "Copy back 2" << std::endl;
                cudaMemcpy(host_cluster_start_pointers[i], cluster_start_pointers[i], sizeof(int) * number_of_clusters * 3, cudaMemcpyDeviceToHost);
                check_error();
                cudaEventRecord(copy2_stop[i], streams[i]);
            }
            check_error();

            cudaDeviceSynchronize();

            for (int i = 0; i < streamsUsed; i++) {
                copy1_duration += microseconds(&copy1_start[i], &copy1_stop[i]);
                kernel1_duration += microseconds(&kernel1_start[i], &kernel1_stop[i]);
                copy2_duration += microseconds(&copy2_start[i], &copy2_stop[i]);
            }

            check_error();

            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < streamsUsed; i++) {
                int* path_ptr = host_cluster_start_pointers[i];
                int chunk_index = index + 2 * i;
                int chunk_start = generating_chunks[chunk_index];
                int chunk_end = generating_chunks[chunk_index + 1];
                size_t number_of_clusters = cluster_start[chunk_end] + pair_count[vertex_length[chunk_end]] - cluster_start[chunk_start];

                for (size_t j = 0; j < number_of_clusters; j++) {
                    int cluster_size = host_cluster_size_pointers[i][j];
                    int source_vertex = path_ptr[3 * j];
                    int neighbor_one = path_ptr[3 * j + 1];
                    int neighbor_two = path_ptr[3 * j + 2];

                    // if two neighbors form a triangle with cluster-size edges
                    // contains() cannot be used -> C++20-only
                    if (neighbor_sets[neighbor_one].find(neighbor_two) != neighbor_sets[neighbor_one].end()) {
                        cluster_size -= 2;
                    }

                    for (size_t k = 0; k < 4; k++) {
                        cluster_sizes[data_offset + k] = cluster_size;
                    }


                    // push twice - two combinations of S->A S->B and S->B S->A (two clusters but with same edges)
                    start_vertex[data_offset] = source_vertex;
                    start_vertex[data_offset + 1] = source_vertex;
                    start_vertex[data_offset + 2] = neighbor_one;
                    start_vertex[data_offset + 3] = neighbor_two;

                    data_offset = data_offset + 4;

                    total_cluster_size[source_vertex] += cluster_size;
                    total_cluster_size[source_vertex] += cluster_size;
                    total_cluster_size[neighbor_one] += cluster_size;
                    total_cluster_size[neighbor_two] += cluster_size;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            mid_process_duration += time;
        }

        for (int i = 0; i < streamCount; i++) {
            cudaFree(index_pointers[i]);
            cudaFree(vertex_start_pointers[i]);
            cudaFree(neighbor_pointers[i]);
            cudaFree(cluster_size_pointers[i]);
            cudaFree(cluster_start_pointers[i]);
            cudaFreeHost(host_cluster_size_pointers[i]);
            cudaFreeHost(host_cluster_start_pointers[i]);
        }     
        cudaFree(length_ptr);
        cudaFree(pairs_ptr);
        check_error();

        // std::cout << "First pointers freed" << std::endl;

        std::vector<int*> input_sizes;
        std::vector<int*> input_vertices;
        std::vector<float*> outputs;
        int array_size = blocks * threads;
        // std::cout << "Allocated vertex_ptr: " << sizeof(int) * array_size << std::endl;
        for (int i = 0; i < streamCount; i++) {
            int *size_ptr, *vertex_ptr; 
            float *devOut;
            cudaMalloc((void**)&size_ptr, (size_t) sizeof(int) * array_size);
            cudaMalloc((void**)&vertex_ptr, (size_t) sizeof(int) * array_size);
            cudaMalloc((void**)&devOut, (size_t) sizeof(float) * array_size);
            // std::cout << "size_ptr: " << size_ptr << std::endl;
            input_sizes.push_back(size_ptr);
            input_vertices.push_back(vertex_ptr);
            outputs.push_back(devOut);
            check_error();
        }
        // std::cout << " allocate total size ptr " << std::endl;
        size_t *total_size_ptr;
        cudaMalloc((void**)&total_size_ptr, sizeof(size_t) * total_cluster_size.size());
        check_error();
        cudaMemcpyAsync(total_size_ptr, total_cluster_size.data(), sizeof(size_t) * total_cluster_size.size(), cudaMemcpyHostToDevice);
        check_error();

        // std::cout << " compute ceil" << std::endl;

        // std::ceil is stupid
        int ceil_share = (int) ((graph_summary.cluster_count * 4 / array_size) + ((graph_summary.cluster_count * 4 % array_size) != 0));
        int chunks = std::max((int) 1, ceil_share);

        // store for outer scope computation of average time
        parallel_block2_repeats = chunks;

        float* normalized_sizes;
        cudaMallocHost((void**) &normalized_sizes, sizeof(float) * formed_clusters_count);
        // std::cout << "Normalized sizes size: " << 4 * graph_summary.cluster_count << " float elements" << std::endl;

        cudaMemGetInfo(&free_stage_2, &total_gpu_memory);
        for (int index = 0; index < chunks; index += streamCount) {
            int streamsUsed = std::min((int) streamCount, (int) chunks - index);
            for (int i = 0; i < streamsUsed; i++) {
                // std::cout << "Round: " << index + i << " out of " << chunks << " CLuster count: " << graph_summary.cluster_count << ", array size: " << array_size << std::endl;
                size_t element_count = std::min((size_t) array_size, 4 * graph_summary.cluster_count - ((size_t) (index + i)) * array_size);
                // std::cout << "Elements counted" << std::endl;
                size_t offset = ((size_t) (index + i)) * array_size;
                // std::cout << "Offset " << offset << std::endl;
                size_t copied_bytes = sizeof(int) * element_count;

                // std::cout << "stream[" << i << "]: " << streams[i] << std::endl;
                // std::cout << "copy3_start[" << i << "]: " << copy3_start[i] << std::endl;
                cudaEventRecord(copy3_start[i], streams[i]);
                check_error();

                // std::cout << "Copied bytes " << copied_bytes << std::endl;
                cudaMemcpyAsync(input_sizes[i], cluster_sizes + offset, copied_bytes, cudaMemcpyHostToDevice);
                check_error();

                // std::cout << "Input sizes copied " << std::endl;
                cudaMemcpyAsync(input_vertices[i], start_vertex + offset, copied_bytes, cudaMemcpyHostToDevice);
                check_error();
                
                cudaEventRecord(copy3_stop[i], streams[i]);

                // std::cout << "Element count " << element_count << ", vertex size " << vertex_length.size() << ", array size " << array_size * i << std::endl;
                cudaEventRecord(kernel2_start[i], streams[i]);
                CountClusterExpectedForce<<<blocks, threads, 0, streams[i]>>>(input_sizes[i], input_vertices[i], total_size_ptr, outputs[i], element_count);
                cudaEventRecord(kernel2_stop[i], streams[i]);
                check_error();
                // std::cout << "Elements copied" << std::endl;
                size_t copied_bytes_back = sizeof(float) * element_count;

                cudaEventRecord(copy4_start[i], streams[i]);
                check_error();

                // std::cout << "Copied bytes back: " << copied_bytes_back << std::endl;
                cudaMemcpyAsync(normalized_sizes + offset, outputs[i], copied_bytes_back, cudaMemcpyDeviceToHost);
                check_error();

                cudaEventRecord(copy4_stop[i], streams[i]);
                check_error();
            }

            cudaDeviceSynchronize();
            check_error();
            
            for(int i = 0; i < streamCount; i++) {
                copy3_duration += microseconds(&copy3_start[i], &copy3_stop[i]);
                kernel2_duration += microseconds(&kernel2_start[i], &kernel2_stop[i]);
                copy4_duration += microseconds(&copy4_start[i], &copy4_stop[i]);
            }
        }

        for (int i = 0; i < streamCount; i++) {
            cudaFree(input_sizes[i]);
            check_error();

            cudaFree(input_vertices[i]);
            check_error();

            cudaFree(outputs[i]);
            check_error();

            cudaStreamDestroy(streams[i]);
            check_error();
        }    
        cudaFree(total_size_ptr);
        check_error();

        cudaFreeHost(cluster_sizes);
        check_error();

        // std::cout << "Memory freed" << std::endl;

        std::vector<float> results(vertex_length.size(), 0);

        auto result_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < formed_clusters_count; i++) {
            results[start_vertex[i]] += normalized_sizes[i];
        }
        auto result_end = std::chrono::high_resolution_clock::now();
        auto result_time = std::chrono::duration_cast<std::chrono::microseconds>(result_end - result_start).count();
        end_process_duration += result_time;

        cudaFreeHost(start_vertex);
        cudaFreeHost(normalized_sizes);

        cudaEventRecord(repetition_stop);
        cudaEventSynchronize(repetition_stop);

        int repetition_microseconds = microseconds(&repetition_start, &repetition_stop);
        repetition_duration += repetition_microseconds;

        std::cout << repetition_microseconds << std::endl;

        int graph_read_microseconds = microseconds(&graph_read_start, &graph_read_stop);
        graph_read_duration += graph_read_microseconds;

        if (rep == 0 && print_results) {
            std::ofstream output_file(argv[6]);
            if (output_file.is_open()) {
                for (size_t i = 0; i < results.size(); i++) {
                    if (isnan(results[i])) {
                        results[i] = 0;
                    }
                    output_file << i << "  " << results[i] << "\n";
                }
            } else {
                std::cout << "Could not open or create file " << argv[5] << ". Results will not be printed." << std::endl;
            }
        } 
    }

    // free after all repetitions - these arrays are pre-computed and common for all runs
    cudaFreeHost(vertex_start);
    cudaFreeHost(neighbors);
    cudaFreeHost(interval_tree);
    cudaFreeHost(interval_tree_node_start);
    cudaFreeHost(interval_tree_node_length);

    cudaEventDestroy(repetition_start);
    cudaEventDestroy(repetition_stop);
    cudaEventDestroy(graph_read_start);
    cudaEventDestroy(graph_read_stop);
    for (size_t i = 0; i < streamCount; i++) {
        cudaEventDestroy(copy1_start[i]);
        cudaEventDestroy(copy1_stop[i]);
        cudaEventDestroy(kernel1_start[i]);
        cudaEventDestroy(kernel1_stop[i]);
        cudaEventDestroy(copy2_start[i]);
        cudaEventDestroy(copy2_stop[i]);
        cudaEventDestroy(copy3_start[i]);
        cudaEventDestroy(copy3_stop[i]);
        cudaEventDestroy(kernel2_start[i]);
        cudaEventDestroy(kernel2_stop[i]);
        cudaEventDestroy(copy4_start[i]);
        cudaEventDestroy(copy4_stop[i]);
    }

    int parallel_block1_repeats = generating_chunks.size() / 2;

    std::cout << "graph_read;" << graph_read_duration << std::endl;
    std::cout << "repetition_time;" << repetition_duration / repetitions << std::endl;
    std::cout << "max_mem_stage_1;" << (free_beginning - free_stage_1) / 1048576 << "MB" << std::endl;
    std::cout << "max_mem_stage_2;" << (free_beginning - free_stage_2) / 1048576 << "MB" << std::endl << std::endl;

    std::cout << "copy1_per_repetition;" << copy1_duration / repetitions << std::endl;
    std::cout << "kernel1_per_repetition;" << kernel1_duration / repetitions << std::endl;
    std::cout << "copy2_per_repetition;" << copy2_duration / repetitions << std::endl;
    std::cout << "mid_process_per_repetition;" << static_cast<int>(mid_process_duration / repetitions) << std::endl;
    std::cout << "copy3_per_repetition;" << copy3_duration /  repetitions << std::endl;
    std::cout << "kernel2_per_repetition;" << kernel2_duration / repetitions << std::endl;
    std::cout << "copy4_per_repetition;" << copy4_duration / repetitions << std::endl;
    std::cout << "result_sum_per_repetition;" << end_process_duration / repetitions << std::endl << std::endl;

    std::cout << "parallel1_chunks;" << parallel_block1_repeats << std::endl;
    std::cout << "parallel1_repeats;" << parallel_block1_repeats / streamCount << std::endl;
    std::cout << "copy1;" << copy1_duration / (parallel_block1_repeats * repetitions / streamCount) << std::endl;
    std::cout << "kernel1;" << kernel1_duration / (parallel_block1_repeats * repetitions / streamCount) << std::endl;
    std::cout << "copy2;" << copy2_duration / (parallel_block1_repeats * repetitions / streamCount) << std::endl;
    std::cout << "mid_process;" << static_cast<int>(mid_process_duration / (repetitions * ceil((double) parallel_block1_repeats / streamCount))) << std::endl;
    std::cout << "parallel2_chunks;" << parallel_block2_repeats << std::endl;
    std::cout << "parallel2_repeats;" << parallel_block2_repeats / streamCount << std::endl;
    std::cout << "copy3;" << copy3_duration / (parallel_block2_repeats * repetitions / streamCount) << std::endl;
    std::cout << "kernel2;" << kernel2_duration / (parallel_block2_repeats * repetitions / streamCount) << std::endl;
    std::cout << "copy4;" << copy4_duration / (parallel_block2_repeats * repetitions / streamCount) << std::endl << std::endl;

    // blocks,threads,streamCount,avg duration in microseconds
    std::cout << blocks << "," << threads << "," << streamCount << std::endl;

	return 0;
}
