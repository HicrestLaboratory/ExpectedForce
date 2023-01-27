//#include "stdafx.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm> // max_element, std::min
#include <utility> // pair
#include <math.h> // ceil
#include <sys/time.h>
// for debugging only
#define OBSERVED_NODE -1
#define DEBUG 0

#ifndef TIMING_H
#define TIMING_H
#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec))
#endif


typedef std::vector<size_t> size_t_vector;
typedef std::vector<int> int_vector;
typedef std::vector<std::pair<int, int>> pair_vector;
typedef std::vector<std::vector<size_t>> interval_tree;
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
__global__ void GeneratePairs(const size_t* __restrict__ indexes, const int* __restrict__ neighbors, const int* __restrict__ vertex_start, const int* __restrict__ pairs, const int* __restrict__ vertex_length, int* cluster_sizes, int* cluster_starts, int vertex_count, size_t maxNum, int vertexOffset, int neighborOffset) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

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
    unsigned int index = 0;
    unsigned int array_start = 0;
    unsigned int array_size = vertex_count - 1;
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
    unsigned int pair_combination = i - indexes[index];

    // get neighbor indexes
    unsigned int edge_one = pairs[2 * pair_combination];
    unsigned int edge_two = pairs[2 * pair_combination + 1];
    // if (i == 5) {
    //     printf("Neighbors: %d and %d\n", edge_one, edge_two);
    // }
    // get neighbors
    unsigned int neighbor_one = neighbors[vertex_start[index] - neighborOffset + edge_one];
    unsigned int neighbor_two = neighbors[vertex_start[index] - neighborOffset + edge_two];

    unsigned int cluster_index = 3 * i;
    unsigned int cluster_size = vertex_length[neighbor_one] + vertex_length[neighbor_two] + vertex_length[vertexOffset + index] - 4;
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
__global__ void CountClusterExpectedForce(const int* __restrict__ cluster_size, const int* __restrict__ cluster_start, const size_t* __restrict__ total_vertex_size, float* output, size_t maxNum) {

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
void read_graph(int_vector &vertex_start, int_vector &vertex_length, int_vector &neighbors, set_vector &neighbor_sets, std::string infilename, char delimiter = ' ', int ignore_weights = 0) 
{
	vertex_start.clear(); 
    vertex_length.clear();
    neighbors.clear();

	std::ifstream infile;
	infile.open(infilename);
	std::string temp;
	
	int last_node = -1;
	int node_count = 0;
    int vertex_count = 0;
    
    // start of vertex 0
    vertex_start.push_back(0);
    std::set<int> neighbor_set;

	while (getline(infile, temp, delimiter)) {
        int node = std::stoi(temp);
        node_count++;

		if (node != last_node) {
            if (vertex_count != 0) {
                vertex_length.push_back(vertex_count);
                vertex_start.push_back(node_count-1);
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
            neighbors.push_back(neighbor); 
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
void split_into_generating_chunks(int_vector &generating_chunks, int_vector &vertex_length, int_vector &pair_count, int_vector &chunk_size, int_vector &cluster_start, interval_tree &intervals, int blocks, int threads) {
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
 * Entry function, parameters: 
 * - filename: name fo the file whence the graph shall be read
 * - blocks - nubmer of blocks to be used
 * - threads - number of threads per block to be used, max. 1024 (GPU limitation)
 * - streams - 1..n
 */
int main(int argc, char* argv[]) { //takes a filename (es: fb_full) as input; print its ExF in result.txt 

	//cout << "This program determines the Expected Force of every node for each graph.\n Stores the results in 'FILENAME_results.txt'" << endl;
	
    if(argc < 5) {
        std::cout << "Insufficient number of arguments: " << argc << std::endl;
        exit(3);
    }

    std::string filename = argv[1];
    int blocks = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int streamCount = atoi(argv[4]);
    float ttime = 0.0;
    TIMER_DEF;
    int_vector vertex_start, vertex_length, neighbors;
    set_vector neighbor_sets;

    //int ignore_weights = std::stoi(argv[2]);

    std::cout << "Evaluating file " << filename << std::endl;
    
    int64_t duration;
    int repetitions = 1;

    //reads graph
    read_graph(vertex_start, vertex_length, neighbors, neighbor_sets, filename, ' ', 1); //converts graph to a v-graph-like structure
    std::cout << "File Read" << std::endl;
    for (int rep = 0; rep < repetitions; rep++) {
        TIMER_START;
        interval_tree intervals;
        int_vector pairs, pair_count, generating_chunks, chunk_size, path_vertex_one, cluster_start;

        auto start = std::chrono::high_resolution_clock::now();

        int highest_degree = *std::max_element(vertex_length.begin(), vertex_length.end());

        generate_pairs(pairs, pair_count, highest_degree);
        //std::cout << "Pair generated" << std::endl;
        split_into_generating_chunks(generating_chunks, vertex_length, pair_count, chunk_size, cluster_start, intervals, blocks, threads);

        int biggest_chunk = graph_summary.biggest_chunk;
        int most_neighbors = graph_summary.longest_neighbor_seq;

        //std::cout << "Allocating first pointers" << std::endl;

        cudaStream_t streams[streamCount];
        std::vector<size_t*> index_pointers;
        std::vector<int*> vertex_start_pointers;
        std::vector<int*> neighbor_pointers;
        std::vector<int*> cluster_size_pointers;
        std::vector<int*> host_cluster_size_pointers;
        std::vector<int*> cluster_start_pointers;
        std::vector<int*> host_cluster_start_pointers;
        for (int i = 0; i < streamCount; i++) {
#ifdef ASYNC
	    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
#else
	    cudaStreamCreate(&streams[i]);
#endif
            check_error();

            size_t* index_ptr;
            cudaMallocHost((void**)&index_ptr, sizeof(size_t) * biggest_chunk);
            index_pointers.push_back(index_ptr);
            
            int* vertex_start_ptr;
            cudaMallocHost((void**)&vertex_start_ptr, sizeof(int) * biggest_chunk);
            vertex_start_pointers.push_back(vertex_start_ptr);

            int* neighbor_ptr;
            cudaMallocHost((void**)&neighbor_ptr, sizeof(int) * most_neighbors);
            neighbor_pointers.push_back(neighbor_ptr);

            int* cluster_size_ptr;
            cudaMallocHost((void**)&cluster_size_ptr, sizeof(int) * graph_summary.cluster_count);
            cluster_size_pointers.push_back(cluster_size_ptr);

            int* host_cluster_size_ptr = (int*) malloc(sizeof(int) * graph_summary.cluster_count);
            host_cluster_size_pointers.push_back(host_cluster_size_ptr);

            int* cluster_start_ptr;
            cudaMallocHost((void**)&cluster_start_ptr, sizeof(int) * graph_summary.cluster_count * 3);
            cluster_start_pointers.push_back(cluster_start_ptr);

            int* host_cluster_start_ptr = (int*) malloc(sizeof(int) * graph_summary.cluster_count * 3);
            host_cluster_start_pointers.push_back(host_cluster_start_ptr);
        }
        //check_error();
        // std::cout << "Allocating common pointers" << std::endl;


        int* pairs_ptr;
        cudaMallocHost((void**)&pairs_ptr, sizeof(int) * pairs.size());
        cudaMemcpy(pairs_ptr, pairs.data(), sizeof(int) * pairs.size(), cudaMemcpyHostToDevice);
        check_error();

        int* length_ptr;
        cudaMallocHost((void**)&length_ptr, sizeof(int) * vertex_length.size());
        cudaMemcpy(length_ptr, vertex_length.data(), sizeof(int) * vertex_length.size(), cudaMemcpyHostToDevice);
        check_error();

        std::vector<int> cluster_sizes;
        std::vector<int> start_vertex;
        std::vector<size_t> total_cluster_size(vertex_length.size(), 0);

        //std::cout << "Allocated" << std::endl;

        for (int index = 0; index < generating_chunks.size(); index += 2 * streamCount) {
            int streamsUsed = std::min((int) (generating_chunks.size() / 2) - index/2, (int) streamCount);
            for (int i = 0; i < streamsUsed; i++) {
                int chunk_index = index + 2 * i;
                int chunk_start = generating_chunks[chunk_index];
                int chunk_end = generating_chunks[chunk_index + 1];
                size_t number_of_clusters = cluster_start[chunk_end] + pair_count[vertex_length[chunk_end]] - cluster_start[chunk_start];
                int neighbor_size = vertex_start[chunk_end] + vertex_length[chunk_end] - vertex_start[chunk_start];
                // std::cout << "Copying index pointers: " << sizeof(int) * intervals[index/2 + i].size() << std::endl;
                cudaMemcpyAsync(index_pointers[i], intervals[index/2 + i].data(), sizeof(size_t) * intervals[index/2 + i].size(), cudaMemcpyHostToDevice, streams[i]);
                check_error();

                // std::cout << "Copying vertex start pointers" << std::endl;
                cudaMemcpyAsync(vertex_start_pointers[i], vertex_start.data() + chunk_start, sizeof(int) * (chunk_end - chunk_start + 1), cudaMemcpyHostToDevice, streams[i]);
                check_error();

                // std::cout << "Copying neighbors" << std::endl;
                cudaMemcpyAsync(neighbor_pointers[i], neighbors.data() + vertex_start[chunk_start], sizeof(int) * neighbor_size, cudaMemcpyHostToDevice, streams[i]);
                check_error();
                
                //std::cout << "Blocks " << blocks << ", threads " << threads << ", i " << i << ", vertex count " << chunk_end - chunk_start + 1 << ", cluster count " << number_of_clusters << ", vertex offset " << chunk_start << ", neighbor offset " << vertex_start[chunk_start] << std::endl;
		//int pt = std::floor(std::log2(chunk_end - chunk_start + 1));
		//threads = std::pow(2,pt-1);
                GeneratePairs<<<blocks, threads, 0, streams[i]>>>(index_pointers[i], neighbor_pointers[i], vertex_start_pointers[i], pairs_ptr, length_ptr, cluster_size_pointers[i], cluster_start_pointers[i], chunk_end - chunk_start + 1, number_of_clusters, chunk_start, vertex_start[chunk_start]);
                //GeneratePairs<<<blocks, threads, 0, streams[i]>>>(index_pointers[i], neighbor_pointers[i], vertex_start_pointers[i], pairs.data(), vertex_length.data(), cluster_size_pointers[i], cluster_start_pointers[i], chunk_end - chunk_start + 1, number_of_clusters, chunk_start, vertex_start[chunk_start]);
                //cudaDeviceSynchronize();
                //check_error();
                //cudaStreamSynchronize(streams[i]);
                cudaMemcpyAsync(host_cluster_size_pointers[i], cluster_size_pointers[i], sizeof(int) * number_of_clusters, cudaMemcpyDeviceToHost, streams[i]);
                check_error();

                cudaMemcpyAsync(host_cluster_start_pointers[i], cluster_start_pointers[i], sizeof(int) * number_of_clusters * 3, cudaMemcpyDeviceToHost, streams[i]);
                check_error();
            }
            cudaDeviceSynchronize();

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

                    cluster_sizes.push_back(cluster_size);
                    cluster_sizes.push_back(cluster_size);
                    cluster_sizes.push_back(cluster_size);
                    cluster_sizes.push_back(cluster_size);

                    // push twice - two combinations of S->A S->B and S->B S->A (two clusters but with same edges) 
                    start_vertex.push_back(source_vertex);
                    start_vertex.push_back(source_vertex);
                    start_vertex.push_back(neighbor_one);
                    start_vertex.push_back(neighbor_two);
                    total_cluster_size[source_vertex] += cluster_size;
                    total_cluster_size[source_vertex] += cluster_size;
                    total_cluster_size[neighbor_one] += cluster_size;
                    total_cluster_size[neighbor_two] += cluster_size;
                }
            }
        }

        for (int i = 0; i < streamCount; i++) {
            cudaFreeHost(index_pointers[i]);
            cudaFreeHost(vertex_start_pointers[i]);
            cudaFreeHost(neighbor_pointers[i]);
            cudaFreeHost(cluster_size_pointers[i]);
            cudaFreeHost(cluster_start_pointers[i]);
            free(host_cluster_size_pointers[i]);
            free(host_cluster_start_pointers[i]);
        }     
        cudaFreeHost(length_ptr);
        cudaFreeHost(pairs_ptr);

        // std::cout << "First pointers freed" << std::endl;

        std::vector<int*> input_sizes;
        std::vector<int*> input_vertices;
        std::vector<float*> outputs;
        int array_size = blocks * threads;
        for (int i = 0; i < streamCount; i++) {
            int *size_ptr, *vertex_ptr; 
            float *devOut;
            cudaMallocHost((void**)&size_ptr, (size_t) sizeof(int) * array_size);
            cudaMallocHost((void**)&vertex_ptr, (size_t) sizeof(int) * array_size);
            cudaMallocHost((void**)&devOut, (size_t) sizeof(float) * array_size);
            input_sizes.push_back(size_ptr);
            input_vertices.push_back(vertex_ptr);
            outputs.push_back(devOut);
        }

        size_t *total_size_ptr;
        cudaMallocHost((void**)&total_size_ptr, sizeof(size_t) * total_cluster_size.size());

        cudaMemcpy(total_size_ptr, total_cluster_size.data(), sizeof(size_t) * total_cluster_size.size(), cudaMemcpyHostToDevice);

        // std::ceil is stupid
        int ceil_share = (int) ((graph_summary.cluster_count * 4 / array_size) + ((graph_summary.cluster_count * 4 % array_size) != 0));
        int chunks = std::max((int) 1, ceil_share);

        std::vector<float> normalized_sizes(cluster_sizes.size(), 0);
        // std::cout << "Normalized sizes size: " << normalized_sizes.size() << std::endl;
        for (int index = 0; index < chunks; index += streamCount) {
            int streamsUsed = std::min((int) streamCount, (int) chunks - index);
            for (int i = 0; i < streamsUsed; i++) {
                // std::cout << "Round: " << index + i << " out of " << chunks << " CLuster count: " << graph_summary.cluster_count << ", array size: " << array_size << std::endl;
                size_t element_count = std::min((size_t) array_size, 4 * graph_summary.cluster_count - ((size_t) (index + i)) * array_size);
                // std::cout << "Elements counted" << std::endl;
                size_t offset = ((size_t) (index + i)) * array_size;
                // std::cout << "Offset " << offset << std::endl;
                size_t copied_bytes = sizeof(int) * element_count;
                // std::cout << "Copied bytes " << copied_bytes << std::endl;
                cudaMemcpyAsync(input_sizes[i], cluster_sizes.data() + offset, copied_bytes, cudaMemcpyHostToDevice, streams[i]);
                // std::cout << "Input sizes copied" << std::endl;
                cudaMemcpyAsync(input_vertices[i], start_vertex.data() + offset, copied_bytes, cudaMemcpyHostToDevice, streams[i]);
                // std::cout << "Element count " << element_count << ", vertex size " << vertex_length.size() << ", array size " << array_size * i << std::endl;
                CountClusterExpectedForce<<<blocks, threads, 0, streams[i]>>>(input_sizes[i], input_vertices[i], total_size_ptr, outputs[i], element_count);
                // std::cout << "Elements copied" << std::endl;
                size_t copied_bytes_back = sizeof(float) * element_count;
                // std::cout << "Copied bytes back: " << copied_bytes_back << std::endl;
	//	cudaStreamSynchronize(streams[i]);
                cudaMemcpyAsync(normalized_sizes.data() + offset, outputs[i], copied_bytes_back, cudaMemcpyDeviceToHost, streams[i]);
            }

            cudaDeviceSynchronize();
            check_error();
        }
        TIMER_STOP;
	ttime +=TIMER_ELAPSED;
        for (int i = 0; i < streamCount; i++) {
            cudaFreeHost(input_sizes[i]);
            check_error();

            cudaFreeHost(input_vertices[i]);
            check_error();

            cudaFreeHost(outputs[i]);
            check_error();

            cudaStreamDestroy(streams[i]);
            check_error();
        }    
        cudaFreeHost(total_size_ptr);

        // std::cout << "Memory freed" << std::endl;

        std::vector<float> results(vertex_length.size(), 0);

        for (size_t i = 0; i < normalized_sizes.size(); i++) {
            results[start_vertex[i]] += normalized_sizes[i];
        }

         std::cout << "ExF computed" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        duration += time;
	
        if (rep == 0) {
      //      for (size_t i = 0; i < results.size(); i++) {
                std::cout << 10 << "  " << results[10] << std::endl;
      //      }
        } 
        std::cout << time << std::endl;
    }

    // blocks,threads,streamCount,avg duration in microseconds
    //std::cout << blocks << "," << threads << "," << streamCount << "," << duration/repetitions << std::endl;
    printf("GPU processing in ms: time = %f\n", ttime/1000.0);
	return 0;
}
