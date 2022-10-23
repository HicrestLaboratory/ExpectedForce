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
// log start and end time (granularity in seconds only)
#include <ctime>
#include <iostream>

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
__global__ void GeneratePairs(size_t* indexes, int* neighbors, int* vertex_start, int* pairs, 
                              int* vertex_neighbor_count_ptr, int* cluster_sizes, int* cluster_starts, 
                              int vertex_count, size_t maxNum, int vertexOffset, int neighborOffset) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= maxNum) {
        return;
    }

    // search in an 'interval tree' to map onto correct vertex's combinations
    int index = 0;
    int array_start = 0;
    int array_size = vertex_count - 1;
    while (array_start <= array_size) {
        index = (array_start + array_size) / 2;
        if (indexes[index] == i || (indexes[index] < i && indexes[index + 1] > i) 
            || (indexes[index] < i && index == vertex_count - 1)) {
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

    int pair_combination = i - indexes[index];

    // get neighbor indexes - every cluster is formed by 2 edges and 3 vertices
    int edge_one = pairs[2 * pair_combination];
    int edge_two = pairs[2 * pair_combination + 1];

    // get neighbors 
    int neighbor_one = neighbors[vertex_start[index] - neighborOffset + edge_one];
    int neighbor_two = neighbors[vertex_start[index] - neighborOffset + edge_two];

    int cluster_index = 3 * i;
    int cluster_size = vertex_neighbor_count_ptr[neighbor_one] + vertex_neighbor_count_ptr[neighbor_two] 
                        + vertex_neighbor_count_ptr[vertexOffset + index] - 4;
    cluster_sizes[i] = cluster_size;

    // add middle vertex and two neighbors
    cluster_starts[cluster_index] = vertexOffset + index;
    cluster_starts[cluster_index + 1] = neighbor_one;
    cluster_starts[cluster_index + 2] = neighbor_two;
}

/**
 * Calculates the expected force for the cluster.
 */
__global__ void CountClusterExpectedForce(int* cluster_size, int* cluster_start, size_t* total_vertex_size, 
                                          float* output, size_t maxNum) {

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
 * Input limitations:
 * - graph represented as a list of edges: 'source target' separated by a single whitespace (one edge per line, 
 *   delimiter may be adjusted)
 * - edges sorted by source vertex, then by target vertex
 * - graph is undirected, for every edge 'v1 v2' there is also an edge 'v2 v1'
 * - no multiedges are present
 * - no vertices without edges
 * - vertex ID starts from 0 and ends at n-1, no gaps
 *   - n = number of vertices in graph
 * - for normalization, you can use attached script transformer.py
 * - the graph does not support processing of edge weights yet
 * 
 * @param start_of_vertices_ptr a pointer to the vector which stores information about at which neighbors_vector 
 *                              position start neighbors of given vertex
 * @param vertex_neighbor_count_ptr a pointer to the vector which stores information about how many neighbors 
 *                                  a given vertex has
 * @param neighbors_vector_ptr a pointer to the vector where all neighbors are stored
 * @param neighbor_sets_ptr a pointer to the vector where sets of neighbors for every vertex are stored
 *                          (set of neighbors of vertex i is at i-th position in the vector)
 * @param input_file_name name of the file where the input graph is stored
 * @param delimiter delimiter of the source and target vertex in the line 
 * @param ignore_weights flag whether to ignore weights (offset 2 in the line) - currently ignored; unsupported
 */
void read_graph(int_vector &start_of_vertices_ptr, int_vector &vertex_neighbor_count_ptr, 
                int_vector &neighbors_vector_ptr, set_vector &neighbors_sets_ptr, std::string input_file_name,
                char delimiter=' ', int ignore_weights = 0) {
	start_of_vertices_ptr.clear(); 
    vertex_neighbor_count_ptr.clear();
    neighbors_vector_ptr.clear();

	std::ifstream infile;
	infile.open(input_file_name);
	std::string temp;
	
	int last_node = -1;
	int node_count = 0;
    int vertex_count = 0;
    
    // start of vertex 0
    start_of_vertices_ptr.push_back(0);
    std::set<int> neighbor_set;

    // iterate over all edges and store processed information properly
	while (getline(infile, temp, delimiter)) {
        int node = std::stoi(temp);
        node_count++;

        // check whether we have ran out of edges for last vertex
		if (node != last_node) {
            if (vertex_count != 0) {
                // if yes and the vertex count was not empty, store all information
                vertex_neighbor_count_ptr.push_back(vertex_count);
                start_of_vertices_ptr.push_back(node_count-1);
                neighbors_sets_ptr.push_back(neighbor_set);
            }
            // empty previous information, do not store info
        	last_node = node;
            vertex_count = 0;
            neighbor_set = std::set<int>();
		}
		
        vertex_count++;

        // get neighbor (target vertex of the edge)
		getline(infile, temp);
        int neighbor = std::stoi(temp);
        if (neighbor == last_node) {
            // edge to itself, e.g. 1->1
            vertex_count--;
            node_count--;
        } else {
            // normal case
            neighbors_vector_ptr.push_back(neighbor); 
            neighbor_set.insert(neighbor);
        }
	}

    // we ran out of edges, store information for the last source vertex
    vertex_neighbor_count_ptr.push_back(vertex_count);
    neighbors_sets_ptr.push_back(neighbor_set);
}

/**
 * Generates pair combinations in 2 values i-j, where i is i-th neighbor of source vertex and j is the source vertex's
 * j-th neighbor. This is used for assigning work to threads during parallel execution. For vertex with degree n, 
 * first (n * (n-1) / 2) combinations are created.
 * 
 * Created combinations:
 * 0-1
 * 0-2
 * 1-2
 * 0-3
 * 1-3
 * 2-3 etc.
 * 
 * For vertex of degree 2, only the first combination is used. For degree 3, first 3 combinations are used.
 * For degree 4, first 6 combinations are used etc. Therefore, we can easily assign work to threads sequentially, 
 * using the thread ID.
 * 
 * @param pairs a pointer to the vector where pairs shall be inserted - every 2 elements in the vector are one pair
 * @param pair_count a pointer to the vector where pair count per every degree are stored
 * @param highest_degree max degree for which pairs need to be computed
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
            // generates pairs for all vertices in range 0..i-1
            pairs.push_back(j);
            pairs.push_back(i);
            count++;
        }
        pair_count.push_back(count);
    }
}

/**
 * Splits all vertices in graph into chunks that can be computed at once. Also calculates several other properties
 * of the graph.
 * 
 * It tries to squeeze as many subsequent vertex computations in one 'CUDA computation' as possible. The limit is 
 * (blocks * threads). Every vertex uses (n * (n-1) / 2)  'slots' - this is the number of different clusters it can
 * form, n being the degree of the 'middle' vertex in the cluster. 
 * 
 * Note: The (blocks * threads) must be at least the number of clusters formed by the vertex with the highest degree.
 * 
 * Note 2: Greatly exceeding the minimum required limit does not result in a significant performance requirement.
 * 
 * @param generating_chunks a pointer to the vector where information about index of starting and ending vector 
 *                          per generating chunk is stored. E.g. if there are two generating chunks, 
 *                          each having 3 vertices, the data in this vector are going to be 0-2 and 3-5.
 * @param vertex_neighbor_count_ptr a pointer to the vector which stores information about how many neighbors 
 *                                  a given vertex has
 * @param pair_count a pointer to the vector where pair count per every degree are stored
 * @param chunk_size stores information about how many clusters are to be generated by the given generating chunk.
 *                   We try to make every chunk as full as possible, but they may have different size, 
 *                   below (blocks * threads).
 * @param cluster_start a pointer to the vector storing information at which position would clusters of vertex i start,
 *                      if we put all generated clusters for every vertex in a long array (clusters of vertex 0
 *                      would be first, n-1 last)
 * @param intervals a pointer to the interval tree structure which helps CUDA threads find their cluster vertices 
 *                  for generating clusters - we process all clusters in parts and every thread needs to know
 *                  how to find correct vertices in relation to given FRAGMENT of data. Not all graphs fit into memory.
 * @param blocks the number of blocks to be used during the computation on the device
 * @param threads the number of threads to be used during the computation on the device
 */
void split_into_generating_chunks(int_vector &generating_chunks, int_vector &vertex_neighbor_count_ptr, int_vector &pair_count, int_vector &chunk_size, int_vector &cluster_start, interval_tree_struct &intervals, int blocks, int threads) {
    size_t cluster_count = 0;
    int size = 0;
    int chunk_start = 0;
    int current_vertices = 0;
    int limit = blocks * threads;
    int neighbors_in_chunk = 0;
    int biggest_chunk = 0;
    int longest_neighbor_seq = 0;
    size_t_vector indexes;

    // go through every vertex sequentially
    for (int i = 0; i < vertex_neighbor_count_ptr.size(); i++) {
        cluster_start.push_back(cluster_count);

        // get the number of clusters in which the vertex is 'in the middle' - based on vertec degree
        int vertex_combinations = pair_count[vertex_neighbor_count_ptr[i]];
        cluster_count += vertex_combinations;

        // if we would surpass limit, save chunk and create a new one with current vertex only
        if (size + vertex_combinations > limit || vertex_combinations > limit) {
            if (current_vertices == 0) {
                std::invalid_argument(
                    "Insufficient size of blocks and threads for generating chunks. Required combined size at least "
                    + std::to_string(vertex_combinations) + ".");
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
            neighbors_in_chunk = vertex_neighbor_count_ptr[i];
        } else {
            // we have not surpassed limit, we can add current vertex to chunk
            indexes.push_back((size_t) size);
            size += vertex_combinations;
            current_vertices++;
            neighbors_in_chunk += vertex_neighbor_count_ptr[i];
        }
    }

    // process the last chunk, which is unsaved
    if (current_vertices != 0) {
        generating_chunks.push_back(chunk_start);
        generating_chunks.push_back(chunk_start + current_vertices - 1);
        intervals.push_back(indexes);
        chunk_size.push_back(size);
        biggest_chunk = std::max(biggest_chunk, (int) indexes.size());
        longest_neighbor_seq = std::max(longest_neighbor_seq, neighbors_in_chunk);
    }

    // store some graph data
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
 * - streams - 1..n, less streams are better for memory with no major performance boost, recommended value is 1
 * - use weighted edges - currently unsupported
 * - results file - if present, prints the ExF to given file, otherwise prints results nowhere
 */
int main(int argc, char* argv[]) { //

    std::cout << "This program determines the Expected Force of every node for each graph.\n Stores the results in 'FILENAME_results.txt'" << std::endl;
	
    if(argc < 5) {
        std::cout << "Insufficient number of arguments: " << argc << std::endl;
        exit(3);
    }

    // parse arguments
    std::string filename = argv[1];
    int blocks = atoi(argv[2]);
    int threads = atoi(argv[3]);
    int streamCount = atoi(argv[4]);
    if (argc <= 6) {
        std::cout << "No output file provided - the program ends." << std::endl;
        exit(7);
    }

    int_vector start_of_vertices_ptr, vertex_neighbor_count_ptr, neighbors_vector_ptr;
    set_vector neighbors_sets_ptr;
    interval_tree_struct intervals;
    int_vector pairs, pair_count, generating_chunks, chunk_size, path_vertex_one, cluster_start;

    std::cout << "Evaluating file " << filename << std::endl;
    std::time_t start = std::time(0);   // get time now
    std::tm* start_time = std::localtime(&start);
    std::cout << "Start: " 
              << start_time->tm_hour << ':' 
              << start_time->tm_min << ":" 
              << start_time->tm_sec 
              << std::endl; 

    // algorithm initialization
    read_graph(start_of_vertices_ptr, vertex_neighbor_count_ptr, neighbors_vector_ptr, neighbors_sets_ptr, filename, ' ', 1);
    int highest_degree = *std::max_element(vertex_neighbor_count_ptr.begin(), vertex_neighbor_count_ptr.end());
    generate_pairs(pairs, pair_count, highest_degree);
    split_into_generating_chunks(generating_chunks, vertex_neighbor_count_ptr, pair_count, chunk_size, cluster_start, intervals, blocks, threads);

    int biggest_chunk = graph_summary.biggest_chunk;
    int most_neighbors = graph_summary.longest_neighbor_seq;

    // transform from C++ vector to arrays that can be copied in async by CUDA
    int* vertex_start;
    cudaMallocHost((void**) &vertex_start, sizeof(int) * start_of_vertices_ptr.size());
    std::copy(start_of_vertices_ptr.begin(), start_of_vertices_ptr.end(), vertex_start);
    start_of_vertices_ptr = std::vector<int>(); // empties the start_of_vertices_ptr, move constructor works like swap()



    int* neighbors;
    cudaMallocHost((void**) &neighbors, sizeof(int) * neighbors_vector_ptr.size());
    std::copy(neighbors_vector_ptr.begin(), neighbors_vector_ptr.end(), neighbors);
    neighbors_vector_ptr = std::vector<int>(); // empties the neighbors_vector_ptr, move constructor works like swap()

    // transform interval tree (std::vector<std::vector<int>>) into CUDA pinned-memory arrays
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

    // initialize all vectors we are going to use - allocate in the device
    cudaStream_t streams[streamCount];
    std::vector<size_t*> index_pointers;
    std::vector<int*> vertex_start_pointers;
    std::vector<int*> neighbor_pointers;
    std::vector<int*> cluster_size_pointers;
    std::vector<int*> host_cluster_size_pointers;
    std::vector<int*> cluster_start_pointers;
    std::vector<int*> host_cluster_start_pointers;

    // allocate arrays per-stream
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

    // allocate globally-needed arrays
    int* pairs_ptr;
    cudaMalloc((void**)&pairs_ptr, sizeof(int) * pairs.size());
    cudaMemcpy(pairs_ptr, pairs.data(), sizeof(int) * pairs.size(), cudaMemcpyHostToDevice);
    check_error();

    int* length_ptr;
    cudaMalloc((void**)&length_ptr, sizeof(int) * vertex_neighbor_count_ptr.size());
    cudaMemcpy(length_ptr, vertex_neighbor_count_ptr.data(), sizeof(int) * vertex_neighbor_count_ptr.size(), cudaMemcpyHostToDevice);
    check_error();

    // measures all generated clusters (one 'computed' cluster can be formed in 4 different configurations -> 4 'actual' clusters)
    size_t formed_clusters_count = 4 * graph_summary.cluster_count;

    int* cluster_sizes;
    cudaMallocHost((void**) &cluster_sizes, sizeof(int) * formed_clusters_count); // cluster sizes * 4 -> each cluster has 4 initiation variants
    int* start_vertex;
    cudaMallocHost((void**) &start_vertex, sizeof(int) * formed_clusters_count); // cluster sizes * 4 -> each cluster has 4 starting points

    // offsets in cluster_sizes and start_vertex to the first unwritten slot
    size_t data_offset = 0;
    std::vector<size_t> total_cluster_size(vertex_neighbor_count_ptr.size(), 0);

    // PARALLEL SECITON 1: generate clusters in the device - per generating chunk
    for (size_t index = 0; index < generating_chunks.size(); index += 2 * streamCount) {
        int streamsUsed = std::min((int) ((generating_chunks.size() / 2) - index/2), (int) streamCount);
        for (int i = 0; i < streamsUsed; i++) {
            size_t chunk_index = index + 2 * i;
            size_t chunk_start = generating_chunks[chunk_index];
            size_t chunk_end = generating_chunks[chunk_index + 1];
            size_t number_of_clusters = cluster_start[chunk_end] + pair_count[vertex_neighbor_count_ptr[chunk_end]] - cluster_start[chunk_start];
            size_t neighbor_size = vertex_start[chunk_end] + vertex_neighbor_count_ptr[chunk_end] - vertex_start[chunk_start];
            size_t interval_tree_offset = index/2 + i;

            // copy data into the device
            cudaMemcpyAsync(index_pointers[i], interval_tree + (*(interval_tree_node_start + interval_tree_offset)), sizeof(size_t) * (*(interval_tree_node_length + interval_tree_offset)), cudaMemcpyHostToDevice);
            check_error();
            cudaMemcpy(vertex_start_pointers[i], vertex_start + chunk_start, sizeof(int) * (chunk_end - chunk_start + 1), cudaMemcpyHostToDevice);
            check_error();
            cudaMemcpy(neighbor_pointers[i], neighbors + vertex_start[chunk_start], sizeof(int) * neighbor_size, cudaMemcpyHostToDevice);
            check_error();

            // compute
            GeneratePairs<<<blocks, threads, 0, streams[i]>>>(index_pointers[i], neighbor_pointers[i], vertex_start_pointers[i], pairs_ptr, length_ptr, cluster_size_pointers[i], cluster_start_pointers[i], chunk_end - chunk_start + 1, number_of_clusters, chunk_start, vertex_start[chunk_start]);
            cudaDeviceSynchronize();
            check_error();

            // retrieve data from device, save in host (in prepared arrays)
            cudaMemcpy(host_cluster_size_pointers[i], cluster_size_pointers[i], sizeof(int) * number_of_clusters, cudaMemcpyDeviceToHost);
            check_error();
            cudaMemcpy(host_cluster_start_pointers[i], cluster_start_pointers[i], sizeof(int) * number_of_clusters * 3, cudaMemcpyDeviceToHost);
            check_error();
        }
        check_error();

        // process all generated clusters
        for (int i = 0; i < streamsUsed; i++) {
            int* path_ptr = host_cluster_start_pointers[i];
            int chunk_index = index + 2 * i;
            int chunk_start = generating_chunks[chunk_index];
            int chunk_end = generating_chunks[chunk_index + 1];
            size_t number_of_clusters = cluster_start[chunk_end] + pair_count[vertex_neighbor_count_ptr[chunk_end]] - cluster_start[chunk_start];

            // sort clusters into convenient representation in arrays
            for (size_t j = 0; j < number_of_clusters; j++) {
                int cluster_size = host_cluster_size_pointers[i][j];
                int source_vertex = path_ptr[3 * j];
                int neighbor_one = path_ptr[3 * j + 1];
                int neighbor_two = path_ptr[3 * j + 2];

                // if two neighbors form a triangle with cluster-size edges
                // contains() cannot be used -> C++20-only
                if (neighbors_sets_ptr[neighbor_one].find(neighbor_two) != neighbors_sets_ptr[neighbor_one].end()) {
                    cluster_size -= 2;
                }

                for (size_t k = 0; k < 4; k++) {
                    // adjust cluster size
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
    }

    // free device stream-specific arrays we are not going to use anymore
    for (int i = 0; i < streamCount; i++) {
        cudaFree(index_pointers[i]);
        cudaFree(vertex_start_pointers[i]);
        cudaFree(neighbor_pointers[i]);
        cudaFree(cluster_size_pointers[i]);
        cudaFree(cluster_start_pointers[i]);
        cudaFreeHost(host_cluster_size_pointers[i]);
        cudaFreeHost(host_cluster_start_pointers[i]);
    }     

    // free global device arrays we are not going to use anymore
    cudaFree(length_ptr);
    cudaFree(pairs_ptr);
    check_error();

    // prepare for parallel section 2
    std::vector<int*> input_sizes;
    std::vector<int*> input_vertices;
    std::vector<float*> outputs;
    int array_size = blocks * threads;

    // allocate stream-specific arrays in the device
    for (int i = 0; i < streamCount; i++) {
        int *size_ptr, *vertex_ptr; 
        float *devOut;
        cudaMalloc((void**)&size_ptr, (size_t) sizeof(int) * array_size);
        cudaMalloc((void**)&vertex_ptr, (size_t) sizeof(int) * array_size);
        cudaMalloc((void**)&devOut, (size_t) sizeof(float) * array_size);
        input_sizes.push_back(size_ptr);
        input_vertices.push_back(vertex_ptr);
        outputs.push_back(devOut);
        check_error();
    }

    // allocate global arrays in the device
    size_t *total_size_ptr;
    cudaMalloc((void**)&total_size_ptr, sizeof(size_t) * total_cluster_size.size());
    check_error();
    cudaMemcpyAsync(total_size_ptr, total_cluster_size.data(), sizeof(size_t) * total_cluster_size.size(), cudaMemcpyHostToDevice);
    check_error();

    // std::ceil is stupid
    int ceil_share = (int) ((graph_summary.cluster_count * 4 / array_size) + ((graph_summary.cluster_count * 4 % array_size) != 0));
    int chunks = std::max((int) 1, ceil_share);

    // allocate device array for storing computed normalized cluster sizes
    float* normalized_sizes;
    cudaMallocHost((void**) &normalized_sizes, sizeof(float) * formed_clusters_count);

    // PARALLEL SECTION 2: compute Expected Force per-cluster
    for (int index = 0; index < chunks; index += streamCount) {
        int streamsUsed = std::min((int) streamCount, (int) chunks - index);
        for (int i = 0; i < streamsUsed; i++) {
            size_t element_count = std::min((size_t) array_size, 4 * graph_summary.cluster_count - ((size_t) (index + i)) * array_size);
            size_t offset = ((size_t) (index + i)) * array_size;
            size_t copied_bytes = sizeof(int) * element_count;

            // move data from host to device
            cudaMemcpyAsync(input_sizes[i], cluster_sizes + offset, copied_bytes, cudaMemcpyHostToDevice);
            check_error();
            cudaMemcpyAsync(input_vertices[i], start_vertex + offset, copied_bytes, cudaMemcpyHostToDevice);
            check_error();

            // compute Expected Force in parallel
            CountClusterExpectedForce<<<blocks, threads, 0, streams[i]>>>(input_sizes[i], input_vertices[i], total_size_ptr, outputs[i], element_count);
            check_error();

            // move Expected Force data back from device to host
            size_t copied_bytes_back = sizeof(float) * element_count;
            check_error();
            cudaMemcpyAsync(normalized_sizes + offset, outputs[i], copied_bytes_back, cudaMemcpyDeviceToHost);
            check_error();
        }
    }

    // free unused CUDA memory
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

    // process resulting Expected Force
    std::vector<float> results(vertex_neighbor_count_ptr.size(), 0);
    for (size_t i = 0; i < formed_clusters_count; i++) {
        results[start_vertex[i]] += normalized_sizes[i];
    }

    // free unused memory
    cudaFreeHost(start_vertex);
    cudaFreeHost(normalized_sizes);

    // print results into the output file
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

    std::time_t end = std::time(0);   // get time now
    std::tm* end_time = std::localtime(&end);
    std::cout << "Done: " 
              << end_time->tm_hour << ':' 
              << end_time->tm_min << ":" 
              << end_time->tm_sec 
              << std::endl; 

	return 0;
}
