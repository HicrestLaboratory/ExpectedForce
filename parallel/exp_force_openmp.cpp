#pragma GCC ivde

//#include "stdafx.h"
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <map>
#include <cmath> // check for nanwhen printing results
#include <atomic> // std::atomic for correct duration summing in streams
#include <algorithm> // max_element, std::min
#include <utility> // pair
#include <math.h> // ceilm
#include <omp.h>

// for debugging only
#define OBSERVED_NODE -1
#define DEBUG 0

typedef std::vector<int> int_vector;
typedef std::vector<std::set<int>> set_vector;

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
 * Entry function, parameters: 
 * - filename: name fo the file whence the graph shall be read
 * - result_file: optional, will print results into the file if present, will be skipped if this argument is missing
 */
int main(int argc, char* argv[]) { //takes a filename (es: fb_full) as input; print its ExF in 'filename' provided as argument

	// cout << "This program determines the Expected Force of every node for each graph.\n Stores the results in 'FILENAME_results.txt'" << endl;
	
    if(argc < 2) {
        std::cout << "Insufficient number of arguments: " << argc << std::endl;
        exit(3);
    }

    std::string filename = argv[1];
    bool print_results = false;
    if (argc > 2) {
        print_results = true;
    }

    int_vector v_start, vertex_length, neighbors_vector, pairs, pair_count;
    set_vector neighbor_sets;
    size_t duration = 0;

    std::cout << "Evaluating file " << filename << std::endl;

    int repetitions = 1;

    //reads graph
    read_graph(v_start, vertex_length, neighbors_vector, neighbor_sets, filename, ' ', 1); //converts graph to a v-graph-like structure

    int highest_degree = *std::max_element(vertex_length.begin(), vertex_length.end());

    generate_pairs(pairs, pair_count, highest_degree);


    for (int rep = 0; rep < repetitions; rep++) {
        auto start = std::chrono::high_resolution_clock::now();

        float* expected_force = (float*) malloc(sizeof(float) * v_start.size());
        size_t* total_cluster_sizes = (size_t*) malloc(sizeof(size_t) * v_start.size());
        std::vector<std::map<int,int>> cluster_map(v_start.size());

        #pragma omp parallel
        {
            #pragma omp for 
            for (int cluster=0; cluster < v_start.size(); cluster++) {
                total_cluster_sizes[cluster] = 0;
            }

            #pragma omp for
            for (int current_vertex=0; current_vertex < v_start.size(); current_vertex++) {
                int neighbors_start = v_start[current_vertex];
                int vector_size = vertex_length[current_vertex];
                int cluster_count = pair_count[vertex_length[current_vertex]];
                for (int computed_pair=0; computed_pair < cluster_count; computed_pair++) {
                    int neighbor_one = neighbors_vector[neighbors_start + pairs[2 * computed_pair]];
                    int neighbor_two = neighbors_vector[neighbors_start + pairs[2 * computed_pair + 1]];

                    int cluster_size = vector_size + vertex_length[neighbor_one] + vertex_length[neighbor_two] - 4;
                    if (neighbor_sets[neighbor_one].find(neighbor_two) != neighbor_sets[neighbor_one].end()) {
                        cluster_size -= 2;
                    }

                    #pragma omp critical
                    {
                        // increase total_cluster_sizes for all 4 cluster combinations
                        total_cluster_sizes[current_vertex] += 2 * cluster_size;
                        total_cluster_sizes[neighbor_one] += cluster_size;
                        total_cluster_sizes[neighbor_two] += cluster_size;

                        // store clusters for later normalization and exf computation
                        cluster_map[current_vertex].insert(std::pair<int,int> (cluster_size, 0));
                        cluster_map[neighbor_one].insert(std::pair<int,int> (cluster_size, 0));
                        cluster_map[neighbor_two].insert(std::pair<int,int> (cluster_size, 0));

                        // 'save' cluster sizes
                        cluster_map[current_vertex][cluster_size] = cluster_map[current_vertex][cluster_size] + 2;
                        cluster_map[neighbor_one][cluster_size]++;
                        cluster_map[neighbor_two][cluster_size]++;
                    }
                }  

                #pragma omp for
                for (int current_vertex=0; current_vertex < v_start.size(); current_vertex++) {
                    std::map<int, int> current_map = cluster_map[current_vertex];
                    float current_expected_force = 0;

                    for (auto const& cluster_pair : current_map) {
                        int cluster_size = cluster_pair.first;
                        float normalized = (float) cluster_size/total_cluster_sizes[current_vertex];
                        // negated entropy -> subtract instead of adding negative; insert as many times as there were occurrences
                        current_expected_force -= (logf(normalized) * (normalized)) * cluster_pair.second; 
                    }

                    expected_force[current_vertex] = current_expected_force;
                }
            }
        }

        free(total_cluster_sizes);

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        duration += time;

        std::cout << time << std::endl;

        if (rep == 0 && print_results) {
            std::ofstream output_file(argv[2]);
            if (output_file.is_open()) {
                for (size_t i = 0; i < v_start.size(); i++) {
                    if (isnan(expected_force[i])) {
                        expected_force[i] = 0;
                    }
                    output_file << i << "  " << expected_force[i] << "\n";
                }
            } else {
                std::cout << "Could not open or create file " << argv[2] << ". Results will not be printed." << std::endl;
            }
        } 

        free(expected_force);
    }

    // blocks,threads,streamCount,avg duration in microseconds
    std::cout << "Average time: " << duration/repetitions << std::endl;

	return 0;
}
