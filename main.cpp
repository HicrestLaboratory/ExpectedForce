#include "stdafx.h"
#include <sys/time.h>
#include <omp.h>
#ifndef TIMING_H
#define TIMING_H
#define TIMER_DEF     struct timeval temp_1, temp_2

#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)

#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)

#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec))
#endif

using namespace std;

typedef vector<int> svi;
typedef vector<int>::iterator svii;

/* HELPER FUNCTION
Converts a sorted, full edgelist text file to a vector edgelist.
 @param[out] egos, alters: The edgelist.
 @param[in] infilename: The edgelist file.
*/
int read_snap_format(svi &egos, svi &alters,
	string infilename, char delimiter = ' ', int ignore_weights = 0) 
{
	egos.clear(); alters.clear();

	ifstream infile;
	infile.open(infilename);
	string temp;
	
	int last_node = -1;
	int node_count = 0;

	while (getline(infile, temp, delimiter)) {
		
		if (stoi(temp) != last_node) { //conta i nodi diversi
			node_count++;
			last_node = stoi(temp);
		}
		
		egos.push_back(stoi(temp)); //out node
		
		if(ignore_weights)
		{
			getline(infile, temp, delimiter);
		}
		
		getline(infile, temp);
		alters.push_back(stoi(temp)); //in node
	}
	//cout << node_count << endl;

	return node_count;
}



int main(int argc, char* argv[]) { //takes a filename (es: fb_full) as input; print its ExF in result.txt 

	cout << "This program determines the Expected Force of every node for each graph.\n Stores the results in 'FILENAME_results.txt'" << endl;
	
	svi egosVect, altersVect;                
	string filename = argv[1];
	int ignore_weights = stoi(argv[2]);
	string out_name = argv[3];
        int num_threads, thread_id;


#pragma omp parallel private(thread_id)
        {
             num_threads = omp_get_num_threads();
             thread_id = omp_get_thread_num();
             if (thread_id == 0)
	     {
		     std::cout << "Using " << num_threads << " threads" << std::endl;
	     }
        }

	//reads graph
	int node_count = read_snap_format(egosVect, altersVect, filename, ' ', ignore_weights); //converts SNAP graph to sorted edgelist.
	//TODO: check if edgelist is full and sorted 

	ofstream outfile;
	outfile.open(out_name);
	cout << "Evaluating Expected Force for graph '" + filename + "'"<< endl;

	double *EXF = new double[node_count];
        TIMER_DEF;
        float ttime = 0.0;
        TIMER_START; 
	int chunk_size = 2;
#pragma omp parallel for schedule(dynamic, chunk_size)
	for (int i = 0; i < node_count; i++) 
	{
		//calculates and prints on file the Expected Force for each node
		EXF[i] = exfcpp(egosVect, altersVect, i);
		//outfile << std::to_string(i) << "  " << std::to_string(EXF) << endl;
		//notificate progress
		//cout << i + 1 << "out of" << node_count << endl;
	}
        TIMER_STOP;
	ttime +=TIMER_ELAPSED;
        for (int i = 0; i < node_count; i++){
              outfile << std::to_string(i) << "  " << std::to_string(EXF[i]) << endl;
	}	
	outfile.close();
	cout << "Results saved as " << out_name << endl;
        cout << "CPU processing in ms: time = " << ttime/1000.0 << endl;
	return 0;
}
