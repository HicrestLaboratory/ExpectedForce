#include "stdafx.h"

// log start and end time (granularity in seconds only)
#include <ctime>
#include <iostream>

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
		
	//reads graph
	int node_count = read_snap_format(egosVect, altersVect, filename, ' ', ignore_weights); //converts SNAP graph to sorted edgelist.
	//TODO: check if edgelist is full and sorted 

	ofstream outfile;
	//outfile.open("results.txt");
	outfile.open(argv[3]);
	cout << "Evaluating Expected Force for graph '" + filename + "'"<< endl;

    std::time_t start = std::time(0);   // get time now
    std::tm* start_time = std::localtime(&start);
    std::cout << "Start: " 
              << start_time->tm_hour << ':' 
              << start_time->tm_min << ":" 
              << start_time->tm_sec 
              << std::endl; 

	double EXF;
	for (int i = 0; i < node_count; i++) 
	{
		//calculates and prints on file the Expected Force for each node
		EXF = exfcpp(egosVect, altersVect, i);
		outfile << to_string(i) << "  " << to_string(EXF) << endl;
		//notificate progress
		// cout << i + 1 << "out of" << node_count << endl;
	}

	outfile.close();
	    std::time_t end = std::time(0);   // get time now
    std::tm* end_time = std::localtime(&end);
    std::cout << "Done: " 
              << end_time->tm_hour << ':' 
              << end_time->tm_min << ":" 
              << end_time->tm_sec 
              << std::endl; 

	// cout << "Results saved as results.txt'" << endl;

	return 0;
}
