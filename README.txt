This repository was created by Andrej Jurco, Paolo Sylos Labini and Flavio Vella. 
The code implements algorithms for the calculation of the Expected Force metric on SNAP graphs in C++ and OpenMP. 

It contains:
**original serial implementation (exffunction.cpp copyright Glenn Lawyer, 2013.)
**OpenMP implementation.
**CUDA GPU implementation (our efficient algorithm, see the code in the directory ``parallel`` )

--------------------------------------------------------------

USAGE via Makefile
"make all" will compile and test the code.

"make compile" will compile the code and create executable named ExpForce. 

"make run_test" will run a test on the graph stored in "fb_full.txt" and produce a result file "fb_full_results.txt".

---------------------------------------------------------------
GENERAL USAGE
Once compiled, an executable named ExpForce should appear. 
Execute it with any number of filenames as arguments;

example: OMP_NUM_THREADS=16 ./ExpForce fb_full.txt 1 fb_exp.score.txt, where fb_full.txt contains a full, sorted edgelists such as
0  2
1  2
2  0
2  1


-----------------------------------------------------------------
CONTENTS
exffunction.cpp is the Glenn Lawyer original function. Calculates the expected force of a node.

main.cpp loads a graph from a text file and calculate the expected force of the nodes.

stdafx.h is an header for standard libraries and the exfccp function.

fb_full.txt is a test graph. 


**Reference**
@INPROCEEDINGS{10495558,
  author={Labini, Paolo Sylos and Jurco, Andrej and Ceccarello, Matteo and Guarino, Stefano and Mastrostefano, Enrico and Vella, Flavio},
  booktitle={2024 32nd Euromicro International Conference on Parallel, Distributed and Network-Based Processing (PDP)}, 
  title={Scaling Expected Force: Efficient Identification of Key Nodes in Network-Based Epidemic Models}, 
  year={2024},
  volume={},
  number={},
  pages={98-107},
  keywords={Measurement;Epidemics;Force measurement;Scalability;Computational modeling;Force;Graphics processing units;Epidemic;SIR;Big Data;Expected Force;Graph Centrality;Network;Parallel Computing},
  doi={10.1109/PDP62718.2024.00021}}

