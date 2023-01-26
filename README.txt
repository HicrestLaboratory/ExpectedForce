This is a repository created by Paolo Sylos Labini and Flavio Velle. 
The code implement the Glenn Lawyer Expected Force algorithm on SNAP graphs in C++ and OpenMP.
A GPU version is also available in ``parallel`` directory. 

It provides:
OpenMP implementation.
CUDA ve

exffunction.cpp copyright Glenn Lawyer, 2013.
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
