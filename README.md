This is a repository created by Paolo Sylos Labini in order to test the Glenn Lawyer Expected Force algorithm on SNAP graphs in C++ .
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

example: ./ExpForce fb_full.txt , where fb_full.txt contains a full, sorted edgelists such as
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
