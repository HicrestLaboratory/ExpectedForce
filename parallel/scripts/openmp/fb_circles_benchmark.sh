#!/bin/bash

# set the number of nodes the job will employ, along with the amount
# of memory and the number of CPU cores it will require, on each node;s
# in this case, select 12 cores and 23 GB of memory for 5 nodes,
# which takes the total to 60 cores and 115 GB of memory
# (simply multiply the numbers)
#PBS -l select=1:ncpus=1:mem=23gb 

# maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually)
#PBS -l walltime=5:00:00

# set the execution queue
#PBS -q common_cpuQ 

#PBS -e ca_network.log

module load gcc75

#gcc ~/ExpectedForce/parallel/exp_force_openmp.cpp -o ~/ExpectedForce/parallel/output/ExForceOpenmp -ftree-vectorize -msse3 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopt-info-vec-missed=output.miss -funroll-loops -std=c++11
g++ ~/ExpectedForce/parallel/exp_force_openmp.cpp -o ~/ExpectedForce/parallel/output/ExForceOpenmp -std=c++11 -lm -fopenmp

~/ExpectedForce/parallel/output/ExForceOpenmp ~/ExpectedForce/parallel/test_graphs/ready/fb-social-circles.txt ~/ExpectedForce/parallel/output/fb_social_circles.txt >> ~/ExpectedForce/parallel/output/stopwatch/fb-social-circles_openmp.txt

