#!/bin/bash

# set the number of nodes the job will employ, along with the amount
# of memory and the number of CPU cores it will require, on each node;
# in this case, select 12 cores and 23 GB of memory for 5 nodes,
# which takes the total to 60 cores and 115 GB of memory
# (simply multiply the numbers)
#PBS -l select=1:ncpus=1:mem=23gb 

# maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually)
#PBS -l walltime=01:00:00

# set the execution queue
#PBS -q common_cpuQ 

#PBS -e rmat_ef8.log

module load cuda-11.0
module load gcc75

/apps/cuda-11.0/bin/nvcc ~/ExpectedForce/parallel/exp_force_main.cu -o ~/ExpectedForce/parallel/output/ExForce -std=c++11

for blocks in 3200 4096 8192
do
    for stream_count in 1 2 4 8
    do
        ~/ExpectedForce/parallel/output/ExForce ~/ExpectedForce/parallel/test_graphs/ready/rmat_S14_EF8.txt $blocks 1024 $stream_count 1 >> ~/ExpectedForce/parallel/output/stopwatch/rmat_S14_EF8_stopwatch.txt
    done 
done
