#!/bin/bash

# set the number of nodes the job will employ, along with the amount
# of memory and the number of CPU cores it will require, on each node;
# in this case, select 12 cores and 23 GB of memory for 5 nodes,
# which takes the total to 60 cores and 115 GB of memory
# (simply multiply the numbers)
#PBS -l select=1:ncpus=1:mem=23gb 

# maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually)
#PBS -l walltime=5:00:00

# set the execution queue
#PBS -q short_gpuQ 

#PBS -e ca_network.log

module load cuda-11.0
module load gcc75

/apps/cuda-11.0/bin/nvcc ~/ExpectedForce/parallel/exp_force_compute.cu -o ~/ExpectedForce/parallel/output/ExForce-Compute -std=c++11
~/ExpectedForce/parallel/output/ExForce-Compute ~/ExpectedForce/parallel/test_graphs/ready/road-net-ca.txt 64 1024 1 1 ~/ExpectedForce/parallel/output/result/road_net_ca.txt >> ~/ExpectedForce/parallel/output/log/road_net_ca.txt

