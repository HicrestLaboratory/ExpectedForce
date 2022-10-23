#!/bin/bash

# set the number of nodes the job will employ, along with the amount
# of memory and the number of CPU cores it will require, on each node;
# in this case, select 12 cores and 23 GB of memory for 5 nodes,
# which takes the total to 60 cores and 115 GB of memory
# (simply multiply the numbers)
#PBS -l select=1:ncpus=1:mem=32gb 

# maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually)
#PBS -l walltime=8:00:00

# set the execution queue
#PBS -q common_cpuQ 

#PBS -e serial_benchmark1.log

# 4min
~/ExpectedForce/ExpForce ~/ExpectedForce/parallel/test_graphs/ready/fb-social-circles.txt 1 ~/ExpectedForce/serial_results/fb-social-circles.txt >> ~/ExpectedForce/serial.log
# 35min
~/ExpectedForce/ExpForce ~/ExpectedForce/parallel/test_graphs/ready/wiki.txt 1 ~/ExpectedForce/serial_results/wiki.txt >> ~/ExpectedForce/serial.log
# 110min
~/ExpectedForce/ExpForce ~/ExpectedForce/parallel/test_graphs/ready/Amazon0302.txt 1 ~/ExpectedForce/serial_results/amazon.txt >> ~/ExpectedForce/serial.log
# 128min
~/ExpectedForce/ExpForce ~/ExpectedForce/parallel/test_graphs/ready/rmat_S14_EF8.txt 1 ~/ExpectedForce/serial_results/rmat-S14-EF8.txt >> ~/ExpectedForce/serial.log

