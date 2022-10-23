#!/bin/bash

# set the number of nodes the job will employ, along with the amount
# of memory and the number of CPU cores it will require, on each node;
# in this case, select 12 cores and 23 GB of memory for 5 nodes,
# which takes the total to 60 cores and 115 GB of memory
# (simply multiply the numbers)
#PBS -l select=1:ncpus=1:mem=23gb 

# maximum execution time (the longer the time, the longer the job
# will stay in the queue before running actually)
#PBS -l walltime=04:00:00

# set the execution queue
#PBS -q short_gpuQ 

#PBS -e benchmark.log

bash ~/ExpectedForce/parallel/scripts/pbs-results-only/amazon_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/ca_network_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/fb_circles_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/google_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/rmat-s14-ef8_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/rmat-s14-ef16_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/rmat-s14-ef32_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/rmat-s14-ef64_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/twitter_benchmark.sh
bash ~/ExpectedForce/parallel/scripts/pbs-results-only/wiki_benchmark.sh