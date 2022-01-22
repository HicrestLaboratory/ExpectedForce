#!/bin/bash

#SBATCH --job-name=hello-cuda
#SBATCH --output=hello-cuda-%j.out
#SBATCH --error=hello-cuda-%j.err

#SBATCH --partition training
#SBATCH --gres=gpu
#SBATCH --nodes 1
#SBATCH --time=01:00:00

#SBATCH --ntasks=1

module load cuda-11.2.1
module load gcc-6.5.0

/usr/local/cuda/bin/nvcc ../exp_force_main.cu -o ../output/ExForce

for blocks in 7200 8192 16384
do
    for stream_count in 2 4 8
    do
        srun -N 1 ../output/ExForce ../input/rmat_S14_EF16.txt $blocks 1024 $stream_count 1 >> rmat_s14_ef16_stopwatch.txt
    done 
done
