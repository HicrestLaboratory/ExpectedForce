This is a parallel implementation of the serial Expected Force algorithm.
---------------------------------------------------------------------------------------------------------------------
Input:
- Graph represented as pairs of edges (source-target)
- undirected graph, for each edge there are two entries: node1-node2 and node2-node1
- edges are sorted by the source vertex
- no edges from vertex to itself
- no multiedges
- vertex numbers start from 0 and no number is skipped

For transformation of the graphs and getting graph stats, attached script 'transformer.py' can be used. 
See the script how to use it (execution takes zero parameters).
---------------------------------------------------------------------------------------------------------------------
Requirements:
- cuda 11.0 or newer
- gcc 6.5.0 or newer
Tested on:
- cuda 11.2.1 and gcc 6.5.0 (example below - slurm)
- cuda 11.0 and gcc 7.5 (example below - pbs)
---------------------------------------------------------------------------------------------------------------------
Usage: 
  - slurm:
      >> nvcc ../exp_force_main.cu -o ../output/ExForce
      >> srun -N 1 ../output/ExForce input_graph blocks threads stream_count ignore_weights output_file
  - pbs:
      >> nvcc ~/ExpectedForce/parallel/exp_force_main.cu -o ~/ExpectedForce/parallel/output/ExForce -std=c++11
      >> ~/ExpectedForce/parallel/output/ExForce input_graph blocks threads stream_count ignore_weights output_file

...where parameters are:
input_graph
    - relative path to the input graph
blocks
    - number of blocks to be used (blocks*threads must be at least the max cluster count of a vertex in the graph)
threads
    - number of threads to be used (see limits of your GPU)
stream_count
    - number of streams to be used
ignore_weights
    - flag whether to ignore weights (currently not implemented, value is ignored)
output_file
    - optional parameter
    - the file into which the results shall be printed (in pairs '<node_id>  <node_ex_force' separated by two whitespaces)
    - if the parameter is not present, no output will be printed (to file nor console)
    - usually relative to the location where the script running the execution is located