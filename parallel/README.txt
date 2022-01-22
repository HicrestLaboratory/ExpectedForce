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
- cuda 11.2.1
- gcc 6.5.0
---------------------------------------------------------------------------------------------------------------------
Usage: 
nvcc ../exp_force_main.cu -o ../output/ExForce
srun -N 1 ../output/ExForce input_graph blocks threads stream_count ignore_weights

...where parameters are:
input_graph
    relative path to the input graph
blocks
    number of blocks to be used (blocks*threads must be at least the max cluster count of a vertex in the graph)
threads
    number of threads to be used (see limits of your GPU)
stream_count
    number of streams to be used
ignore_weights
    flag whether to ignore weights (currently not implemented, value is ignored)