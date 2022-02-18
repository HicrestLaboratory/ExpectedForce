# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:16:01 2020

@author: Paolo



Graph Converter
"""
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converts graph to be run with Expected Force")
    
    parser.add_argument("--input-file", default="fb_full.txt",
        help="input graph")
    parser.add_argument("--delimiter", default="  ",
        help="delimiter between parent and children node")
    parser.add_argument("--outname", default="test",
        help="name and location of output (no file extension)")

    args = parser.parse_args()

    input_file = args.input_file;
    delimiter = args.delimiter;
    outname = args.outname;

mappingfile = outname + "_mapping.txt"
outfile = outname + "_converted.txt"

mapping = {}
with open(input_file) as f:
    n = 0;
    inc = "!"
    for line in f:
        linesplit = line.split(delimiter)
        inc = int(linesplit[0]);
        if inc not in mapping:
            mapping[inc] = n;
            n += 1;
        out = int(linesplit[1]);
        if out not in mapping:
            mapping[out] = n;
            n += 1;

print(n, "Nodes found");

with open(mappingfile, "w") as f:
    for node,nodeid in mapping.items():
        f.writelines(str(node) + " " + str(nodeid) + "\n")
    
print("mapping saved in ", mappingfile)
graph = {}
with open(input_file) as f:
    inc = "!"
    for line in f:
        linesplit = line.split(delimiter)
        inc = mapping[int(linesplit[0])];
        out = mapping[int(linesplit[1])];
        if inc not in graph:
            graph[inc] = [out,]
        else:
            if out not in graph[inc]:
                graph[inc].append(out);
        if out not in graph:
            graph[out] = [inc,]
        else:
            if inc not in graph[out]:
                graph[out].append(inc);
    for parent, children in graph.items():
        children.sort();

with open(outfile, "w") as f:
    for parent in range(len(graph)):
        for child in graph[parent]:
            f.writelines(str(parent) + " " + str(child) + "\n");

print("graph converted saved in ", outfile)



