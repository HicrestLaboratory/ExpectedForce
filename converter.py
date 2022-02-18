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
        help="file that contains the already run experiments")
    parser.add_argument("--delimiter", default=" ",
        help="file that contains the already run experiments")

    args = parser.parse_args()

    input_file = args.input_file;
    delimiter = args.delimiter;

filename = "fb_full.txt"
outfile = filename.split(".")[0] + "mapping.txt"
outfile = filename.split(".")[0] + "converted.txt"

mapping = {}
with open(filename) as f:
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

with open(outfile, "w") as f:
    for node,id in mapping.items():
        f.writeline(str(node) + " " + str(id))
    
graph = {}
with open(filename) as f:
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
    for parent, children in graph:
        children.sort();


with open(outfile, "w") as f:
    for parent in graph:
        for child in graph[parent]:
            f.writeline(str(parent) + " " + str(child));
    


