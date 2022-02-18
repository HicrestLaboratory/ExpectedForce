# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:16:01 2020

@author: Paolo



Graph Converter

"""
filename = "graph_to_convert.txt"


graph = {}
with open(filename) as f:
    #search for 3D data
    n = 0;
    inc = "!"
    while (inc):
        n += 1;
        inc = f.readline();
        reaction = f.readline();
        out = f.readline();
        if inc != out:
            if not (inc in graph):
                graph[inc] = [out,]
            else:
                graph[inc].append(out)
            
            if not (out in graph):
                graph[out] = [inc,]
            else:
                graph[out].append(inc)
            

outfile = "mapping.txt"
mapping = {}
i = 0;

with open(outfile, "w") as f:
    for parent in graph:
        mapping[parent] = i;
        i+= 1;
        f.writelines(str(i) + "  " + parent);

outfile = "converted.txt"
with open(outfile, "w") as f:
    for parent in graph:
        i = mapping[parent];
        for elem in graph[parent]:
            j = mapping[elem]
            f.writelines(str(i) + "  " + str(j) + "\n");
    


