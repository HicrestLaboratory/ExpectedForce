# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:22:02 2022

@author: Paolo
"""

import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Converts results from a mapping file")
    
    parser.add_argument("--input-file",
        help="input data")
    parser.add_argument("--mapping",
        help="input mapping")
    parser.add_argument("--outfile",
        help="converted file")
    parser.add_argument("--delimiter", default=" ",
        help="delimiter between parent and children node")
    
    
    args = parser.parse_args()

    input_file = args.input_file;
    mapping_file = args.mapping;
    outfile = args.outfile;
    delimiter = args.delimiter;


def find_delimiter(file):
    delimiters = [" ", "  ", "-", "\t"]
    with open(file) as f:
        for delimiter in delimiters:
            line = f.readline()
            if len(line.split(delimiter)) == 2:
                if line.split[0] != "" and line.split[1] != "":
                    return delimiter


delimiter = find_delimiter(mapping_file)
print("found delimiter: (", delimiter, ")")

mapping = {}
with open(mapping_file) as f:
    n = 0;
    inc = "!"
    for line in f:
        linesplit = line.split(delimiter)
        node = int(linesplit[0]);
        node_id = int(linesplit[1]);
        mapping[node_id] = node;


delimiter = find_delimiter(mapping_file)
print("found delimiter: (", delimiter, ")")

with open(input_file) as f:
    with open(outfile, "w") as o:
        for line in f:
            linesplit = line.split(delimiter);
            node = int(linesplit[0])
            value = linesplit[1]
            o.writelines(str(mapping[node]) + delimiter + value + "\n")
            
            
print("saved converted file to", outfile)