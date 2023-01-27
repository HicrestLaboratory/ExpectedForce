#!/bin/bash


grep time $1 | cut -d " " -f7 | sort -n | head -n 3
