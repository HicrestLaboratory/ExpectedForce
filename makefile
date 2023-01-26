CC = g++ -fopenmp -O3

all: compile run_test

compile: main.cpp exffunction.cpp stdafx.h
	${CC} main.cpp exffunction.cpp -o ExpForce

run_test: ExpForce
	./ExpForce dataset/rmat_10_4.txt 1 /tmp/out.txt
	
