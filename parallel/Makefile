CC=g++
NVCC=nvcc
#CXXFLAGS= -fopenmp -O3 -Wextra -std=c++11
CUDAFLAGS= -std=c++11 -O2 --use_fast_math --ptxas-options=-v  -gencode=arch=compute_80,code=compute_80
#LIBS= -lopenblas -lpthread -lcudart -lcublas
#LIBDIRS=-L/usr/local/cuda-7.5/lib64
#INCDIRS=-I/usr/local/cuda-7.5/include


default: exp_force_main.cu
	$(NVCC) $(CUDAFLAGS) exp_force_main.cu  -o expected_force $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)

opt: exp_force_main-opt.cu
	$(NVCC) $(CUDAFLAGS) exp_force_main-opt.cu  -o expected_force_opt $(LIBDIRS) $(INCDIRS) $(LIBS) $(CXXFLAGS)

clean: 
	rm -rf expected_force *.o

run_test: 
	echo "TO BE Implement"

.PHONY: clean all run_test
