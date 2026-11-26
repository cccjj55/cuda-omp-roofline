CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -Wall -Wextra -std=c11
LDFLAGS ?= -fopenmp -lm

SRC := src/kernels.c src/main.c
BIN := build/bench

.PHONY: all build test bench clean

all: build

build: $(BIN)

$(BIN): $(SRC) src/kernels.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SRC) -o $(BIN) $(LDFLAGS)

# correctness first: every optimized variant is diffed against the reference
test: build
	./$(BIN) --quick > build/quick.csv
	@echo "--- quick run ---"
	@cat build/quick.csv

bench: test
	@mkdir -p results
	./$(BIN) > results/bench.csv
	@echo "--- full run ---"
	@cat results/bench.csv

clean:
	rm -rf build results

NVCC ?= nvcc
CUDA_ARCH ?= sm_120
HOSTCXX ?= g++

cuda:
	@mkdir -p build
	$(NVCC) -O3 -Xcompiler -O3 -arch=$(CUDA_ARCH) -ccbin $(HOSTCXX) src/kernels_cuda.cu -o build/kernels_cuda

gpu-bench: cuda
	./build/kernels_cuda 16777216 1048576 512
