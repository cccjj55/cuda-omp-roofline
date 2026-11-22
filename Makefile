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
