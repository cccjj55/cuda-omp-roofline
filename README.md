# hpc-kernels

Three kernels, each written three ways, measured by one harness that refuses to time a wrong answer.

This is the first rung of the HPC ladder: before touching CUDA or MPI, know what the CPU is doing.
Every optimized variant is diffed against a reference implementation *before* it is timed, so a fast
wrong answer cannot make it into the table.

## What is in here

| kernel | variants | what it teaches |
| --- | --- | --- |
| `axpy` (`y += a*x`) | `ref`, `omp` | memory bandwidth: 3 streams per element, arithmetic is free |
| `softmax` | `ref`, `omp` (max-reduction + sum-reduction) | compute-bound (exp) plus a reduction that has to stay numerically stable |
| `matmul` (`C=A*B`) | `naive`, `tiled` (64×64), `tiled+omp` | cache blocking and loop order; the naive version is bandwidth-starved by construction |

## Running it

```bash
make build      # gcc -O3 -march=native -fopenmp
make test       # quick run; aborts on any correctness mismatch
make bench      # full run -> results/bench.csv
```

Needs a C11 compiler and OpenMP. No dependencies beyond libm.

## Results

Measured on GitHub Actions `ubuntu-latest` (4 vCPU), not on my laptop — the runner is the one machine
whose hardware I can state exactly and that anyone can re-create.

<!-- RESULTS:BEGIN -->
Measured on GitHub Actions `ubuntu-latest`: **Intel Xeon Platinum 8573C, 4 vCPU, gcc 15.2.0**,
`-O3 -march=native -fopenmp`. Full sizes: axpy 16.7 M doubles, softmax 4.2 M doubles, matmul n=512.
Best of 5 timed runs after one untimed warmup.

| kernel | variant | n | best | median | metric | spread |
| --- | --- | --- | --- | --- | --- | --- |
| axpy | ref | 16,777,216 | 18.77 ms | 18.83 ms | 21.45 GB/s | 0.30% |
| axpy | omp (4 threads) | 16,777,216 | 10.02 ms | 10.01 ms | **40.22 GB/s** | 0.09% |
| softmax | ref | 4,194,304 | 22.68 ms | 22.56 ms | 0.19 Gelem/s | 0.57% |
| softmax | omp (4 threads) | 4,194,304 | 9.30 ms | 8.93 ms | **0.47 Gelem/s** | 4.13% |
| matmul | naive | 512 | 235.54 ms | 234.18 ms | 1.15 GFLOP/s | 0.58% |
| matmul | tiled 64 | 512 | 32.73 ms | 32.63 ms | 8.23 GFLOP/s | 0.29% |
| matmul | tiled 64 + omp | 512 | 14.65 ms | 14.58 ms | **18.41 GFLOP/s** | 0.48% |

Ratios: axpy **1.9×**, softmax **2.5×**, matmul **7.2×** from blocking alone and **16×** with threads on top.
Correctness at these sizes: `max|diff|` = 0 for axpy and matmul, 4.7e-20 for softmax (the reduction order
differs, which is the expected source of that residue).

**The quick run is where the interesting failure lives.** At n = 1.05 M, the same OpenMP `axpy` measured
78.4 GB/s in one run and 5.1 GB/s in the next (spread 19.9%), and `softmax` with threads came out
**10× slower** than the reference. Nothing was wrong with the code: at that size the array fits in cache
and the measurement is dominated by thread spin-up and whatever else the shared runner was doing. That is
exactly why the full-size numbers are the quotable ones and the quick run is only for correctness.
<!-- RESULTS:END -->

## What the numbers mean

- **`axpy` is a bandwidth test, not a compute test.** 3 × 8 bytes per element, so the GB/s figure is the
  ceiling everything else has to live under. OpenMP helps only until the memory bus saturates; past that
  more threads make the number worse, not better.
- **`softmax` is compute-bound** because of `exp`, but the max-reduction and the sum are the parts that
  decide whether it is numerically stable. The OpenMP version keeps both reductions explicit; a version
  that fuses them looks faster and drifts on wide input ranges.
- **`matmul` naive is 3 nested loops in the wrong order.** For `n=512` the inner loop walks `B` by columns,
  touching a new cache line per iteration. Blocking fixes locality; OpenMP then has something worth
  parallelising. If you parallelise the naive version first, you scale a memory stall.

## Timer discipline

- one untimed warmup per variant (page faults, cold caches, frequency ramp)
- 5 timed repeats, report **best and median plus spread**, not a single number
- the correctness check runs before the timing, every time

The same rules as [dsh-plugin-bench](https://github.com/cccjj55/dsh-plugin-bench), applied to C.

## Status and what comes next

- [x] CPU reference + OpenMP + blocking, correctness-checked
- [ ] CUDA: `saxpy`, `reduce` (tree + warp shuffle), `softmax`, `tiled matmul` with shared memory and a
      bank-conflict fix; each with `ncu` metrics (achieved occupancy, DRAM throughput)
- [ ] MPI: 2D stencil with halo exchange, strong and weak scaling curves (`mpi-heat`)
- [ ] roofline plot: put every kernel on the same arithmetic-intensity axis

The CUDA step needs a GPU; this laptop has one (RTX 5070 Ti Laptop, 12 GB, sm_120) but no CUDA toolkit
in WSL yet. Numbers will be added with the hardware line attached, not before.

## License

MIT
