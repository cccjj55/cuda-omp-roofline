# hpc-kernels

Three kernels, each written three ways, measured by one harness that refuses to time a wrong answer.

This is the first rung of the HPC ladder: before touching CUDA or MPI, know what the CPU is doing.
Every optimized variant is diffed against a reference implementation *before* it is timed, so a fast
wrong answer cannot make it into the table.

![hpc-kernels Cache Blocking & OpenMP Scaling](docs/kernels.svg)

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
Measured in CI on GitHub Actions `ubuntu-latest`: **Intel Xeon Platinum 8573C, 4 vCPU, gcc 15.2.0**,
`-O3 -march=native -fopenmp`, 4 threads. Sizes: axpy 16.7 M doubles, softmax 4.2 M doubles, matmul n=512.
Best and median of 5 timed runs after one untimed warmup. Two separate CI runs are shown because the
difference between them is the most useful thing on this page.

| kernel | variant | run A (metric) | run B (metric) | run A spread | run B spread |
| --- | --- | --- | --- | --- | --- |
| axpy | ref | 21.45 GB/s | 62.77 GB/s | 0.30% | 0.98% |
| axpy | omp | 40.22 GB/s | 72.96 GB/s | 0.09% | **47.17%** |
| softmax | ref | 0.19 Gelem/s | 0.28 Gelem/s | 0.57% | 0.50% |
| softmax | omp | 0.47 Gelem/s | 0.88 Gelem/s | 4.13% | 0.88% |
| matmul | naive | 1.15 GFLOP/s | 0.36 GFLOP/s | 0.58% | 0.48% |
| matmul | tiled 64 | 8.23 GFLOP/s | 12.35 GFLOP/s | 0.29% | 0.16% |
| matmul | tiled 64 + omp | 18.41 GFLOP/s | 30.08 GFLOP/s | 0.48% | 0.12% |

Correctness at these sizes: `max|diff|` = 0 for axpy and matmul, 4.7e-20 for softmax (different reduction
order; that residue is expected, not a bug).

**Run C: this laptop, not a runner.** WSL2 Ubuntu 24.04, `i7-14650HX`, 12 threads visible to OpenMP,
gcc 16.2.0 (conda-forge). Same binary, same sizes:

| kernel | variant | metric | spread |
| --- | --- | --- | --- |
| axpy | ref | 31.94 GB/s | 9.82% |
| axpy | omp | **58.35 GB/s** | 5.77% |
| softmax | ref | 0.26 Gelem/s | 1.33% |
| softmax | omp | **1.20 Gelem/s** | 12.37% |
| matmul | naive | 2.28 GFLOP/s | 0.51% |
| matmul | tiled 64 | 16.05 GFLOP/s | 0.93% |
| matmul | tiled 64 + omp | **70.62 GFLOP/s** | 3.81% |

Same conclusions, different absolute numbers: blocking gives ~7×, threads on top give another 4.4× here versus
2.2× on the 4-vCPU runner. **Twelve cores change the ratio, not the ranking** — which is the reason every row
carries a hardware line.

**What is stable across runs and what is not**

- **Stable: tiled → tiled+omp is about 2.2–2.4×.** Both runs agree. Threads on top of blocking is a real,
  repeatable gain.
- **Stable: softmax with threads is 2.5–3.1×.** The work is compute-bound, so scaling survives the shared
  machine.
- **Not stable: everything measured against `naive`.** Blocking looks like 7× in run A and 34× in run B —
  not because blocking changed, but because the naive baseline moved by 3× (1.15 → 0.36 GFLOP/s). A speedup
  whose denominator is noisy is not a speedup you can quote.
- **Not quotable: `axpy omp` in run B (47% spread).** The best sample was 5.52 ms and the median 8.12 ms:
  that row says "the runner was busy", nothing about the kernel.

**The quick run is worse, on purpose.** At n = 1.05 M the same OpenMP `axpy` measured 78.4 GB/s in one run and
5.1 GB/s in the next, and threaded `softmax` came out 10× *slower* than the reference. At that size the array
fits in cache and thread spin-up dominates. The quick run exists to check correctness, never to produce a number.

This is why the repo reports best, median and spread instead of a single figure: on a shared runner, a lone
number is mostly a description of the runner.
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
