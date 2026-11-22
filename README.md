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
_Filled by CI: see `results/bench.md` in the `bench-ubuntu-latest` artifact of the latest run._
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
