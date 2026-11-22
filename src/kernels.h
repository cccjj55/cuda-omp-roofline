/* Reference and optimized kernels, CPU side.
 *
 * Everything here is deliberately boring C11. The point is not clever code; it is
 * that every variant is measured against the same reference and the same timer,
 * so the numbers in the README can be reproduced by anyone with a compiler.
 */
#ifndef HPC_KERNELS_H
#define HPC_KERNELS_H

#include <stddef.h>

/* y[i] = a * x[i] + y[i] */
void axpy_ref(double a, const double *x, double *y, size_t n);
void axpy_omp(double a, const double *x, double *y, size_t n);

/* out[i] = exp(in[i] - max) / sum */
void softmax_ref(const double *in, double *out, size_t n);
void softmax_omp(const double *in, double *out, size_t n);

/* C = A * B, all row-major n x n */
void matmul_naive(const double *a, const double *b, double *c, size_t n);
void matmul_tiled(const double *a, const double *b, double *c, size_t n, size_t block);
void matmul_tiled_omp(const double *a, const double *b, double *c, size_t n, size_t block);

/* Deterministic pseudo-random fill: same input on every machine. */
void fill_pseudo(double *x, size_t n, unsigned seed);

#endif /* HPC_KERNELS_H */
