#include "kernels.h"

#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* xorshift64*: deterministic, no libc rand() differences between platforms. */
static uint64_t next_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 2685821657736338717ULL;
}

void fill_pseudo(double *x, size_t n, unsigned seed) {
    uint64_t state = 0x9E3779B97F4A7C15ULL ^ (uint64_t)seed;
    for (size_t i = 0; i < n; ++i) {
        /* map to [-1, 1) so the values stay well conditioned */
        x[i] = ((double)(next_u64(&state) >> 11) / 9007199254740992.0) * 2.0 - 1.0;
    }
}

void axpy_ref(double a, const double *x, double *y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
}

void axpy_omp(double a, const double *x, double *y, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
}

void softmax_ref(const double *in, double *out, size_t n) {
    double m = in[0];
    for (size_t i = 1; i < n; ++i) m = in[i] > m ? in[i] : m;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        out[i] = exp(in[i] - m);
        s += out[i];
    }
    const double inv = 1.0 / s;
    for (size_t i = 0; i < n; ++i) out[i] *= inv;
}

void softmax_omp(const double *in, double *out, size_t n) {
    double m = in[0];
#pragma omp parallel for schedule(static) reduction(max : m)
    for (size_t i = 1; i < n; ++i) m = in[i] > m ? in[i] : m;

    double s = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : s)
    for (size_t i = 0; i < n; ++i) {
        out[i] = exp(in[i] - m);
        s += out[i];
    }

    const double inv = 1.0 / s;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] *= inv;
}

void matmul_naive(const double *a, const double *b, double *c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < n; ++k) acc += a[i * n + k] * b[k * n + j];
            c[i * n + j] = acc;
        }
    }
}

/* Blocked matmul: keeps a block of B in cache while sweeping i, and the i-k block
 * of A in registers across the inner j loop. Same arithmetic, different order. */
void matmul_tiled(const double *a, const double *b, double *c, size_t n, size_t block) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) c[i * n + j] = 0.0;

    for (size_t ii = 0; ii < n; ii += block) {
        const size_t i_max = (ii + block < n) ? ii + block : n;
        for (size_t kk = 0; kk < n; kk += block) {
            const size_t k_max = (kk + block < n) ? kk + block : n;
            for (size_t jj = 0; jj < n; jj += block) {
                const size_t j_max = (jj + block < n) ? jj + block : n;
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        const double aik = a[i * n + k];
                        double *c_row = &c[i * n];
                        const double *b_row = &b[k * n];
                        for (size_t j = jj; j < j_max; ++j) c_row[j] += aik * b_row[j];
                    }
                }
            }
        }
    }
}

void matmul_tiled_omp(const double *a, const double *b, double *c, size_t n, size_t block) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) c[i * n + j] = 0.0;

#pragma omp parallel for schedule(dynamic) collapse(1)
    for (size_t ii = 0; ii < n; ii += block) {
        const size_t i_max = (ii + block < n) ? ii + block : n;
        for (size_t kk = 0; kk < n; kk += block) {
            const size_t k_max = (kk + block < n) ? kk + block : n;
            for (size_t jj = 0; jj < n; jj += block) {
                const size_t j_max = (jj + block < n) ? jj + block : n;
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        const double aik = a[i * n + k];
                        double *c_row = &c[i * n];
                        const double *b_row = &b[k * n];
                        for (size_t j = jj; j < j_max; ++j) c_row[j] += aik * b_row[j];
                    }
                }
            }
        }
    }
}
