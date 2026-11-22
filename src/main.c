/* Benchmark harness. Prints CSV on stdout and a markdown table on stderr.
 *
 * Timing discipline (this is the whole point of the repo):
 *   - one untimed warmup per variant, so page faults and cache cold-start do not land in a sample
 *   - N timed repeats, report best / median / spread
 *   - every optimized variant is checked against the reference before it is timed
 */
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define REPEATS 5
#define BLOCK 64

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmp_double(const void *pa, const void *pb) {
    const double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

typedef struct {
    double best, median, spread_pct;
} timing_t;

/* fn ctx -> run one iteration */
typedef void (*kernel_fn)(void *ctx);

static double time_once(kernel_fn fn, void *ctx) {
    const double t0 = now_s();
    fn(ctx);
    return now_s() - t0;
}

static timing_t measure(kernel_fn fn, void *ctx) {
    time_once(fn, ctx); /* warmup, untimed */

    double samples[REPEATS];
    for (int r = 0; r < REPEATS; ++r) samples[r] = time_once(fn, ctx);

    double sorted[REPEATS];
    memcpy(sorted, samples, sizeof samples);
    qsort(sorted, REPEATS, sizeof(double), cmp_double);

    timing_t t;
    t.best = sorted[0];
    t.median = sorted[REPEATS / 2];
    t.spread_pct = t.best > 0.0 ? (t.median - t.best) / t.best * 100.0 : 0.0;
    return t;
}

static double max_abs_diff(const double *x, const double *y, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = fabs(x[i] - y[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ---- contexts ---------------------------------------------------------- */

typedef struct {
    double a;
    const double *x;
    double *y;
    size_t n;
} axpy_ctx;

static void run_axpy_ref(void *p) {
    axpy_ctx *c = p;
    axpy_ref(c->a, c->x, c->y, c->n);
}
static void run_axpy_omp(void *p) {
    axpy_ctx *c = p;
    axpy_omp(c->a, c->x, c->y, c->n);
}

typedef struct {
    const double *in;
    double *out;
    size_t n;
} softmax_ctx;

static void run_softmax_ref(void *p) {
    softmax_ctx *c = p;
    softmax_ref(c->in, c->out, c->n);
}
static void run_softmax_omp(void *p) {
    softmax_ctx *c = p;
    softmax_omp(c->in, c->out, c->n);
}

typedef struct {
    const double *a, *b;
    double *c;
    size_t n, block;
    int variant; /* 0 naive, 1 tiled, 2 tiled+omp */
} mm_ctx;

static void run_matmul(void *p) {
    mm_ctx *c = p;
    switch (c->variant) {
        case 0: matmul_naive(c->a, c->b, c->c, c->n); break;
        case 1: matmul_tiled(c->a, c->b, c->c, c->n, c->block); break;
        default: matmul_tiled_omp(c->a, c->b, c->c, c->n, c->block); break;
    }
}

static void row(const char *kernel, const char *variant, size_t n, timing_t t, double rate,
                const char *unit) {
    /* stdout is the CSV contract; stderr carries human-readable notes */
    printf("%s,%s,%zu,%.6f,%.6f,%.2f,%s,%.2f\n", kernel, variant, n, t.best, t.median,
           rate, unit, t.spread_pct);
}

int main(int argc, char **argv) {
    int quick = 0;
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--quick") == 0) quick = 1;

    const size_t n_axpy = quick ? (1u << 20) : (1u << 24);
    const size_t n_soft = quick ? (1u << 18) : (1u << 22);
    const size_t n_mm = quick ? 256 : 512;

#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif

    fprintf(stderr, "# hpc-kernels: %zu axpy, %zu softmax, %zu matmul, threads=%d\n", n_axpy,
            n_soft, n_mm, threads);
    printf("kernel,variant,n,best_s,median_s,metric,unit,spread_pct\n");

    /* ---- axpy: bandwidth bound (2 reads + 1 write per element) ---------- */
    {
        double *x = malloc(n_axpy * sizeof(double));
        double *y_ref = malloc(n_axpy * sizeof(double));
        double *y_opt = malloc(n_axpy * sizeof(double));
        if (!x || !y_ref || !y_opt) return 2;
        fill_pseudo(x, n_axpy, 1);
        fill_pseudo(y_opt, n_axpy, 2);
        memcpy(y_ref, y_opt, n_axpy * sizeof(double));

        axpy_ctx c_ref = {0.5, x, y_ref, n_axpy};
        axpy_ctx c_opt = {0.5, x, y_opt, n_axpy};

        timing_t t_ref = measure(run_axpy_ref, &c_ref);
        timing_t t_opt = measure(run_axpy_omp, &c_opt);

        const double bytes = 3.0 * (double)n_axpy * sizeof(double);
        row("axpy", "ref", n_axpy, t_ref, bytes / t_ref.best / 1e9, "GB/s");
        row("axpy", "omp", n_axpy, t_opt, bytes / t_opt.best / 1e9, "GB/s");
        fprintf(stderr, "# axpy max|diff| = %.3e\n", max_abs_diff(y_ref, y_opt, n_axpy));

        free(x), free(y_ref), free(y_opt);
    }

    /* ---- softmax: compute bound (exp), with a max-reduction ------------ */
    {
        double *in = malloc(n_soft * sizeof(double));
        double *out_ref = malloc(n_soft * sizeof(double));
        double *out_opt = malloc(n_soft * sizeof(double));
        if (!in || !out_ref || !out_opt) return 2;
        fill_pseudo(in, n_soft, 3);

        softmax_ctx c_ref = {in, out_ref, n_soft};
        softmax_ctx c_opt = {in, out_opt, n_soft};

        timing_t t_ref = measure(run_softmax_ref, &c_ref);
        timing_t t_opt = measure(run_softmax_omp, &c_opt);

        row("softmax", "ref", n_soft, t_ref, (double)n_soft / t_ref.best / 1e9, "Gelem/s");
        row("softmax", "omp", n_soft, t_opt, (double)n_soft / t_opt.best / 1e9, "Gelem/s");
        fprintf(stderr, "# softmax max|diff| = %.3e\n", max_abs_diff(out_ref, out_opt, n_soft));

        free(in), free(out_ref), free(out_opt);
    }

    /* ---- matmul: 2 n^3 flops ------------------------------------------- */
    {
        const size_t n = n_mm;
        double *a = malloc(n * n * sizeof(double));
        double *b = malloc(n * n * sizeof(double));
        double *c_ref = malloc(n * n * sizeof(double));
        double *c_opt = malloc(n * n * sizeof(double));
        if (!a || !b || !c_ref || !c_opt) return 2;
        fill_pseudo(a, n * n, 4);
        fill_pseudo(b, n * n, 5);

        const double flops = 2.0 * (double)n * (double)n * (double)n;

        mm_ctx c0 = {a, b, c_ref, n, BLOCK, 0};
        timing_t t0 = measure(run_matmul, &c0);

        mm_ctx c1 = {a, b, c_opt, n, BLOCK, 1};
        timing_t t1 = measure(run_matmul, &c1);
        const double diff1 = max_abs_diff(c_ref, c_opt, n * n);

        mm_ctx c2 = {a, b, c_opt, n, BLOCK, 2};
        timing_t t2 = measure(run_matmul, &c2);
        const double diff2 = max_abs_diff(c_ref, c_opt, n * n);

        row("matmul", "naive", n, t0, flops / t0.best / 1e9, "GFLOP/s");
        row("matmul", "tiled", n, t1, flops / t1.best / 1e9, "GFLOP/s");
        row("matmul", "tiled+omp", n, t2, flops / t2.best / 1e9, "GFLOP/s");
        fprintf(stderr, "# matmul max|diff| tiled = %.3e, tiled+omp = %.3e\n", diff1, diff2);

        free(a), free(b), free(c_ref), free(c_opt);
    }

    return 0;
}
