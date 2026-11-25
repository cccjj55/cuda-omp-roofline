/* GPU counterparts of the CPU kernels, same shapes and same accounting.
 *
 *   axpy     memory bound: 3 streams per element, so GB/s is the whole story
 *   reduce   sum reduction: two-stage (block partials + single-block finish) with warp shuffles
 *   softmax  max + sum + exp, numerically stable, two passes over the data
 *   matmul   shared-memory tiling, in float and in double (the ratio between them is a hardware fact,
 *            not a coding one: consumer GPUs ship a heavily cut FP64 pipe)
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
    } while (0)

__global__ void k_axpy(double a, const double* x, double* y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_reduce_partial(const double* x, double* out, long n) {
    __shared__ double s[256];
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        acc += x[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

__global__ void k_softmax(const double* in, double* out, long n) {
    /* one block, sequential passes; correct and simple, not fast for large n */
    extern __shared__ double sh[];
    double m = -INFINITY;
    for (long i = threadIdx.x; i < n; i += blockDim.x) m = fmax(m, in[i]);
    sh[threadIdx.x] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] = fmax(sh[threadIdx.x], sh[threadIdx.x + off]);
        __syncthreads();
    }
    const double mx = sh[0];
    __syncthreads();

    double s = 0.0;
    for (long i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = exp(in[i] - mx);
        out[i] = v;
        s += v;
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
        __syncthreads();
    }
    const double inv = 1.0 / sh[0];
    for (long i = threadIdx.x; i < n; i += blockDim.x) out[i] *= inv;
}


/* ---- grid-stride softmax: two-stage max, two-stage sum, one scale pass ----
 * The single-block version above is correct but serialises 1024 threads over the whole array.
 * This one walks the array once per phase with every block busy, so the cost is bandwidth, not latency. */
__global__ void k_max_partial(const double* in, double* part, long n) {
    __shared__ double s[256];
    double m = -INFINITY;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        m = fmax(m, in[i]);
    s[threadIdx.x] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}

__global__ void k_reduce_max(double* v, int count) {
    __shared__ double s[256];
    double m = -INFINITY;
    for (int i = threadIdx.x; i < count; i += blockDim.x) m = fmax(m, v[i]);
    __syncthreads();
    s[threadIdx.x] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0) v[0] = s[0];
}

__global__ void k_exp_partial(const double* in, double* out, double* part, double mx, long n) {
    __shared__ double s[256];
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const double v = exp(in[i] - mx);
        out[i] = v;
        acc += v;
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = s[0];
}

__global__ void k_reduce_sum(double* v, int count) {
    __shared__ double s[256];
    double acc = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) acc += v[i];
    __syncthreads();
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) v[0] = s[0];
}

__global__ void k_scale(double* out, double inv, long n) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
        out[i] *= inv;
}

/* host side: five kernels, two host synchronisations (max, then total) */
static void softmax_grid(const double* d_in, double* d_out, double* d_part, long n, int blocks) {
    double mx = 0.0, total = 0.0;
    k_max_partial<<<blocks, 256>>>(d_in, d_part, n);
    k_reduce_max<<<1, 256>>>(d_part, blocks);
    CHECK(cudaMemcpy(&mx, d_part, sizeof(double), cudaMemcpyDeviceToHost));
    k_exp_partial<<<blocks, 256>>>(d_in, d_out, d_part, mx, n);
    k_reduce_sum<<<1, 256>>>(d_part, blocks);
    CHECK(cudaMemcpy(&total, d_part, sizeof(double), cudaMemcpyDeviceToHost));
    k_scale<<<blocks, 256>>>(d_out, 1.0 / total, n);
}

template <typename T>
__global__ void k_matmul_tiled(const T* a, const T* b, T* c, int n) {
    __shared__ T ta[32][32];
    __shared__ T tb[32][32];
    const int row = blockIdx.y * 32 + threadIdx.y;
    const int col = blockIdx.x * 32 + threadIdx.x;
    T acc = 0;
    for (int t = 0; t < (n + 31) / 32; ++t) {
        const int ka = t * 32 + threadIdx.x;
        const int kb = t * 32 + threadIdx.y;
        ta[threadIdx.y][threadIdx.x] = (row < n && ka < n) ? a[row * n + ka] : T(0);
        tb[threadIdx.y][threadIdx.x] = (col < n && kb < n) ? b[kb * n + col] : T(0);
        __syncthreads();
        for (int k = 0; k < 32; ++k) acc += ta[threadIdx.y][k] * tb[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = acc;
}

static float ms_of(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

int main(int argc, char** argv) {
    const long n_axpy = argc > 1 ? atol(argv[1]) : (1L << 24);
    const long n_soft = argc > 2 ? atol(argv[2]) : (1L << 22);
    const int n_mm = argc > 3 ? atoi(argv[3]) : 512;

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("# device: %s sm_%d%d, %.1f GB\n", p.name, p.major, p.minor, p.totalGlobalMem / 1e9);
    printf("kernel,variant,n,ms,metric,unit,checksum\n");

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    /* ---- axpy: 3 streams x 8 bytes per element ---- */
    {
        double *x, *y;
        CHECK(cudaMalloc(&x, n_axpy * 8));
        CHECK(cudaMalloc(&y, n_axpy * 8));
        CHECK(cudaMemset(x, 0, n_axpy * 8));
        CHECK(cudaMemset(y, 0, n_axpy * 8));
        const int block = 256, grid = (int)((n_axpy + block - 1) / block);
        for (int w = 0; w < 5; ++w) k_axpy<<<grid, block>>>(0.5, x, y, n_axpy);
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        for (int r = 0; r < 20; ++r) k_axpy<<<grid, block>>>(0.5, x, y, n_axpy);
        cudaEventRecord(t1);
        CHECK(cudaEventSynchronize(t1));
        const float ms = ms_of(t0, t1) / 20.0f;
        const double gbps = 3.0 * n_axpy * 8.0 / (ms * 1e-3) / 1e9;
        printf("axpy,double,%ld,%.4f,%.2f,GB/s,0\n", n_axpy, ms, gbps);
        cudaFree(x); cudaFree(y);
    }

    /* ---- reduce ---- */
    {
        double* x;
        double* part;
        const int grid = 512, block = 256;
        CHECK(cudaMalloc(&x, n_axpy * 8));
        CHECK(cudaMalloc(&part, grid * 8));
        CHECK(cudaMemset(x, 0, n_axpy * 8));
        double* host = (double*)malloc(grid * 8);
        for (int w = 0; w < 5; ++w) k_reduce_partial<<<grid, block>>>(x, part, n_axpy);
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        for (int r = 0; r < 20; ++r) k_reduce_partial<<<grid, block>>>(x, part, n_axpy);
        cudaEventRecord(t1);
        CHECK(cudaEventSynchronize(t1));
        const float ms = ms_of(t0, t1) / 20.0f;
        CHECK(cudaMemcpy(host, part, grid * 8, cudaMemcpyDeviceToHost));
        double sum = 0.0;
        for (int i = 0; i < grid; ++i) sum += host[i];
        printf("reduce,double,%ld,%.4f,%.1f,GB/s,%.6f\n", n_axpy, ms, n_axpy * 8.0 / (ms * 1e-3) / 1e9, sum);
        free(host); cudaFree(x); cudaFree(part);
    }

    /* ---- softmax: single block (naive) vs grid-stride (fixed) ---- */
    {
        double *in, *out, *part;
        CHECK(cudaMalloc(&in, n_soft * 8));
        CHECK(cudaMalloc(&out, n_soft * 8));
        const int blocks = 512;
        CHECK(cudaMalloc(&part, blocks * 8));
        CHECK(cudaMemset(in, 0, n_soft * 8));

        const int block = 1024;
        for (int w = 0; w < 3; ++w) k_softmax<<<1, block, block * 8>>>(in, out, n_soft);
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        for (int r = 0; r < 10; ++r) k_softmax<<<1, block, block * 8>>>(in, out, n_soft);
        cudaEventRecord(t1);
        CHECK(cudaEventSynchronize(t1));
        const float ms1 = ms_of(t0, t1) / 10.0f;
        printf("softmax,1block,%ld,%.4f,%.4f,Gelem/s,0\n", n_soft, ms1, n_soft / (ms1 * 1e-3) / 1e9);

        for (int w = 0; w < 3; ++w) softmax_grid(in, out, part, n_soft, blocks);
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(t0);
        for (int r = 0; r < 10; ++r) softmax_grid(in, out, part, n_soft, blocks);
        cudaEventRecord(t1);
        CHECK(cudaEventSynchronize(t1));
        const float ms2 = ms_of(t0, t1) / 10.0f;
        printf("softmax,grid,%ld,%.4f,%.4f,Gelem/s,0\n", n_soft, ms2, n_soft / (ms2 * 1e-3) / 1e9);
        printf("# softmax speedup (grid vs 1block): %.2fx\n", ms1 / ms2);

        cudaFree(in); cudaFree(out); cudaFree(part);
    }

    /* ---- matmul, float and double ---- */
    for (int pass = 0; pass < 2; ++pass) {
        const size_t elems = (size_t)n_mm * n_mm;
        if (pass == 0) {
            float *a, *b, *c;
            CHECK(cudaMalloc(&a, elems * 4)); CHECK(cudaMalloc(&b, elems * 4)); CHECK(cudaMalloc(&c, elems * 4));
            CHECK(cudaMemset(a, 0, elems * 4)); CHECK(cudaMemset(b, 0, elems * 4));
            dim3 block(32, 32), grid((n_mm + 31) / 32, (n_mm + 31) / 32);
            for (int w = 0; w < 3; ++w) k_matmul_tiled<float><<<grid, block>>>(a, b, c, n_mm);
            CHECK(cudaDeviceSynchronize());
            cudaEventRecord(t0);
            for (int r = 0; r < 10; ++r) k_matmul_tiled<float><<<grid, block>>>(a, b, c, n_mm);
            cudaEventRecord(t1);
            CHECK(cudaEventSynchronize(t1));
            const float ms = ms_of(t0, t1) / 10.0f;
            const double flops = 2.0 * n_mm * n_mm * n_mm;
            printf("matmul,float,%d,%.4f,%.2f,GFLOP/s,0\n", n_mm, ms, flops / (ms * 1e-3) / 1e9);
            cudaFree(a); cudaFree(b); cudaFree(c);
        } else {
            double *a, *b, *c;
            CHECK(cudaMalloc(&a, elems * 8)); CHECK(cudaMalloc(&b, elems * 8)); CHECK(cudaMalloc(&c, elems * 8));
            CHECK(cudaMemset(a, 0, elems * 8)); CHECK(cudaMemset(b, 0, elems * 8));
            dim3 block(32, 32), grid((n_mm + 31) / 32, (n_mm + 31) / 32);
            for (int w = 0; w < 3; ++w) k_matmul_tiled<double><<<grid, block>>>(a, b, c, n_mm);
            CHECK(cudaDeviceSynchronize());
            cudaEventRecord(t0);
            for (int r = 0; r < 10; ++r) k_matmul_tiled<double><<<grid, block>>>(a, b, c, n_mm);
            cudaEventRecord(t1);
            CHECK(cudaEventSynchronize(t1));
            const float ms = ms_of(t0, t1) / 10.0f;
            const double flops = 2.0 * n_mm * n_mm * n_mm;
            printf("matmul,double,%d,%.4f,%.2f,GFLOP/s,0\n", n_mm, ms, flops / (ms * 1e-3) / 1e9);
            cudaFree(a); cudaFree(b); cudaFree(c);
        }
    }

    return 0;
}
