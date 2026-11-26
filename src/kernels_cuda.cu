/* GPU counterparts of the CPU kernels, verified against CPU references before reporting timings.
 *
 *   axpy     memory bound: 3 streams per element, so GB/s is the whole story
 *   reduce   sum reduction: two-stage (warp shuffle + block partials + finish)
 *   softmax  max + sum + exp, numerically stable two-pass (1-block naive vs grid-stride)
 *   matmul   shared-memory 32x32 tiling, in float and in double (reveals cut FP64 pipe on consumer GPUs)
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(call)                                                                        \
    do {                                                                                   \
        cudaError_t e = (call);                                                            \
        if (e != cudaSuccess) {                                                            \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));      \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int off = 16; off > 0; off >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, off);
    }
    return val;
}

__device__ __forceinline__ double warp_reduce_max(double val) {
    for (int off = 16; off > 0; off >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffffu, val, off));
    }
    return val;
}

__device__ __forceinline__ double block_reduce_sum(double val) {
    __shared__ double warp_vals[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? warp_vals[threadIdx.x] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__device__ __forceinline__ double block_reduce_max(double val) {
    __shared__ double warp_vals[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) warp_vals[wid] = val;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nwarps) ? warp_vals[threadIdx.x] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    return val;
}

__global__ void k_axpy(double a, const double* x, double* y, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_reduce_partial(const double* x, double* out, long n) {
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        acc += x[i];
    }
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

__global__ void k_softmax(const double* in, double* out, long n) {
    /* single-block baseline: correct, but 1024 threads walk n sequentially */
    __shared__ double s_mx;
    __shared__ double s_sum;
    double m = -INFINITY;
    for (long i = threadIdx.x; i < n; i += blockDim.x) m = fmax(m, in[i]);
    m = block_reduce_max(m);
    if (threadIdx.x == 0) s_mx = m;
    __syncthreads();
    const double mx = s_mx;

    double s = 0.0;
    for (long i = threadIdx.x; i < n; i += blockDim.x) {
        const double v = exp(in[i] - mx);
        out[i] = v;
        s += v;
    }
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) s_sum = s;
    __syncthreads();
    const double inv = 1.0 / s_sum;
    for (long i = threadIdx.x; i < n; i += blockDim.x) out[i] *= inv;
}

__global__ void k_max_partial(const double* in, double* part, long n) {
    double m = -INFINITY;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        m = fmax(m, in[i]);
    }
    m = block_reduce_max(m);
    if (threadIdx.x == 0) part[blockIdx.x] = m;
}

__global__ void k_reduce_max(const double* v, double* out_scalar, int count) {
    double m = -INFINITY;
    for (int i = threadIdx.x; i < count; i += blockDim.x) m = fmax(m, v[i]);
    m = block_reduce_max(m);
    if (threadIdx.x == 0) *out_scalar = m;
}

__global__ void k_exp_partial(const double* in, double* out, double* part, const double* mx_ptr, long n) {
    const double mx = *mx_ptr;
    double acc = 0.0;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        const double v = exp(in[i] - mx);
        out[i] = v;
        acc += v;
    }
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) part[blockIdx.x] = acc;
}

__global__ void k_reduce_sum(const double* v, double* out_scalar, int count) {
    double acc = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) acc += v[i];
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) *out_scalar = acc;
}

__global__ void k_scale(double* out, const double* sum_ptr, long n) {
    const double inv = 1.0 / (*sum_ptr);
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        out[i] *= inv;
    }
}

static void softmax_grid(const double* d_in, double* d_out, double* d_part, double* d_scalars,
                         long n, int blocks) {
    double* d_mx = d_scalars + 0;
    double* d_sum = d_scalars + 1;
    k_max_partial<<<blocks, 256>>>(d_in, d_part, n);
    k_reduce_max<<<1, 256>>>(d_part, d_mx, blocks);
    k_exp_partial<<<blocks, 256>>>(d_in, d_out, d_part, d_mx, n);
    k_reduce_sum<<<1, 256>>>(d_part, d_sum, blocks);
    k_scale<<<blocks, 256>>>(d_out, d_sum, n);
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

static void check_tol(const char* name, double maxdiff, double tol) {
    if (std::isnan(maxdiff) || maxdiff > tol) {
        fprintf(stderr, "FAIL: %s maxdiff=%.3e exceeds tol=%.3e\n", name, maxdiff, tol);
        exit(2);
    }
}

int main(int argc, char** argv) {
    const long n_axpy = argc > 1 ? atol(argv[1]) : (1L << 24);
    const long n_soft = argc > 2 ? atol(argv[2]) : (1L << 22);
    const int n_mm = argc > 3 ? atoi(argv[3]) : 512;

    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));
    printf("# device: %s sm_%d%d, %.1f GB\n", p.name, p.major, p.minor, p.totalGlobalMem / 1e9);
    printf("kernel,variant,n,ms,metric,unit,maxdiff\n");

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    /* ---- axpy: non-zero deterministic input + CPU reference check ---- */
    {
        const double alpha = 0.375;
        double* hx = (double*)malloc((size_t)n_axpy * sizeof(double));
        double* hy = (double*)malloc((size_t)n_axpy * sizeof(double));
        double* h_gpu = (double*)malloc((size_t)n_axpy * sizeof(double));
        for (long i = 0; i < n_axpy; ++i) {
            hx[i] = 0.5 + 0.25 * sin(0.001 * (double)(i & 4095));
            hy[i] = 1.0 - 0.125 * cos(0.002 * (double)(i & 4095));
        }

        double *dx, *dy;
        CHECK(cudaMalloc(&dx, (size_t)n_axpy * sizeof(double)));
        CHECK(cudaMalloc(&dy, (size_t)n_axpy * sizeof(double)));
        CHECK(cudaMemcpy(dx, hx, (size_t)n_axpy * sizeof(double), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dy, hy, (size_t)n_axpy * sizeof(double), cudaMemcpyHostToDevice));

        const int block = 256, grid = (int)((n_axpy + block - 1) / block);
        k_axpy<<<grid, block>>>(alpha, dx, dy, n_axpy);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_gpu, dy, (size_t)n_axpy * sizeof(double), cudaMemcpyDeviceToHost));

        double maxdiff = 0.0;
        for (long i = 0; i < n_axpy; ++i) {
            const double ref = hy[i] + alpha * hx[i];
            const double d = fabs(h_gpu[i] - ref);
            if (d > maxdiff) maxdiff = d;
        }
        check_tol("axpy", maxdiff, 1e-13);

        for (int w = 0; w < 5; ++w) k_axpy<<<grid, block>>>(alpha, dx, dy, n_axpy);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 20; ++r) k_axpy<<<grid, block>>>(alpha, dx, dy, n_axpy);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        const float ms = ms_of(t0, t1) / 20.0f;
        const double gbps = 3.0 * (double)n_axpy * 8.0 / (ms * 1e-3) / 1e9;
        printf("axpy,double,%ld,%.4f,%.2f,GB/s,%.3e\n", n_axpy, ms, gbps, maxdiff);

        /* ---- reduce: reuse non-zero hx ---- */
        const int rgrid = 512;
        double* dpart;
        CHECK(cudaMalloc(&dpart, (size_t)rgrid * sizeof(double)));
        double* hpart = (double*)malloc((size_t)rgrid * sizeof(double));

        double cpu_sum = 0.0, comp = 0.0;
        for (long i = 0; i < n_axpy; ++i) {
            const double yv = hx[i] - comp;
            const double tv = cpu_sum + yv;
            comp = (tv - cpu_sum) - yv;
            cpu_sum = tv;
        }

        k_reduce_partial<<<rgrid, block>>>(dx, dpart, n_axpy);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hpart, dpart, (size_t)rgrid * sizeof(double), cudaMemcpyDeviceToHost));
        double gpu_sum = 0.0;
        for (int i = 0; i < rgrid; ++i) gpu_sum += hpart[i];
        const double rdiff = fabs(gpu_sum - cpu_sum) / fmax(1.0, fabs(cpu_sum));
        check_tol("reduce", rdiff, 1e-11);

        for (int w = 0; w < 5; ++w) k_reduce_partial<<<rgrid, block>>>(dx, dpart, n_axpy);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 20; ++r) k_reduce_partial<<<rgrid, block>>>(dx, dpart, n_axpy);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        const float rms = ms_of(t0, t1) / 20.0f;
        printf("reduce,double,%ld,%.4f,%.1f,GB/s,%.3e\n", n_axpy, rms,
               (double)n_axpy * 8.0 / (rms * 1e-3) / 1e9, rdiff);

        free(hpart);
        cudaFree(dpart);
        cudaFree(dx);
        cudaFree(dy);
        free(hx);
        free(hy);
        free(h_gpu);
    }

    /* ---- softmax: single block (naive) vs grid-stride (fixed) ---- */
    {
        double* hin = (double*)malloc((size_t)n_soft * sizeof(double));
        double* href = (double*)malloc((size_t)n_soft * sizeof(double));
        double* hgpu = (double*)malloc((size_t)n_soft * sizeof(double));

        double mx = -INFINITY;
        for (long i = 0; i < n_soft; ++i) {
            hin[i] = 3.0 * sin(0.003 * (double)(i & 8191));
            if (hin[i] > mx) mx = hin[i];
        }
        double sum = 0.0;
        for (long i = 0; i < n_soft; ++i) {
            href[i] = exp(hin[i] - mx);
            sum += href[i];
        }
        for (long i = 0; i < n_soft; ++i) href[i] /= sum;

        double *din, *dout, *dpart, *dscalars;
        const int blocks = 512;
        CHECK(cudaMalloc(&din, (size_t)n_soft * sizeof(double)));
        CHECK(cudaMalloc(&dout, (size_t)n_soft * sizeof(double)));
        CHECK(cudaMalloc(&dpart, (size_t)blocks * sizeof(double)));
        CHECK(cudaMalloc(&dscalars, 2 * sizeof(double)));
        CHECK(cudaMemcpy(din, hin, (size_t)n_soft * sizeof(double), cudaMemcpyHostToDevice));

        k_softmax<<<1, 1024>>>(din, dout, n_soft);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hgpu, dout, (size_t)n_soft * sizeof(double), cudaMemcpyDeviceToHost));
        double diff1 = 0.0;
        for (long i = 0; i < n_soft; ++i) {
            const double d = fabs(hgpu[i] - href[i]);
            if (d > diff1) diff1 = d;
        }
        check_tol("softmax_1block", diff1, 1e-13);

        for (int w = 0; w < 3; ++w) k_softmax<<<1, 1024>>>(din, dout, n_soft);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 10; ++r) k_softmax<<<1, 1024>>>(din, dout, n_soft);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        const float ms1 = ms_of(t0, t1) / 10.0f;
        printf("softmax,1block,%ld,%.4f,%.4f,Gelem/s,%.3e\n", n_soft, ms1,
               n_soft / (ms1 * 1e-3) / 1e9, diff1);

        softmax_grid(din, dout, dpart, dscalars, n_soft, blocks);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hgpu, dout, (size_t)n_soft * sizeof(double), cudaMemcpyDeviceToHost));
        double diff2 = 0.0;
        for (long i = 0; i < n_soft; ++i) {
            const double d = fabs(hgpu[i] - href[i]);
            if (d > diff2) diff2 = d;
        }
        check_tol("softmax_grid", diff2, 1e-13);

        for (int w = 0; w < 3; ++w) softmax_grid(din, dout, dpart, dscalars, n_soft, blocks);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 10; ++r) softmax_grid(din, dout, dpart, dscalars, n_soft, blocks);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        const float ms2 = ms_of(t0, t1) / 10.0f;
        printf("softmax,grid,%ld,%.4f,%.4f,Gelem/s,%.3e\n", n_soft, ms2,
               n_soft / (ms2 * 1e-3) / 1e9, diff2);
        printf("# softmax speedup (grid vs 1block): %.2fx\n", ms1 / ms2);

        cudaFree(din); cudaFree(dout); cudaFree(dpart); cudaFree(dscalars);
        free(hin); free(href); free(hgpu);
    }

    /* ---- matmul, float and double with full CPU reference verification ---- */
    const size_t elems = (size_t)n_mm * n_mm;
    double* ha_d = (double*)malloc(elems * sizeof(double));
    double* hb_d = (double*)malloc(elems * sizeof(double));
    double* hc_d = (double*)malloc(elems * sizeof(double));
    double* href_d = (double*)malloc(elems * sizeof(double));
    float* ha_f = (float*)malloc(elems * sizeof(float));
    float* hb_f = (float*)malloc(elems * sizeof(float));
    float* hc_f = (float*)malloc(elems * sizeof(float));

    for (int i = 0; i < n_mm; ++i) {
        for (int j = 0; j < n_mm; ++j) {
            const double va = sin(0.01 * (double)(i + 1)) * cos(0.02 * (double)(j + 1));
            const double vb = cos(0.015 * (double)(i + 1)) + 0.5 * sin(0.01 * (double)(j + 1));
            ha_d[(size_t)i * n_mm + j] = va;
            hb_d[(size_t)i * n_mm + j] = vb;
            ha_f[(size_t)i * n_mm + j] = (float)va;
            hb_f[(size_t)i * n_mm + j] = (float)vb;
        }
    }
    for (int i = 0; i < n_mm; ++i) {
        for (int j = 0; j < n_mm; ++j) {
            double s = 0.0;
            for (int k = 0; k < n_mm; ++k) {
                s += ha_d[(size_t)i * n_mm + k] * hb_d[(size_t)k * n_mm + j];
            }
            href_d[(size_t)i * n_mm + j] = s;
        }
    }

    {
        float *a, *b, *c;
        CHECK(cudaMalloc(&a, elems * 4)); CHECK(cudaMalloc(&b, elems * 4)); CHECK(cudaMalloc(&c, elems * 4));
        CHECK(cudaMemcpy(a, ha_f, elems * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b, hb_f, elems * 4, cudaMemcpyHostToDevice));
        dim3 block(32, 32), grid((n_mm + 31) / 32, (n_mm + 31) / 32);
        k_matmul_tiled<float><<<grid, block>>>(a, b, c, n_mm);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hc_f, c, elems * 4, cudaMemcpyDeviceToHost));
        double mdiff_f = 0.0;
        for (size_t idx = 0; idx < elems; ++idx) {
            const double d = fabs((double)hc_f[idx] - href_d[idx]);
            if (d > mdiff_f) mdiff_f = d;
        }
        check_tol("matmul_float", mdiff_f, 1e-3);

        for (int w = 0; w < 3; ++w) k_matmul_tiled<float><<<grid, block>>>(a, b, c, n_mm);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 10; ++r) k_matmul_tiled<float><<<grid, block>>>(a, b, c, n_mm);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        const float ms = ms_of(t0, t1) / 10.0f;
        const double flops = 2.0 * n_mm * n_mm * n_mm;
        printf("matmul,float,%d,%.4f,%.2f,GFLOP/s,%.3e\n", n_mm, ms, flops / (ms * 1e-3) / 1e9, mdiff_f);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }

    {
        double *a, *b, *c;
        CHECK(cudaMalloc(&a, elems * 8)); CHECK(cudaMalloc(&b, elems * 8)); CHECK(cudaMalloc(&c, elems * 8));
        CHECK(cudaMemcpy(a, ha_d, elems * 8, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(b, hb_d, elems * 8, cudaMemcpyHostToDevice));
        dim3 block(32, 32), grid((n_mm + 31) / 32, (n_mm + 31) / 32);
        k_matmul_tiled<double><<<grid, block>>>(a, b, c, n_mm);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(hc_d, c, elems * 8, cudaMemcpyDeviceToHost));
        double mdiff_d = 0.0;
        for (size_t idx = 0; idx < elems; ++idx) {
            const double d = fabs(hc_d[idx] - href_d[idx]);
            if (d > mdiff_d) mdiff_d = d;
        }
        check_tol("matmul_double", mdiff_d, 1e-11);

        for (int w = 0; w < 3; ++w) k_matmul_tiled<double><<<grid, block>>>(a, b, c, n_mm);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(t0));
        for (int r = 0; r < 10; ++r) k_matmul_tiled<double><<<grid, block>>>(a, b, c, n_mm);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        const float ms = ms_of(t0, t1) / 10.0f;
        const double flops = 2.0 * n_mm * n_mm * n_mm;
        printf("matmul,double,%d,%.4f,%.2f,GFLOP/s,%.3e\n", n_mm, ms, flops / (ms * 1e-3) / 1e9, mdiff_d);
        cudaFree(a); cudaFree(b); cudaFree(c);
    }

    free(ha_d); free(hb_d); free(hc_d); free(href_d);
    free(ha_f); free(hb_f); free(hc_f);
    return 0;
}
