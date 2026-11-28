/* Compare custom CUDA GEMM kernels (shared-mem 32x32 vs 4x4 register-tiled)
 * against NVIDIA cuBLAS FP32 SIMT, TF32, and FP16 Tensor Core (cublasGemmEx),
 * checking numerical error before timing.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                          \
    do {                                                                                          \
        cudaError_t e = (call);                                                                   \
        if (e != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));  \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

#define CHECK_CUBLAS(call)                                                                        \
    do {                                                                                          \
        cublasStatus_t s = (call);                                                                \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                         \
            std::fprintf(stderr, "cuBLAS %s:%d: code=%d\n", __FILE__, __LINE__, (int)s);          \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

// Baseline: simple 32x32 shared-memory tiled SGEMM (1 output element per thread)
__global__ void gemm_smem_32x32(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int n) {
    __shared__ float ta[32][33];
    __shared__ float tb[32][33];
    const int row = blockIdx.y * 32 + threadIdx.y;
    const int col = blockIdx.x * 32 + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < n; t += 32) {
        ta[threadIdx.y][threadIdx.x] = A[row * n + (t + threadIdx.x)];
        tb[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * n + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 32; ++k) acc += ta[threadIdx.y][k] * tb[k][threadIdx.x];
        __syncthreads();
    }
    C[row * n + col] = acc;
}

// Optimized: 64x64 block tile, BK=16, 16x16 threads computing 4x4 register tiles
// Bank-conflict-free B layout (tx + c*16 stride = 1 bank across half-warp)
__global__ void gemm_reg_4x4(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int n) {
    __shared__ float As[16][65];
    __shared__ float Bs[16][65];
    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * 16 + tx;
    const int row0 = blockIdx.y * 64;
    const int col0 = blockIdx.x * 64;

    float acc[4][4] = {{0}};
    const int a_row = tid >> 2;          // 0..63
    const int a_col = (tid & 3) << 2;    // 0, 4, 8, 12
    const int b_row = tid >> 4;          // 0..15
    const int b_col = (tid & 15) << 2;   // 0, 4..60

    for (int t = 0; t < n; t += 16) {
        const float4 va = *reinterpret_cast<const float4*>(&A[(row0 + a_row) * n + (t + a_col)]);
        As[a_col + 0][a_row] = va.x;
        As[a_col + 1][a_row] = va.y;
        As[a_col + 2][a_row] = va.z;
        As[a_col + 3][a_row] = va.w;

        const float4 vb = *reinterpret_cast<const float4*>(&B[(t + b_row) * n + (col0 + b_col)]);
        Bs[b_row][b_col + 0] = vb.x;
        Bs[b_row][b_col + 1] = vb.y;
        Bs[b_row][b_col + 2] = vb.z;
        Bs[b_row][b_col + 3] = vb.w;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            float af[4];
            float bf[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) af[r] = As[k][ty * 4 + r];
            #pragma unroll
            for (int c = 0; c < 4; ++c) bf[c] = Bs[k][tx + c * 16];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) acc[r][c] += af[r] * bf[c];
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < 4; ++r) {
        const int gr = row0 + ty * 4 + r;
        for (int c = 0; c < 4; ++c) {
            const int gc = col0 + tx + c * 16;
            C[gr * n + gc] = acc[r][c];
        }
    }
}

__global__ void fp32_to_fp16(size_t total, const float* __restrict__ in, __half* __restrict__ out) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)blockDim.x * gridDim.x) {
        out[i] = __float2half(in[i]);
    }
}

static float time_ms(cudaEvent_t t0, cudaEvent_t t1) {
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    return ms;
}

static double max_rel_err_f32(const std::vector<float>& ref, const std::vector<float>& got) {
    double max_d = 0.0, max_r = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_d = fmax(max_d, fabs((double)ref[i] - (double)got[i]));
        max_r = fmax(max_r, fabs((double)ref[i]));
    }
    return max_d / fmax(1.0, max_r);
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : 2048;
    int reps = argc > 2 ? std::atoi(argv[2]) : 20;

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("# GPU: %s (CC %d.%d, SMs=%d) | n=%d | reps=%d\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, n, reps);
    std::printf("variant,n,ms,tflops,rel_err_vs_cublas\n");

    const size_t elems = (size_t)n * n;
    const double flops = 2.0 * (double)n * (double)n * (double)n;
    std::vector<float> hA(elems), hB(elems), hRef(elems), hOut(elems);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            hA[(size_t)i * n + j] = 0.25f * sinf(0.01f * (i + 1)) * cosf(0.02f * (j + 1));
            hB[(size_t)i * n + j] = 0.25f * (cosf(0.015f * (i + 1)) + 0.5f * sinf(0.01f * (j + 1)));
        }
    }

    float *dA, *dB, *dC;
    __half *dAh, *dBh, *dCh;
    CHECK_CUDA(cudaMalloc(&dA, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dAh, elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dBh, elems * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dCh, elems * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
    fp32_to_fp16<<<(int)std::min<size_t>(4096, (elems + 255) / 256), 256>>>(elems, dA, dAh);
    fp32_to_fp16<<<(int)std::min<size_t>(4096, (elems + 255) / 256), 256>>>(elems, dB, dBh);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    // 1) cuBLAS FP32 SIMT (ground-truth reference on row-major A*B via B^T * A^T trick -> col-major C^T = B^T * A^T?
    // Wait: row-major C = A*B <=> col-major C^T = B^T * A^T, i.e., cublasSgemm(N, N, n, n, n, B, A, C)!)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hRef.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
    for (int w = 0; w < 3; ++w)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dB, n, dA, n, &beta, dC, n));
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms_cublas_fp32 = time_ms(t0, t1) / reps;
    std::printf("cublas_fp32_simt,%d,%.4f,%.2f,%.3e\n", n, ms_cublas_fp32, flops / (ms_cublas_fp32 * 1e-3) / 1e12, 0.0);

    // 2) Custom 32x32 shared-memory tiled
    {
        dim3 block(32, 32), grid(n / 32, n / 32);
        gemm_smem_32x32<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(hOut.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
        double err = max_rel_err_f32(hRef, hOut);
        for (int w = 0; w < 3; ++w) gemm_smem_32x32<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaEventRecord(t0));
        for (int r = 0; r < reps; ++r) gemm_smem_32x32<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms = time_ms(t0, t1) / reps;
        std::printf("custom_smem_32x32,%d,%.4f,%.2f,%.3e\n", n, ms, flops / (ms * 1e-3) / 1e12, err);
    }

    // 3) Custom 4x4 register-tiled (bank-conflict-free)
    {
        dim3 block(16, 16), grid(n / 64, n / 64);
        gemm_reg_4x4<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(hOut.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
        double err = max_rel_err_f32(hRef, hOut);
        for (int w = 0; w < 3; ++w) gemm_reg_4x4<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaEventRecord(t0));
        for (int r = 0; r < reps; ++r) gemm_reg_4x4<<<grid, block>>>(dA, dB, dC, n);
        CHECK_CUDA(cudaEventRecord(t1));
        CHECK_CUDA(cudaEventSynchronize(t1));
        float ms = time_ms(t0, t1) / reps;
        std::printf("custom_reg_4x4,%d,%.4f,%.2f,%.3e\n", n, ms, flops / (ms * 1e-3) / 1e12, err);
    }

    // 4) cuBLAS TF32 Tensor Core
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                              dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &beta,
                              dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOut.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
    double err_tf32 = max_rel_err_f32(hRef, hOut);
    for (int w = 0; w < 3; ++w)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                  dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &beta,
                                  dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                  dB, CUDA_R_32F, n, dA, CUDA_R_32F, n, &beta,
                                  dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms_tf32 = time_ms(t0, t1) / reps;
    std::printf("cublas_tf32_tc,%d,%.4f,%.2f,%.3e\n", n, ms_tf32, flops / (ms_tf32 * 1e-3) / 1e12, err_tf32);

    // 5) cuBLAS FP16 Tensor Core (FP32 accumulate, output FP16 or FP32)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                              dBh, CUDA_R_16F, n, dAh, CUDA_R_16F, n, &beta,
                              dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOut.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost));
    double err_fp16 = max_rel_err_f32(hRef, hOut);
    for (int w = 0; w < 3; ++w)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                  dBh, CUDA_R_16F, n, dAh, CUDA_R_16F, n, &beta,
                                  dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int r = 0; r < reps; ++r)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                                  dBh, CUDA_R_16F, n, dAh, CUDA_R_16F, n, &beta,
                                  dC, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaEventSynchronize(t1));
    float ms_fp16 = time_ms(t0, t1) / reps;
    std::printf("cublas_fp16_tc,%d,%.4f,%.2f,%.3e\n", n, ms_fp16, flops / (ms_fp16 * 1e-3) / 1e12, err_fp16);

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dAh); cudaFree(dBh); cudaFree(dCh);
    return 0;
}
