/*
 * test_affine_cuda.cu — GPU correctness test for affine group matmul
 *
 * Tests N=1 decode and N>1 prefill CUDA kernels against CPU reference.
 *
 * Build: nvcc -O2 -Iinclude test/test_affine_cuda.cu lib/hxq_affine_cuda.o
 *        lib/hxq_affine.o -o test/test_affine_cuda -lm
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef HXQ_HAVE_CUDA
#define HXQ_HAVE_CUDA
#endif
#include "hxq_affine.h"

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("  CUDA ERROR: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        return; \
    } \
} while(0)

/* Simple LCG */
static uint32_t rng_state = 42;
static float rand_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / (float)(1 << 24) - 0.5f;
}
static void rng_seed(uint32_t s) { rng_state = s; }

/* ── Test: N=1 decode kernel vs CPU reference ────────────────── */

void test_decode_kernel(void) {
    printf("test_decode_kernel (N=1):\n");

    uint32_t OUT = 256, IN = 512, GS = 128;
    uint32_t n_groups = IN / GS;
    uint32_t N = 1;

    /* Allocate host */
    uint8_t *h_indices = (uint8_t *)malloc(OUT * IN);
    float *h_scales  = (float *)malloc(OUT * n_groups * sizeof(float));
    float *h_offsets = (float *)malloc(OUT * n_groups * sizeof(float));
    float *h_x       = (float *)malloc(N * IN * sizeof(float));
    float *h_out_cpu = (float *)malloc(N * OUT * sizeof(float));
    float *h_out_gpu = (float *)malloc(N * OUT * sizeof(float));

    rng_seed(100);
    for (uint32_t i = 0; i < OUT * IN; i++)
        h_indices[i] = (uint8_t)((rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63));
    for (uint32_t i = 0; i < OUT * n_groups; i++) h_scales[i]  = rand_float() * 0.1f;
    for (uint32_t i = 0; i < OUT * n_groups; i++) h_offsets[i] = rand_float() * 0.01f;
    for (uint32_t i = 0; i < IN; i++) h_x[i] = rand_float();

    /* CPU reference */
    hxq_affine_group_matmul_ref(h_x, h_indices, h_scales, h_offsets, NULL,
                                 h_out_cpu, N, IN, OUT, GS);

    /* GPU */
    float *d_x, *d_scales, *d_offsets, *d_output;
    uint8_t *d_indices;
    CHECK_CUDA(cudaMalloc(&d_x, N * IN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, OUT * IN));
    CHECK_CUDA(cudaMalloc(&d_scales, OUT * n_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_offsets, OUT * n_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * OUT * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * IN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices, OUT * IN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice));

    hxq_affine_tensor_t tensor;
    tensor.indices      = d_indices;
    tensor.scales       = d_scales;
    tensor.offsets      = d_offsets;
    tensor.bias         = NULL;
    tensor.out_features = OUT;
    tensor.in_features  = IN;
    tensor.group_size   = GS;

    int ret = hxq_affine_group_matmul(d_x, &tensor, d_output, N);
    ASSERT(ret == 0, "kernel launch success");

    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_output, N * OUT * sizeof(float), cudaMemcpyDeviceToHost));

    /* Compare */
    float max_err = 0.0f;
    for (uint32_t i = 0; i < N * OUT; i++) {
        float err = fabsf(h_out_gpu[i] - h_out_cpu[i]);
        if (err > max_err) max_err = err;
    }

    ASSERT(max_err < 1e-4f, "N=1 GPU vs CPU max_err < 1e-4");
    printf("  max_err = %.2e\n", max_err);

    cudaFree(d_x); cudaFree(d_indices); cudaFree(d_scales);
    cudaFree(d_offsets); cudaFree(d_output);
    free(h_indices); free(h_scales); free(h_offsets);
    free(h_x); free(h_out_cpu); free(h_out_gpu);
    printf("  done\n");
}

/* ── Test: N>1 prefill kernel vs CPU reference ───────────────── */

void test_prefill_kernel(void) {
    printf("test_prefill_kernel (N=4):\n");

    uint32_t OUT = 256, IN = 512, GS = 128;
    uint32_t n_groups = IN / GS;
    uint32_t N = 4;

    uint8_t *h_indices = (uint8_t *)malloc(OUT * IN);
    float *h_scales  = (float *)malloc(OUT * n_groups * sizeof(float));
    float *h_offsets = (float *)malloc(OUT * n_groups * sizeof(float));
    float *h_x       = (float *)malloc(N * IN * sizeof(float));
    float *h_out_cpu = (float *)malloc(N * OUT * sizeof(float));
    float *h_out_gpu = (float *)malloc(N * OUT * sizeof(float));

    rng_seed(200);
    for (uint32_t i = 0; i < OUT * IN; i++)
        h_indices[i] = (uint8_t)((rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63));
    for (uint32_t i = 0; i < OUT * n_groups; i++) h_scales[i]  = rand_float() * 0.1f;
    for (uint32_t i = 0; i < OUT * n_groups; i++) h_offsets[i] = rand_float() * 0.01f;
    for (uint32_t i = 0; i < N * IN; i++) h_x[i] = rand_float();

    hxq_affine_group_matmul_ref(h_x, h_indices, h_scales, h_offsets, NULL,
                                 h_out_cpu, N, IN, OUT, GS);

    float *d_x, *d_scales, *d_offsets, *d_output;
    uint8_t *d_indices;
    CHECK_CUDA(cudaMalloc(&d_x, N * IN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, OUT * IN));
    CHECK_CUDA(cudaMalloc(&d_scales, OUT * n_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_offsets, OUT * n_groups * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, N * OUT * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * IN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices, OUT * IN, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales, h_scales, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, h_offsets, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice));

    hxq_affine_tensor_t tensor;
    tensor.indices      = d_indices;
    tensor.scales       = d_scales;
    tensor.offsets      = d_offsets;
    tensor.bias         = NULL;
    tensor.out_features = OUT;
    tensor.in_features  = IN;
    tensor.group_size   = GS;

    int ret = hxq_affine_group_matmul(d_x, &tensor, d_output, N);
    ASSERT(ret == 0, "prefill kernel launch success");

    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_output, N * OUT * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (uint32_t i = 0; i < N * OUT; i++) {
        float err = fabsf(h_out_gpu[i] - h_out_cpu[i]);
        if (err > max_err) max_err = err;
    }

    ASSERT(max_err < 1e-4f, "N=4 GPU vs CPU max_err < 1e-4");
    printf("  max_err = %.2e\n", max_err);

    cudaFree(d_x); cudaFree(d_indices); cudaFree(d_scales);
    cudaFree(d_offsets); cudaFree(d_output);
    free(h_indices); free(h_scales); free(h_offsets);
    free(h_x); free(h_out_cpu); free(h_out_gpu);
    printf("  done\n");
}

/* ── Test: Real Qwen shapes, N=1 decode ──────────────────────── */

void test_qwen_shapes(void) {
    printf("test_qwen_shapes (N=1, real layer sizes):\n");

    uint32_t shapes[][2] = {
        { 2048, 2048 },    /* q_proj */
        { 11008, 2048 },   /* gate_proj */
        { 2048, 11008 },   /* down_proj */
    };

    uint32_t GS = 128;

    for (int s = 0; s < 3; s++) {
        uint32_t OUT = shapes[s][0];
        uint32_t IN  = shapes[s][1];
        uint32_t n_groups = IN / GS;
        uint32_t N = 1;

        uint8_t *h_indices = (uint8_t *)malloc(OUT * IN);
        float *h_scales  = (float *)malloc(OUT * n_groups * sizeof(float));
        float *h_offsets = (float *)malloc(OUT * n_groups * sizeof(float));
        float *h_x       = (float *)malloc(N * IN * sizeof(float));
        float *h_out_cpu = (float *)malloc(N * OUT * sizeof(float));
        float *h_out_gpu = (float *)malloc(N * OUT * sizeof(float));

        rng_seed(300 + s);
        for (size_t i = 0; i < (size_t)OUT * IN; i++)
            h_indices[i] = (uint8_t)((rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63));
        for (uint32_t i = 0; i < OUT * n_groups; i++) h_scales[i]  = rand_float() * 0.01f;
        for (uint32_t i = 0; i < OUT * n_groups; i++) h_offsets[i] = rand_float() * 0.001f;
        for (uint32_t i = 0; i < IN; i++) h_x[i] = rand_float();

        hxq_affine_group_matmul_ref(h_x, h_indices, h_scales, h_offsets, NULL,
                                     h_out_cpu, N, IN, OUT, GS);

        float *d_x, *d_scales, *d_offsets, *d_output;
        uint8_t *d_indices;
        cudaMalloc(&d_x, N * IN * sizeof(float));
        cudaMalloc(&d_indices, (size_t)OUT * IN);
        cudaMalloc(&d_scales, OUT * n_groups * sizeof(float));
        cudaMalloc(&d_offsets, OUT * n_groups * sizeof(float));
        cudaMalloc(&d_output, N * OUT * sizeof(float));

        cudaMemcpy(d_x, h_x, N * IN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_indices, h_indices, (size_t)OUT * IN, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, h_scales, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_offsets, h_offsets, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice);

        hxq_affine_tensor_t tensor;
        tensor.indices      = d_indices;
        tensor.scales       = d_scales;
        tensor.offsets      = d_offsets;
        tensor.bias         = NULL;
        tensor.out_features = OUT;
        tensor.in_features  = IN;
        tensor.group_size   = GS;

        int ret = hxq_affine_group_matmul(d_x, &tensor, d_output, N);

        cudaMemcpy(h_out_gpu, d_output, N * OUT * sizeof(float), cudaMemcpyDeviceToHost);

        float max_err = 0.0f;
        for (uint32_t i = 0; i < N * OUT; i++) {
            float err = fabsf(h_out_gpu[i] - h_out_cpu[i]);
            if (err > max_err) max_err = err;
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "%ux%u GPU vs CPU max_err < 1e-3", OUT, IN);
        ASSERT(ret == 0 && max_err < 1e-3f, msg);
        printf("  %ux%u: max_err = %.2e\n", OUT, IN, max_err);

        cudaFree(d_x); cudaFree(d_indices); cudaFree(d_scales);
        cudaFree(d_offsets); cudaFree(d_output);
        free(h_indices); free(h_scales); free(h_offsets);
        free(h_x); free(h_out_cpu); free(h_out_gpu);
    }

    printf("  done\n");
}

/* ── Test: Bias on GPU ───────────────────────────────────────── */

void test_bias_gpu(void) {
    printf("test_bias_gpu:\n");

    uint32_t OUT = 64, IN = 128, GS = 128, N = 1;
    uint32_t n_groups = IN / GS;

    uint8_t h_indices[64 * 128];
    memset(h_indices, 0, sizeof(h_indices));
    float h_scales[64], h_offsets[64], h_bias[64];
    float h_x[128], h_out_gpu[64], h_out_cpu[64];

    for (uint32_t i = 0; i < OUT; i++) {
        h_scales[i]  = 1.0f;
        h_offsets[i] = 0.0f;
        h_bias[i]    = (float)(i + 1) * 10.0f;
    }
    memset(h_x, 0, sizeof(h_x));

    /* CPU: all-zero indices + all-zero x → output = bias */
    hxq_affine_group_matmul_ref(h_x, h_indices, h_scales, h_offsets, h_bias,
                                 h_out_cpu, N, IN, OUT, GS);

    float *d_x, *d_scales, *d_offsets, *d_bias, *d_output;
    uint8_t *d_indices;
    cudaMalloc(&d_x, IN * sizeof(float));
    cudaMalloc(&d_indices, OUT * IN);
    cudaMalloc(&d_scales, OUT * n_groups * sizeof(float));
    cudaMalloc(&d_offsets, OUT * n_groups * sizeof(float));
    cudaMalloc(&d_bias, OUT * sizeof(float));
    cudaMalloc(&d_output, OUT * sizeof(float));

    cudaMemcpy(d_x, h_x, IN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, OUT * IN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, OUT * sizeof(float), cudaMemcpyHostToDevice);

    hxq_affine_tensor_t tensor;
    tensor.indices      = d_indices;
    tensor.scales       = d_scales;
    tensor.offsets      = d_offsets;
    tensor.bias         = d_bias;
    tensor.out_features = OUT;
    tensor.in_features  = IN;
    tensor.group_size   = GS;

    int ret = hxq_affine_group_matmul(d_x, &tensor, d_output, N);
    ASSERT(ret == 0, "bias kernel launch");

    cudaMemcpy(h_out_gpu, d_output, OUT * sizeof(float), cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (uint32_t i = 0; i < OUT; i++) {
        float err = fabsf(h_out_gpu[i] - h_out_cpu[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-5f, "bias GPU vs CPU max_err < 1e-5");
    printf("  max_err = %.2e\n", max_err);

    cudaFree(d_x); cudaFree(d_indices); cudaFree(d_scales);
    cudaFree(d_offsets); cudaFree(d_bias); cudaFree(d_output);
    printf("  done\n");
}

/* ── Microbenchmark: N=1 decode, Qwen q_proj shape ───────────── */

void bench_decode_qwen(void) {
    printf("bench_decode_qwen (2048x2048, N=1):\n");

    uint32_t OUT = 2048, IN = 2048, GS = 128;
    uint32_t n_groups = IN / GS;
    uint32_t N = 1;

    uint8_t *h_indices = (uint8_t *)malloc((size_t)OUT * IN);
    float *h_scales  = (float *)malloc(OUT * n_groups * sizeof(float));
    float *h_offsets = (float *)malloc(OUT * n_groups * sizeof(float));
    float *h_x       = (float *)malloc(IN * sizeof(float));

    rng_seed(999);
    for (size_t i = 0; i < (size_t)OUT * IN; i++)
        h_indices[i] = (uint8_t)((rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63));
    for (uint32_t i = 0; i < OUT * n_groups; i++) h_scales[i]  = rand_float() * 0.01f;
    for (uint32_t i = 0; i < OUT * n_groups; i++) h_offsets[i] = rand_float() * 0.001f;
    for (uint32_t i = 0; i < IN; i++) h_x[i] = rand_float();

    float *d_x, *d_scales, *d_offsets, *d_output;
    uint8_t *d_indices;
    cudaMalloc(&d_x, IN * sizeof(float));
    cudaMalloc(&d_indices, (size_t)OUT * IN);
    cudaMalloc(&d_scales, OUT * n_groups * sizeof(float));
    cudaMalloc(&d_offsets, OUT * n_groups * sizeof(float));
    cudaMalloc(&d_output, OUT * sizeof(float));

    cudaMemcpy(d_x, h_x, IN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, (size_t)OUT * IN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, OUT * n_groups * sizeof(float), cudaMemcpyHostToDevice);

    hxq_affine_tensor_t tensor;
    tensor.indices      = d_indices;
    tensor.scales       = d_scales;
    tensor.offsets      = d_offsets;
    tensor.bias         = NULL;
    tensor.out_features = OUT;
    tensor.in_features  = IN;
    tensor.group_size   = GS;

    /* Warmup */
    for (int i = 0; i < 10; i++)
        hxq_affine_group_matmul(d_x, &tensor, d_output, N);
    cudaDeviceSynchronize();

    /* Benchmark */
    int iters = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        hxq_affine_group_matmul(d_x, &tensor, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float per_call_us = (ms * 1000.0f) / iters;

    printf("  %d iters: %.2f ms total, %.1f us/call\n", iters, ms, per_call_us);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_x); cudaFree(d_indices); cudaFree(d_scales);
    cudaFree(d_offsets); cudaFree(d_output);
    free(h_indices); free(h_scales); free(h_offsets); free(h_x);
    printf("  done\n");
}

/* ── Main ────────────────────────────────────────────────────── */

int main(void) {
    printf("=== HXQ Affine CUDA Kernel Tests ===\n\n");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("GPU: %s (SM %d.%d, %d MB)\n\n",
           prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / (1024 * 1024)));

    test_decode_kernel();
    test_prefill_kernel();
    test_qwen_shapes();
    test_bias_gpu();
    bench_decode_qwen();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
