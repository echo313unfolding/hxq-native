/*
 * test_affine.c — Unit tests for HXQ affine group quantization
 *
 * Tests CPU reference matmul + decompress paths.
 *
 * Build: gcc -O2 -I../include -o test_affine test_affine.c ../src/hxq_affine.c -lm
 * Run:   ./test_affine
 */

#include "hxq_affine.h"
#include "hxq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

#define ASSERT_FLOAT_EQ(a, b, tol, msg) do { \
    tests_run++; \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (tol)) { \
        printf("  FAIL: %s — got %.6f, expected %.6f, diff %.2e (line %d)\n", \
               msg, _a, _b, fabsf(_a - _b), __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ── Helpers ─────────────────────────────────────────────────── */

/* Simple LCG for deterministic pseudo-random floats */
static uint32_t rng_state = 42;
static float rand_float(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) / (float)(1 << 24) - 0.5f;
}
static void rng_seed(uint32_t s) { rng_state = s; }

/* Naive matmul reference: Y = X @ W^T + bias, where W is dense */
static void naive_matmul(
    const float *x, const float *W, const float *bias,
    float *out, uint32_t N, uint32_t IN, uint32_t OUT
) {
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t o = 0; o < OUT; o++) {
            float acc = 0.0f;
            for (uint32_t i = 0; i < IN; i++) {
                acc += x[n * IN + i] * W[o * IN + i];
            }
            if (bias) acc += bias[o];
            out[n * OUT + o] = acc;
        }
    }
}

/* ── Test 1: CPU reference correctness — small 4x4 ───────────── */

void test_small_known_values(void) {
    printf("test_small_known_values:\n");

    /*
     * 2x4 weight matrix, group_size=4 (1 group), N=1
     * indices = [[0, 1, 2, 3], [4, 5, 6, 7]]
     * scales  = [[2.0], [0.5]]
     * offsets = [[1.0], [-1.0]]
     *
     * W[0,:] = [0*2+1, 1*2+1, 2*2+1, 3*2+1] = [1, 3, 5, 7]
     * W[1,:] = [4*0.5-1, 5*0.5-1, 6*0.5-1, 7*0.5-1] = [1, 1.5, 2, 2.5]
     *
     * x = [1, 1, 1, 1]
     * Y[0] = 1+3+5+7 = 16
     * Y[1] = 1+1.5+2+2.5 = 7
     */
    uint8_t indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    float scales[2]    = { 2.0f, 0.5f };
    float offsets[2]   = { 1.0f, -1.0f };
    float x[4]         = { 1.0f, 1.0f, 1.0f, 1.0f };
    float output[2];

    hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, output,
                                 1, 4, 2, 4);

    ASSERT_FLOAT_EQ(output[0], 16.0f, 1e-5, "Y[0]=16");
    ASSERT_FLOAT_EQ(output[1],  7.0f, 1e-5, "Y[1]=7");

    printf("  done\n");
}

/* ── Test 2: CPU reference with multiple groups ──────────────── */

void test_multiple_groups(void) {
    printf("test_multiple_groups:\n");

    /*
     * 1x8 weight, group_size=4 → 2 groups, N=1
     * out_f=1, in_f=8
     * indices = [1, 1, 1, 1,  2, 2, 2, 2]
     * scales  = [1.0, 3.0]   (group 0: scale=1, group 1: scale=3)
     * offsets = [0.0, 0.0]
     *
     * W = [1,1,1,1,  6,6,6,6]
     * x = [1,1,1,1,  1,1,1,1]
     * Y = 1*4 + 6*4 = 4 + 24 = 28
     */
    uint8_t indices[8] = { 1, 1, 1, 1, 2, 2, 2, 2 };
    float scales[2]    = { 1.0f, 3.0f };
    float offsets[2]   = { 0.0f, 0.0f };
    float x[8]         = { 1,1,1,1, 1,1,1,1 };
    float output[1];

    hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, output,
                                 1, 8, 1, 4);

    ASSERT_FLOAT_EQ(output[0], 28.0f, 1e-5, "multi_group=28");

    printf("  done\n");
}

/* ── Test 3: N=1 fused vs decompress+matmul ──────────────────── */

void test_n1_vs_decompress(void) {
    printf("test_n1_vs_decompress:\n");

    uint32_t OUT = 16, IN = 128, GS = 128;
    uint32_t n_groups = IN / GS;

    uint8_t *indices = (uint8_t *)malloc(OUT * IN);
    float *scales  = (float *)malloc(OUT * n_groups * sizeof(float));
    float *offsets = (float *)malloc(OUT * n_groups * sizeof(float));
    float *x       = (float *)malloc(IN * sizeof(float));
    float *out_fused = (float *)malloc(OUT * sizeof(float));
    float *out_naive = (float *)malloc(OUT * sizeof(float));
    float *W_dense   = (float *)malloc(OUT * IN * sizeof(float));

    rng_seed(123);
    for (uint32_t i = 0; i < OUT * IN; i++) indices[i] = (uint8_t)(rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63);
    for (uint32_t i = 0; i < OUT * n_groups; i++) scales[i]  = rand_float() * 0.1f;
    for (uint32_t i = 0; i < OUT * n_groups; i++) offsets[i] = rand_float() * 0.01f;
    for (uint32_t i = 0; i < IN; i++) x[i] = rand_float();

    /* Fused path */
    hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, out_fused,
                                 1, IN, OUT, GS);

    /* Decompress + naive matmul */
    hxq_affine_decompress_ref(indices, scales, offsets, W_dense, OUT, IN, GS);
    naive_matmul(x, W_dense, NULL, out_naive, 1, IN, OUT);

    float max_err = 0.0f;
    for (uint32_t o = 0; o < OUT; o++) {
        float err = fabsf(out_fused[o] - out_naive[o]);
        if (err > max_err) max_err = err;
    }

    ASSERT(max_err < 1e-4f, "N=1 fused vs decompress max_err < 1e-4");
    printf("  max_err = %.2e\n", max_err);

    free(indices); free(scales); free(offsets); free(x);
    free(out_fused); free(out_naive); free(W_dense);
    printf("  done\n");
}

/* ── Test 4: N>1 fused vs decompress+matmul ──────────────────── */

void test_n4_vs_decompress(void) {
    printf("test_n4_vs_decompress:\n");

    uint32_t N = 4, OUT = 32, IN = 256, GS = 128;
    uint32_t n_groups = IN / GS;

    uint8_t *indices = (uint8_t *)malloc(OUT * IN);
    float *scales  = (float *)malloc(OUT * n_groups * sizeof(float));
    float *offsets = (float *)malloc(OUT * n_groups * sizeof(float));
    float *x       = (float *)malloc(N * IN * sizeof(float));
    float *out_fused = (float *)malloc(N * OUT * sizeof(float));
    float *out_naive = (float *)malloc(N * OUT * sizeof(float));
    float *W_dense   = (float *)malloc(OUT * IN * sizeof(float));

    rng_seed(456);
    for (uint32_t i = 0; i < OUT * IN; i++) indices[i] = (uint8_t)(rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63);
    for (uint32_t i = 0; i < OUT * n_groups; i++) scales[i]  = rand_float() * 0.1f;
    for (uint32_t i = 0; i < OUT * n_groups; i++) offsets[i] = rand_float() * 0.01f;
    for (uint32_t i = 0; i < N * IN; i++) x[i] = rand_float();

    hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, out_fused,
                                 N, IN, OUT, GS);

    hxq_affine_decompress_ref(indices, scales, offsets, W_dense, OUT, IN, GS);
    naive_matmul(x, W_dense, NULL, out_naive, N, IN, OUT);

    float max_err = 0.0f;
    for (uint32_t i = 0; i < N * OUT; i++) {
        float err = fabsf(out_fused[i] - out_naive[i]);
        if (err > max_err) max_err = err;
    }

    ASSERT(max_err < 1e-3f, "N=4 fused vs decompress max_err < 1e-3");
    printf("  max_err = %.2e\n", max_err);

    free(indices); free(scales); free(offsets); free(x);
    free(out_fused); free(out_naive); free(W_dense);
    printf("  done\n");
}

/* ── Test 5: Bias handling ───────────────────────────────────── */

void test_bias(void) {
    printf("test_bias:\n");

    uint8_t indices[4] = { 0, 0, 0, 0 };
    float scales[2]    = { 1.0f, 1.0f };
    float offsets[2]   = { 0.0f, 0.0f };
    float bias[2]      = { 10.0f, -5.0f };
    float x[2]         = { 0.0f, 0.0f };
    float output[2];

    /* All indices=0, all x=0, so Y should be just bias */
    hxq_affine_group_matmul_ref(x, indices, scales, offsets, bias, output,
                                 1, 2, 2, 2);

    ASSERT_FLOAT_EQ(output[0], 10.0f, 1e-6, "bias_only[0]=10");
    ASSERT_FLOAT_EQ(output[1], -5.0f, 1e-6, "bias_only[1]=-5");

    printf("  done\n");
}

/* ── Test 6: Zero indices → all-offset output ────────────────── */

void test_zero_indices(void) {
    printf("test_zero_indices:\n");

    uint32_t OUT = 4, IN = 8, GS = 4;
    uint32_t n_groups = IN / GS;

    uint8_t indices[32]; /* 4 * 8 */
    memset(indices, 0, sizeof(indices));

    float scales[8];  /* 4 * 2 */
    float offsets[8];
    for (uint32_t i = 0; i < OUT * n_groups; i++) {
        scales[i]  = 1.0f;
        offsets[i] = (float)(i + 1) * 0.5f;  /* 0.5, 1.0, 1.5, ... */
    }

    /* x = all ones → x_sum per group = GS = 4 */
    float x[8] = { 1,1,1,1, 1,1,1,1 };
    float output[4];

    hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, output,
                                 1, IN, OUT, GS);

    /* For each output o: Y[o] = sum_g { scale*0 + offset[o,g]*4 }
     *                         = sum_g { offset[o,g] * 4 } */
    for (uint32_t o = 0; o < OUT; o++) {
        float expected = 0.0f;
        for (uint32_t g = 0; g < n_groups; g++) {
            expected += offsets[o * n_groups + g] * (float)GS;
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "zero_idx_Y[%u]", o);
        ASSERT_FLOAT_EQ(output[o], expected, 1e-4, msg);
    }

    printf("  done\n");
}

/* ── Test 7: Max indices (63) ────────────────────────────────── */

void test_max_indices(void) {
    printf("test_max_indices:\n");

    uint32_t OUT = 2, IN = 4, GS = 4;

    uint8_t indices[8];
    memset(indices, 63, sizeof(indices));

    float scales[2]  = { 0.1f, 0.2f };
    float offsets[2]  = { 0.0f, 0.0f };
    float x[4]        = { 1.0f, 1.0f, 1.0f, 1.0f };
    float output[2];

    hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, output,
                                 1, IN, OUT, GS);

    /* W[o,i] = 63 * scale[o,0] + 0
     * Y[o] = sum_i(x[i] * W[o,i]) = 4 * 63 * scale[o,0] */
    ASSERT_FLOAT_EQ(output[0], 4.0f * 63.0f * 0.1f, 1e-4, "max_idx_Y[0]");
    ASSERT_FLOAT_EQ(output[1], 4.0f * 63.0f * 0.2f, 1e-4, "max_idx_Y[1]");

    printf("  done\n");
}

/* ── Test 8: Decompress correctness ──────────────────────────── */

void test_decompress(void) {
    printf("test_decompress:\n");

    uint32_t OUT = 2, IN = 4, GS = 4;

    uint8_t indices[8] = { 0, 10, 20, 30, 5, 15, 25, 63 };
    float scales[2]   = { 0.5f, 2.0f };
    float offsets[2]   = { 1.0f, -3.0f };
    float W[8];

    hxq_affine_decompress_ref(indices, scales, offsets, W, OUT, IN, GS);

    /* Row 0: W[0,i] = idx * 0.5 + 1.0 */
    ASSERT_FLOAT_EQ(W[0],  0 * 0.5f + 1.0f,  1e-6, "W[0][0]");
    ASSERT_FLOAT_EQ(W[1], 10 * 0.5f + 1.0f,  1e-6, "W[0][1]");
    ASSERT_FLOAT_EQ(W[2], 20 * 0.5f + 1.0f,  1e-6, "W[0][2]");
    ASSERT_FLOAT_EQ(W[3], 30 * 0.5f + 1.0f,  1e-6, "W[0][3]");

    /* Row 1: W[1,i] = idx * 2.0 - 3.0 */
    ASSERT_FLOAT_EQ(W[4],  5 * 2.0f - 3.0f,  1e-6, "W[1][0]");
    ASSERT_FLOAT_EQ(W[5], 15 * 2.0f - 3.0f,  1e-6, "W[1][1]");
    ASSERT_FLOAT_EQ(W[6], 25 * 2.0f - 3.0f,  1e-6, "W[1][2]");
    ASSERT_FLOAT_EQ(W[7], 63 * 2.0f - 3.0f,  1e-6, "W[1][3]");

    printf("  done\n");
}

/* ── Test 9: Qwen2.5-3B shapes ───────────────────────────────── */

void test_real_shapes(void) {
    printf("test_real_shapes:\n");

    /* Qwen2.5-3B: hidden=2048, intermediate=11008 */
    uint32_t shapes[][2] = {
        { 2048, 2048 },    /* self_attn.q_proj */
        { 11008, 2048 },   /* mlp.gate_proj */
        { 2048, 11008 },   /* mlp.down_proj */
    };

    uint32_t GS = 128;

    for (int s = 0; s < 3; s++) {
        uint32_t OUT = shapes[s][0];
        uint32_t IN  = shapes[s][1];
        uint32_t n_groups = IN / GS;

        /* Skip if IN not divisible by GS */
        if (IN % GS != 0) {
            printf("  SKIP: %ux%u not divisible by %u\n", OUT, IN, GS);
            continue;
        }

        uint8_t *indices = (uint8_t *)malloc(OUT * IN);
        float *scales  = (float *)malloc(OUT * n_groups * sizeof(float));
        float *offsets = (float *)malloc(OUT * n_groups * sizeof(float));
        float *x       = (float *)malloc(IN * sizeof(float));
        float *out_fused = (float *)malloc(OUT * sizeof(float));
        float *W_dense   = (float *)malloc(OUT * IN * sizeof(float));
        float *out_naive = (float *)malloc(OUT * sizeof(float));

        rng_seed(789 + s);
        for (uint32_t i = 0; i < OUT * IN; i++)
            indices[i] = (uint8_t)(rng_state = rng_state * 1664525u + 1013904223u, (rng_state >> 16) & 63);
        for (uint32_t i = 0; i < OUT * n_groups; i++) scales[i]  = rand_float() * 0.01f;
        for (uint32_t i = 0; i < OUT * n_groups; i++) offsets[i] = rand_float() * 0.001f;
        for (uint32_t i = 0; i < IN; i++) x[i] = rand_float();

        hxq_affine_group_matmul_ref(x, indices, scales, offsets, NULL, out_fused,
                                     1, IN, OUT, GS);

        hxq_affine_decompress_ref(indices, scales, offsets, W_dense, OUT, IN, GS);
        naive_matmul(x, W_dense, NULL, out_naive, 1, IN, OUT);

        float max_err = 0.0f;
        for (uint32_t o = 0; o < OUT; o++) {
            float err = fabsf(out_fused[o] - out_naive[o]);
            if (err > max_err) max_err = err;
        }

        char msg[128];
        snprintf(msg, sizeof(msg), "shape_%ux%u_max_err < 1e-2", OUT, IN);
        ASSERT(max_err < 1e-2f, msg);
        printf("  %ux%u: max_err = %.2e\n", OUT, IN, max_err);

        free(indices); free(scales); free(offsets); free(x);
        free(out_fused); free(out_naive); free(W_dense);
    }

    printf("  done\n");
}

/* ── Test 10: N=1 bias + nonzero ─────────────────────────────── */

void test_bias_with_values(void) {
    printf("test_bias_with_values:\n");

    /*
     * 2x4, GS=4, N=1
     * indices = [[1,1,1,1], [2,2,2,2]]
     * scales = [1.0, 1.0], offsets = [0.0, 0.0]
     * bias = [100.0, 200.0]
     * x = [1,1,1,1]
     *
     * W = [[1,1,1,1], [2,2,2,2]]
     * Y[0] = 4 + 100 = 104
     * Y[1] = 8 + 200 = 208
     */
    uint8_t indices[8] = { 1,1,1,1, 2,2,2,2 };
    float scales[2]    = { 1.0f, 1.0f };
    float offsets[2]   = { 0.0f, 0.0f };
    float bias[2]      = { 100.0f, 200.0f };
    float x[4]         = { 1,1,1,1 };
    float output[2];

    hxq_affine_group_matmul_ref(x, indices, scales, offsets, bias, output,
                                 1, 4, 2, 4);

    ASSERT_FLOAT_EQ(output[0], 104.0f, 1e-5, "bias+val Y[0]=104");
    ASSERT_FLOAT_EQ(output[1], 208.0f, 1e-5, "bias+val Y[1]=208");

    printf("  done\n");
}

/* ── Test 11: Meta-kernel decompress path (hxq_dequant) ──────── */

void test_meta_kernel_affine(void) {
    printf("test_meta_kernel_affine:\n");

    uint32_t OUT = 2, IN = 4, GS = 4;

    uint8_t indices[8] = { 0, 10, 20, 30, 5, 15, 25, 63 };
    float scales[2]   = { 0.5f, 2.0f };
    float offsets[2]   = { 1.0f, -3.0f };

    hxq_tensor_t t;
    hxq_tensor_init(&t);
    hxq_error_t err = hxq_tensor_load_affine(&t, indices, scales, offsets, OUT, IN, GS);
    ASSERT(err == HXQ_OK, "load_affine_ok");

    /* Decompress via hxq_dequant */
    hxq_shared_buffer_t buf;
    hxq_shared_buffer_init(&buf, 64);
    hxq_result_t result;
    err = hxq_dequant(&t, &buf, HXQ_BACKEND_AUTO, &result);
    ASSERT(err == HXQ_OK, "dequant_affine_ok");
    ASSERT(result.out_features == OUT, "meta_out_f");
    ASSERT(result.in_features == IN, "meta_in_f");

    /* W[0,1] = 10 * 0.5 + 1.0 = 6.0 */
    ASSERT_FLOAT_EQ(result.weights[1], 6.0f, 1e-6, "dequant_W[0][1]");
    /* W[1,3] = 63 * 2.0 - 3.0 = 123.0 */
    ASSERT_FLOAT_EQ(result.weights[7], 123.0f, 1e-6, "dequant_W[1][3]");

    hxq_tensor_free(&t);
    hxq_shared_buffer_free(&buf);
    printf("  done\n");
}

/* ── Main ────────────────────────────────────────────────────── */

int main(void) {
    printf("=== HXQ Affine Group Quantization Tests ===\n\n");

    test_small_known_values();
    test_multiple_groups();
    test_n1_vs_decompress();
    test_n4_vs_decompress();
    test_bias();
    test_zero_indices();
    test_max_indices();
    test_decompress();
    test_real_shapes();
    test_bias_with_values();
    test_meta_kernel_affine();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
