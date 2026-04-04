/*
 * test_hxq.c — Unit tests for HXQ native decompression
 *
 * Tests all three codec modes (scalar, 2D 8-bit, 2D 12-bit),
 * sidecar application, shared buffer, and confidence signal.
 *
 * Build: gcc -O2 -I../include -o test_hxq test_hxq.c ../src/hxq.c -lm
 * Run:   ./test_hxq
 */

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
    if (fabsf((a) - (b)) > (tol)) { \
        printf("  FAIL: %s — got %f, expected %f (line %d)\n", msg, (a), (b), __LINE__); \
    } else { \
        tests_passed++; \
    } \
} while(0)

/* ── Test: scalar VQ decompress ──────────────────────────────── */

void test_scalar_vq(void) {
    printf("test_scalar_vq:\n");

    hxq_tensor_t t;
    hxq_tensor_init(&t);

    /* Codebook: 256 entries, scalar (vdim=1) */
    float codebook[256];
    for (int i = 0; i < 256; i++) {
        codebook[i] = (float)i * 0.01f;  /* 0.00, 0.01, ... 2.55 */
    }

    /* 4x4 weight matrix */
    uint8_t indices[16] = {
        0,   1,   2,   3,
        10,  20,  30,  40,
        100, 150, 200, 250,
        255, 128, 64,  32
    };

    hxq_error_t err;
    err = hxq_tensor_load_codebook(&t, codebook, 256, 1);
    ASSERT(err == HXQ_OK, "load_codebook");

    err = hxq_tensor_load_indices_8bit(&t, indices, 4, 4);
    ASSERT(err == HXQ_OK, "load_indices_8bit");

    float output[16];
    err = hxq_tensor_decompress(&t, output);
    ASSERT(err == HXQ_OK, "decompress");

    /* Verify first row */
    ASSERT_FLOAT_EQ(output[0], 0.00f, 1e-6, "W[0][0]");
    ASSERT_FLOAT_EQ(output[1], 0.01f, 1e-6, "W[0][1]");
    ASSERT_FLOAT_EQ(output[2], 0.02f, 1e-6, "W[0][2]");
    ASSERT_FLOAT_EQ(output[3], 0.03f, 1e-6, "W[0][3]");

    /* Verify last row */
    ASSERT_FLOAT_EQ(output[12], 2.55f, 1e-6, "W[3][0]");
    ASSERT_FLOAT_EQ(output[13], 1.28f, 1e-6, "W[3][1]");
    ASSERT_FLOAT_EQ(output[14], 0.64f, 1e-6, "W[3][2]");
    ASSERT_FLOAT_EQ(output[15], 0.32f, 1e-6, "W[3][3]");

    /* No sidecar → confidence should be 0 */
    ASSERT_FLOAT_EQ(hxq_get_sidecar_confidence(&t), 0.0f, 1e-6, "no_sidecar_confidence");

    hxq_tensor_free(&t);
    printf("  scalar VQ: done\n");
}

/* ── Test: 12-bit packing/unpacking ──────────────────────────── */

void test_12bit_packing(void) {
    printf("test_12bit_packing:\n");

    /* Pack two 12-bit values */
    uint16_t a = 0xABC;  /* 2748 */
    uint16_t b = 0x123;  /* 291  */
    uint8_t packed[3];

    hxq_pack_12bit_pair(a, b, packed);

    uint16_t ra, rb;
    hxq_unpack_12bit_pair(packed, &ra, &rb);

    ASSERT(ra == a, "unpack_a");
    ASSERT(rb == b, "unpack_b");

    /* Edge cases */
    hxq_pack_12bit_pair(0, 0, packed);
    hxq_unpack_12bit_pair(packed, &ra, &rb);
    ASSERT(ra == 0 && rb == 0, "zero_pair");

    hxq_pack_12bit_pair(4095, 4095, packed);
    hxq_unpack_12bit_pair(packed, &ra, &rb);
    ASSERT(ra == 4095 && rb == 4095, "max_pair");

    /* All values 0-4095 round-trip */
    int round_trip_ok = 1;
    for (uint16_t v = 0; v < 4096; v++) {
        hxq_pack_12bit_pair(v, 4095 - v, packed);
        hxq_unpack_12bit_pair(packed, &ra, &rb);
        if (ra != v || rb != (4095 - v)) {
            round_trip_ok = 0;
            break;
        }
    }
    ASSERT(round_trip_ok, "full_round_trip_4096");

    printf("  12-bit packing: done\n");
}

/* ── Test: sidecar application + confidence signal ───────────── */

void test_sidecar(void) {
    printf("test_sidecar:\n");

    hxq_tensor_t t;
    hxq_tensor_init(&t);

    /* Simple 2x2 codebook + indices */
    float codebook[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    uint8_t indices[4] = { 0, 1, 2, 3 };

    hxq_tensor_load_codebook(&t, codebook, 4, 1);
    /* Load as "256" for the scalar path — we'll use k=4 entries, rest are 0 */
    /* Actually let's use proper k=256 with only first 4 entries non-zero */
    float codebook256[256];
    memset(codebook256, 0, sizeof(codebook256));
    codebook256[0] = 1.0f;
    codebook256[1] = 2.0f;
    codebook256[2] = 3.0f;
    codebook256[3] = 4.0f;

    hxq_tensor_free(&t);
    hxq_tensor_init(&t);
    hxq_tensor_load_codebook(&t, codebook256, 256, 1);
    hxq_tensor_load_indices_8bit(&t, indices, 2, 2);

    /* Sidecar: correct W[0][1] by +0.5 and W[1][0] by -0.3 */
    uint32_t rows[2] = { 0, 1 };
    uint32_t cols[2] = { 1, 0 };
    float    vals[2] = { 0.5f, -0.3f };
    hxq_tensor_load_sidecar(&t, rows, cols, vals, 2);

    float output[4];
    hxq_tensor_decompress(&t, output);

    /* W[0][0] = codebook[0] = 1.0, no correction */
    ASSERT_FLOAT_EQ(output[0], 1.0f, 1e-6, "W[0][0]_no_sidecar");

    /* W[0][1] = codebook[1] + 0.5 = 2.5 */
    ASSERT_FLOAT_EQ(output[1], 2.5f, 1e-6, "W[0][1]_with_sidecar");

    /* W[1][0] = codebook[2] - 0.3 = 2.7 */
    ASSERT_FLOAT_EQ(output[2], 2.7f, 1e-6, "W[1][0]_with_sidecar");

    /* W[1][1] = codebook[3] = 4.0, no correction */
    ASSERT_FLOAT_EQ(output[3], 4.0f, 1e-6, "W[1][1]_no_sidecar");

    /* Confidence signal: L2 norm of sidecar = sqrt(0.25 + 0.09) = sqrt(0.34) */
    float expected_l2 = sqrtf(0.5f * 0.5f + 0.3f * 0.3f);
    ASSERT_FLOAT_EQ(hxq_get_sidecar_confidence(&t), expected_l2, 1e-5, "sidecar_l2");

    hxq_tensor_free(&t);
    printf("  sidecar: done\n");
}

/* ── Test: shared buffer ─────────────────────────────────────── */

void test_shared_buffer(void) {
    printf("test_shared_buffer:\n");

    hxq_shared_buffer_t buf;
    hxq_shared_buffer_init(&buf, 16);
    ASSERT(buf.capacity == 16, "initial_capacity");

    /* Create two tensors of different sizes */
    hxq_tensor_t t1, t2;
    hxq_tensor_init(&t1);
    hxq_tensor_init(&t2);

    float cb[256];
    for (int i = 0; i < 256; i++) cb[i] = (float)i;

    /* t1: 2x2 = 4 floats (fits in buffer) */
    uint8_t idx1[4] = { 0, 1, 2, 3 };
    hxq_tensor_load_codebook(&t1, cb, 256, 1);
    hxq_tensor_load_indices_8bit(&t1, idx1, 2, 2);

    /* t2: 4x4 = 16 floats (exactly fits) */
    uint8_t idx2[16] = { 10, 20, 30, 40, 50, 60, 70, 80,
                         90, 100, 110, 120, 130, 140, 150, 160 };
    hxq_tensor_load_codebook(&t2, cb, 256, 1);
    hxq_tensor_load_indices_8bit(&t2, idx2, 4, 4);

    float *out;

    /* Decompress t1 into shared buffer */
    hxq_error_t err = hxq_tensor_decompress_shared(&t1, &buf, &out);
    ASSERT(err == HXQ_OK, "shared_decompress_t1");
    ASSERT_FLOAT_EQ(out[0], 0.0f, 1e-6, "shared_t1[0]");

    /* Decompress t2 — reuses same buffer */
    err = hxq_tensor_decompress_shared(&t2, &buf, &out);
    ASSERT(err == HXQ_OK, "shared_decompress_t2");
    ASSERT_FLOAT_EQ(out[0], 10.0f, 1e-6, "shared_t2[0]");

    /* High water mark should be 16 (from t2) */
    ASSERT(buf.high_water == 16, "high_water");

    hxq_tensor_free(&t1);
    hxq_tensor_free(&t2);
    hxq_shared_buffer_free(&buf);
    printf("  shared buffer: done\n");
}

/* ── Test: 2D VQ 12-bit decompress ───────────────────────────── */

void test_vq2d_12bit(void) {
    printf("test_vq2d_12bit:\n");

    hxq_tensor_t t;
    hxq_tensor_init(&t);

    /* Codebook: 4096 entries, vdim=2 → [4096, 2] */
    float codebook[4096 * 2];
    for (int i = 0; i < 4096; i++) {
        codebook[i * 2]     = (float)i * 0.001f;       /* first element */
        codebook[i * 2 + 1] = (float)i * 0.001f + 0.5f; /* second element */
    }

    hxq_error_t err = hxq_tensor_load_codebook(&t, codebook, 4096, 2);
    ASSERT(err == HXQ_OK, "load_codebook_4096");

    /* 2x4 matrix: 2 rows, 4 cols → 2 pairs per row → 4 pairs total
     * 4 pairs = 4 indices, packed as 2 groups of 2 = 6 bytes */
    uint8_t packed[6];
    /* Row 0: pair(0,1) → idx=100, pair(2,3) → idx=200 */
    hxq_pack_12bit_pair(100, 200, &packed[0]);
    /* Row 1: pair(0,1) → idx=300, pair(2,3) → idx=400 */
    hxq_pack_12bit_pair(300, 400, &packed[3]);

    err = hxq_tensor_load_indices_12bit(&t, packed, 6, 2, 4);
    ASSERT(err == HXQ_OK, "load_indices_12bit");

    float output[8];
    memset(output, 0, sizeof(output));
    err = hxq_tensor_decompress(&t, output);
    ASSERT(err == HXQ_OK, "decompress_vq2d");

    /* Row 0, cols 0-1: codebook[100] = (0.100, 0.600) */
    ASSERT_FLOAT_EQ(output[0], 100.0f * 0.001f,       1e-4, "W[0][0]");
    ASSERT_FLOAT_EQ(output[1], 100.0f * 0.001f + 0.5f, 1e-4, "W[0][1]");

    /* Row 0, cols 2-3: codebook[200] = (0.200, 0.700) */
    ASSERT_FLOAT_EQ(output[2], 200.0f * 0.001f,       1e-4, "W[0][2]");
    ASSERT_FLOAT_EQ(output[3], 200.0f * 0.001f + 0.5f, 1e-4, "W[0][3]");

    hxq_tensor_free(&t);
    printf("  vq2d 12-bit: done\n");
}

/* ── Test: meta-kernel hxq_dequant ────────────────────────────── */

void test_meta_kernel(void) {
    printf("test_meta_kernel:\n");

    hxq_tensor_t t;
    hxq_tensor_init(&t);

    float cb[256];
    for (int i = 0; i < 256; i++) cb[i] = (float)i;

    uint8_t idx[6] = { 10, 20, 30, 40, 50, 60 };
    hxq_tensor_load_codebook(&t, cb, 256, 1);
    hxq_tensor_load_indices_8bit(&t, idx, 2, 3);

    /* Sidecar on W[1][2] */
    uint32_t sr[1] = { 1 };
    uint32_t sc[1] = { 2 };
    float    sv[1] = { 0.7f };
    hxq_tensor_load_sidecar(&t, sr, sc, sv, 1);

    hxq_shared_buffer_t buf;
    hxq_shared_buffer_init(&buf, 64);

    hxq_result_t result;
    hxq_error_t err = hxq_dequant(&t, &buf, HXQ_BACKEND_AUTO, &result);
    ASSERT(err == HXQ_OK, "dequant_ok");

    /* Check weights came through */
    ASSERT_FLOAT_EQ(result.weights[0], 10.0f, 1e-6, "meta_W[0][0]");
    ASSERT_FLOAT_EQ(result.weights[5], 60.7f, 1e-6, "meta_W[1][2]_sidecar");

    /* Check shape */
    ASSERT(result.out_features == 2, "meta_out_f");
    ASSERT(result.in_features == 3, "meta_in_f");

    /* Check confidence signal */
    float expected_conf = sqrtf(0.7f * 0.7f);
    ASSERT_FLOAT_EQ(result.confidence, expected_conf, 1e-5, "meta_confidence");

    hxq_tensor_free(&t);
    hxq_shared_buffer_free(&buf);
    printf("  meta-kernel: done\n");
}

/* ── Main ────────────────────────────────────────────────────── */

int main(void) {
    printf("=== HXQ Native Library Tests ===\n\n");

    test_scalar_vq();
    test_12bit_packing();
    test_sidecar();
    test_shared_buffer();
    test_vq2d_12bit();
    test_meta_kernel();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
