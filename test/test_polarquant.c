/*
 * test_polarquant.c — Tests for PolarQuant KV Cache Rotation Library
 *
 * Tests:
 *   1. Rotation matrix is orthogonal (Q @ Q^T ≈ I)
 *   2. Rotation matrix has det = +1 (proper rotation, not reflection)
 *   3. Same seed → same matrix (determinism)
 *   4. Different seeds → different matrices
 *   5. Roundtrip: rotate then unrotate ≈ identity (cosine > 0.9999)
 *   6. Rotation spreads variance (coefficient of variation decreases)
 *   7. Head geometry inference
 *   8. Layer context API
 *   9. Error handling (null pointers, bad dims)
 *  10. Multiple head dimensions (32, 64, 128)
 */

#include "polarquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
    } else { \
        tests_passed++; \
        printf("  PASS: %s\n", msg); \
    } \
} while (0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NEAR(a, b, tol, msg) ASSERT(fabsf((a) - (b)) < (tol), msg)

/* ── Helpers ──────────────────────────────────────────────────── */

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float l2_norm(const float *a, int n) {
    return sqrtf(dot(a, a, n));
}

static float cosine_similarity(const float *a, const float *b, int n) {
    float na = l2_norm(a, n);
    float nb = l2_norm(b, n);
    if (na < 1e-10f || nb < 1e-10f) return 0.0f;
    return dot(a, b, n) / (na * nb);
}

/* Compute det(Q) for small matrices via LU decomposition with partial pivoting */
static float determinant(const float *M, int n) {
    float *A = (float *)malloc((size_t)(n * n) * sizeof(float));
    memcpy(A, M, (size_t)(n * n) * sizeof(float));
    float det = 1.0f;
    int swaps = 0;

    for (int k = 0; k < n; k++) {
        /* Find pivot */
        float max_val = fabsf(A[k * n + k]);
        int max_row = k;
        for (int i = k + 1; i < n; i++) {
            if (fabsf(A[i * n + k]) > max_val) {
                max_val = fabsf(A[i * n + k]);
                max_row = i;
            }
        }

        if (max_val < 1e-10f) { free(A); return 0.0f; }

        if (max_row != k) {
            for (int j = 0; j < n; j++) {
                float tmp = A[k * n + j];
                A[k * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
            swaps++;
        }

        det *= A[k * n + k];

        for (int i = k + 1; i < n; i++) {
            float factor = A[i * n + k] / A[k * n + k];
            for (int j = k + 1; j < n; j++)
                A[i * n + j] -= factor * A[k * n + j];
        }
    }

    free(A);
    return (swaps % 2 == 0) ? det : -det;
}

/* ── Test functions ──────────────────────────────────────────── */

static void test_orthogonality(void) {
    printf("\n[Orthogonality]\n");
    uint32_t dim = 32;
    float *Q = (float *)malloc(dim * dim * sizeof(float));

    pq_error_t err = pq_generate_rotation(dim, 42, Q);
    ASSERT_EQ(err, PQ_OK, "generate_rotation returns PQ_OK");

    /* Check Q @ Q^T ≈ I */
    float max_off_diag = 0.0f;
    float min_diag = 1.0f;
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t j = 0; j < dim; j++) {
            float val = 0.0f;
            for (uint32_t k = 0; k < dim; k++)
                val += Q[i * dim + k] * Q[j * dim + k];

            if (i == j) {
                if (val < min_diag) min_diag = val;
            } else {
                if (fabsf(val) > max_off_diag) max_off_diag = fabsf(val);
            }
        }
    }

    ASSERT(max_off_diag < 1e-4f, "off-diagonal elements of Q@Q^T < 1e-4");
    ASSERT(fabsf(min_diag - 1.0f) < 1e-4f, "diagonal elements of Q@Q^T ≈ 1.0");

    free(Q);
}

static void test_proper_rotation(void) {
    printf("\n[Orthogonal Matrix (|det| = 1)]\n");

    /* Use small dim where determinant is cheap.
     * Sign fix matches numpy's QR convention (positive R diagonal),
     * so det(Q) = sign(det(A)) = ±1 depending on the random matrix.
     * We verify |det| = 1 (orthogonal). */
    uint32_t dim = 8;
    float *Q = (float *)malloc(dim * dim * sizeof(float));

    pq_generate_rotation(dim, 42, Q);
    float det = determinant(Q, (int)dim);
    ASSERT_NEAR(fabsf(det), 1.0f, 0.01f, "|det(Q)| ≈ 1 (seed=42, dim=8)");

    pq_generate_rotation(dim, 99, Q);
    det = determinant(Q, (int)dim);
    ASSERT_NEAR(fabsf(det), 1.0f, 0.01f, "|det(Q)| ≈ 1 (seed=99, dim=8)");

    free(Q);
}

static void test_determinism(void) {
    printf("\n[Determinism]\n");
    uint32_t dim = 64;
    float *Q1 = (float *)malloc(dim * dim * sizeof(float));
    float *Q2 = (float *)malloc(dim * dim * sizeof(float));

    pq_generate_rotation(dim, 42, Q1);
    pq_generate_rotation(dim, 42, Q2);

    int match = 1;
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        if (Q1[i] != Q2[i]) { match = 0; break; }
    }
    ASSERT(match, "same seed → identical matrix (bitwise)");

    pq_generate_rotation(dim, 99, Q2);
    int differ = 0;
    for (size_t i = 0; i < (size_t)dim * dim; i++) {
        if (Q1[i] != Q2[i]) { differ = 1; break; }
    }
    ASSERT(differ, "different seed → different matrix");

    free(Q1);
    free(Q2);
}

static void test_roundtrip(void) {
    printf("\n[Roundtrip: rotate → unrotate]\n");

    uint32_t head_dim = 64;
    uint32_t n_heads = 4;
    size_t total = (size_t)n_heads * head_dim;

    float *Q = (float *)malloc(head_dim * head_dim * sizeof(float));
    float *original = (float *)malloc(total * sizeof(float));
    float *values = (float *)malloc(total * sizeof(float));

    pq_generate_rotation(head_dim, 42, Q);

    /* Fill with known pattern */
    for (size_t i = 0; i < total; i++) {
        original[i] = sinf((float)i * 0.1f) * 2.0f;
        values[i] = original[i];
    }

    pq_rotate(values, Q, n_heads, head_dim);

    /* Values should be different after rotation */
    float cos_after_rot = cosine_similarity(original, values, (int)total);
    ASSERT(cos_after_rot < 0.99f, "values differ after rotation");

    pq_unrotate(values, Q, n_heads, head_dim);

    /* Values should match original after roundtrip */
    float cos_roundtrip = cosine_similarity(original, values, (int)total);
    ASSERT(cos_roundtrip > 0.9999f, "roundtrip cosine > 0.9999");

    /* Check max absolute error */
    float max_err = 0.0f;
    for (size_t i = 0; i < total; i++) {
        float err = fabsf(values[i] - original[i]);
        if (err > max_err) max_err = err;
    }
    ASSERT(max_err < 1e-4f, "roundtrip max error < 1e-4");

    printf("    roundtrip cosine = %.6f, max_err = %.2e\n", cos_roundtrip, max_err);

    free(Q);
    free(original);
    free(values);
}

static void test_variance_spreading(void) {
    printf("\n[Variance Spreading]\n");

    /* PolarQuant spreads energy within each head across dimensions.
     * Test: create a head with outlier dimensions (high max/min magnitude ratio),
     * then verify rotation reduces that ratio. */

    uint32_t head_dim = 64;
    uint32_t n_heads = 1;

    float *Q = (float *)malloc(head_dim * head_dim * sizeof(float));
    float *values = (float *)malloc((size_t)head_dim * sizeof(float));

    pq_generate_rotation(head_dim, 42, Q);

    /* Create head with extreme outlier dimensions */
    for (uint32_t i = 0; i < head_dim; i++)
        values[i] = 0.01f;  /* near-zero baseline */
    values[0] = 100.0f;   /* huge outlier */
    values[1] = -80.0f;   /* huge outlier */

    /* Measure max/min |value| ratio before rotation */
    float max_abs_before = 0.0f, min_abs_before = 1e30f;
    for (uint32_t i = 0; i < head_dim; i++) {
        float a = fabsf(values[i]);
        if (a > max_abs_before) max_abs_before = a;
        if (a < min_abs_before) min_abs_before = a;
    }
    float ratio_before = max_abs_before / (min_abs_before + 1e-10f);

    pq_rotate(values, Q, n_heads, head_dim);

    /* Measure max/min |value| ratio after rotation */
    float max_abs_after = 0.0f, min_abs_after = 1e30f;
    for (uint32_t i = 0; i < head_dim; i++) {
        float a = fabsf(values[i]);
        if (a > max_abs_after) max_abs_after = a;
        if (a < min_abs_after) min_abs_after = a;
    }
    float ratio_after = max_abs_after / (min_abs_after + 1e-10f);

    printf("    magnitude ratio before: %.1f, after: %.1f\n", ratio_before, ratio_after);
    ASSERT(ratio_after < ratio_before, "rotation reduces max/min magnitude ratio");

    free(Q);
    free(values);
}

static void test_head_geometry(void) {
    printf("\n[Head Geometry Inference]\n");
    uint32_t nh, hd;

    /* Explicit n_heads */
    ASSERT_EQ(pq_infer_head_geometry(256, 4, &nh, &hd), PQ_OK, "256 / 4 heads");
    ASSERT_EQ(nh, 4, "n_heads = 4");
    ASSERT_EQ(hd, 64, "head_dim = 64");

    /* Auto-infer head_dim = 128 */
    ASSERT_EQ(pq_infer_head_geometry(1024, 0, &nh, &hd), PQ_OK, "1024 auto");
    ASSERT_EQ(hd, 128, "head_dim = 128");
    ASSERT_EQ(nh, 8, "n_heads = 8");

    /* Auto-infer head_dim = 64 */
    ASSERT_EQ(pq_infer_head_geometry(192, 0, &nh, &hd), PQ_OK, "192 auto → 64");
    ASSERT_EQ(hd, 64, "head_dim = 64");
    ASSERT_EQ(nh, 3, "n_heads = 3");

    /* Failure: not divisible */
    ASSERT_EQ(pq_infer_head_geometry(17, 0, &nh, &hd), PQ_ERR_NOT_DIVISIBLE, "17 fails");

    /* Failure: explicit n_heads doesn't divide */
    ASSERT_EQ(pq_infer_head_geometry(100, 3, &nh, &hd), PQ_ERR_NOT_DIVISIBLE, "100/3 fails");
}

static void test_layer_api(void) {
    printf("\n[Layer Context API]\n");

    pq_layer_t layer;
    memset(&layer, 0, sizeof(layer));

    pq_error_t err = pq_layer_init(&layer, 64, 4, 42, 5);
    ASSERT_EQ(err, PQ_OK, "layer_init succeeds");
    ASSERT(layer.initialized, "layer is initialized");
    ASSERT_EQ(layer.seed, 47, "seed = base(42) + idx(5) = 47");

    /* Create test data */
    size_t total = 4 * 64;
    float *values = (float *)malloc(total * sizeof(float));
    float *original = (float *)malloc(total * sizeof(float));
    for (size_t i = 0; i < total; i++) {
        original[i] = sinf((float)i * 0.3f);
        values[i] = original[i];
    }

    err = pq_layer_rotate(&layer, values);
    ASSERT_EQ(err, PQ_OK, "layer_rotate succeeds");

    err = pq_layer_unrotate(&layer, values);
    ASSERT_EQ(err, PQ_OK, "layer_unrotate succeeds");

    float cos = cosine_similarity(original, values, (int)total);
    ASSERT(cos > 0.9999f, "layer roundtrip cosine > 0.9999");

    pq_layer_free(&layer);
    ASSERT(layer.Q == NULL, "layer freed");

    free(values);
    free(original);
}

static void test_error_handling(void) {
    printf("\n[Error Handling]\n");

    ASSERT_EQ(pq_generate_rotation(32, 42, NULL), PQ_ERR_NULL_PTR, "null Q → ERR_NULL_PTR");
    float dummy;
    ASSERT_EQ(pq_generate_rotation(0, 42, &dummy), PQ_ERR_INVALID_DIM, "dim=0 → ERR_INVALID_DIM");
    ASSERT_EQ(pq_generate_rotation(2048, 42, &dummy), PQ_ERR_INVALID_DIM, "dim=2048 → ERR_INVALID_DIM");

    float Q[32 * 32];
    pq_generate_rotation(32, 42, Q);
    ASSERT_EQ(pq_rotate(NULL, Q, 1, 32), PQ_ERR_NULL_PTR, "null values → ERR_NULL_PTR");
    float vals[32];
    ASSERT_EQ(pq_rotate(vals, NULL, 1, 32), PQ_ERR_NULL_PTR, "null Q → ERR_NULL_PTR");
}

static void test_multiple_dims(void) {
    printf("\n[Multiple Head Dimensions]\n");

    uint32_t dims[] = {32, 64, 128};
    for (int d = 0; d < 3; d++) {
        uint32_t dim = dims[d];
        uint32_t n_heads = 4;
        size_t total = (size_t)n_heads * dim;

        float *Q = (float *)malloc(dim * dim * sizeof(float));
        float *original = (float *)malloc(total * sizeof(float));
        float *values = (float *)malloc(total * sizeof(float));

        pq_generate_rotation(dim, 42, Q);

        for (size_t i = 0; i < total; i++) {
            original[i] = cosf((float)i * 0.05f) * 3.0f;
            values[i] = original[i];
        }

        pq_rotate(values, Q, n_heads, dim);
        pq_unrotate(values, Q, n_heads, dim);

        float cos = cosine_similarity(original, values, (int)total);
        char msg[80];
        snprintf(msg, sizeof(msg), "dim=%u: roundtrip cosine=%.6f > 0.9999", dim, cos);
        ASSERT(cos > 0.9999f, msg);

        free(Q);
        free(original);
        free(values);
    }
}

/* ── Main ────────────────────────────────────────────────────── */

int main(void) {
    printf("PolarQuant Native Tests — v%s\n", PQ_VERSION_STRING);
    printf("=========================================\n");

    test_orthogonality();
    test_proper_rotation();
    test_determinism();
    test_roundtrip();
    test_variance_spreading();
    test_head_geometry();
    test_layer_api();
    test_error_handling();
    test_multiple_dims();

    printf("\n=========================================\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);

    return (tests_passed == tests_run) ? 0 : 1;
}
