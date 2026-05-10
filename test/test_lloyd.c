/*
 * test_lloyd.c — Unit tests for hxq_lloyd_reassign
 *
 * Tests:
 *   1-7.   Input validation
 *   8.     Scalar d=1: assign_only idempotent on fixed codebook
 *   9-10.  Grouped d=2, d=4: same
 *   11.    Centroid update: distortion decreases after iteration
 *   12.    Dead centroid reinitialization
 *   13-14. Production-scale benchmarks
 */

#include "hxq_lloyd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int tests_run = 0;
static int tests_pass = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  [%2d] %-50s ", tests_run, name); \
    fflush(stdout); \
} while(0)

#define PASS() do { tests_pass++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

/* ── Helper: simple RNG ───────────────────────────────────── */

static float randf(unsigned int *seed) {
    *seed = *seed * 1103515245u + 12345u;
    return ((float)(*seed >> 16) / 32768.0f) - 1.0f;
}

/* ── Helper: total distortion ─────────────────────────────── */

static double total_distortion(
    const float *w, const float *cb, const uint8_t *idx,
    size_t n, int k, int d
) {
    (void)k;
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        const float *wi = w + i * d;
        const float *cj = cb + (size_t)idx[i] * d;
        for (int p = 0; p < d; p++) {
            double diff = (double)wi[p] - (double)cj[p];
            total += diff * diff;
        }
    }
    return total;
}

/* ── Test: input validation ───────────────────────────────── */

static void test_validation(void) {
    float w[4] = {1, 2, 3, 4};
    float cb[4] = {0, 0, 0, 0};
    uint8_t idx[2] = {0, 0};
    int n_dead;

    TEST("null weights");
    if (hxq_lloyd_reassign(NULL, cb, idx, 2, 2, 2, &n_dead) == HXQ_LLOYD_ERR_NULL)
        PASS(); else FAIL("expected ERR_NULL");

    TEST("null codebook");
    if (hxq_lloyd_reassign(w, NULL, idx, 2, 2, 2, &n_dead) == HXQ_LLOYD_ERR_NULL)
        PASS(); else FAIL("expected ERR_NULL");

    TEST("null indices");
    if (hxq_lloyd_reassign(w, cb, NULL, 2, 2, 2, &n_dead) == HXQ_LLOYD_ERR_NULL)
        PASS(); else FAIL("expected ERR_NULL");

    TEST("k=0");
    if (hxq_lloyd_reassign(w, cb, idx, 2, 0, 2, &n_dead) == HXQ_LLOYD_ERR_K)
        PASS(); else FAIL("expected ERR_K");

    TEST("k=257");
    if (hxq_lloyd_reassign(w, cb, idx, 2, 257, 2, &n_dead) == HXQ_LLOYD_ERR_K)
        PASS(); else FAIL("expected ERR_K");

    TEST("d=3 (invalid)");
    if (hxq_lloyd_reassign(w, cb, idx, 2, 4, 3, &n_dead) == HXQ_LLOYD_ERR_DIM)
        PASS(); else FAIL("expected ERR_DIM");

    TEST("n_vectors=0 (no-op)");
    if (hxq_lloyd_reassign(w, cb, idx, 0, 4, 2, &n_dead) == HXQ_LLOYD_OK)
        PASS(); else FAIL("expected OK");
}

/* ── Test: assign_only idempotent ─────────────────────────── */

static void test_assign_idempotent(int d, const char *label) {
    char name[80];
    snprintf(name, 80, "%s assign_only is idempotent", label);
    TEST(name);

    int k = 32, n = 500;
    unsigned int seed = 42 + d;

    float *w = (float *)malloc((size_t)n * d * sizeof(float));
    float *cb = (float *)malloc((size_t)k * d * sizeof(float));
    uint8_t *idx1 = (uint8_t *)malloc((size_t)n);
    uint8_t *idx2 = (uint8_t *)malloc((size_t)n);

    for (int i = 0; i < n * d; i++) w[i] = randf(&seed);
    for (int j = 0; j < k * d; j++) cb[j] = randf(&seed);

    /* Two calls with same inputs should produce identical output */
    hxq_lloyd_assign_only(w, cb, idx1, (size_t)n, k, d);
    hxq_lloyd_assign_only(w, cb, idx2, (size_t)n, k, d);

    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (idx1[i] != idx2[i]) mismatches++;
    }

    if (mismatches == 0) PASS();
    else { char buf[64]; snprintf(buf, 64, "%d mismatches", mismatches); FAIL(buf); }

    free(w); free(cb); free(idx1); free(idx2);
}

/* ── Test: distortion decreases after Lloyd's ─────────────── */

static void test_distortion_decreases(void) {
    TEST("distortion decreases after Lloyd iteration");

    int k = 16, d = 2, n = 1000;
    unsigned int seed = 789;

    float *w = (float *)malloc((size_t)n * d * sizeof(float));
    float *cb = (float *)malloc((size_t)k * d * sizeof(float));
    uint8_t *idx = (uint8_t *)malloc((size_t)n);

    for (int i = 0; i < n * d; i++) w[i] = randf(&seed);
    for (int j = 0; j < k * d; j++) cb[j] = randf(&seed);

    /* Initial assignment */
    hxq_lloyd_assign_only(w, cb, idx, (size_t)n, k, d);
    double dist_before = total_distortion(w, cb, idx, (size_t)n, k, d);

    /* One Lloyd iteration */
    int n_dead;
    hxq_lloyd_reassign(w, cb, idx, (size_t)n, k, d, &n_dead);

    /* After full Lloyd (reassign + update), re-assign to get consistent indices */
    hxq_lloyd_assign_only(w, cb, idx, (size_t)n, k, d);
    double dist_after = total_distortion(w, cb, idx, (size_t)n, k, d);

    if (dist_after <= dist_before) {
        char buf[128];
        snprintf(buf, 128, "%.2f -> %.2f (%.1f%% reduction)",
                 dist_before, dist_after,
                 100.0 * (1.0 - dist_after / dist_before));
        printf("PASS %s\n", buf);
        tests_pass++;
    } else {
        char buf[128];
        snprintf(buf, 128, "increased: %.2f -> %.2f", dist_before, dist_after);
        FAIL(buf);
    }

    free(w); free(cb); free(idx);
}

/* ── Test: dead centroid reinitialization ──────────────────── */

static void test_dead_reinit(void) {
    TEST("dead centroid reinitialization");

    /* All vectors near (1,1). 4 centroids: c0=(1,1), c1/c2/c3 far away.
     * All vectors should assign to c0, leaving c1/c2/c3 dead.
     * Dead centroids should be reinitialized near the data. */
    int n = 10, k = 4, d = 2;
    float w[20];
    for (int i = 0; i < n; i++) {
        w[i * 2] = 1.0f + (float)i * 0.01f;
        w[i * 2 + 1] = 1.0f;
    }
    float cb[8] = {1.0f, 1.0f,  100.0f, 100.0f,  -100.0f, -100.0f,  200.0f, 200.0f};
    uint8_t idx[10];
    memset(idx, 0, 10);

    int n_dead;
    hxq_lloyd_reassign(w, cb, idx, (size_t)n, k, d, &n_dead);

    /* After reinit, dead centroids should be moved near the data */
    int reinitialized = 0;
    for (int j = 1; j < k; j++) {
        float dist = 0.0f;
        for (int p = 0; p < d; p++) {
            float diff = cb[j * d + p] - 1.0f;
            dist += diff * diff;
        }
        if (dist < 1.0f) reinitialized++;
    }

    if (n_dead >= 2 && reinitialized >= 2) PASS();
    else {
        char buf[64];
        snprintf(buf, 64, "n_dead=%d, reinit=%d", n_dead, reinitialized);
        FAIL(buf);
    }
}

/* ── Benchmark: production scale ──────────────────────────── */

static void bench_production_scale(void) {
    int k = 256, d = 2;
    size_t n = 590000;

    TEST("benchmark: 590K vectors, k=256, d=2");

    float *w = (float *)malloc(n * d * sizeof(float));
    float *cb = (float *)malloc((size_t)k * d * sizeof(float));
    uint8_t *idx = (uint8_t *)malloc(n);

    unsigned int seed = 999;
    for (size_t i = 0; i < n * d; i++) w[i] = randf(&seed);
    for (int j = 0; j < k * d; j++) cb[j] = randf(&seed);

    int n_dead;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int rc = hxq_lloyd_reassign(w, cb, idx, n, k, d, &n_dead);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                (t1.tv_nsec - t0.tv_nsec) / 1e6;

    if (rc == HXQ_LLOYD_OK) {
        printf("PASS (%.1f ms, %d dead)\n", ms, n_dead);
        tests_pass++;
    } else {
        char buf[64]; snprintf(buf, 64, "rc=%d", rc); FAIL(buf);
    }

    free(w); free(cb); free(idx);
}

static void bench_d4_scale(void) {
    int k = 256, d = 4;
    size_t n = 295000;

    TEST("benchmark: 295K vectors, k=256, d=4");

    float *w = (float *)malloc(n * d * sizeof(float));
    float *cb = (float *)malloc((size_t)k * d * sizeof(float));
    uint8_t *idx = (uint8_t *)malloc(n);

    unsigned int seed = 777;
    for (size_t i = 0; i < n * d; i++) w[i] = randf(&seed);
    for (int j = 0; j < k * d; j++) cb[j] = randf(&seed);

    int n_dead;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int rc = hxq_lloyd_reassign(w, cb, idx, n, k, d, &n_dead);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                (t1.tv_nsec - t0.tv_nsec) / 1e6;

    if (rc == HXQ_LLOYD_OK) {
        printf("PASS (%.1f ms, %d dead)\n", ms, n_dead);
        tests_pass++;
    } else {
        char buf[64]; snprintf(buf, 64, "rc=%d", rc); FAIL(buf);
    }

    free(w); free(cb); free(idx);
}

/* ── Main ─────────────────────────────────────────────────── */

int main(void) {
    printf("\nHXQ Lloyd's Reassignment — Test Suite\n");
    printf("======================================\n\n");

    /* Validation */
    test_validation();

    /* Correctness */
    test_assign_idempotent(1, "scalar d=1");
    test_assign_idempotent(2, "grouped d=2");
    test_assign_idempotent(4, "grouped d=4");
    test_distortion_decreases();
    test_dead_reinit();

    /* Benchmarks */
    printf("\n  Benchmarks:\n");
    bench_production_scale();
    bench_d4_scale();

    printf("\n======================================\n");
    printf("Results: %d/%d passed\n\n", tests_pass, tests_run);

    return (tests_pass == tests_run) ? 0 : 1;
}
