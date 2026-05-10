/*
 * hxq_lloyd.c — Lloyd's Reassignment for Born-Compressed Training
 *
 * Pure C99 + OpenMP. Uses SIMD-friendly memory layout and loop patterns
 * that GCC/Clang auto-vectorize with -O3 -march=native.
 *
 * The hot loop is Phase 1 (distance matrix). For n_vectors=590K, k=256, d=2:
 *   590K * 256 * 2 = ~302M FMA operations → ~1ms on modern AVX2 hardware.
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 */

#include "hxq_lloyd.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Compile-time tuning ────────────────────────────────────── */

/* Process vectors in chunks to stay in L2 cache.
 * At d=4, each vector is 16B. 4096 vectors = 64KB = fits in L2. */
#define CHUNK_SIZE 4096

/* ── Phase 1: Reassign ──────────────────────────────────────── */

/*
 * Find nearest centroid for each vector.
 * dist(i,j) = ||w_i||^2 + ||c_j||^2 - 2*(w_i · c_j)
 *
 * Precompute ||c_j||^2 once, then for each chunk:
 *   - Precompute ||w_i||^2 for the chunk
 *   - Inner loop: accumulate dot(w_i, c_j) and find argmin
 */
static void reassign_vectors(
    const float *weights,   /* [n_vectors * d] */
    const float *codebook,  /* [k * d] */
    uint8_t     *indices,   /* [n_vectors] */
    size_t       n_vectors,
    int          k,
    int          d
) {
    /* Precompute ||c_j||^2 */
    float *c_sq = (float *)malloc((size_t)k * sizeof(float));
    for (int j = 0; j < k; j++) {
        float s = 0.0f;
        const float *cj = codebook + (size_t)j * d;
        for (int p = 0; p < d; p++) {
            s += cj[p] * cj[p];
        }
        c_sq[j] = s;
    }

    /* Process in chunks (OpenMP parallel over chunks) */
    size_t n_chunks = (n_vectors + CHUNK_SIZE - 1) / CHUNK_SIZE;

    #pragma omp parallel for schedule(dynamic)
    for (size_t ci = 0; ci < n_chunks; ci++) {
        size_t start = ci * CHUNK_SIZE;
        size_t end = start + CHUNK_SIZE;
        if (end > n_vectors) end = n_vectors;

        for (size_t i = start; i < end; i++) {
            const float *wi = weights + i * d;

            /* ||w_i||^2 */
            float w_sq = 0.0f;
            for (int p = 0; p < d; p++) {
                w_sq += wi[p] * wi[p];
            }

            /* Find nearest centroid */
            float best_dist = FLT_MAX;
            int best_j = 0;

            for (int j = 0; j < k; j++) {
                /* dot(w_i, c_j) */
                float dot = 0.0f;
                const float *cj = codebook + (size_t)j * d;
                for (int p = 0; p < d; p++) {
                    dot += wi[p] * cj[p];
                }
                float dist = w_sq + c_sq[j] - 2.0f * dot;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_j = j;
                }
            }
            indices[i] = (uint8_t)best_j;
        }
    }

    free(c_sq);
}


/* ── Phase 2: Update centroids ──────────────────────────────── */

/*
 * Recompute centroids as mean of assigned vectors.
 * Uses per-thread accumulators to avoid atomics, then reduces.
 */
static void update_centroids(
    const float   *weights,   /* [n_vectors * d] */
    const uint8_t *indices,   /* [n_vectors] */
    float         *codebook,  /* [k * d] — UPDATED */
    int           *counts,    /* [k] — OUTPUT: assignment counts */
    size_t         n_vectors,
    int            k,
    int            d
) {
    size_t kd = (size_t)k * d;

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    /* Per-thread accumulators: [n_threads][k * d] for sums, [n_threads][k] for counts */
    double *all_sums = (double *)calloc((size_t)n_threads * kd, sizeof(double));
    int *all_counts = (int *)calloc((size_t)n_threads * k, sizeof(int));

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        double *my_sums = all_sums + (size_t)tid * kd;
        int *my_counts = all_counts + (size_t)tid * k;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n_vectors; i++) {
            int j = indices[i];
            my_counts[j]++;
            double *dst = my_sums + (size_t)j * d;
            const float *src = weights + i * d;
            for (int p = 0; p < d; p++) {
                dst[p] += (double)src[p];
            }
        }
    }

    /* Reduce across threads */
    memset(counts, 0, (size_t)k * sizeof(int));
    double *final_sums = (double *)calloc(kd, sizeof(double));

    for (int t = 0; t < n_threads; t++) {
        double *t_sums = all_sums + (size_t)t * kd;
        int *t_counts = all_counts + (size_t)t * k;
        for (int j = 0; j < k; j++) {
            counts[j] += t_counts[j];
        }
        for (size_t s = 0; s < kd; s++) {
            final_sums[s] += t_sums[s];
        }
    }

    /* Update alive centroids */
    for (int j = 0; j < k; j++) {
        if (counts[j] > 0) {
            double *sum_j = final_sums + (size_t)j * d;
            float *cb_j = codebook + (size_t)j * d;
            double inv_count = 1.0 / (double)counts[j];
            for (int p = 0; p < d; p++) {
                cb_j[p] = (float)(sum_j[p] * inv_count);
            }
        }
    }

    free(all_sums);
    free(all_counts);
    free(final_sums);
}


/* ── Phase 3: Reinitialize dead centroids ───────────────────── */

/*
 * For each dead centroid (count == 0):
 *   Find the largest cluster.
 *   Find the member farthest from that cluster's centroid.
 *   Move dead centroid there (+ small jitter).
 */
static int reinit_dead_centroids(
    const float   *weights,
    const uint8_t *indices,
    float         *codebook,
    const int     *counts,
    size_t         n_vectors,
    int            k,
    int            d
) {
    /* Count dead centroids */
    int n_dead = 0;
    for (int j = 0; j < k; j++) {
        if (counts[j] == 0) n_dead++;
    }
    if (n_dead == 0) return 0;

    /* Find largest cluster */
    int largest = 0;
    for (int j = 1; j < k; j++) {
        if (counts[j] > counts[largest]) largest = j;
    }

    /* For each dead centroid, steal from largest cluster */
    int dead_idx = 0;
    for (int j = 0; j < k && dead_idx < n_dead; j++) {
        if (counts[j] != 0) continue;

        /* Find farthest member from largest cluster's centroid */
        const float *lc = codebook + (size_t)largest * d;
        float max_dist = -1.0f;
        size_t farthest = 0;

        for (size_t i = 0; i < n_vectors; i++) {
            if (indices[i] != (uint8_t)largest) continue;
            const float *wi = weights + i * d;
            float dist = 0.0f;
            for (int p = 0; p < d; p++) {
                float diff = wi[p] - lc[p];
                dist += diff * diff;
            }
            if (dist > max_dist) {
                max_dist = dist;
                farthest = i;
            }
        }

        /* Move dead centroid to farthest member + tiny jitter */
        float *cb_j = codebook + (size_t)j * d;
        const float *wf = weights + farthest * d;
        for (int p = 0; p < d; p++) {
            cb_j[p] = wf[p] * 1.001f;  /* 0.1% jitter */
        }

        dead_idx++;
    }

    return n_dead;
}


/* ── Public API ─────────────────────────────────────────────── */

int hxq_lloyd_reassign(
    const float *weights,
    float       *codebook,
    uint8_t     *indices,
    size_t       n_vectors,
    int          k,
    int          d,
    int         *n_dead
) {
    /* Validate */
    if (!weights || !codebook || !indices) return HXQ_LLOYD_ERR_NULL;
    if (k < 1 || k > 256) return HXQ_LLOYD_ERR_K;
    if (d != 1 && d != 2 && d != 4 && d != 8) return HXQ_LLOYD_ERR_DIM;
    if (n_vectors == 0) {
        if (n_dead) *n_dead = 0;
        return HXQ_LLOYD_OK;
    }

    /* Allocate counts */
    int *counts = (int *)calloc((size_t)k, sizeof(int));
    if (!counts) return HXQ_LLOYD_ERR_ALLOC;

    /* Phase 1: Reassign */
    reassign_vectors(weights, codebook, indices, n_vectors, k, d);

    /* Phase 2: Update centroids */
    update_centroids(weights, indices, codebook, counts, n_vectors, k, d);

    /* Phase 3: Reinitialize dead centroids */
    int dead = reinit_dead_centroids(weights, indices, codebook, counts,
                                      n_vectors, k, d);
    if (n_dead) *n_dead = dead;

    free(counts);
    return HXQ_LLOYD_OK;
}


int hxq_lloyd_assign_only(
    const float *weights,
    const float *codebook,
    uint8_t     *indices,
    size_t       n_vectors,
    int          k,
    int          d
) {
    if (!weights || !codebook || !indices) return HXQ_LLOYD_ERR_NULL;
    if (k < 1 || k > 256) return HXQ_LLOYD_ERR_K;
    if (d != 1 && d != 2 && d != 4 && d != 8) return HXQ_LLOYD_ERR_DIM;
    if (n_vectors == 0) return HXQ_LLOYD_OK;

    reassign_vectors(weights, (float *)codebook, indices, n_vectors, k, d);
    return HXQ_LLOYD_OK;
}
