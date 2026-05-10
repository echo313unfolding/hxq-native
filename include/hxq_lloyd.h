/*
 * hxq_lloyd.h — Lloyd's Reassignment for Born-Compressed Training
 *
 * One iteration of Lloyd's algorithm on grouped VQ codebooks:
 *   Phase 1: Reassign — find nearest centroid for each weight vector
 *   Phase 2: Update  — recompute centroids as mean of assignments
 *   Phase 3: Reinit  — replace dead centroids from largest cluster
 *
 * Designed for integration with the born-compressed training loop.
 * Replaces the Python/numpy implementation (~206s) with SIMD + OpenMP (~ms).
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 * License: Apache 2.0
 *
 * Usage:
 *   #include "hxq_lloyd.h"
 *
 *   int n_dead = 0;
 *   int rc = hxq_lloyd_reassign(
 *       weights, codebook, indices,
 *       n_vectors, k, d, &n_dead
 *   );
 *
 * Build:
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o lib/libhxq_lloyd.so src/hxq_lloyd.c -lm
 */

#ifndef HXQ_LLOYD_H
#define HXQ_LLOYD_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Error codes (extend hxq_error_t) ────────────────────────── */

#define HXQ_LLOYD_OK         0
#define HXQ_LLOYD_ERR_NULL  -1
#define HXQ_LLOYD_ERR_K     -2   /* k out of range [1, 256] */
#define HXQ_LLOYD_ERR_DIM   -3   /* d not in {1, 2, 4, 8}  */
#define HXQ_LLOYD_ERR_ALLOC -4

/* ── Main API ────────────────────────────────────────────────── */

/*
 * Grouped VQ Lloyd's reassignment — one iteration.
 *
 * Phase 1: For each vector, find nearest centroid (||w - c||^2).
 * Phase 2: Recompute centroids as mean of assigned vectors.
 * Phase 3: Reinitialize dead centroids from the largest cluster.
 *
 * @param weights    [n_vectors * d] float32, row-major weight vectors
 * @param codebook   [k * d] float32, row-major codebook entries (UPDATED)
 * @param indices    [n_vectors] uint8, assignments (UPDATED)
 * @param n_vectors  number of weight groups
 * @param k          codebook size (1..256 for uint8)
 * @param d          vector dimension (1, 2, 4, or 8)
 * @param n_dead     output: number of dead centroids found (may be NULL)
 *
 * @return HXQ_LLOYD_OK on success, negative on error
 */
int hxq_lloyd_reassign(
    const float *weights,
    float       *codebook,
    uint8_t     *indices,
    size_t       n_vectors,
    int          k,
    int          d,
    int         *n_dead
);

/*
 * Distance-only pass — compute squared L2 distances without updating.
 * Useful for diagnostics (codebook utilization, quality metrics).
 *
 * @param weights    [n_vectors * d] float32
 * @param codebook   [k * d] float32
 * @param indices    [n_vectors] uint8, nearest centroid per vector (OUTPUT)
 * @param n_vectors  number of weight groups
 * @param k          codebook size
 * @param d          vector dimension
 *
 * @return HXQ_LLOYD_OK on success
 */
int hxq_lloyd_assign_only(
    const float *weights,
    const float *codebook,
    uint8_t     *indices,
    size_t       n_vectors,
    int          k,
    int          d
);

#ifdef __cplusplus
}
#endif

#endif /* HXQ_LLOYD_H */
