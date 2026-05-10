/*
 * polarquant.h — PolarQuant KV Cache Rotation Library
 *
 * Random orthogonal rotation before scalar VQ on KV cache tensors.
 * Spreads outlier-dimension energy uniformly, improving centroid
 * utilization by 55-59% MSE on K-cache (receipted).
 *
 * The rotation matrix is deterministic per seed — zero storage overhead.
 * Reconstructed via QR decomposition of a seeded Gaussian matrix.
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 * License: Apache 2.0
 *
 * Usage:
 *   #include "polarquant.h"
 *
 *   // Generate rotation matrix (once per layer)
 *   float Q[128*128];
 *   pq_generate_rotation(128, 42, Q);
 *
 *   // Rotate before quantization
 *   pq_rotate(kv_values, Q, n_heads, head_dim);
 *
 *   // ... quantize ...
 *
 *   // Unrotate after dequantization
 *   pq_unrotate(kv_values, Q, n_heads, head_dim);
 */

#ifndef POLARQUANT_H
#define POLARQUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ─────────────────────────────────────────────────── */

#define PQ_VERSION_MAJOR 0
#define PQ_VERSION_MINOR 1
#define PQ_VERSION_PATCH 0
#define PQ_VERSION_STRING "0.1.0"

/* ── Error codes ─────────────────────────────────────────────── */

typedef enum {
    PQ_OK = 0,
    PQ_ERR_NULL_PTR,
    PQ_ERR_INVALID_DIM,
    PQ_ERR_ALLOC_FAILED,
    PQ_ERR_NOT_DIVISIBLE,
} pq_error_t;

/* ── Core API ────────────────────────────────────────────────── */

/**
 * Generate a deterministic orthogonal rotation matrix via Householder QR.
 *
 * Algorithm:
 *   1. Seed a Gaussian random matrix H[dim, dim] (Box-Muller)
 *   2. Compute QR decomposition via Householder reflections
 *   3. Fix sign: Q *= sign(diag(R)) to match numpy convention (positive R diagonal)
 *
 * The result matches numpy.linalg.qr on the same RandomState(seed)
 * Gaussian draws.
 *
 * @param dim   Dimension (head_dim, typically 32/64/128)
 * @param seed  Random seed (use base_seed + layer_idx for per-layer rotation)
 * @param Q     Output: pre-allocated float buffer of size [dim * dim], row-major
 * @return PQ_OK on success
 */
pq_error_t pq_generate_rotation(uint32_t dim, uint32_t seed, float *Q);

/**
 * Rotate KV values in-place before quantization.
 *
 * For each of n_heads attention heads:
 *   head[i] = head[i] @ Q^T
 *
 * This spreads outlier energy across all dimensions, making scalar VQ
 * more effective (55-59% K MSE improvement on layers 5-20).
 *
 * @param values    Float array of size [n_heads * head_dim], modified in-place
 * @param Q         Rotation matrix [head_dim * head_dim], row-major
 * @param n_heads   Number of attention heads
 * @param head_dim  Dimension per head (must match Q)
 * @return PQ_OK on success
 */
pq_error_t pq_rotate(
    float          *values,
    const float    *Q,
    uint32_t        n_heads,
    uint32_t        head_dim
);

/**
 * Unrotate KV values in-place after dequantization.
 *
 * Since Q is orthogonal, Q^{-1} = Q^T, so:
 *   head[i] = head[i] @ Q
 *
 * @param values    Float array of size [n_heads * head_dim], modified in-place
 * @param Q         Rotation matrix [head_dim * head_dim], row-major
 * @param n_heads   Number of attention heads
 * @param head_dim  Dimension per head (must match Q)
 * @return PQ_OK on success
 */
pq_error_t pq_unrotate(
    float          *values,
    const float    *Q,
    uint32_t        n_heads,
    uint32_t        head_dim
);

/**
 * Infer (n_heads, head_dim) from flat entry size.
 *
 * If n_heads_hint > 0, uses it directly.
 * Otherwise tries common head_dim values: 128, 64, 32.
 *
 * @param entry_size     Total flat size (n_heads * head_dim)
 * @param n_heads_hint   Hint (0 = auto-infer)
 * @param out_n_heads    Output: number of heads
 * @param out_head_dim   Output: dimension per head
 * @return PQ_OK on success, PQ_ERR_NOT_DIVISIBLE if geometry can't be inferred
 */
pq_error_t pq_infer_head_geometry(
    uint32_t  entry_size,
    uint32_t  n_heads_hint,
    uint32_t *out_n_heads,
    uint32_t *out_head_dim
);

/* ── Layer context (optional convenience wrapper) ────────────── */

/**
 * Per-layer PolarQuant state. Holds the precomputed rotation matrix
 * and head geometry for a single KV cache layer.
 */
typedef struct {
    float   *Q;           /* [head_dim * head_dim] rotation matrix  */
    uint32_t head_dim;
    uint32_t n_heads;
    uint32_t seed;        /* base_seed + layer_idx                  */
    int      initialized; /* 0 until first pq_layer_init call       */
} pq_layer_t;

/**
 * Initialize a layer context. Generates the rotation matrix.
 *
 * @param layer       Layer context to initialize
 * @param head_dim    Head dimension
 * @param n_heads     Number of attention heads
 * @param base_seed   Base random seed
 * @param layer_idx   Layer index (seed = base_seed + layer_idx)
 * @return PQ_OK on success
 */
pq_error_t pq_layer_init(
    pq_layer_t *layer,
    uint32_t    head_dim,
    uint32_t    n_heads,
    uint32_t    base_seed,
    uint32_t    layer_idx
);

/**
 * Rotate values using a layer context.
 */
pq_error_t pq_layer_rotate(pq_layer_t *layer, float *values);

/**
 * Unrotate values using a layer context.
 */
pq_error_t pq_layer_unrotate(pq_layer_t *layer, float *values);

/**
 * Free rotation matrix owned by the layer context.
 */
void pq_layer_free(pq_layer_t *layer);

#ifdef __cplusplus
}
#endif

#endif /* POLARQUANT_H */
