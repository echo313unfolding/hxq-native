/*
 * hxq.h — HXQ Native Decompression Library
 *
 * Portable C implementation of the HXQ vector quantization codec.
 * Reads codebook + packed indices + sidecar corrections and
 * reconstructs weight matrices at runtime.
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 * License: Apache 2.0
 *
 * Usage:
 *   #include "hxq.h"
 *
 *   hxq_tensor_t tensor;
 *   hxq_tensor_init(&tensor);
 *   hxq_tensor_load_codebook(&tensor, codebook_data, k, vector_dim);
 *   hxq_tensor_load_indices_8bit(&tensor, indices, out_features);
 *   hxq_tensor_load_sidecar(&tensor, rows, cols, deltas, nnz);
 *   hxq_tensor_decompress(&tensor, output_buffer, out_features, in_features);
 *   hxq_tensor_free(&tensor);
 */

#ifndef HXQ_H
#define HXQ_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ─────────────────────────────────────────────────── */

#define HXQ_VERSION_MAJOR 0
#define HXQ_VERSION_MINOR 4
#define HXQ_VERSION_PATCH 0
#define HXQ_VERSION_STRING "0.4.0"

/* ── Error codes ─────────────────────────────────────────────── */

typedef enum {
    HXQ_OK = 0,
    HXQ_ERR_NULL_PTR,
    HXQ_ERR_INVALID_K,
    HXQ_ERR_INVALID_DIM,
    HXQ_ERR_ALLOC_FAILED,
    HXQ_ERR_INDEX_OOB,
    HXQ_ERR_PACK_FORMAT,
    HXQ_ERR_SIDECAR_OOB,
} hxq_error_t;

/* ── Codec mode ──────────────────────────────────────────────── */

typedef enum {
    HXQ_SCALAR_VQ,    /* k=256, uint8 indices, 1 weight per entry         */
    HXQ_VQ2D_8BIT,    /* k=4096, uint8 pair (two 8-bit indices)           */
    HXQ_VQ2D_12BIT,   /* k=4096, 12-bit packed indices (3 bytes per pair) */
} hxq_mode_t;

/* ── Tensor descriptor ───────────────────────────────────────── */

typedef struct {
    /* Codebook: [k, vector_dim] stored as float32 */
    float       *codebook;
    uint32_t     k;             /* Number of codebook entries (256 or 4096) */
    uint32_t     vector_dim;    /* 1 for scalar, 2 for 2D VQ               */

    /* Indices: packed or unpacked depending on mode */
    uint8_t     *indices_raw;   /* Raw packed bytes                        */
    size_t       indices_len;   /* Length of raw index buffer in bytes      */

    /* Sidecar: sparse COO corrections */
    uint32_t    *sidecar_rows;
    uint32_t    *sidecar_cols;
    float       *sidecar_vals;
    uint32_t     sidecar_nnz;

    /* Shape */
    uint32_t     out_features;
    uint32_t     in_features;

    /* Codec mode */
    hxq_mode_t   mode;

    /* Runtime: sidecar L2 norm per decompress call (confidence signal) */
    float        sidecar_l2_norm;

} hxq_tensor_t;

/* ── Shared buffer (cross-tensor, cross-model) ───────────────── */

typedef struct {
    float   *data;
    size_t   capacity;    /* In floats */
    size_t   high_water;  /* Largest tensor seen */
} hxq_shared_buffer_t;

/* ── Core API ────────────────────────────────────────────────── */

/**
 * Initialize a tensor descriptor to zero state.
 */
hxq_error_t hxq_tensor_init(hxq_tensor_t *t);

/**
 * Load codebook data.
 * @param data   Flat float32 array of shape [k, vector_dim]
 * @param k      Number of codebook entries (256 or 4096)
 * @param vdim   Vector dimension (1 for scalar, 2 for 2D VQ)
 *
 * Data is COPIED into the tensor descriptor.
 */
hxq_error_t hxq_tensor_load_codebook(
    hxq_tensor_t *t,
    const float  *data,
    uint32_t      k,
    uint32_t      vdim
);

/**
 * Load 8-bit indices (scalar VQ k=256).
 * @param indices  uint8 array, length = out_features * in_features
 * @param out_f    Output features (rows)
 * @param in_f     Input features (columns)
 */
hxq_error_t hxq_tensor_load_indices_8bit(
    hxq_tensor_t  *t,
    const uint8_t *indices,
    uint32_t       out_f,
    uint32_t       in_f
);

/**
 * Load 12-bit packed indices (2D VQ k=4096).
 * Every 3 bytes encode 2 indices: [A11:A8][A7:A0|B11:B8][B7:B0]
 * @param packed   Packed uint8 array
 * @param len      Length of packed buffer in bytes
 * @param out_f    Output features (rows)
 * @param in_f     Input features (columns)
 */
hxq_error_t hxq_tensor_load_indices_12bit(
    hxq_tensor_t  *t,
    const uint8_t *packed,
    size_t         len,
    uint32_t       out_f,
    uint32_t       in_f
);

/**
 * Load sidecar sparse corrections.
 * @param rows    Row indices (uint32)
 * @param cols    Column indices (uint32)
 * @param vals    Correction values (float32)
 * @param nnz     Number of nonzero corrections
 */
hxq_error_t hxq_tensor_load_sidecar(
    hxq_tensor_t   *t,
    const uint32_t *rows,
    const uint32_t *cols,
    const float    *vals,
    uint32_t        nnz
);

/**
 * Decompress weights into output buffer.
 *
 * Reconstructs the full weight matrix:
 *   W[i][j] = codebook[indices[i*in_f + j]] + sidecar_correction(i, j)
 *
 * For 2D VQ, pairs of weights are reconstructed from each codebook entry.
 *
 * Also computes sidecar_l2_norm (the confidence signal):
 *   t->sidecar_l2_norm = sqrt(sum(sidecar_contribution^2))
 *
 * @param t       Tensor descriptor (must have codebook + indices loaded)
 * @param output  Pre-allocated float32 buffer of size [out_features * in_features]
 * @return HXQ_OK on success
 */
hxq_error_t hxq_tensor_decompress(
    hxq_tensor_t *t,
    float        *output
);

/**
 * Decompress into shared buffer (for buffered forward path).
 * Grows the buffer if needed. Returns pointer to decompressed data.
 *
 * @param t       Tensor descriptor
 * @param buf     Shared buffer (may be reallocated)
 * @param out     On success, points to decompressed weights in buf->data
 * @return HXQ_OK on success
 */
hxq_error_t hxq_tensor_decompress_shared(
    hxq_tensor_t        *t,
    hxq_shared_buffer_t *buf,
    float              **out
);

/**
 * Free all memory owned by the tensor descriptor.
 */
void hxq_tensor_free(hxq_tensor_t *t);

/* ── Shared buffer management ────────────────────────────────── */

hxq_error_t hxq_shared_buffer_init(hxq_shared_buffer_t *buf, size_t initial_capacity);
void        hxq_shared_buffer_free(hxq_shared_buffer_t *buf);

/* ── 12-bit packing utilities ────────────────────────────────── */

/**
 * Extract two 12-bit indices from 3 packed bytes.
 * Layout: byte0 = A[11:4], byte1 = A[3:0]|B[11:8], byte2 = B[7:0]
 */
static inline void hxq_unpack_12bit_pair(
    const uint8_t *packed,
    uint16_t      *idx_a,
    uint16_t      *idx_b
) {
    *idx_a = ((uint16_t)packed[0] << 4) | (packed[1] >> 4);
    *idx_b = ((uint16_t)(packed[1] & 0x0F) << 8) | packed[2];
}

/**
 * Pack two 12-bit indices into 3 bytes.
 */
static inline void hxq_pack_12bit_pair(
    uint16_t  idx_a,
    uint16_t  idx_b,
    uint8_t  *packed
) {
    packed[0] = (uint8_t)(idx_a >> 4);
    packed[1] = (uint8_t)(((idx_a & 0x0F) << 4) | (idx_b >> 8));
    packed[2] = (uint8_t)(idx_b & 0xFF);
}

/* ── Confidence signal ───────────────────────────────────────── */

/**
 * Get the sidecar L2 norm from the last decompress call.
 * This is the confidence signal: higher norm = harder input = lower confidence.
 * Correlation with chunk-level CE: ρ=0.574 (proven, receipted).
 */
static inline float hxq_get_sidecar_confidence(const hxq_tensor_t *t) {
    return t->sidecar_l2_norm;
}

/* ── Meta-kernel: universal dequantization interface ─────────── */

/**
 * Result structure for the meta-kernel.
 * Returned by hxq_dequant() — contains decompressed weights
 * and the confidence signal in one call.
 */
typedef struct {
    float       *weights;       /* Dense weight buffer (in shared buf)  */
    float        confidence;    /* Sidecar L2 norm: 0=easy, high=hard   */
    uint32_t     out_features;
    uint32_t     in_features;
} hxq_result_t;

/**
 * Backend hint for the meta-kernel dispatcher.
 */
typedef enum {
    HXQ_BACKEND_AUTO,      /* Pick best available               */
    HXQ_BACKEND_CPU,       /* Force CPU (portable, always works) */
    HXQ_BACKEND_CUDA,      /* Force CUDA (requires GPU)          */
    HXQ_BACKEND_METAL,     /* Apple Silicon (future)              */
    HXQ_BACKEND_VULKAN,    /* Cross-platform GPU (future)        */
} hxq_backend_t;

/**
 * Universal dequantization — the ONE function frameworks call.
 *
 * Takes a compressed tensor descriptor and produces dense weights
 * in a shared buffer. The caller does not need to know which VQ
 * variant is stored or which hardware is running.
 *
 * Usage from any framework:
 *
 *   hxq_result_t result;
 *   hxq_dequant(&tensor, &shared_buf, HXQ_BACKEND_AUTO, &result);
 *   // result.weights → dense float buffer, ready for matmul
 *   // result.confidence → quality signal (ρ=0.574 with CE loss)
 *   framework_matmul(input, result.weights, output);
 *
 * The shared buffer is reused across ALL tensors and ALL models.
 * The confidence signal is computed as a byproduct of decompression.
 * Both are free — zero additional cost beyond the dequant itself.
 *
 * @param tensor   Compressed tensor (codebook + indices + sidecar)
 * @param buf      Shared buffer (grows as needed, reused across calls)
 * @param backend  Hardware hint (AUTO picks best available)
 * @param result   Output: weights pointer + confidence + shape
 * @return HXQ_OK on success
 */
hxq_error_t hxq_dequant(
    hxq_tensor_t        *tensor,
    hxq_shared_buffer_t *buf,
    hxq_backend_t        backend,
    hxq_result_t        *result
);

#ifdef __cplusplus
}
#endif

#endif /* HXQ_H */
