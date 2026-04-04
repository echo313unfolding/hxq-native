/*
 * hxq.c — HXQ Native Decompression Library Implementation
 *
 * Pure C, zero external dependencies. Portable to any platform
 * with a C99 compiler: Linux, macOS, Windows, Jetson, mobile.
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 */

#include "hxq.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Tensor lifecycle ────────────────────────────────────────── */

hxq_error_t hxq_tensor_init(hxq_tensor_t *t) {
    if (!t) return HXQ_ERR_NULL_PTR;
    memset(t, 0, sizeof(hxq_tensor_t));
    return HXQ_OK;
}

void hxq_tensor_free(hxq_tensor_t *t) {
    if (!t) return;
    free(t->codebook);
    free(t->indices_raw);
    free(t->sidecar_rows);
    free(t->sidecar_cols);
    free(t->sidecar_vals);
    memset(t, 0, sizeof(hxq_tensor_t));
}

/* ── Codebook loading ────────────────────────────────────────── */

hxq_error_t hxq_tensor_load_codebook(
    hxq_tensor_t *t,
    const float  *data,
    uint32_t      k,
    uint32_t      vdim
) {
    if (!t || !data) return HXQ_ERR_NULL_PTR;
    if (k == 0 || (k != 256 && k != 4096)) return HXQ_ERR_INVALID_K;
    if (vdim == 0 || vdim > 2) return HXQ_ERR_INVALID_DIM;

    size_t cb_size = (size_t)k * vdim * sizeof(float);
    float *cb = (float *)malloc(cb_size);
    if (!cb) return HXQ_ERR_ALLOC_FAILED;

    memcpy(cb, data, cb_size);

    free(t->codebook);
    t->codebook   = cb;
    t->k          = k;
    t->vector_dim = vdim;

    return HXQ_OK;
}

/* ── Index loading: 8-bit (scalar VQ, k=256) ─────────────────── */

hxq_error_t hxq_tensor_load_indices_8bit(
    hxq_tensor_t  *t,
    const uint8_t *indices,
    uint32_t       out_f,
    uint32_t       in_f
) {
    if (!t || !indices) return HXQ_ERR_NULL_PTR;

    size_t n = (size_t)out_f * in_f;
    uint8_t *buf = (uint8_t *)malloc(n);
    if (!buf) return HXQ_ERR_ALLOC_FAILED;

    memcpy(buf, indices, n);

    free(t->indices_raw);
    t->indices_raw  = buf;
    t->indices_len  = n;
    t->out_features = out_f;
    t->in_features  = in_f;
    t->mode         = HXQ_SCALAR_VQ;

    return HXQ_OK;
}

/* ── Index loading: 12-bit packed (2D VQ, k=4096) ────────────── */

hxq_error_t hxq_tensor_load_indices_12bit(
    hxq_tensor_t  *t,
    const uint8_t *packed,
    size_t         len,
    uint32_t       out_f,
    uint32_t       in_f
) {
    if (!t || !packed) return HXQ_ERR_NULL_PTR;

    /*
     * 2D VQ pairs adjacent columns. Each pair needs one 12-bit index.
     * Total pairs = out_features * (in_features / 2)
     * Each pair of 12-bit indices = 3 bytes
     * But we pack pairs of PAIRS: every 3 bytes = 2 indices = 2 weight pairs = 4 weights.
     *
     * Actually, the 12-bit packing is:
     *   Each weight pair gets one 12-bit index.
     *   Two 12-bit indices pack into 3 bytes.
     *   So: 3 bytes → 2 indices → 2 pairs → 4 weights.
     *
     * Total indices = out_features * (in_features / 2)
     * Total packed bytes = (total_indices / 2) * 3  (if even)
     */

    uint8_t *buf = (uint8_t *)malloc(len);
    if (!buf) return HXQ_ERR_ALLOC_FAILED;

    memcpy(buf, packed, len);

    free(t->indices_raw);
    t->indices_raw  = buf;
    t->indices_len  = len;
    t->out_features = out_f;
    t->in_features  = in_f;
    t->mode         = HXQ_VQ2D_12BIT;

    return HXQ_OK;
}

/* ── Sidecar loading ─────────────────────────────────────────── */

hxq_error_t hxq_tensor_load_sidecar(
    hxq_tensor_t   *t,
    const uint32_t *rows,
    const uint32_t *cols,
    const float    *vals,
    uint32_t        nnz
) {
    if (!t) return HXQ_ERR_NULL_PTR;
    if (nnz == 0) {
        t->sidecar_nnz = 0;
        return HXQ_OK;
    }
    if (!rows || !cols || !vals) return HXQ_ERR_NULL_PTR;

    uint32_t *r = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    uint32_t *c = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    float    *v = (float    *)malloc(nnz * sizeof(float));
    if (!r || !c || !v) {
        free(r); free(c); free(v);
        return HXQ_ERR_ALLOC_FAILED;
    }

    memcpy(r, rows, nnz * sizeof(uint32_t));
    memcpy(c, cols, nnz * sizeof(uint32_t));
    memcpy(v, vals, nnz * sizeof(float));

    free(t->sidecar_rows);
    free(t->sidecar_cols);
    free(t->sidecar_vals);

    t->sidecar_rows = r;
    t->sidecar_cols = c;
    t->sidecar_vals = v;
    t->sidecar_nnz  = nnz;

    return HXQ_OK;
}

/* ── Decompress: scalar VQ (k=256, uint8 indices) ────────────── */

static hxq_error_t decompress_scalar(
    const hxq_tensor_t *t,
    float              *output
) {
    const uint32_t out_f = t->out_features;
    const uint32_t in_f  = t->in_features;
    const float   *cb    = t->codebook;
    const uint8_t *idx   = t->indices_raw;

    for (uint32_t i = 0; i < out_f; i++) {
        for (uint32_t j = 0; j < in_f; j++) {
            uint8_t code = idx[i * in_f + j];
            output[i * in_f + j] = cb[code];
        }
    }

    return HXQ_OK;
}

/* ── Decompress: 2D VQ 12-bit packed (k=4096, vector_dim=2) ─── */

static hxq_error_t decompress_vq2d_12bit(
    const hxq_tensor_t *t,
    float              *output
) {
    const uint32_t out_f = t->out_features;
    const uint32_t in_f  = t->in_features;
    const float   *cb    = t->codebook;   /* [4096, 2] */
    const uint8_t *raw   = t->indices_raw;

    /*
     * Weight layout: rows of in_features weights.
     * 2D VQ pairs adjacent columns: (col 0,1), (col 2,3), ...
     * Each pair → one 12-bit index → codebook[idx] gives 2 floats.
     *
     * Packed layout: every 3 bytes = 2 indices = 2 pairs = 4 weights.
     */

    uint32_t pairs_per_row = in_f / 2;
    size_t   byte_offset   = 0;

    for (uint32_t i = 0; i < out_f; i++) {
        for (uint32_t p = 0; p < pairs_per_row; p += 2) {
            /* Unpack two 12-bit indices from 3 bytes */
            if (byte_offset + 3 > t->indices_len) {
                return HXQ_ERR_PACK_FORMAT;
            }

            uint16_t idx_a, idx_b;
            hxq_unpack_12bit_pair(&raw[byte_offset], &idx_a, &idx_b);
            byte_offset += 3;

            /* Bounds check */
            if (idx_a >= t->k || idx_b >= t->k) {
                return HXQ_ERR_INDEX_OOB;
            }

            /* First pair: codebook[idx_a] → (w0, w1) */
            uint32_t col_a = p * 2;
            if (col_a + 1 < in_f) {
                output[i * in_f + col_a]     = cb[idx_a * 2];
                output[i * in_f + col_a + 1] = cb[idx_a * 2 + 1];
            }

            /* Second pair: codebook[idx_b] → (w2, w3) */
            uint32_t col_b = col_a + 2;
            if (col_b + 1 < in_f) {
                output[i * in_f + col_b]     = cb[idx_b * 2];
                output[i * in_f + col_b + 1] = cb[idx_b * 2 + 1];
            }
        }

        /* Handle odd pair at end of row (single 12-bit index) */
        if (pairs_per_row % 2 == 1) {
            /* Read one more index — need to handle partial 3-byte pack */
            /* For now: this case requires the packer to pad */
        }
    }

    return HXQ_OK;
}

/* ── Apply sidecar corrections + compute L2 norm ─────────────── */

static void apply_sidecar(
    const hxq_tensor_t *t,
    float              *output,
    float              *l2_norm_out
) {
    float l2_sum = 0.0f;

    for (uint32_t s = 0; s < t->sidecar_nnz; s++) {
        uint32_t r = t->sidecar_rows[s];
        uint32_t c = t->sidecar_cols[s];
        float    v = t->sidecar_vals[s];

        /* Bounds check */
        if (r < t->out_features && c < t->in_features) {
            output[r * t->in_features + c] += v;
            l2_sum += v * v;
        }
    }

    *l2_norm_out = sqrtf(l2_sum);
}

/* ── Main decompress entry point ─────────────────────────────── */

hxq_error_t hxq_tensor_decompress(
    hxq_tensor_t *t,
    float        *output
) {
    if (!t || !output) return HXQ_ERR_NULL_PTR;
    if (!t->codebook || !t->indices_raw) return HXQ_ERR_NULL_PTR;

    hxq_error_t err;

    /* Phase 1: Codebook reconstruction */
    switch (t->mode) {
        case HXQ_SCALAR_VQ:
            err = decompress_scalar(t, output);
            break;
        case HXQ_VQ2D_12BIT:
            err = decompress_vq2d_12bit(t, output);
            break;
        default:
            return HXQ_ERR_INVALID_DIM;
    }

    if (err != HXQ_OK) return err;

    /* Phase 2: Sidecar correction + confidence signal */
    t->sidecar_l2_norm = 0.0f;
    if (t->sidecar_nnz > 0) {
        apply_sidecar(t, output, &t->sidecar_l2_norm);
    }

    return HXQ_OK;
}

/* ── Shared buffer decompress ────────────────────────────────── */

hxq_error_t hxq_tensor_decompress_shared(
    hxq_tensor_t        *t,
    hxq_shared_buffer_t *buf,
    float              **out
) {
    if (!t || !buf || !out) return HXQ_ERR_NULL_PTR;

    size_t needed = (size_t)t->out_features * t->in_features;

    /* Grow buffer if needed */
    if (needed > buf->capacity) {
        float *new_data = (float *)realloc(buf->data, needed * sizeof(float));
        if (!new_data) return HXQ_ERR_ALLOC_FAILED;
        buf->data     = new_data;
        buf->capacity = needed;
    }

    /* Track high water mark */
    if (needed > buf->high_water) {
        buf->high_water = needed;
    }

    hxq_error_t err = hxq_tensor_decompress(t, buf->data);
    if (err != HXQ_OK) return err;

    *out = buf->data;
    return HXQ_OK;
}

/* ── Shared buffer lifecycle ─────────────────────────────────── */

hxq_error_t hxq_shared_buffer_init(hxq_shared_buffer_t *buf, size_t initial_capacity) {
    if (!buf) return HXQ_ERR_NULL_PTR;

    buf->data = (float *)calloc(initial_capacity, sizeof(float));
    if (!buf->data && initial_capacity > 0) return HXQ_ERR_ALLOC_FAILED;

    buf->capacity   = initial_capacity;
    buf->high_water = 0;

    return HXQ_OK;
}

void hxq_shared_buffer_free(hxq_shared_buffer_t *buf) {
    if (!buf) return;
    free(buf->data);
    buf->data       = NULL;
    buf->capacity   = 0;
    buf->high_water = 0;
}

/* ── Meta-kernel: universal dequantization ───────────────────── */

hxq_error_t hxq_dequant(
    hxq_tensor_t        *tensor,
    hxq_shared_buffer_t *buf,
    hxq_backend_t        backend,
    hxq_result_t        *result
) {
    if (!tensor || !buf || !result) return HXQ_ERR_NULL_PTR;

    (void)backend;  /* CPU-only for now; CUDA/Metal dispatched in future */

    /* Decompress into shared buffer */
    float *out;
    hxq_error_t err = hxq_tensor_decompress_shared(tensor, buf, &out);
    if (err != HXQ_OK) return err;

    /* Pack result */
    result->weights      = out;
    result->confidence   = tensor->sidecar_l2_norm;
    result->out_features = tensor->out_features;
    result->in_features  = tensor->in_features;

    return HXQ_OK;
}
