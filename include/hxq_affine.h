/*
 * hxq_affine.h — Affine Group Quantization for HXQ Native
 *
 * Per-group affine dequant: W[o,i] = indices[o,i] * scale[o, i//G] + offset[o, i//G]
 * Fused matmul: Y = X @ W^T without materializing W.
 *
 * Two paths:
 *   1. hxq_affine_group_matmul()     — fused GPU matmul (no W materialization)
 *   2. hxq_affine_decompress()       — decompress to dense (for hxq_dequant compat)
 *
 * Port of triton_affine_group_matmul.py (WO-AFFINE-GROUP-MATMUL-01).
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 * License: Apache 2.0
 */

#ifndef HXQ_AFFINE_H
#define HXQ_AFFINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Affine tensor descriptor ────────────────────────────────── */

typedef struct {
    uint8_t  *indices;      /* [out_f, in_f] uint8 (0..63 for 6-bit)   */
    float    *scales;       /* [out_f, n_groups] float32                */
    float    *offsets;      /* [out_f, n_groups] float32                */
    float    *bias;         /* [out_f] float32, or NULL                 */
    uint32_t  out_features;
    uint32_t  in_features;
    uint32_t  group_size;   /* 128 for v4                               */
} hxq_affine_tensor_t;

/* ── CPU reference: fused matmul ─────────────────────────────── */

/**
 * CPU reference affine group matmul.
 *
 * Y[n, o] = sum_g { scale[o,g] * sum_k{ x[n, g*G+k] * idx[o, g*G+k] }
 *                  + offset[o,g] * sum_k{ x[n, g*G+k] } }
 *         + bias[o]   (if bias != NULL)
 *
 * @param x        Input activations [N, in_f], row-major
 * @param indices  Quantized weight indices [out_f, in_f], row-major
 * @param scales   Per-group scales [out_f, n_groups], row-major
 * @param offsets  Per-group offsets [out_f, n_groups], row-major
 * @param bias     Per-output bias [out_f], or NULL
 * @param output   Output buffer [N, out_f], row-major
 * @param N        Batch size (number of tokens)
 * @param in_f     Input features
 * @param out_f    Output features
 * @param group_size  Group size (must divide in_f evenly)
 */
void hxq_affine_group_matmul_ref(
    const float    *x,
    const uint8_t  *indices,
    const float    *scales,
    const float    *offsets,
    const float    *bias,
    float          *output,
    uint32_t        N,
    uint32_t        in_f,
    uint32_t        out_f,
    uint32_t        group_size
);

/**
 * CPU reference: decompress affine-quantized weights to dense float32.
 *
 * W[o, i] = indices[o, i] * scale[o, i/G] + offset[o, i/G]
 *
 * @param indices    [out_f, in_f] uint8
 * @param scales     [out_f, n_groups] float32
 * @param offsets    [out_f, n_groups] float32
 * @param output     [out_f, in_f] float32 (pre-allocated)
 * @param out_f      Output features
 * @param in_f       Input features
 * @param group_size Group size
 */
void hxq_affine_decompress_ref(
    const uint8_t  *indices,
    const float    *scales,
    const float    *offsets,
    float          *output,
    uint32_t        out_f,
    uint32_t        in_f,
    uint32_t        group_size
);

/* ── GPU fused matmul (defined in hxq_affine_cuda.cu) ────────── */

#ifdef HXQ_HAVE_CUDA

/**
 * Fused affine group matmul on GPU.
 * Dispatches N=1 decode kernel vs N>1 prefill kernel.
 *
 * All pointers must be device pointers.
 *
 * @param d_x       [N, in_f] float32 on device
 * @param tensor    Affine tensor descriptor (device pointers)
 * @param d_output  [N, out_f] float32 on device
 * @param N         Batch size
 * @return 0 on success, -1 on CUDA error
 */
int hxq_affine_group_matmul(
    const float                *d_x,
    const hxq_affine_tensor_t  *tensor,
    float                      *d_output,
    uint32_t                    N
);

#endif /* HXQ_HAVE_CUDA */

#ifdef __cplusplus
}
#endif

#endif /* HXQ_AFFINE_H */
