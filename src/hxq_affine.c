/*
 * hxq_affine.c — CPU Reference for Affine Group Quantization
 *
 * Pure C99, zero external dependencies.
 *
 * Math:
 *   W[o, i] = indices[o, i] * scale[o, i//G] + offset[o, i//G]
 *   Y[n, o] = sum_i { X[n, i] * W[o, i] } + bias[o]
 *
 * Algebraic decomposition (matches Triton decode kernel):
 *   Y[n, o] = sum_g { scale[o,g] * sum_k(x[n,g*G+k] * idx[o,g*G+k])
 *                    + offset[o,g] * sum_k(x[n,g*G+k]) }
 *           + bias[o]
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 */

#include "hxq_affine.h"

/* ── Fused matmul reference ──────────────────────────────────── */

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
) {
    uint32_t n_groups = in_f / group_size;

    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t o = 0; o < out_f; o++) {
            float acc = 0.0f;

            for (uint32_t g = 0; g < n_groups; g++) {
                float s   = scales[o * n_groups + g];
                float off = offsets[o * n_groups + g];

                float partial = 0.0f;
                float x_sum   = 0.0f;

                for (uint32_t k = 0; k < group_size; k++) {
                    uint32_t i = g * group_size + k;
                    float xv = x[n * in_f + i];
                    float iv = (float)indices[o * in_f + i];
                    partial += xv * iv;
                    x_sum   += xv;
                }

                acc += s * partial + off * x_sum;
            }

            if (bias) {
                acc += bias[o];
            }

            output[n * out_f + o] = acc;
        }
    }
}

/* ── Decompress to dense ─────────────────────────────────────── */

void hxq_affine_decompress_ref(
    const uint8_t  *indices,
    const float    *scales,
    const float    *offsets,
    float          *output,
    uint32_t        out_f,
    uint32_t        in_f,
    uint32_t        group_size
) {
    uint32_t n_groups = in_f / group_size;

    for (uint32_t o = 0; o < out_f; o++) {
        for (uint32_t g = 0; g < n_groups; g++) {
            float s   = scales[o * n_groups + g];
            float off = offsets[o * n_groups + g];

            for (uint32_t k = 0; k < group_size; k++) {
                uint32_t i = g * group_size + k;
                float iv = (float)indices[o * in_f + i];
                output[o * in_f + i] = iv * s + off;
            }
        }
    }
}
