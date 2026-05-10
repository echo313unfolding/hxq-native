/*
 * hxq_affine_cuda.cu — GPU Kernels for Affine Group Quantization
 *
 * Port of triton_affine_group_matmul.py (WO-AFFINE-GROUP-MATMUL-01).
 *
 * Two kernels:
 *   Kernel A — N=1 decode: 1D accumulator, BLOCK_OUT=64, element-wise
 *   Kernel B — N>1 prefill: naive double-loop (correct, not optimized)
 *
 * Math:
 *   W[o, i] = indices[o, i] * scale[o, i//G] + offset[o, i//G]
 *   Y[n, o] = sum_i { X[n, i] * W[o, i] } + bias[o]
 *
 * Algebraic decomposition (decode kernel):
 *   Y[o] = sum_g { scale[o,g] * partial_g + offset[o,g] * x_sum_g }
 *   where partial_g = sum_k(x[g*G+k] * idx[o,g*G+k])
 *         x_sum_g   = sum_k(x[g*G+k])
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

/* ── Kernel A: N=1 Decode ────────────────────────────────────── */
/*
 * Grid:  ((OUT + BLOCK_OUT-1) / BLOCK_OUT,)
 * Block: (BLOCK_OUT,)  — each thread handles one output dimension
 *
 * Per group g:
 *   1. Load x_vec[GROUP_SIZE] from global (shared across all threads in block)
 *   2. Each thread loads idx[GROUP_SIZE] for its output dim
 *   3. partial = dot(x_vec, idx_fp)   [sequential loop over GROUP_SIZE]
 *   4. x_sum  = sum(x_vec)            [shared across block]
 *   5. acc += scale * partial + offset * x_sum
 */

#define DECODE_BLOCK_OUT 64
#define DEFAULT_GROUP_SIZE 128

__global__ void hxq_affine_decode_kernel(
    const float    *__restrict__ x,          /* [IN] flattened             */
    const uint8_t  *__restrict__ indices,    /* [OUT, IN]                  */
    const float    *__restrict__ scales,     /* [OUT, n_groups]            */
    const float    *__restrict__ offsets,    /* [OUT, n_groups]            */
    float          *__restrict__ output,     /* [OUT]                      */
    uint32_t        IN,
    uint32_t        OUT,
    uint32_t        n_groups,
    uint32_t        group_size
) {
    uint32_t o = blockIdx.x * DECODE_BLOCK_OUT + threadIdx.x;

    /* Shared memory for x_vec and x_sum per group */
    __shared__ float x_smem[DEFAULT_GROUP_SIZE];
    __shared__ float x_sum_smem;

    float acc = 0.0f;

    for (uint32_t g = 0; g < n_groups; g++) {
        uint32_t k_start = g * group_size;

        /* Cooperatively load x_vec into shared memory */
        for (uint32_t k = threadIdx.x; k < group_size; k += DECODE_BLOCK_OUT) {
            x_smem[k] = x[k_start + k];
        }
        __syncthreads();

        /* Thread 0 computes x_sum (reused by all threads) */
        if (threadIdx.x == 0) {
            float s = 0.0f;
            for (uint32_t k = 0; k < group_size; k++) {
                s += x_smem[k];
            }
            x_sum_smem = s;
        }
        __syncthreads();

        if (o < OUT) {
            /* Compute partial = dot(x_vec, indices_fp) for this output dim */
            float partial = 0.0f;
            const uint8_t *idx_row = indices + (size_t)o * IN + k_start;

            for (uint32_t k = 0; k < group_size; k++) {
                partial += x_smem[k] * (float)idx_row[k];
            }

            float s   = scales[o * n_groups + g];
            float off = offsets[o * n_groups + g];

            acc += s * partial + off * x_sum_smem;
        }

        __syncthreads();
    }

    if (o < OUT) {
        output[o] = acc;
    }
}

/* ── Kernel B: N>1 Prefill (naive correct) ───────────────────── */
/*
 * Grid:  ((N+15)/16, (OUT+63)/64)
 * Block: (16, 4)  — 16 tokens x 4 outputs per thread (covering 64 outputs)
 *
 * Simplified: each thread computes one (n, o) output element.
 * Grid: ((OUT + 63) / 64, (N + 15) / 16)
 * Block: (64, 1) — but each block handles a tile of outputs for a range of tokens.
 *
 * For correctness-first: 1D grid, each thread = one (n, o) pair.
 */

__global__ void hxq_affine_prefill_kernel(
    const float    *__restrict__ x,          /* [N, IN]                    */
    const uint8_t  *__restrict__ indices,    /* [OUT, IN]                  */
    const float    *__restrict__ scales,     /* [OUT, n_groups]            */
    const float    *__restrict__ offsets,    /* [OUT, n_groups]            */
    float          *__restrict__ output,     /* [N, OUT]                   */
    uint32_t        N,
    uint32_t        IN,
    uint32_t        OUT,
    uint32_t        n_groups,
    uint32_t        group_size
) {
    /* 2D grid: blockIdx.x covers output dim, blockIdx.y covers batch */
    uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = blockIdx.y * blockDim.y + threadIdx.y;

    if (n >= N || o >= OUT) return;

    const float   *x_row   = x + (size_t)n * IN;
    const uint8_t *idx_row = indices + (size_t)o * IN;

    float acc = 0.0f;

    for (uint32_t g = 0; g < n_groups; g++) {
        uint32_t k_start = g * group_size;
        float s   = scales[o * n_groups + g];
        float off = offsets[o * n_groups + g];

        float partial = 0.0f;
        float x_sum   = 0.0f;

        for (uint32_t k = 0; k < group_size; k++) {
            float xv = x_row[k_start + k];
            partial += xv * (float)idx_row[k_start + k];
            x_sum   += xv;
        }

        acc += s * partial + off * x_sum;
    }

    output[(size_t)n * OUT + o] = acc;
}

/* ── Bias addition kernel ────────────────────────────────────── */

__global__ void hxq_affine_add_bias_kernel(
    float          *__restrict__ output,     /* [N, OUT] */
    const float    *__restrict__ bias,       /* [OUT]    */
    uint32_t        N,
    uint32_t        OUT
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * OUT;
    if (idx >= total) return;

    uint32_t o = idx % OUT;
    output[idx] += bias[o];
}

/* ── Host-side dispatch ──────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

#include "hxq_affine.h"

int hxq_affine_group_matmul(
    const float                *d_x,
    const hxq_affine_tensor_t  *tensor,
    float                      *d_output,
    uint32_t                    N
) {
    uint32_t IN  = tensor->in_features;
    uint32_t OUT = tensor->out_features;
    uint32_t GS  = tensor->group_size;
    uint32_t n_groups = IN / GS;

    if (N == 1) {
        /* Decode path: 1D grid, BLOCK_OUT=64 threads per block */
        uint32_t blocks = (OUT + DECODE_BLOCK_OUT - 1) / DECODE_BLOCK_OUT;
        hxq_affine_decode_kernel<<<blocks, DECODE_BLOCK_OUT>>>(
            d_x,
            tensor->indices,
            tensor->scales,
            tensor->offsets,
            d_output,
            IN, OUT, n_groups, GS
        );
    } else {
        /* Prefill path: 2D grid, 64 threads on output dim, 1 on batch */
        dim3 threads(64, 1);
        dim3 blocks_2d(
            (OUT + threads.x - 1) / threads.x,
            (N   + threads.y - 1) / threads.y
        );
        hxq_affine_prefill_kernel<<<blocks_2d, threads>>>(
            d_x,
            tensor->indices,
            tensor->scales,
            tensor->offsets,
            d_output,
            N, IN, OUT, n_groups, GS
        );
    }

    /* Add bias if present */
    if (tensor->bias != NULL) {
        uint32_t total = N * OUT;
        uint32_t thr = 256;
        uint32_t blk = (total + thr - 1) / thr;
        hxq_affine_add_bias_kernel<<<blk, thr>>>(
            d_output, tensor->bias, N, OUT
        );
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/**
 * Flat entry point for ctypes / Python integration.
 * All pointers are raw device pointers. No struct needed.
 *
 * @param d_x        [N, in_f] float32 on device
 * @param d_indices  [out_f, in_f] uint8 on device
 * @param d_scales   [out_f, n_groups] float32 on device
 * @param d_offsets  [out_f, n_groups] float32 on device
 * @param d_bias     [out_f] float32 on device, or NULL
 * @param d_output   [N, out_f] float32 on device
 * @param N          batch size
 * @param in_f       input features
 * @param out_f      output features
 * @param group_size group size (128)
 * @return 0 on success, -1 on CUDA error
 */
int hxq_affine_matmul_flat(
    const float   *d_x,
    const uint8_t *d_indices,
    const float   *d_scales,
    const float   *d_offsets,
    const float   *d_bias,
    float         *d_output,
    uint32_t       N,
    uint32_t       in_f,
    uint32_t       out_f,
    uint32_t       group_size
) {
    uint32_t n_groups = in_f / group_size;

    if (N == 1) {
        uint32_t blocks = (out_f + DECODE_BLOCK_OUT - 1) / DECODE_BLOCK_OUT;
        hxq_affine_decode_kernel<<<blocks, DECODE_BLOCK_OUT>>>(
            d_x, d_indices, d_scales, d_offsets, d_output,
            in_f, out_f, n_groups, group_size
        );
    } else {
        dim3 threads(64, 1);
        dim3 blocks_2d(
            (out_f + threads.x - 1) / threads.x,
            (N     + threads.y - 1) / threads.y
        );
        hxq_affine_prefill_kernel<<<blocks_2d, threads>>>(
            d_x, d_indices, d_scales, d_offsets, d_output,
            N, in_f, out_f, n_groups, group_size
        );
    }

    if (d_bias != NULL) {
        uint32_t total = N * out_f;
        uint32_t thr = 256;
        uint32_t blk = (total + thr - 1) / thr;
        hxq_affine_add_bias_kernel<<<blk, thr>>>(
            d_output, d_bias, N, out_f
        );
    }

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

#ifdef __cplusplus
}
#endif
