/*
 * hxq_cuda.cu — GPU Decompression Kernel for HXQ
 *
 * Fused gather: reads packed 12-bit indices, looks up codebook
 * entries from shared memory, writes BF16/FP32 weights directly
 * to output buffer. One kernel launch per tensor.
 *
 * Codebook (32 KB for k=4096, vdim=2) fits in shared memory.
 * Only index reads hit global memory.
 *
 * Also computes per-block sidecar L2 partial sums for the
 * confidence signal (reduced on host).
 *
 * Part of helix-substrate (Echo Labs LLC)
 * Author: Joshua P Fellows
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

/* ── 12-bit unpacking (device) ───────────────────────────────── */

__device__ __forceinline__ void unpack_12bit(
    const uint8_t *packed,
    uint16_t      *idx_a,
    uint16_t      *idx_b
) {
    uint8_t b0 = packed[0];
    uint8_t b1 = packed[1];
    uint8_t b2 = packed[2];
    *idx_a = ((uint16_t)b0 << 4) | (b1 >> 4);
    *idx_b = ((uint16_t)(b1 & 0x0F) << 8) | b2;
}

/* ── Scalar VQ kernel (k=256, uint8 indices) ─────────────────── */

__global__ void hxq_decompress_scalar_kernel(
    const float   *__restrict__ codebook,   /* [256]              */
    const uint8_t *__restrict__ indices,    /* [out_f * in_f]     */
    float         *__restrict__ output,     /* [out_f * in_f]     */
    uint32_t       out_f,
    uint32_t       in_f
) {
    /* Load codebook into shared memory (256 * 4 = 1 KB) */
    __shared__ float cb_smem[256];

    if (threadIdx.x < 256) {
        cb_smem[threadIdx.x] = codebook[threadIdx.x];
    }
    __syncthreads();

    /* Each thread handles one weight */
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = out_f * in_f;

    if (gid < total) {
        uint8_t code = indices[gid];
        output[gid] = cb_smem[code];
    }
}

/* ── 2D VQ 12-bit kernel (k=4096, vdim=2) ───────────────────── */

/*
 * Codebook: [4096, 2] = 32 KB of float32. Fits in shared memory
 * on all Ampere/Ada/Hopper GPUs (48-228 KB shared per SM).
 *
 * Each thread processes one index → two output weights.
 * Global reads: 3 bytes (packed index pair) per 2 threads.
 * Shared reads: 2 floats (codebook entry) per thread.
 */

__global__ void hxq_decompress_vq2d_12bit_kernel(
    const float   *__restrict__ codebook,   /* [4096, 2]                   */
    const uint8_t *__restrict__ packed,     /* 12-bit packed indices       */
    float         *__restrict__ output,     /* [out_f * in_f]              */
    uint32_t       out_f,
    uint32_t       in_f,
    uint32_t       total_pairs              /* out_f * (in_f / 2)          */
) {
    /* Load codebook into shared memory (4096 * 2 * 4 = 32 KB) */
    extern __shared__ float cb_smem[];

    uint32_t cb_size = 4096 * 2;
    for (uint32_t i = threadIdx.x; i < cb_size; i += blockDim.x) {
        cb_smem[i] = codebook[i];
    }
    __syncthreads();

    /*
     * Each thread handles one weight pair (one 12-bit index).
     * Pairs are packed in groups of 2: every 3 bytes = 2 indices.
     *
     * Thread gid handles pair #gid.
     * pair_group = gid / 2  → which 3-byte group
     * pair_slot  = gid % 2  → first or second index in the group
     */
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_pairs) return;

    uint32_t pair_group = gid / 2;
    uint32_t pair_slot  = gid % 2;

    /* Read 3 packed bytes */
    const uint8_t *pack_ptr = packed + pair_group * 3;
    uint16_t idx_a, idx_b;
    unpack_12bit(pack_ptr, &idx_a, &idx_b);

    uint16_t my_idx = (pair_slot == 0) ? idx_a : idx_b;

    /* Bounds check */
    if (my_idx >= 4096) return;

    /* Lookup codebook entry (2 weights) from shared memory */
    float w0 = cb_smem[my_idx * 2];
    float w1 = cb_smem[my_idx * 2 + 1];

    /* Write to output: pair #gid maps to columns gid*2, gid*2+1
     * within the row. Pair index within the row:
     *   row = gid / (in_f / 2)
     *   col = (gid % (in_f / 2)) * 2
     */
    uint32_t pairs_per_row = in_f / 2;
    uint32_t row = gid / pairs_per_row;
    uint32_t col = (gid % pairs_per_row) * 2;

    if (row < out_f && col + 1 < in_f) {
        output[row * in_f + col]     = w0;
        output[row * in_f + col + 1] = w1;
    }
}

/* ── Sidecar application kernel ──────────────────────────────── */

__global__ void hxq_apply_sidecar_kernel(
    const uint32_t *__restrict__ rows,
    const uint32_t *__restrict__ cols,
    const float    *__restrict__ vals,
    float          *__restrict__ output,
    float          *__restrict__ partial_l2,  /* per-block partial sums */
    uint32_t        in_f,
    uint32_t        nnz
) {
    extern __shared__ float smem[];

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    float my_sq = 0.0f;

    if (gid < nnz) {
        uint32_t r = rows[gid];
        uint32_t c = cols[gid];
        float    v = vals[gid];

        output[r * in_f + c] += v;
        my_sq = v * v;
    }

    /* Block-level reduction for L2 partial sum */
    smem[threadIdx.x] = my_sq;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_l2[blockIdx.x] = smem[0];
    }
}

/* ── Host-side launch wrappers ───────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Device memory pointers */
    float    *d_codebook;
    uint8_t  *d_indices;
    float    *d_output;
    uint32_t *d_sidecar_rows;
    uint32_t *d_sidecar_cols;
    float    *d_sidecar_vals;
    float    *d_partial_l2;

    /* Shape */
    uint32_t  out_features;
    uint32_t  in_features;
    uint32_t  k;
    uint32_t  vector_dim;
    uint32_t  sidecar_nnz;
    size_t    indices_len;

    /* Shared output buffer (reused across tensors) */
    float    *d_shared_buffer;
    size_t    shared_capacity;

} hxq_cuda_tensor_t;

/**
 * Launch scalar VQ decompression on GPU.
 */
int hxq_cuda_decompress_scalar(
    const float   *d_codebook,
    const uint8_t *d_indices,
    float         *d_output,
    uint32_t       out_f,
    uint32_t       in_f
) {
    uint32_t total = out_f * in_f;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;

    hxq_decompress_scalar_kernel<<<blocks, threads>>>(
        d_codebook, d_indices, d_output, out_f, in_f
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/**
 * Launch 2D VQ 12-bit decompression on GPU.
 */
int hxq_cuda_decompress_vq2d_12bit(
    const float   *d_codebook,
    const uint8_t *d_packed,
    float         *d_output,
    uint32_t       out_f,
    uint32_t       in_f
) {
    uint32_t total_pairs = out_f * (in_f / 2);
    uint32_t threads = 256;
    uint32_t blocks = (total_pairs + threads - 1) / threads;
    size_t   smem_size = 4096 * 2 * sizeof(float);  /* 32 KB codebook */

    hxq_decompress_vq2d_12bit_kernel<<<blocks, threads, smem_size>>>(
        d_codebook, d_packed, d_output, out_f, in_f, total_pairs
    );

    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/**
 * Launch sidecar application + L2 norm computation.
 * Returns total sidecar L2 norm (the confidence signal).
 */
float hxq_cuda_apply_sidecar(
    const uint32_t *d_rows,
    const uint32_t *d_cols,
    const float    *d_vals,
    float          *d_output,
    uint32_t        in_f,
    uint32_t        nnz
) {
    if (nnz == 0) return 0.0f;

    uint32_t threads = 256;
    uint32_t blocks = (nnz + threads - 1) / threads;
    size_t   smem_size = threads * sizeof(float);

    /* Allocate partial L2 sums on device */
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    hxq_apply_sidecar_kernel<<<blocks, threads, smem_size>>>(
        d_rows, d_cols, d_vals, d_output, d_partial, in_f, nnz
    );

    /* Reduce partial sums on host */
    float *h_partial = (float *)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (uint32_t i = 0; i < blocks; i++) {
        total += h_partial[i];
    }

    free(h_partial);
    cudaFree(d_partial);

    return sqrtf(total);
}

#ifdef __cplusplus
}
#endif
