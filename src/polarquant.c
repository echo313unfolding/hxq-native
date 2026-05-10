/*
 * polarquant.c — PolarQuant KV Cache Rotation Library
 *
 * Pure C99 implementation. No external dependencies beyond <math.h>.
 *
 * The rotation matrix must bit-match numpy.linalg.qr(RandomState(seed).randn(dim,dim))
 * so that C-generated rotations interoperate with Python-compressed KV caches.
 *
 * Implementation:
 *   - Mersenne Twister 19937 (matches numpy.random.RandomState)
 *   - Box-Muller transform for Gaussian draws (matches numpy.random.randn)
 *   - Householder QR decomposition (matches numpy.linalg.qr)
 *   - Sign fix to match numpy convention (positive R diagonal)
 */

#include "polarquant.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════════
 *  Mersenne Twister 19937 — matches numpy.random.RandomState
 * ══════════════════════════════════════════════════════════════════ */

#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A   0x9908b0dfUL
#define MT_UPPER_MASK 0x80000000UL
#define MT_LOWER_MASK 0x7fffffffUL

typedef struct {
    uint32_t mt[MT_N];
    int      mti;
    /* Box-Muller state */
    int      has_gauss;
    double   gauss_next;
} mt_state_t;

static void mt_seed(mt_state_t *s, uint32_t seed) {
    s->mt[0] = seed;
    for (int i = 1; i < MT_N; i++) {
        s->mt[i] = 1812433253UL * (s->mt[i-1] ^ (s->mt[i-1] >> 30)) + (uint32_t)i;
    }
    s->mti = MT_N;
    s->has_gauss = 0;
    s->gauss_next = 0.0;
}

static uint32_t mt_uint32(mt_state_t *s) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MT_MATRIX_A};

    if (s->mti >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (s->mt[kk] & MT_UPPER_MASK) | (s->mt[kk+1] & MT_LOWER_MASK);
            s->mt[kk] = s->mt[kk + MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (s->mt[kk] & MT_UPPER_MASK) | (s->mt[kk+1] & MT_LOWER_MASK);
            s->mt[kk] = s->mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (s->mt[MT_N-1] & MT_UPPER_MASK) | (s->mt[0] & MT_LOWER_MASK);
        s->mt[MT_N-1] = s->mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];
        s->mti = 0;
    }

    y = s->mt[s->mti++];
    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

/* Uniform double in [0, 1) — matches numpy's 53-bit method */
static double mt_double(mt_state_t *s) {
    uint32_t a = mt_uint32(s) >> 5;  /* 27 bits */
    uint32_t b = mt_uint32(s) >> 6;  /* 26 bits */
    return (a * 67108864.0 + b) / 9007199254740992.0;  /* (a*2^26 + b) / 2^53 */
}

/* Standard normal via Box-Muller — matches numpy.random.RandomState.randn */
static double mt_randn(mt_state_t *s) {
    if (s->has_gauss) {
        s->has_gauss = 0;
        return s->gauss_next;
    }

    double x1, x2, r2;
    do {
        x1 = 2.0 * mt_double(s) - 1.0;
        x2 = 2.0 * mt_double(s) - 1.0;
        r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    double f = sqrt(-2.0 * log(r2) / r2);
    s->gauss_next = f * x1;
    s->has_gauss = 1;
    return f * x2;
}

/* ══════════════════════════════════════════════════════════════════
 *  Householder QR decomposition
 *
 *  Computes A = Q*R where Q is orthogonal and R is upper triangular.
 *  Operates in-place on a row-major dim x dim matrix.
 * ══════════════════════════════════════════════════════════════════ */

/*
 * Compute Q from Householder QR of an n x n matrix A.
 * A is overwritten. Q is written to Q_out (n x n, row-major).
 * diag_R receives the diagonal of R (for sign-fixing).
 */
static void householder_qr(float *A, float *Q_out, float *diag_R, int n) {
    /* Work buffer for Householder vectors */
    float *v = (float *)malloc((size_t)n * sizeof(float));

    /* Store Householder vectors and tau values for later Q reconstruction */
    float *taus = (float *)calloc((size_t)n, sizeof(float));

    /* Apply Householder reflections to reduce A to upper triangular R */
    for (int k = 0; k < n; k++) {
        /* Extract column k below diagonal */
        float norm_sq = 0.0f;
        for (int i = k; i < n; i++) {
            float val = A[i * n + k];
            v[i] = val;
            norm_sq += val * val;
        }
        float norm = sqrtf(norm_sq);
        if (norm < 1e-30f) {
            taus[k] = 0.0f;
            continue;
        }

        /* Choose sign to avoid cancellation */
        float sign = (A[k * n + k] >= 0.0f) ? 1.0f : -1.0f;
        v[k] += sign * norm;

        /* Recompute norm of v */
        float v_norm_sq = 0.0f;
        for (int i = k; i < n; i++)
            v_norm_sq += v[i] * v[i];

        if (v_norm_sq < 1e-30f) {
            taus[k] = 0.0f;
            continue;
        }

        float tau = 2.0f / v_norm_sq;
        taus[k] = tau;

        /* Store the Householder vector in the lower part of A */
        /* First, apply reflection to remaining columns of A: */
        /*   A[k:, j] -= tau * v * (v^T * A[k:, j]) for j = k..n-1 */
        for (int j = k; j < n; j++) {
            float dot = 0.0f;
            for (int i = k; i < n; i++)
                dot += v[i] * A[i * n + j];
            for (int i = k; i < n; i++)
                A[i * n + j] -= tau * v[i] * dot;
        }

        /* Store v (normalized) below diagonal for Q reconstruction */
        for (int i = k + 1; i < n; i++)
            A[i * n + k] = v[i] / v[k];  /* Store as v[i]/v[k], v[k] implicit = 1 */
    }

    /* Extract diagonal of R */
    for (int i = 0; i < n; i++)
        diag_R[i] = A[i * n + i];

    /* Reconstruct Q by accumulating Householder reflections backwards */
    /* Start with Q = I */
    memset(Q_out, 0, (size_t)(n * n) * sizeof(float));
    for (int i = 0; i < n; i++)
        Q_out[i * n + i] = 1.0f;

    for (int k = n - 1; k >= 0; k--) {
        if (taus[k] == 0.0f) continue;

        /* Reconstruct v from stored values */
        v[k] = 1.0f;
        for (int i = k + 1; i < n; i++)
            v[i] = A[i * n + k];

        /* Recompute tau from v */
        float v_norm_sq = 0.0f;
        for (int i = k; i < n; i++)
            v_norm_sq += v[i] * v[i];
        float tau = 2.0f / v_norm_sq;

        /* Apply H_k = I - tau*v*v^T to Q from the left: Q = H_k @ Q */
        /* Q[k:, j] -= tau * v * (v^T @ Q[k:, j]) */
        for (int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (int i = k; i < n; i++)
                dot += v[i] * Q_out[i * n + j];
            for (int i = k; i < n; i++)
                Q_out[i * n + j] -= tau * v[i] * dot;
        }
    }

    free(v);
    free(taus);
}

/* ══════════════════════════════════════════════════════════════════
 *  Core API implementation
 * ══════════════════════════════════════════════════════════════════ */

pq_error_t pq_generate_rotation(uint32_t dim, uint32_t seed, float *Q) {
    if (!Q) return PQ_ERR_NULL_PTR;
    if (dim == 0 || dim > 1024) return PQ_ERR_INVALID_DIM;

    size_t n = (size_t)dim;

    /* 1. Generate random Gaussian matrix H (matches numpy RandomState) */
    float *H = (float *)malloc(n * n * sizeof(float));
    if (!H) return PQ_ERR_ALLOC_FAILED;

    mt_state_t rng;
    mt_seed(&rng, seed);

    for (size_t i = 0; i < n * n; i++)
        H[i] = (float)mt_randn(&rng);

    /* 2. QR decomposition: H = Q * R */
    float *diag_R = (float *)malloc(n * sizeof(float));
    if (!diag_R) { free(H); return PQ_ERR_ALLOC_FAILED; }

    householder_qr(H, Q, diag_R, (int)dim);

    /* 3. Fix sign: Q *= sign(diag(R)) to match numpy.linalg.qr convention */
    /*    (positive R diagonal). det(Q) = ±1 depending on det(H). */
    for (uint32_t j = 0; j < dim; j++) {
        if (diag_R[j] < 0.0f) {
            for (uint32_t i = 0; i < dim; i++)
                Q[i * dim + j] = -Q[i * dim + j];
        }
    }

    free(diag_R);
    free(H);
    return PQ_OK;
}

pq_error_t pq_rotate(float *values, const float *Q, uint32_t n_heads, uint32_t head_dim) {
    if (!values || !Q) return PQ_ERR_NULL_PTR;
    if (head_dim == 0 || n_heads == 0) return PQ_ERR_INVALID_DIM;

    /* Temp buffer for one head */
    float *tmp = (float *)malloc((size_t)head_dim * sizeof(float));
    if (!tmp) return PQ_ERR_ALLOC_FAILED;

    /* For each head: head = head @ Q^T */
    for (uint32_t h = 0; h < n_heads; h++) {
        float *head = values + (size_t)h * head_dim;

        for (uint32_t j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < head_dim; k++) {
                /* Q^T[k, j] = Q[j, k] */
                sum += head[k] * Q[j * head_dim + k];
            }
            tmp[j] = sum;
        }
        memcpy(head, tmp, (size_t)head_dim * sizeof(float));
    }

    free(tmp);
    return PQ_OK;
}

pq_error_t pq_unrotate(float *values, const float *Q, uint32_t n_heads, uint32_t head_dim) {
    if (!values || !Q) return PQ_ERR_NULL_PTR;
    if (head_dim == 0 || n_heads == 0) return PQ_ERR_INVALID_DIM;

    /* Temp buffer for one head */
    float *tmp = (float *)malloc((size_t)head_dim * sizeof(float));
    if (!tmp) return PQ_ERR_ALLOC_FAILED;

    /* For each head: head = head @ Q */
    for (uint32_t h = 0; h < n_heads; h++) {
        float *head = values + (size_t)h * head_dim;

        for (uint32_t j = 0; j < head_dim; j++) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < head_dim; k++) {
                sum += head[k] * Q[k * head_dim + j];
            }
            tmp[j] = sum;
        }
        memcpy(head, tmp, (size_t)head_dim * sizeof(float));
    }

    free(tmp);
    return PQ_OK;
}

pq_error_t pq_infer_head_geometry(
    uint32_t  entry_size,
    uint32_t  n_heads_hint,
    uint32_t *out_n_heads,
    uint32_t *out_head_dim
) {
    if (!out_n_heads || !out_head_dim) return PQ_ERR_NULL_PTR;

    if (n_heads_hint > 0) {
        if (entry_size % n_heads_hint != 0)
            return PQ_ERR_NOT_DIVISIBLE;
        *out_n_heads = n_heads_hint;
        *out_head_dim = entry_size / n_heads_hint;
        return PQ_OK;
    }

    /* Auto-infer: try common head_dim values */
    static const uint32_t common_dims[] = {128, 64, 32};
    for (int i = 0; i < 3; i++) {
        if (entry_size % common_dims[i] == 0) {
            *out_n_heads = entry_size / common_dims[i];
            *out_head_dim = common_dims[i];
            return PQ_OK;
        }
    }

    return PQ_ERR_NOT_DIVISIBLE;
}

/* ── Layer context convenience API ───────────────────────────── */

pq_error_t pq_layer_init(
    pq_layer_t *layer,
    uint32_t    head_dim,
    uint32_t    n_heads,
    uint32_t    base_seed,
    uint32_t    layer_idx
) {
    if (!layer) return PQ_ERR_NULL_PTR;

    layer->head_dim = head_dim;
    layer->n_heads = n_heads;
    layer->seed = base_seed + layer_idx;

    layer->Q = (float *)malloc((size_t)head_dim * head_dim * sizeof(float));
    if (!layer->Q) return PQ_ERR_ALLOC_FAILED;

    pq_error_t err = pq_generate_rotation(head_dim, layer->seed, layer->Q);
    if (err != PQ_OK) {
        free(layer->Q);
        layer->Q = NULL;
        return err;
    }

    layer->initialized = 1;
    return PQ_OK;
}

pq_error_t pq_layer_rotate(pq_layer_t *layer, float *values) {
    if (!layer || !layer->initialized) return PQ_ERR_NULL_PTR;
    return pq_rotate(values, layer->Q, layer->n_heads, layer->head_dim);
}

pq_error_t pq_layer_unrotate(pq_layer_t *layer, float *values) {
    if (!layer || !layer->initialized) return PQ_ERR_NULL_PTR;
    return pq_unrotate(values, layer->Q, layer->n_heads, layer->head_dim);
}

void pq_layer_free(pq_layer_t *layer) {
    if (layer && layer->Q) {
        free(layer->Q);
        layer->Q = NULL;
        layer->initialized = 0;
    }
}
