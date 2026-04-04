/*
 * verify_against_python.c — Load real HXQ tensor data exported from Python,
 * decompress with the C library, compare against Python's reference output.
 *
 * This is the codec verification test: if the C output matches Python
 * bit-for-bit (within float tolerance), the native library is correct.
 *
 * Usage:
 *   1. Run export_tensor.py on the GPU box to dump binary files
 *   2. SCP the files to this machine
 *   3. gcc -O2 -Iinclude verify_against_python.c -Llib -lhxq -lm -o verify
 *   4. ./verify test_meta.json
 *
 * Expected output:
 *   Max absolute error: 0.000000  (should be < 1e-5)
 *   Mean absolute error: 0.000000
 *   Sidecar L2 norm: X.XXXX  (the confidence signal)
 *   PASS: C library matches Python reference
 */

#include "hxq.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Minimal JSON parser (just enough for our metadata) ──────── */

static int parse_int(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return 0;
    pos = strchr(pos, ':');
    if (!pos) return 0;
    return atoi(pos + 1);
}

static void parse_string(const char *json, const char *key, char *out, size_t out_len) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) { out[0] = '\0'; return; }
    pos = strchr(pos, ':');
    if (!pos) { out[0] = '\0'; return; }
    pos = strchr(pos, '"');
    if (!pos) { out[0] = '\0'; return; }
    pos++;
    const char *end = strchr(pos, '"');
    if (!end) { out[0] = '\0'; return; }
    size_t len = (size_t)(end - pos);
    if (len >= out_len) len = out_len - 1;
    memcpy(out, pos, len);
    out[len] = '\0';
}

/* ── File loading helpers ────────────────────────────────────── */

static void *load_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    size_t sz = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    void *buf = malloc(sz);
    if (!buf) { fclose(f); return NULL; }

    size_t rd = fread(buf, 1, sz, f);
    fclose(f);

    if (rd != sz) { free(buf); return NULL; }
    if (size_out) *size_out = sz;
    return buf;
}

static char *load_text(const char *path) {
    size_t sz;
    char *buf = (char *)load_file(path, &sz);
    if (!buf) return NULL;
    /* Null-terminate */
    char *txt = (char *)realloc(buf, sz + 1);
    if (!txt) { free(buf); return NULL; }
    txt[sz] = '\0';
    return txt;
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test_meta.json>\n", argv[0]);
        return 1;
    }

    const char *dir = ".";  /* Expect all files in current directory */
    (void)dir;

    /* Load metadata */
    char *json = load_text(argv[1]);
    if (!json) { fprintf(stderr, "Cannot read metadata\n"); return 1; }

    char name[256];
    parse_string(json, "name", name, sizeof(name));
    int out_f      = parse_int(json, "out_features");
    int in_f       = parse_int(json, "in_features");
    int k          = parse_int(json, "k");
    int vdim       = parse_int(json, "vector_dim");
    int sc_nnz     = parse_int(json, "sidecar_nnz");
    int idx_bytes  = parse_int(json, "indices_bytes");

    printf("=== HXQ C Library vs Python Reference ===\n\n");
    printf("Tensor:       %s\n", name);
    printf("Shape:        [%d, %d]\n", out_f, in_f);
    printf("Codebook:     k=%d, vdim=%d\n", k, vdim);
    printf("Sidecar:      %d corrections\n", sc_nnz);
    printf("Index bytes:  %d\n", idx_bytes);
    printf("\n");

    /* Load codebook */
    size_t cb_sz;
    float *cb_data = (float *)load_file("test_codebook.bin", &cb_sz);
    if (!cb_data) { fprintf(stderr, "Cannot load codebook\n"); return 1; }
    printf("Codebook loaded: %zu bytes (%zu floats)\n", cb_sz, cb_sz / sizeof(float));

    /* Load indices */
    size_t idx_sz;
    uint8_t *idx_data = (uint8_t *)load_file("test_indices.bin", &idx_sz);
    if (!idx_data) { fprintf(stderr, "Cannot load indices\n"); return 1; }
    printf("Indices loaded: %zu bytes\n", idx_sz);

    /* Load sidecar */
    uint32_t *sc_rows = NULL, *sc_cols = NULL;
    float    *sc_vals = NULL;
    if (sc_nnz > 0) {
        sc_rows = (uint32_t *)load_file("test_sc_rows.bin", NULL);
        sc_cols = (uint32_t *)load_file("test_sc_cols.bin", NULL);
        sc_vals = (float    *)load_file("test_sc_vals.bin", NULL);
        if (!sc_rows || !sc_cols || !sc_vals) {
            fprintf(stderr, "Cannot load sidecar\n"); return 1;
        }
        printf("Sidecar loaded: %d corrections\n", sc_nnz);
    }

    /* Load Python reference weights */
    size_t ref_sz;
    float *reference = (float *)load_file("test_reference.bin", &ref_sz);
    if (!reference) { fprintf(stderr, "Cannot load reference\n"); return 1; }
    size_t n_weights = ref_sz / sizeof(float);
    printf("Reference loaded: %zu weights\n\n", n_weights);

    /* ── Build HXQ tensor and decompress ──────────────────────── */

    hxq_tensor_t tensor;
    hxq_tensor_init(&tensor);

    hxq_error_t err;
    err = hxq_tensor_load_codebook(&tensor, cb_data, (uint32_t)k, (uint32_t)vdim);
    if (err != HXQ_OK) { fprintf(stderr, "load_codebook failed: %d\n", err); return 1; }

    if (vdim == 1 && k == 256) {
        err = hxq_tensor_load_indices_8bit(&tensor, idx_data, (uint32_t)out_f, (uint32_t)in_f);
    } else if (vdim == 2 && k == 4096) {
        err = hxq_tensor_load_indices_12bit(&tensor, idx_data, idx_sz, (uint32_t)out_f, (uint32_t)in_f);
    } else {
        fprintf(stderr, "Unsupported mode: k=%d vdim=%d\n", k, vdim);
        return 1;
    }
    if (err != HXQ_OK) { fprintf(stderr, "load_indices failed: %d\n", err); return 1; }

    if (sc_nnz > 0) {
        err = hxq_tensor_load_sidecar(&tensor, sc_rows, sc_cols, sc_vals, (uint32_t)sc_nnz);
        if (err != HXQ_OK) { fprintf(stderr, "load_sidecar failed: %d\n", err); return 1; }
    }

    /* Use meta-kernel for decompress */
    hxq_shared_buffer_t buf;
    hxq_shared_buffer_init(&buf, (size_t)out_f * in_f);

    hxq_result_t result;
    err = hxq_dequant(&tensor, &buf, HXQ_BACKEND_CPU, &result);
    if (err != HXQ_OK) { fprintf(stderr, "dequant failed: %d\n", err); return 1; }

    printf("Decompressed: [%u, %u]\n", result.out_features, result.in_features);
    printf("Confidence:   %.6f (sidecar L2 norm)\n\n", result.confidence);

    /* ── Compare against Python reference ─────────────────────── */

    double max_err = 0.0;
    double sum_err = 0.0;
    int    mismatches = 0;

    for (size_t i = 0; i < n_weights; i++) {
        double diff = fabs((double)result.weights[i] - (double)reference[i]);
        if (diff > max_err) max_err = diff;
        sum_err += diff;
        if (diff > 1e-4) mismatches++;
    }

    double mean_err = sum_err / (double)n_weights;

    printf("=== Comparison Results ===\n\n");
    printf("  Weights compared: %zu\n", n_weights);
    printf("  Max absolute error:  %.8f\n", max_err);
    printf("  Mean absolute error: %.8f\n", mean_err);
    printf("  Mismatches (>1e-4):  %d\n", mismatches);
    printf("\n");

    if (max_err < 1e-4 && mismatches == 0) {
        printf("  *** PASS: C library matches Python reference ***\n");
    } else if (max_err < 1e-2) {
        printf("  ** MARGINAL: Small differences (likely float rounding) **\n");
    } else {
        printf("  !! FAIL: C library diverges from Python !!\n");
    }

    printf("\n  Confidence signal: %.6f\n", result.confidence);
    printf("  (Higher = model working harder on hard-to-compress weights)\n");

    /* Cleanup */
    hxq_tensor_free(&tensor);
    hxq_shared_buffer_free(&buf);
    free(cb_data);
    free(idx_data);
    free(sc_rows);
    free(sc_cols);
    free(sc_vals);
    free(reference);
    free(json);

    return (max_err < 1e-4) ? 0 : 1;
}
