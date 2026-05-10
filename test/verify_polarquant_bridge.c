/*
 * verify_polarquant_bridge.c — Write C rotation matrix to binary file
 *
 * Called by verify_polarquant.py to compare C output against Python.
 *
 * Usage: ./verify_polarquant_bridge <dim> <seed> <output_path>
 *   Writes dim*dim float32 values (row-major) to output_path.
 */

#include "polarquant.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <dim> <seed> <output.bin>\n", argv[0]);
        return 1;
    }

    uint32_t dim  = (uint32_t)atoi(argv[1]);
    uint32_t seed = (uint32_t)atoi(argv[2]);
    const char *path = argv[3];

    float *Q = (float *)malloc((size_t)dim * dim * sizeof(float));
    if (!Q) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    pq_error_t err = pq_generate_rotation(dim, seed, Q);
    if (err != PQ_OK) {
        fprintf(stderr, "pq_generate_rotation failed: %d\n", err);
        free(Q);
        return 1;
    }

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        free(Q);
        return 1;
    }

    size_t written = fwrite(Q, sizeof(float), (size_t)dim * dim, f);
    fclose(f);
    free(Q);

    if (written != (size_t)dim * dim) {
        fprintf(stderr, "write error\n");
        return 1;
    }

    return 0;
}
