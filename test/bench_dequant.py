#!/usr/bin/env python3
"""Benchmark hxq_dequant() C speed vs Python HelixLinear decompression.

Usage:
    python3 test/bench_dequant.py [MODEL_ID]

Exports each tensor, times C dequant and Python dequant, compares.
"""

import torch
import json
import numpy as np
import sys
import os
import subprocess
import time
import resource
import platform

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "EchoLabs33/zamba2-1.2b-hxq"
WORK_DIR = "/tmp/hxq-bench"
VERIFY_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_against_python")

t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

os.makedirs(WORK_DIR, exist_ok=True)

# Write a small C benchmark tool inline
BENCH_C_SRC = r"""
#define _POSIX_C_SOURCE 199309L
#include "hxq.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int parse_int(const char *json, const char *key) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char *pos = strstr(json, search);
    if (!pos) return 0;
    pos = strchr(pos, ':');
    if (!pos) return 0;
    return atoi(pos + 1);
}

static void *load_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
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
    char *txt = (char *)realloc(buf, sz + 1);
    if (!txt) { free(buf); return NULL; }
    txt[sz] = '\0';
    return txt;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <meta.json> [warmup] [iters]\n", argv[0]); return 1; }

    int warmup = argc > 2 ? atoi(argv[2]) : 3;
    int iters  = argc > 3 ? atoi(argv[3]) : 10;

    char *json = load_text(argv[1]);
    if (!json) return 1;

    int out_f = parse_int(json, "out_features");
    int in_f  = parse_int(json, "in_features");
    int k     = parse_int(json, "k");
    int vdim  = parse_int(json, "vector_dim");
    int sc_nnz = parse_int(json, "sidecar_nnz");

    size_t cb_sz, idx_sz;
    float *cb_data = (float *)load_file("test_codebook.bin", &cb_sz);
    uint8_t *idx_data = (uint8_t *)load_file("test_indices.bin", &idx_sz);
    if (!cb_data || !idx_data) return 1;

    uint32_t *sc_rows = NULL, *sc_cols = NULL;
    float *sc_vals = NULL;
    if (sc_nnz > 0) {
        sc_rows = (uint32_t *)load_file("test_sc_rows.bin", NULL);
        sc_cols = (uint32_t *)load_file("test_sc_cols.bin", NULL);
        sc_vals = (float *)load_file("test_sc_vals.bin", NULL);
    }

    hxq_tensor_t tensor;
    hxq_tensor_init(&tensor);
    hxq_tensor_load_codebook(&tensor, cb_data, (uint32_t)k, (uint32_t)vdim);

    if (vdim == 1 && k == 256)
        hxq_tensor_load_indices_8bit(&tensor, idx_data, (uint32_t)out_f, (uint32_t)in_f);
    else if (vdim == 2 && k == 4096)
        hxq_tensor_load_indices_12bit(&tensor, idx_data, idx_sz, (uint32_t)out_f, (uint32_t)in_f);

    if (sc_nnz > 0)
        hxq_tensor_load_sidecar(&tensor, sc_rows, sc_cols, sc_vals, (uint32_t)sc_nnz);

    hxq_shared_buffer_t buf;
    hxq_shared_buffer_init(&buf, (size_t)out_f * in_f);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        hxq_result_t r;
        hxq_dequant(&tensor, &buf, HXQ_BACKEND_CPU, &r);
    }

    /* Timed runs */
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int i = 0; i < iters; i++) {
        hxq_result_t r;
        hxq_dequant(&tensor, &buf, HXQ_BACKEND_CPU, &r);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    double elapsed_ns = (ts_end.tv_sec - ts_start.tv_sec) * 1e9 +
                        (ts_end.tv_nsec - ts_start.tv_nsec);
    double per_call_us = elapsed_ns / iters / 1000.0;
    size_t n_weights = (size_t)out_f * in_f;
    double gweights_per_sec = (double)n_weights / per_call_us / 1000.0;

    /* JSON output for easy parsing */
    printf("{\"per_call_us\": %.2f, \"n_weights\": %zu, \"gweights_per_sec\": %.3f, "
           "\"iters\": %d, \"out_f\": %d, \"in_f\": %d}\n",
           per_call_us, n_weights, gweights_per_sec, iters, out_f, in_f);

    hxq_tensor_free(&tensor);
    hxq_shared_buffer_free(&buf);
    free(cb_data); free(idx_data);
    free(sc_rows); free(sc_cols); free(sc_vals);
    free(json);
    return 0;
}
"""

# Write and compile C benchmark
bench_c_path = os.path.join(WORK_DIR, "bench_dequant.c")
with open(bench_c_path, "w") as f:
    f.write(BENCH_C_SRC)

hxq_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
compile_cmd = (f"gcc -O2 -Wall -std=c99 -I{hxq_root}/include "
               f"{bench_c_path} -L{hxq_root}/lib -lhxq -lm -o {WORK_DIR}/bench_dequant")
ret = os.system(compile_cmd)
if ret != 0:
    print("ERROR: Failed to compile C benchmark")
    sys.exit(1)

BENCH_BIN = os.path.join(WORK_DIR, "bench_dequant")

print(f"Loading {MODEL_ID}...")
from helix_substrate.hf_quantizer import HelixHfQuantizer
from helix_substrate.helix_linear import HelixLinear
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")

modules = []
for name, mod in model.named_modules():
    if isinstance(mod, HelixLinear):
        modules.append((name, mod))

print(f"Found {len(modules)} HelixLinear modules\n")
print(f"{'Module':55s} {'Shape':16s} {'C (us)':>10s} {'Py (us)':>10s} {'Speedup':>8s} {'GW/s(C)':>8s}")
print("-" * 115)

all_results = []

for mod_idx, (name, mod) in enumerate(modules):
    cb = mod.codebook.cpu().float()
    k = cb.shape[0]
    vdim = cb.shape[-1] if cb.dim() > 1 else 1

    # Export to work dir
    cb.reshape(-1).numpy().tofile(os.path.join(WORK_DIR, "test_codebook.bin"))
    idx_data = mod.indices.cpu().numpy().astype(np.uint8).ravel()
    idx_data.tofile(os.path.join(WORK_DIR, "test_indices.bin"))

    sc_nnz = 0
    if (hasattr(mod, '_sidecar_rows') and mod._sidecar_rows is not None and
            len(mod._sidecar_rows) > 0):
        sc_rows = mod._sidecar_rows.cpu().numpy().astype(np.uint32)
        sc_cols = mod._sidecar_cols.cpu().numpy().astype(np.uint32)
        sc_vals = mod._sidecar_deltas.cpu().float().numpy()
        sc_rows.tofile(os.path.join(WORK_DIR, "test_sc_rows.bin"))
        sc_cols.tofile(os.path.join(WORK_DIR, "test_sc_cols.bin"))
        sc_vals.tofile(os.path.join(WORK_DIR, "test_sc_vals.bin"))
        sc_nnz = len(sc_rows)

    meta = {
        "name": name,
        "out_features": mod.out_features,
        "in_features": mod.in_features,
        "k": int(k),
        "vector_dim": int(vdim),
        "sidecar_nnz": int(sc_nnz),
        "indices_bytes": int(len(idx_data)),
    }
    meta_path = os.path.join(WORK_DIR, "test_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # C benchmark
    proc = subprocess.run(
        [BENCH_BIN, meta_path, "3", "10"],
        capture_output=True, text=True, cwd=WORK_DIR, timeout=30
    )
    c_result = json.loads(proc.stdout.strip()) if proc.returncode == 0 else None

    # Python benchmark
    n_weights = mod.out_features * mod.in_features
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            W = cb[mod.indices.long().cpu()]
        # Timed
        py_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            W = cb[mod.indices.long().cpu()]
            if sc_nnz > 0:
                for i in range(sc_nnz):
                    W_flat = W.reshape(-1)
                    pos = int(sc_rows[i]) * mod.in_features + int(sc_cols[i])
                    W_flat[pos] += float(sc_vals[i])
            t1 = time.perf_counter()
            py_times.append((t1 - t0) * 1e6)

    py_us = np.median(py_times)
    c_us = c_result["per_call_us"] if c_result else float('inf')
    speedup = py_us / c_us if c_us > 0 else 0
    gws = c_result["gweights_per_sec"] if c_result else 0

    shape_str = f"[{mod.out_features}x{mod.in_features}]"
    print(f"  {name:53s} {shape_str:16s} {c_us:10.1f} {py_us:10.1f} {speedup:7.2f}x {gws:7.2f}")

    all_results.append({
        "name": name,
        "shape": [mod.out_features, mod.in_features],
        "n_weights": n_weights,
        "c_us": round(c_us, 2),
        "py_us": round(py_us, 2),
        "speedup": round(speedup, 2),
        "gweights_per_sec": round(gws, 3),
    })

# Summary
wall_time = time.time() - t_start
c_times = [r["c_us"] for r in all_results]
py_times_all = [r["py_us"] for r in all_results]
speedups = [r["speedup"] for r in all_results]

print(f"\n{'='*60}")
print(f"=== BENCHMARK SUMMARY ===")
print(f"{'='*60}")
print(f"  Modules:        {len(modules)}")
print(f"  C median:       {np.median(c_times):.1f} us/tensor")
print(f"  Python median:  {np.median(py_times_all):.1f} us/tensor")
print(f"  Median speedup: {np.median(speedups):.2f}x")
print(f"  Mean speedup:   {np.mean(speedups):.2f}x")
print(f"  Min speedup:    {np.min(speedups):.2f}x")
print(f"  Max speedup:    {np.max(speedups):.2f}x")
print(f"  Wall time:      {wall_time:.1f}s")

# Receipt
receipt = {
    "wo": "WO-HXQ-NATIVE-BENCH",
    "model": MODEL_ID,
    "total_modules": len(modules),
    "c_median_us": round(float(np.median(c_times)), 2),
    "py_median_us": round(float(np.median(py_times_all)), 2),
    "median_speedup": round(float(np.median(speedups)), 2),
    "mean_speedup": round(float(np.mean(speedups)), 2),
    "results": all_results,
    "cost": {
        "wall_time_s": round(wall_time, 3),
        "cpu_time_s": round(time.process_time() - cpu_start, 3),
        "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
}

receipt_path = os.path.join(hxq_root, "receipts", "bench_dequant.json")
os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
with open(receipt_path, "w") as f:
    json.dump(receipt, f, indent=2)
print(f"\nReceipt: {receipt_path}")
