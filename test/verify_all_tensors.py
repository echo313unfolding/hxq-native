#!/usr/bin/env python3
"""Verify ALL HelixLinear tensors in a model: export each, run C verifier, report.

Usage:
    python3 test/verify_all_tensors.py [MODEL_ID]

Requires: hxq-native built (make all verify)
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
WORK_DIR = "/tmp/hxq-verify-all"
VERIFY_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_against_python")

t_start = time.time()
cpu_start = time.process_time()
start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

# Check verify binary exists
if not os.path.isfile(VERIFY_BIN):
    print(f"ERROR: {VERIFY_BIN} not found. Run 'make verify' first.")
    sys.exit(1)

os.makedirs(WORK_DIR, exist_ok=True)

print(f"Loading {MODEL_ID}...")
from helix_substrate.hf_quantizer import HelixHfQuantizer
from helix_substrate.helix_linear import HelixLinear
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")

# Collect all HelixLinear modules
modules = []
for name, mod in model.named_modules():
    if isinstance(mod, HelixLinear):
        modules.append((name, mod))

print(f"Found {len(modules)} HelixLinear modules\n")

results = []
passed = 0
failed = 0
total_weights = 0

for idx, (name, mod) in enumerate(modules):
    cb = mod.codebook.cpu().float()
    k = cb.shape[0]
    vdim = cb.shape[-1] if cb.dim() > 1 else 1

    # Export to work dir
    cb_flat = cb.reshape(-1).numpy()
    cb_path = os.path.join(WORK_DIR, "test_codebook.bin")
    cb_flat.tofile(cb_path)

    idx_data = mod.indices.cpu().numpy().astype(np.uint8).ravel()
    idx_path = os.path.join(WORK_DIR, "test_indices.bin")
    idx_data.tofile(idx_path)

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

    # Python reference decompression
    with torch.no_grad():
        if vdim == 1:
            W = cb[mod.indices.long().cpu()]
        else:
            if hasattr(mod, 'decode_weight'):
                W = mod.decode_weight()
            elif hasattr(mod, '_dequant_tile'):
                W = mod._dequant_tile(0, mod.out_features)
            else:
                W = cb[mod.indices.long().cpu()]

        W_ref = W.clone().float().reshape(mod.out_features, mod.in_features)
        if sc_nnz > 0:
            for i in range(sc_nnz):
                W_ref[int(sc_rows[i]), int(sc_cols[i])] += float(sc_vals[i])

    W_ref.numpy().tofile(os.path.join(WORK_DIR, "test_reference.bin"))

    n_weights = mod.out_features * mod.in_features
    total_weights += n_weights

    meta = {
        "name": name,
        "out_features": mod.out_features,
        "in_features": mod.in_features,
        "k": int(k),
        "vector_dim": int(vdim),
        "sidecar_nnz": int(sc_nnz),
        "indices_bytes": int(len(idx_data)),
        "model": MODEL_ID,
    }
    meta_path = os.path.join(WORK_DIR, "test_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Run C verifier
    proc = subprocess.run(
        [VERIFY_BIN, meta_path],
        capture_output=True, text=True, cwd=WORK_DIR, timeout=60
    )

    is_pass = "*** PASS" in proc.stdout
    # Extract max error from output
    max_err_str = "?"
    conf_str = "?"
    for line in proc.stdout.split('\n'):
        if "Max absolute error:" in line:
            max_err_str = line.split(':')[-1].strip()
        if "Confidence:" in line:
            conf_str = line.split(':')[1].strip().split()[0]

    status = "PASS" if is_pass else "FAIL"
    if is_pass:
        passed += 1
    else:
        failed += 1

    shape_str = f"[{mod.out_features}x{mod.in_features}]"
    sc_str = f"sc={sc_nnz}" if sc_nnz > 0 else "no-sc"
    print(f"  [{idx+1:3d}/{len(modules)}] {status}  {name:50s} {shape_str:16s} k={k} {sc_str:8s} err={max_err_str} conf={conf_str}")

    results.append({
        "name": name,
        "shape": [mod.out_features, mod.in_features],
        "k": int(k),
        "vdim": int(vdim),
        "sidecar_nnz": int(sc_nnz),
        "n_weights": n_weights,
        "max_error": max_err_str,
        "confidence": conf_str,
        "status": status,
    })

    if not is_pass:
        print(f"    STDERR: {proc.stderr.strip()}")
        print(f"    STDOUT: {proc.stdout.strip()[:200]}")

# Summary
wall_time = time.time() - t_start
cpu_time = time.process_time() - cpu_start
peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print(f"\n{'='*60}")
print(f"=== RESULTS: {passed}/{len(modules)} PASS, {failed} FAIL ===")
print(f"{'='*60}")
print(f"  Total weights verified: {total_weights:,}")
print(f"  Wall time: {wall_time:.1f}s")
print(f"  Peak memory: {peak_mem:.1f} MB")

# Write receipt
receipt = {
    "wo": "WO-HXQ-NATIVE-VERIFY-ALL",
    "model": MODEL_ID,
    "total_modules": len(modules),
    "passed": passed,
    "failed": failed,
    "total_weights": total_weights,
    "gate": "PASS" if failed == 0 else "FAIL",
    "results": results,
    "cost": {
        "wall_time_s": round(wall_time, 3),
        "cpu_time_s": round(cpu_time, 3),
        "peak_memory_mb": round(peak_mem, 1),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "timestamp_start": start_iso,
        "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
}

receipt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "receipts",
                            "verify_all_tensors.json")
os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
with open(receipt_path, "w") as f:
    json.dump(receipt, f, indent=2)
print(f"\nReceipt: {receipt_path}")

sys.exit(0 if failed == 0 else 1)
