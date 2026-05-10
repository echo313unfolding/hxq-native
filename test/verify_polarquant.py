#!/usr/bin/env python3
"""
verify_polarquant.py — Verify C PolarQuant matches Python PolarQuant

Generates rotation matrices in Python (numpy) and C, compares them.
This is the critical interop test: C-generated rotations must match
Python-compressed KV caches bit-for-bit (within float32 tolerance).

Usage:
    python3 test/verify_polarquant.py           # from hxq-native/
    python3 test/verify_polarquant.py --dim 128 --seeds 42,99,0,1000

Requires: numpy, compiled test/verify_polarquant_bridge (built by make)
"""

import argparse
import json
import subprocess
import sys
import struct
import tempfile
import time
import resource
import platform
from pathlib import Path

import numpy as np


def generate_rotation_python(dim: int, seed: int) -> np.ndarray:
    """Generate rotation matrix matching helix-online-kv/polar_rotation.py"""
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    signs = np.sign(np.diag(R))
    Q = Q * signs[np.newaxis, :]
    return Q.astype(np.float32)


def write_binary_matrix(path: str, Q: np.ndarray):
    """Write float32 matrix as raw binary (row-major)."""
    Q.tofile(path)


def read_binary_matrix(path: str, dim: int) -> np.ndarray:
    """Read float32 matrix from raw binary."""
    return np.fromfile(path, dtype=np.float32).reshape(dim, dim)


def main():
    parser = argparse.ArgumentParser(description="Verify C PolarQuant vs Python")
    parser.add_argument("--dim", type=int, default=64, help="Matrix dimension")
    parser.add_argument("--seeds", type=str, default="42,99,0,7,256,1000",
                        help="Comma-separated seeds to test")
    parser.add_argument("--bridge", type=str, default="test/verify_polarquant_bridge",
                        help="Path to C bridge binary")
    parser.add_argument("--output", type=str, default=None,
                        help="Output receipt JSON path")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    dim = args.dim
    bridge = Path(args.bridge)

    if not bridge.exists():
        print(f"ERROR: Bridge binary not found at {bridge}")
        print("Build it with: make verify-polarquant")
        sys.exit(1)

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    results = []
    all_pass = True

    for seed in seeds:
        # Python reference
        Q_py = generate_rotation_python(dim, seed)

        # C output via bridge
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            c_out_path = f.name

        proc = subprocess.run(
            [str(bridge), str(dim), str(seed), c_out_path],
            capture_output=True, text=True
        )
        if proc.returncode != 0:
            print(f"  FAIL: C bridge failed for seed={seed}: {proc.stderr}")
            results.append({"seed": seed, "status": "FAIL", "error": proc.stderr})
            all_pass = False
            continue

        Q_c = read_binary_matrix(c_out_path, dim)
        Path(c_out_path).unlink()

        # Compare
        max_abs_diff = float(np.max(np.abs(Q_py - Q_c)))
        mean_abs_diff = float(np.mean(np.abs(Q_py - Q_c)))
        cos_sim = float(np.sum(Q_py * Q_c) / (np.linalg.norm(Q_py) * np.linalg.norm(Q_c)))

        # Orthogonality check (both)
        I_py = Q_py @ Q_py.T
        I_c = Q_c @ Q_c.T
        orth_err_py = float(np.max(np.abs(I_py - np.eye(dim))))
        orth_err_c = float(np.max(np.abs(I_c - np.eye(dim))))

        passed = max_abs_diff < 1e-4
        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  {status}: seed={seed:5d}  max_diff={max_abs_diff:.2e}  "
              f"mean_diff={mean_abs_diff:.2e}  cos={cos_sim:.8f}  "
              f"orth_err(py={orth_err_py:.2e}, c={orth_err_c:.2e})")

        results.append({
            "seed": seed,
            "status": status,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "cosine_similarity": cos_sim,
            "orthogonality_error_python": orth_err_py,
            "orthogonality_error_c": orth_err_c,
        })

    # Summary
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{'='*60}")
    print(f"PolarQuant C↔Python verification: {n_pass}/{len(results)} seeds match")
    print(f"Dimension: {dim}, Tolerance: 1e-4")

    if all_pass:
        print("VERDICT: PASS — C implementation matches Python reference")
    else:
        print("VERDICT: FAIL — C and Python diverge beyond tolerance")

    # Receipt
    receipt = {
        "test": "verify_polarquant_c_vs_python",
        "verdict": "PASS" if all_pass else "FAIL",
        "dim": dim,
        "n_seeds": len(seeds),
        "n_pass": n_pass,
        "tolerance": 1e-4,
        "results": results,
        "cost": {
            "wall_time_s": round(time.time() - t_start, 3),
            "cpu_time_s": round(time.process_time() - cpu_start, 3),
            "peak_memory_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "hostname": platform.node(),
            "timestamp_start": start_iso,
            "timestamp_end": time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
    }

    out_path = args.output or f"receipts/polarquant_verify_dim{dim}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {out_path}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
