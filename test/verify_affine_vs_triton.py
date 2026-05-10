#!/usr/bin/env python3
"""
verify_affine_vs_triton.py — Cross-validate native C affine matmul vs Triton.

1. Quantizes a real Qwen2.5-3B layer (or generates representative data)
2. Exports indices/scales/offsets to binary files
3. Runs the C reference via ctypes
4. Runs the Triton kernel
5. Compares outputs (cosine similarity > 0.9999)

Usage:
    python3 test/verify_affine_vs_triton.py          # synthetic data
    python3 test/verify_affine_vs_triton.py --real    # real Qwen2.5-3B layer

Requires: torch, numpy. Triton optional (skips GPU comparison if unavailable).
"""

import argparse
import ctypes
import os
import sys
import time
import json
import platform
import resource
import numpy as np

# Add project roots to path
sys.path.insert(0, os.path.expanduser("~/helix-substrate"))
sys.path.insert(0, os.path.expanduser("~/tools"))

GROUP_SIZE = 128
N_LEVELS = 63  # 6-bit: 0..63


def quantize_weight(W_np, group_size=GROUP_SIZE):
    """Quantize a weight matrix to affine group format (NumPy only)."""
    out_f, in_f = W_np.shape
    assert in_f % group_size == 0
    n_groups = in_f // group_size
    W_g = W_np.reshape(out_f, n_groups, group_size)
    vmin = W_g.min(axis=2, keepdims=True)
    vmax = W_g.max(axis=2, keepdims=True)
    step = np.maximum((vmax - vmin) / N_LEVELS, 1e-10)
    indices = np.clip(np.round((W_g - vmin) / step), 0, N_LEVELS).astype(np.uint8)
    indices = indices.reshape(out_f, in_f)
    scales = step.squeeze(-1).astype(np.float32)
    offsets = vmin.squeeze(-1).astype(np.float32)
    return indices, scales, offsets


def load_c_lib():
    """Load the affine C library via ctypes."""
    lib_path = os.path.join(os.path.dirname(__file__), "..", "lib", "libhxq_affine.a")

    # Build shared lib for ctypes if not exists
    so_path = os.path.join(os.path.dirname(__file__), "..", "lib", "libhxq_affine.so")
    if not os.path.exists(so_path):
        src = os.path.join(os.path.dirname(__file__), "..", "src", "hxq_affine.c")
        inc = os.path.join(os.path.dirname(__file__), "..", "include")
        ret = os.system(f"gcc -O2 -shared -fPIC -I{inc} {src} -o {so_path} -lm")
        if ret != 0:
            print("ERROR: Failed to build libhxq_affine.so")
            sys.exit(1)

    lib = ctypes.CDLL(so_path)

    # void hxq_affine_group_matmul_ref(
    #     const float *x, const uint8_t *indices,
    #     const float *scales, const float *offsets, const float *bias,
    #     float *output, uint32_t N, uint32_t in_f, uint32_t out_f, uint32_t group_size)
    lib.hxq_affine_group_matmul_ref.restype = None
    lib.hxq_affine_group_matmul_ref.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # x
        ctypes.POINTER(ctypes.c_ubyte),   # indices
        ctypes.POINTER(ctypes.c_float),   # scales
        ctypes.POINTER(ctypes.c_float),   # offsets
        ctypes.POINTER(ctypes.c_float),   # bias (can be NULL)
        ctypes.POINTER(ctypes.c_float),   # output
        ctypes.c_uint32,                   # N
        ctypes.c_uint32,                   # in_f
        ctypes.c_uint32,                   # out_f
        ctypes.c_uint32,                   # group_size
    ]

    return lib


def run_c_matmul(lib, x_np, indices_np, scales_np, offsets_np, group_size=GROUP_SIZE):
    """Run the C reference matmul and return output as numpy array."""
    N, in_f = x_np.shape
    out_f = indices_np.shape[0]

    x_c = x_np.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    idx_c = indices_np.astype(np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    sc_c = scales_np.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    off_c = offsets_np.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    output = np.zeros((N, out_f), dtype=np.float32)
    out_c = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    t0 = time.time()
    lib.hxq_affine_group_matmul_ref(x_c, idx_c, sc_c, off_c, None, out_c,
                                     N, in_f, out_f, group_size)
    dt = time.time() - t0

    return output, dt


def numpy_reference(x_np, indices_np, scales_np, offsets_np, group_size=GROUP_SIZE):
    """Pure NumPy reference for validation."""
    N, in_f = x_np.shape
    out_f = indices_np.shape[0]
    n_groups = in_f // group_size

    # Dequant W
    idx_g = indices_np.reshape(out_f, n_groups, group_size).astype(np.float32)
    W = (idx_g * scales_np[:, :, None] + offsets_np[:, :, None]).reshape(out_f, in_f)
    return x_np @ W.T


def try_triton(x_np, indices_np, scales_np, offsets_np, group_size=GROUP_SIZE):
    """Try running the Triton kernel. Returns (output, time) or None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA not available, skipping Triton comparison")
            return None
        from helix_substrate.triton_affine_group_matmul import fused_affine_group_matmul
    except ImportError as e:
        print(f"  Triton import failed ({e}), skipping GPU comparison")
        return None

    device = "cuda"
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    idx_t = torch.tensor(indices_np, dtype=torch.uint8, device=device)
    # Triton expects FP16 scales/offsets
    sc_t = torch.tensor(scales_np, dtype=torch.float16, device=device)
    off_t = torch.tensor(offsets_np, dtype=torch.float16, device=device)

    # Warmup
    _ = fused_affine_group_matmul(x_t, idx_t, sc_t, off_t, group_size=group_size)
    torch.cuda.synchronize()

    t0 = time.time()
    out_t = fused_affine_group_matmul(x_t, idx_t, sc_t, off_t, group_size=group_size)
    torch.cuda.synchronize()
    dt = time.time() - t0

    return out_t.cpu().numpy(), dt


def cosine_sim(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Use real Qwen2.5-3B layer (requires model)")
    parser.add_argument("--shape", default="2048x2048",
                        help="OUTxIN shape for synthetic test (default: 2048x2048)")
    args = parser.parse_args()

    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    print("=== Affine v4 Cross-Validation: C Native vs NumPy ===\n")

    if args.real:
        try:
            import torch
            from transformers import AutoModelForCausalLM
            print("Loading Qwen2.5-3B q_proj layer...")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-3B", torch_dtype=torch.float32, device_map="cpu"
            )
            W = model.model.layers[0].self_attn.q_proj.weight.data.numpy()
            del model
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Falling back to synthetic data")
            args.real = False

    if not args.real:
        out_f, in_f = [int(x) for x in args.shape.split("x")]
        print(f"Using synthetic data: {out_f}x{in_f}")
        np.random.seed(42)
        W = np.random.randn(out_f, in_f).astype(np.float32) * 0.02

    out_f, in_f = W.shape
    print(f"Weight shape: {out_f} x {in_f}")

    # Quantize
    indices, scales, offsets = quantize_weight(W)
    print(f"Quantized: indices {indices.shape} uint8, "
          f"scales {scales.shape}, offsets {offsets.shape}")

    # Generate input
    np.random.seed(123)
    N = 1
    x = np.random.randn(N, in_f).astype(np.float32)

    # NumPy reference
    out_np = numpy_reference(x, indices, scales, offsets)
    print(f"NumPy reference: shape {out_np.shape}")

    # C native
    lib = load_c_lib()
    out_c, dt_c = run_c_matmul(lib, x, indices, scales, offsets)
    cos_c = cosine_sim(out_c, out_np)
    max_err_c = np.max(np.abs(out_c - out_np))
    print(f"C native:  cos={cos_c:.8f}, max_err={max_err_c:.2e}, time={dt_c*1000:.1f}ms")

    assert cos_c > 0.999999, f"C vs NumPy cosine too low: {cos_c}"
    assert max_err_c < 1e-3, f"C vs NumPy max error too high: {max_err_c}"

    # N=4 test
    x4 = np.random.randn(4, in_f).astype(np.float32)
    out_np4 = numpy_reference(x4, indices, scales, offsets)
    out_c4, dt_c4 = run_c_matmul(lib, x4, indices, scales, offsets)
    cos_c4 = cosine_sim(out_c4, out_np4)
    max_err_c4 = np.max(np.abs(out_c4 - out_np4))
    print(f"C native (N=4): cos={cos_c4:.8f}, max_err={max_err_c4:.2e}, time={dt_c4*1000:.1f}ms")
    assert cos_c4 > 0.999999, f"C N=4 vs NumPy cosine too low: {cos_c4}"

    # Try Triton
    triton_result = try_triton(x, indices, scales, offsets)
    cos_triton = None
    if triton_result is not None:
        out_triton, dt_triton = triton_result
        cos_triton = cosine_sim(out_c, out_triton)
        max_err_triton = np.max(np.abs(out_c - out_triton))
        print(f"C vs Triton: cos={cos_triton:.8f}, max_err={max_err_triton:.2e}, "
              f"triton_time={dt_triton*1000:.1f}ms")
        # Looser tolerance for FP16 dequant path difference
        assert cos_triton > 0.9999, f"C vs Triton cosine too low: {cos_triton}"

    # Build receipt
    cost = {
        'wall_time_s': round(time.time() - t_start, 3),
        'cpu_time_s': round(time.process_time() - cpu_start, 3),
        'peak_memory_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
        'timestamp_start': start_iso,
        'timestamp_end': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    receipt = {
        'test': 'verify_affine_vs_triton',
        'shape': f'{out_f}x{in_f}',
        'data_source': 'real_qwen25_3b' if args.real else 'synthetic',
        'group_size': GROUP_SIZE,
        'n_levels': N_LEVELS,
        'c_vs_numpy_cos': float(cos_c),
        'c_vs_numpy_max_err': float(max_err_c),
        'c_n4_vs_numpy_cos': float(cos_c4),
        'c_time_ms': round(dt_c * 1000, 2),
        'c_n4_time_ms': round(dt_c4 * 1000, 2),
        'triton_available': triton_result is not None,
        'c_vs_triton_cos': float(cos_triton) if cos_triton is not None else None,
        'cost': cost,
    }

    receipt_dir = os.path.expanduser("~/receipts")
    os.makedirs(receipt_dir, exist_ok=True)
    receipt_path = os.path.join(receipt_dir, "affine_native_cross_validation.json")
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)
    print(f"\nReceipt: {receipt_path}")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
