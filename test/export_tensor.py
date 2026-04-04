#!/usr/bin/env python3
"""Export one HelixLinear tensor from a real HXQ model as raw binary files.

Run on any box with the model cached:
    python3 test/export_tensor.py [MODEL_ID]

Produces:
    test_codebook.bin   — float32 codebook [k, vdim]
    test_indices.bin    — raw packed index bytes
    test_sc_rows.bin    — uint32 sidecar row indices
    test_sc_cols.bin    — uint32 sidecar column indices
    test_sc_vals.bin    — float32 sidecar correction values
    test_reference.bin  — float32 fully decompressed weights (Python path)
    test_meta.json      — tensor metadata

Then run:
    make verify
    ./test/verify_against_python test_meta.json
"""

import torch
import json
import numpy as np
import sys

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "EchoLabs33/zamba2-1.2b-hxq"
print(f"Loading {MODEL_ID}...")

try:
    from helix_substrate.hf_quantizer import HelixHfQuantizer
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

from helix_substrate.helix_linear import HelixLinear

# Find the first HelixLinear module with a sidecar
target_name = None
target_mod = None

for name, mod in model.named_modules():
    if isinstance(mod, HelixLinear):
        has_sidecar = (hasattr(mod, '_sidecar_rows') and
                       mod._sidecar_rows is not None and
                       len(mod._sidecar_rows) > 0)
        if has_sidecar:
            target_name = name
            target_mod = mod
            cb = mod.codebook
            print(f"Found: {name} [{mod.out_features}x{mod.in_features}], "
                  f"k={cb.shape[0]}, "
                  f"vdim={cb.shape[-1] if cb.dim() > 1 else 1}, "
                  f"sidecar_nnz={len(mod._sidecar_rows)}")
            break

if target_mod is None:
    for name, mod in model.named_modules():
        if isinstance(mod, HelixLinear):
            target_name = name
            target_mod = mod
            print(f"Found (no sidecar): {name} [{mod.out_features}x{mod.in_features}]")
            break

if target_mod is None:
    print("No HelixLinear modules found!")
    sys.exit(1)

m = target_mod

# Determine codec mode — use .codebook (the registered buffer)
cb = m.codebook.cpu().float()
k = cb.shape[0]
vdim = cb.shape[-1] if cb.dim() > 1 else 1

print(f"\nExporting: {target_name}")
print(f"  Shape: [{m.out_features}, {m.in_features}]")
print(f"  Codebook: k={k}, vdim={vdim}")

# Export codebook
cb_flat = cb.reshape(-1).numpy()
cb_flat.tofile("test_codebook.bin")
print(f"  Codebook: {len(cb_flat)} floats -> test_codebook.bin")

# Export indices — use .indices (the registered buffer)
idx = m.indices.cpu().numpy().astype(np.uint8)
idx_flat = idx.ravel()
idx_flat.tofile("test_indices.bin")
print(f"  Indices (uint8): {len(idx_flat)} bytes -> test_indices.bin")

# Export sidecar
sc_nnz = 0
if (hasattr(m, '_sidecar_rows') and m._sidecar_rows is not None and
    len(m._sidecar_rows) > 0):
    # Convert int64 -> uint32 for C library
    sc_rows = m._sidecar_rows.cpu().numpy().astype(np.uint32)
    sc_cols = m._sidecar_cols.cpu().numpy().astype(np.uint32)
    sc_vals = m._sidecar_deltas.cpu().float().numpy()
    sc_rows.tofile("test_sc_rows.bin")
    sc_cols.tofile("test_sc_cols.bin")
    sc_vals.tofile("test_sc_vals.bin")
    sc_nnz = len(sc_rows)
    print(f"  Sidecar: {sc_nnz} corrections -> test_sc_*.bin")
else:
    print("  Sidecar: none")

# Export reference (fully decompressed weights from Python)
print("  Decompressing via Python path...")
with torch.no_grad():
    if vdim == 1:
        # Scalar VQ: W = codebook[indices]
        W = cb[m.indices.long().cpu()]
    else:
        # 2D VQ: use the module's decode method
        if hasattr(m, 'decode_weight'):
            W = m.decode_weight()
        elif hasattr(m, '_dequant_tile'):
            W = m._dequant_tile(0, m.out_features)
        else:
            print("  WARNING: Can't determine 2D decode path")
            W = cb[m.indices.long().cpu()]

    # Apply sidecar corrections
    W_ref = W.clone().float().reshape(m.out_features, m.in_features)
    if sc_nnz > 0:
        for i in range(sc_nnz):
            r = int(sc_rows[i])
            c = int(sc_cols[i])
            v = float(sc_vals[i])
            W_ref[r, c] += v

    W_ref_np = W_ref.numpy()
    W_ref_np.tofile("test_reference.bin")
    print(f"  Reference: {W_ref_np.size} weights -> test_reference.bin")

# Export metadata
meta = {
    "name": target_name,
    "out_features": m.out_features,
    "in_features": m.in_features,
    "k": int(k),
    "vector_dim": int(vdim),
    "sidecar_nnz": int(sc_nnz),
    "indices_bytes": int(len(idx_flat)),
    "model": MODEL_ID,
}
with open("test_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n  Metadata -> test_meta.json")
print(f"\nDone. Run:")
print(f"  ./test/verify_against_python test_meta.json")

# Quick self-check
print(f"\n  Self-check:")
print(f"    W_ref range: [{W_ref_np.min():.6f}, {W_ref_np.max():.6f}]")
print(f"    W_ref mean:  {W_ref_np.mean():.6f}")
print(f"    W_ref std:   {W_ref_np.std():.6f}")
