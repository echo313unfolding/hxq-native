#!/usr/bin/env python3
"""Convert an HXQ model from HuggingFace safetensors to GGUF format.

Two modes:
  --mode=compressed  (default) Store codebook+indices+sidecar as separate tensors.
                     Compact file, requires HXQ-aware GGUF reader to dequant.
  --mode=dense       Dequantize all HXQ tensors to F32/F16. Works with stock
                     llama.cpp but loses compression benefit.

Usage:
    python3 tools/convert_hxq_to_gguf.py EchoLabs33/zamba2-1.2b-helix -o zamba2-1.2b-hxq.gguf
    python3 tools/convert_hxq_to_gguf.py EchoLabs33/zamba2-1.2b-helix --mode=dense -o zamba2-1.2b-f32.gguf
"""

import argparse
import json
import os
import sys
import time
import resource
import platform
import glob

import numpy as np
import torch
from safetensors import safe_open
from gguf import GGUFWriter, GGMLQuantizationType


def find_safetensors(model_id: str) -> list:
    """Find safetensors files from HF cache or local path."""
    if os.path.isdir(model_id):
        return sorted(glob.glob(os.path.join(model_id, "*.safetensors")))

    # Search HF cache
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")
    safe_name = model_id.replace("/", "--")
    model_dir = os.path.join(cache_base, f"models--{safe_name}")
    if not os.path.isdir(model_dir):
        # Try downloading
        print(f"Model not cached. Downloading {model_id}...")
        from huggingface_hub import snapshot_download
        snapshot_download(model_id)
        if not os.path.isdir(model_dir):
            print(f"ERROR: Cannot find model {model_id}")
            sys.exit(1)

    files = sorted(glob.glob(os.path.join(model_dir, "**/*.safetensors"), recursive=True))
    if not files:
        print(f"ERROR: No safetensors files found in {model_dir}")
        sys.exit(1)
    return files


def load_config(model_id: str) -> dict:
    """Load config.json from HF cache or local path."""
    if os.path.isdir(model_id):
        cfg_path = os.path.join(model_id, "config.json")
    else:
        cache_base = os.path.expanduser("~/.cache/huggingface/hub")
        safe_name = model_id.replace("/", "--")
        model_dir = os.path.join(cache_base, f"models--{safe_name}")
        cfg_files = glob.glob(os.path.join(model_dir, "**/config.json"), recursive=True)
        cfg_path = cfg_files[0] if cfg_files else None

    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def load_tokenizer_json(model_id: str) -> dict | None:
    """Load tokenizer.json from HF cache."""
    if os.path.isdir(model_id):
        path = os.path.join(model_id, "tokenizer.json")
    else:
        cache_base = os.path.expanduser("~/.cache/huggingface/hub")
        safe_name = model_id.replace("/", "--")
        model_dir = os.path.join(cache_base, f"models--{safe_name}")
        files = glob.glob(os.path.join(model_dir, "**/tokenizer.json"), recursive=True)
        path = files[0] if files else None

    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def identify_hxq_modules(tensor_names: list) -> dict:
    """Group HXQ tensor names by module prefix.

    Returns dict: {prefix: {"codebook": name, "indices": name,
                             "sidecar_positions": name, "sidecar_values": name}}
    """
    modules = {}
    for name in tensor_names:
        if name.endswith(".codebook"):
            prefix = name[:-len(".codebook")]
            if prefix not in modules:
                modules[prefix] = {}
            modules[prefix]["codebook"] = name
        elif name.endswith(".indices"):
            prefix = name[:-len(".indices")]
            if prefix not in modules:
                modules[prefix] = {}
            modules[prefix]["indices"] = name
        elif name.endswith(".sidecar_positions"):
            prefix = name[:-len(".sidecar_positions")]
            if prefix not in modules:
                modules[prefix] = {}
            modules[prefix]["sidecar_positions"] = name
        elif name.endswith(".sidecar_values"):
            prefix = name[:-len(".sidecar_values")]
            if prefix not in modules:
                modules[prefix] = {}
            modules[prefix]["sidecar_values"] = name
    return modules


def dequant_hxq_tensor(codebook, indices, sidecar_pos, sidecar_vals, out_f, in_f):
    """Dequantize one HXQ tensor to dense F32."""
    cb = codebook.float()
    idx = indices.long()

    if cb.dim() == 1:
        # Scalar VQ
        W = cb[idx]
    else:
        # 2D VQ
        W = cb[idx].reshape(out_f, in_f)

    W = W.reshape(out_f, in_f)

    # Apply sidecar
    if sidecar_pos is not None and len(sidecar_pos) > 0:
        for i in range(len(sidecar_pos)):
            pos = int(sidecar_pos[i])
            row = pos // in_f
            col = pos % in_f
            W[row, col] += sidecar_vals[i]

    return W


def add_tokenizer(writer: GGUFWriter, model_id: str, config: dict):
    """Add tokenizer data to GGUF."""
    tok_json = load_tokenizer_json(model_id)
    if not tok_json:
        print("  WARNING: No tokenizer.json found, skipping tokenizer")
        return

    # Extract token list from tokenizer.json
    model_data = tok_json.get("model", {})
    vocab = model_data.get("vocab", {})

    if vocab:
        # Sort by token ID
        tokens = [""] * len(vocab)
        scores = [0.0] * len(vocab)
        token_types = [0] * len(vocab)  # 0=normal

        for token_str, token_id in vocab.items():
            if token_id < len(tokens):
                tokens[token_id] = token_str
                # SentencePiece scores aren't in tokenizer.json directly
                # Use 0.0 as default

        writer.add_tokenizer_model("llama")  # SentencePiece/BPE
        writer.add_token_list(tokens)
        writer.add_token_scores(scores)
        writer.add_token_types(token_types)

    bos = config.get("bos_token_id", 1)
    eos = config.get("eos_token_id", 2)
    pad = config.get("pad_token_id", 0)
    writer.add_bos_token_id(bos)
    writer.add_eos_token_id(eos)
    writer.add_pad_token_id(pad)

    print(f"  Tokenizer: {len(vocab)} tokens, BOS={bos}, EOS={eos}")


def convert(model_id: str, output_path: str, mode: str, dtype: str):
    """Main conversion."""
    t_start = time.time()
    cpu_start = time.process_time()
    start_iso = time.strftime('%Y-%m-%dT%H:%M:%S')

    print(f"=== HXQ to GGUF Converter ===\n")
    print(f"Model:  {model_id}")
    print(f"Output: {output_path}")
    print(f"Mode:   {mode}")
    print(f"Dtype:  {dtype}")
    print()

    # Load config
    config = load_config(model_id)
    arch = config.get("model_type", "llama")
    # Map model_type to GGUF arch
    arch_map = {"zamba2": "mamba", "mamba": "mamba", "mamba2": "mamba",
                "llama": "llama", "qwen2": "qwen2"}
    gguf_arch = arch_map.get(arch, "llama")

    print(f"Architecture: {arch} -> GGUF arch '{gguf_arch}'")

    # Find safetensors
    st_files = find_safetensors(model_id)
    print(f"Safetensors: {len(st_files)} files")

    # Collect all tensor names
    all_tensors = {}
    for sf_path in st_files:
        with safe_open(sf_path, framework="pt") as sf:
            for name in sf.keys():
                all_tensors[name] = sf_path

    print(f"Total tensors in safetensors: {len(all_tensors)}")

    # Identify HXQ modules
    hxq_modules = identify_hxq_modules(list(all_tensors.keys()))
    hxq_prefixes = set(hxq_modules.keys())

    # Separate HXQ and non-HXQ tensor names
    hxq_tensor_names = set()
    for mod_info in hxq_modules.values():
        hxq_tensor_names.update(mod_info.values())

    non_hxq_names = [n for n in all_tensors.keys() if n not in hxq_tensor_names]

    print(f"HXQ modules: {len(hxq_modules)}")
    print(f"Non-HXQ tensors: {len(non_hxq_names)}")
    print()

    # Create GGUF writer
    writer = GGUFWriter(output_path, arch=gguf_arch)

    # Add metadata
    writer.add_name(os.path.basename(model_id))
    writer.add_description(f"HXQ compressed model converted from {model_id}")

    # Architecture params
    hidden_size = config.get("hidden_size", 2048)
    n_layers = config.get("num_hidden_layers", 38)
    vocab_size = config.get("vocab_size", 32000)
    n_heads = config.get("num_attention_heads", 32)
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    ctx_len = config.get("max_position_embeddings", 4096)
    ffn_size = config.get("intermediate_size", config.get("ffn_hidden_size", 8192))

    writer.add_uint32(f"{gguf_arch}.vocab_size", vocab_size)
    writer.add_uint32(f"{gguf_arch}.context_length", ctx_len)
    writer.add_uint32(f"{gguf_arch}.embedding_length", hidden_size)
    writer.add_uint32(f"{gguf_arch}.block_count", n_layers)
    writer.add_uint32(f"{gguf_arch}.feed_forward_length", ffn_size)

    if gguf_arch == "mamba":
        # SSM-specific metadata
        d_conv = config.get("mamba_d_conv", 4)
        d_state = config.get("mamba_d_state", 128)
        d_inner = config.get("mamba_expand", 2) * hidden_size
        head_dim = config.get("mamba_headdim", 64)
        n_heads_ssm = config.get("n_mamba_heads", 64)
        dt_rank = config.get("time_step_rank", hidden_size // 16)

        writer.add_uint32(f"{gguf_arch}.ssm.conv_kernel", d_conv)
        writer.add_uint32(f"{gguf_arch}.ssm.state_size", d_state)
        writer.add_uint32(f"{gguf_arch}.ssm.inner_size", d_inner)
        writer.add_uint32(f"{gguf_arch}.ssm.time_step_rank", dt_rank)

    if n_heads != n_kv_heads:
        writer.add_uint32(f"{gguf_arch}.attention.head_count", n_heads)
        writer.add_uint32(f"{gguf_arch}.attention.head_count_kv", n_kv_heads)

    # HXQ-specific metadata
    writer.add_string("hxq.version", "0.4.0")
    writer.add_string("hxq.mode", mode)
    writer.add_uint32("hxq.n_clusters", 256)
    writer.add_uint32("hxq.vector_dim", 1)
    writer.add_uint32("hxq.module_count", len(hxq_modules))

    # Tokenizer
    add_tokenizer(writer, model_id, config)

    print(f"\nWriting tensors...")

    # Output dtype for dense/non-HXQ
    np_dtype = np.float32
    ggml_dtype = GGMLQuantizationType.F32
    if dtype == "f16":
        np_dtype = np.float16
        ggml_dtype = GGMLQuantizationType.F16

    tensor_count = 0
    total_bytes = 0

    # Write non-HXQ tensors (as-is, converted to target dtype)
    for name in sorted(non_hxq_names):
        sf_path = all_tensors[name]
        with safe_open(sf_path, framework="pt") as sf:
            t = sf.get_tensor(name).cpu()

        arr = t.numpy().astype(np_dtype)
        writer.add_tensor(name, arr, raw_dtype=ggml_dtype)
        tensor_count += 1
        total_bytes += arr.nbytes

    print(f"  Non-HXQ: {len(non_hxq_names)} tensors ({total_bytes / 1e6:.1f} MB)")

    # Write HXQ tensors
    hxq_bytes = 0
    hxq_count = 0

    for prefix in sorted(hxq_modules.keys()):
        mod_info = hxq_modules[prefix]

        # Load all parts
        sf_path = all_tensors[mod_info["codebook"]]
        with safe_open(sf_path, framework="pt") as sf:
            codebook = sf.get_tensor(mod_info["codebook"]).cpu()
            indices = sf.get_tensor(mod_info["indices"]).cpu()

            sc_pos = None
            sc_vals = None
            if "sidecar_positions" in mod_info:
                sc_pos = sf.get_tensor(mod_info["sidecar_positions"]).cpu()
                sc_vals = sf.get_tensor(mod_info["sidecar_values"]).cpu()

        out_f, in_f = indices.shape[0], indices.shape[1]

        if mode == "dense":
            # Dequantize to dense
            W = dequant_hxq_tensor(codebook, indices, sc_pos, sc_vals, out_f, in_f)
            arr = W.numpy().astype(np_dtype)
            # Store as the original weight name (without .codebook/.indices)
            writer.add_tensor(prefix + ".weight", arr, raw_dtype=ggml_dtype)
            tensor_count += 1
            total_bytes += arr.nbytes
            hxq_bytes += arr.nbytes
            hxq_count += 1

        elif mode == "compressed":
            # Store codebook, indices, sidecar as separate GGUF tensors
            # Codebook: float32 [k] or [k, vdim]
            cb_arr = codebook.float().numpy()
            writer.add_tensor(mod_info["codebook"], cb_arr,
                              raw_dtype=GGMLQuantizationType.F32)

            # Indices: uint8 [out_f, in_f] — store as I8 (closest GGML type)
            idx_arr = indices.numpy().astype(np.int8)  # GGML I8 is signed
            writer.add_tensor(mod_info["indices"], idx_arr,
                              raw_dtype=GGMLQuantizationType.I8)

            # Sidecar positions: int64
            if sc_pos is not None and len(sc_pos) > 0:
                sc_pos_arr = sc_pos.numpy().astype(np.int64)
                writer.add_tensor(mod_info["sidecar_positions"], sc_pos_arr,
                                  raw_dtype=GGMLQuantizationType.I64)

                sc_vals_arr = sc_vals.float().numpy()
                writer.add_tensor(mod_info["sidecar_values"], sc_vals_arr,
                                  raw_dtype=GGMLQuantizationType.F32)

                tensor_count += 4
                sz = cb_arr.nbytes + idx_arr.nbytes + sc_pos_arr.nbytes + sc_vals_arr.nbytes
            else:
                tensor_count += 2
                sz = cb_arr.nbytes + idx_arr.nbytes

            total_bytes += sz
            hxq_bytes += sz
            hxq_count += 1

    print(f"  HXQ:     {hxq_count} modules ({hxq_bytes / 1e6:.1f} MB in {mode} mode)")
    print(f"  Total:   {tensor_count} tensors ({total_bytes / 1e6:.1f} MB)")

    # Finalize
    print(f"\nWriting GGUF file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path)
    wall_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"=== DONE ===")
    print(f"{'='*60}")
    print(f"  Output:    {output_path}")
    print(f"  File size: {file_size / 1e6:.1f} MB ({file_size / 1e9:.2f} GB)")
    print(f"  Tensors:   {tensor_count}")
    print(f"  Mode:      {mode}")
    print(f"  Wall time: {wall_time:.1f}s")

    # Receipt
    receipt = {
        "wo": "WO-HXQ-GGUF-CONVERT",
        "model": model_id,
        "output": output_path,
        "mode": mode,
        "dtype": dtype,
        "file_size_bytes": file_size,
        "tensor_count": tensor_count,
        "hxq_modules": hxq_count,
        "non_hxq_tensors": len(non_hxq_names),
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

    receipt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "receipts")
    os.makedirs(receipt_dir, exist_ok=True)
    receipt_path = os.path.join(receipt_dir, "convert_hxq_to_gguf.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"  Receipt:   {receipt_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HXQ model to GGUF")
    parser.add_argument("model_id", help="HuggingFace model ID or local path")
    parser.add_argument("-o", "--output", default=None, help="Output GGUF path")
    parser.add_argument("--mode", choices=["compressed", "dense"], default="compressed",
                        help="compressed=keep HXQ format, dense=dequantize to F32/F16")
    parser.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                        help="Output dtype for dense tensors (default: f32)")
    args = parser.parse_args()

    if args.output is None:
        basename = args.model_id.replace("/", "-").replace("--", "-")
        args.output = f"{basename}-{args.mode}.gguf"

    convert(args.model_id, args.output, args.mode, args.dtype)


if __name__ == "__main__":
    main()
