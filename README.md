# HXQ Native — C/CUDA Decompression Library

Portable C implementation of the HXQ vector quantization codec for weight decompression and inference.

**Author:** Joshua P Fellows, Echo Labs LLC

## What This Is

The core HXQ decompression engine in pure C99. No Python, no PyTorch, no Triton. Reads codebook + packed indices + sidecar corrections and reconstructs weight matrices. Runs anywhere a C compiler runs.

## Build

```bash
make        # Build library + tests
make test   # Build + run 41 unit tests
make cuda   # Build GPU kernel (requires nvcc)
make clean
```

## Features

| Feature | Status |
|---|---|
| Scalar VQ (k=256, uint8 indices) | ✅ tested |
| 2D VQ (k=4096, 12-bit packed indices) | ✅ tested |
| Sidecar sparse corrections | ✅ tested |
| Shared materialization buffer | ✅ tested |
| Confidence signal (sidecar L2 norm) | ✅ tested |
| Meta-kernel `hxq_dequant()` | ✅ tested |
| CUDA GPU kernel | ✅ compiles (needs GPU test) |
| **Total** | **41/41 tests** |

## Architecture

```
hxq.h          — Public API (one header, all types + functions)
hxq.c          — Pure C implementation (~390 lines)
hxq_cuda.cu    — GPU kernels (codebook in shared memory, fused gather)
test_hxq.c     — Unit tests (41 tests)
```

## API

```c
#include "hxq.h"

// Load a tensor
hxq_tensor_t tensor;
hxq_tensor_init(&tensor);
hxq_tensor_load_codebook(&tensor, codebook_data, 4096, 2);
hxq_tensor_load_indices_12bit(&tensor, packed_data, len, out_f, in_f);
hxq_tensor_load_sidecar(&tensor, rows, cols, deltas, nnz);

// Decompress
float *weights = malloc(out_f * in_f * sizeof(float));
hxq_tensor_decompress(&tensor, weights);

// Read confidence signal (the sidecar finding, ρ=0.574)
float confidence = hxq_get_sidecar_confidence(&tensor);

// Meta-kernel: ONE function for all frameworks
hxq_shared_buffer_t buf;
hxq_shared_buffer_init(&buf, 1024 * 1024);

hxq_result_t result;
hxq_dequant(&tensor, &buf, HXQ_BACKEND_AUTO, &result);
// result.weights → dense float buffer, ready for matmul
// result.confidence → quality signal (ρ=0.574 with CE loss)
// buf is reused across ALL tensors, ALL models

framework_matmul(input, result.weights, output);

hxq_tensor_free(&tensor);
hxq_shared_buffer_free(&buf);
```

## Integration Paths

| Target | What's Needed |
|---|---|
| llama.cpp / GGUF | Register HXQ as quantization type, add dequant kernel |
| vLLM | C++ quantization plugin wrapping hxq.h |
| TensorRT-LLM | Custom plugin using CUDA kernel |
| Mobile (Android/iOS) | Compile hxq.c with NDK/Xcode — zero dependencies |
| Jetson | CUDA kernel + hxq.c |

## The Confidence Signal

Every `hxq_tensor_decompress()` call computes `sidecar_l2_norm` — the L2 norm of the sidecar correction applied to the output. This correlates with per-chunk cross-entropy loss at ρ=0.574 (p=1.42e-50, n=562 chunks, Zamba2-1.2B).

Meaning: when the sidecar works harder, the model is less confident. This is a zero-cost quality signal — computed as a byproduct of decompression. No other quantization method provides this signal because no other method maintains sparse residual corrections during inference.

## Relation to helix-substrate

This is the native companion to the Python `helix-substrate` package (PyPI). Same codec, same format, different runtime. Models compressed with `helix-substrate` produce safetensors files that this library can read (with a format adapter for the safetensors container).

## License

MIT
