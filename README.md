# Efficient LLM Inference: 4-Bit Quantized Engine

Custom 4-bit post-training quantization pipeline with fused Triton GPU kernels for accelerated LLM inference.

## Module Structure

```
quantize.py   — Quantization functions: quantize_4bit(), dequantize_4bit(), pack_4bit(), unpack_4bit()
kernels.py    — Triton kernel: matmul_4bit_kernel() with @autotune, matmul_4bit() Python wrapper
model.py      — Linear4Bit nn.Module, quantize_linear_layer(), quantize_model()
benchmark.py  — Kernel-level and model-level benchmarking suite
generate.py   — Text generation pipeline with FP16 vs 4-bit comparison
visualize.py  — Matplotlib chart generation for reports
main.py       — Master script orchestrating the full pipeline
```

## Requirements

- Python 3.10+
- PyTorch 2.1+ with CUDA 11.8+
- OpenAI Triton 2.1+
- HuggingFace Transformers 4.36+
- NumPy, Matplotlib
- (Optional) bitsandbytes 0.41+ for NF4 baseline comparison

## Usage

```bash
# Run the full pipeline (tests, benchmarks, generation, charts)
python main.py

# Run individual module tests
python quantize.py    # quantization self-test
python kernels.py     # kernel correctness test
```

## Results (TinyLlama-1.1B on Tesla T4)

| Configuration      | VRAM (GB) | Tokens/sec | Perplexity |
|---------------------|-----------|------------|------------|
| FP16 Baseline       | 2.21      | 32.7       | 17.9       |
| Ours (4-bit Triton) | 1.08      | 22.5       | 19.5       |
| bitsandbytes NF4    | 2.21      | 20.4       | 18.7       |

**51% VRAM reduction** · **10% faster than bitsandbytes** · **+1.6 perplexity**

## License

Copyright 2026 Anshuman Agrawal.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text,
and [NOTICE](NOTICE) for attribution information. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.