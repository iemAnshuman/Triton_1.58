# Triton 1.58

Post-training quantization of transformer weights down to ternary
`{-1, 0, +1}`, with custom Triton GPU kernels for the quantized matmul.
Follows the BitNet b1.58 formulation (Ma et al., 2024), applied as a
post-training transform on an existing FP16 checkpoint rather than as a
training-time constraint.

The repository holds two phases that share one benchmark harness: a 4-bit
pipeline, and the ternary pipeline built on top of it.

## Status

Research code. Post-training ternary quantization is substantially harder
than 4-bit, and this implementation shows a real perplexity gap against
FP16. Read [Quality caveat](#quality-caveat) before relying on it.

## Requirements

A CUDA GPU is required. The Triton kernels have no CPU fallback.

```bash
pip install -r requirements.txt
```

The default model, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, downloads on
first run.

## Quickstart

Four-way benchmark across FP16, 4-bit, ternary, and bitsandbytes NF4:

```bash
python main_ternary.py
```

Writes `ternary_benchmarks.png` and `ternary_report.json`.

The 4-bit phase on its own:

```bash
python main.py
```

Writes `benchmark_charts.png`, `kernel_charts.png`, and `report_data.json`.

### Correctness self-tests

Each quantizer and kernel module checks itself against a PyTorch reference
when run directly:

```bash
python quantize_ternary.py    # pack/unpack round trip, sparsity, effective bits
python kernels_ternary.py     # kernel output vs dequantized reference matmul
python quantize.py            # 4-bit equivalents
python kernels.py
```

### Quality search

Pure ternary PTQ can be too destructive at this model size, so this sweeps
hybrid configurations that hold selected projections or edge blocks in FP16:

```bash
python ternary_quality_search.py --target-ppl 100 --max-samples 100
```

Writes `ternary_quality_search.json` and stops early once the target
perplexity is reached.

### Calibration experiments

```bash
python qat_calibration.py     # knowledge distillation from the FP16 teacher
python qat_simple.py          # cross-entropy only, lower learning rate
```

## Layout

| File | Role |
|---|---|
| `quantize.py`, `quantize_ternary.py` | Weight quantization and bit packing |
| `kernels.py`, `kernels_ternary.py` | Triton matmul kernels with fused dequantization |
| `model.py`, `model_ternary.py` | Quantized `nn.Module` replacements and model traversal |
| `main.py`, `main_ternary.py` | Benchmark pipelines |
| `benchmark.py` | Perplexity, latency, and VRAM measurement |
| `generate.py` | Text generation helper |
| `visualize.py` | Chart rendering |
| `ternary_quality_search.py` | Hybrid FP16 and ternary configuration sweep |
| `qat_calibration.py`, `qat_simple.py` | Post-quantization calibration experiments |

## How the ternary path works

**Quantization.** Absmean over groups of 128, following BitNet, plus an
optional Lloyd-style variant that iteratively updates the scale and the
zeroing threshold. The Lloyd variant is the default, because it
reconstructs a checkpoint that never saw ternary weights better than raw
absmean rounding does.

**Packing.** 16 ternary values per INT32 at 2 bits each. The encoding is
`0 -> 00`, `+1 -> 01`, `-1 -> 10`, leaving `11` unused.

**Kernel.** Loads packed INT32 weights, unpacks them in SRAM with bitwise
ops, applies the per-group scale, then dispatches to Tensor Cores through
`tl.dot()`. Current NVIDIA consumer hardware has no native ternary matmul,
so the practical win is memory bandwidth rather than multiplication-free
execution.

**Held in FP16.** Embeddings, LM head, and rotary layers.

## Compression math

Scales are FP16, one per group of 128 weights, so they add
`16 / 128 = 0.125` bits to every weight. The table counts that overhead.

| Storage | Bits/weight | vs FP16 |
|---|---|---|
| FP16 | 16.000 | 1.00x |
| 4-bit plus scales | 4.125 | 3.88x |
| Ternary plus scales | 2.125 | **7.53x** |
| Ternary information-theoretic floor | 1.585 | 10.09x |

The floor is `log2(3)` and is unreachable with a fixed 2-bit-per-weight
layout, which is what the packing format uses.

## Quality caveat

Post-training ternary quantization is fundamentally harder than 4-bit,
because the representable capacity is 4x lower. Natively trained BitNet
models match FP16, but ternary PTQ applied to a checkpoint that never saw
ternary weights typically shows a noticeable perplexity gap. The
calibration scripts are a first attempt at closing it, and the quality
search exists because the pure ternary configuration alone was not good
enough.

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
