# Ternary (1.58-bit) Phase

Extends the 4-bit pipeline from the previous commit to ternary weight
quantization, following the BitNet b1.58 paradigm (Ma et al., 2024) applied
as a post-training transform.

## What's new

```
quantize_ternary.py  — absmean ternary quantization + 16-per-INT32 packing
kernels_ternary.py   — Triton ternary matmul kernel with 2-bit unpacking
model_ternary.py     — LinearTernary nn.Module + model traversal
main_ternary.py      — 4-way benchmark pipeline (FP16 / 4-bit / ternary / NF4)
```

The existing `benchmark.py`, `generate.py`, and `visualize.py` modules from
the 4-bit phase are reused unchanged.

## Key design decisions

- **Quantization scheme**: absmean per-group (128), following BitNet's
  quantization function but applied post-training rather than during training.
- **Packing**: 16 ternary values per INT32. Encoding `0 -> 00`, `+1 -> 01`,
  `-1 -> 10` (code `11` unused).
- **Kernel**: unpacks to FP16 `{-1.0, 0.0, +1.0}`, applies per-group scale,
  dispatches to Tensor Cores via `tl.dot()`. True multiplication-free
  execution requires hardware with native ternary matmul (not available on
  current NVIDIA consumer GPUs).
- **Skipped layers**: embeddings, LM head, and rotary — kept in FP16.

## Compression math

| Storage                          | Bits/weight |
|----------------------------------|-------------|
| FP16                             | 16.00       |
| 4-bit (previous phase)           | 4.00 + scales ~= 4.12 |
| Ternary (this phase)             | 2.00 + scales ~= 2.12 |
| Information-theoretic minimum    | 1.58        |

Practical VRAM reduction vs FP16: roughly 8x on quantized layers.

## Running

```bash
python main_ternary.py
```

Outputs `ternary_benchmarks.png` and `ternary_report.json`.

## Expected quality caveat

Post-training ternary quantization is fundamentally harder than 4-bit — the
information-theoretic capacity is 4x lower. Unlike natively trained BitNet
models (which match FP16), PTQ ternary typically shows a noticeable
perplexity gap. This implementation is a foundation for future
quantization-aware fine-tuning work that can close the gap.
