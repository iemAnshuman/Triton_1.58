# Copyright 2026 Anshuman Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
benchmark.py — Kernel-level and model-level benchmarking suite.

Measures VRAM usage, inference speed (tokens/sec), perplexity, and
standalone kernel latency across multiple matrix configurations.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Optional

from quantize import quantize_4bit, dequantize_4bit, pack_4bit
from kernels import matmul_4bit

__all__ = [
    "benchmark_kernel",
    "measure_latency",
    "compute_perplexity",
    "run_kernel_benchmarks",
    "run_model_benchmarks",
]

DEVICE = "cuda"


def benchmark_kernel(
    M: int,
    K: int,
    N: int,
    group_size: int = 128,
    warmup: int = 5,
    runs: int = 20,
) -> Dict:
    """
    Benchmark the custom 4-bit kernel against torch.matmul (cuBLAS).

    Args:
        M, K, N: Matrix dimensions.
        group_size: Quantization group size.
        warmup: Number of warmup iterations.
        runs: Number of timed iterations.

    Returns:
        Dict with timing results, speedup, and compression ratio.
    """
    W = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    q, s = quantize_4bit(W, group_size)
    packed = pack_4bit(q)
    w_T = packed.T.contiguous()
    s_T = s.T.contiguous()
    W_deq_T = dequantize_4bit(q, s, group_size).T.contiguous()
    X = torch.randn(M, K, dtype=torch.float16, device=DEVICE)

    # Benchmark torch.matmul (cuBLAS)
    for _ in range(warmup):
        _ = X @ W_deq_T
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = X @ W_deq_T
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / runs * 1000

    # Benchmark our kernel
    for _ in range(warmup):
        _ = matmul_4bit(X, w_T, s_T, group_size)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = matmul_4bit(X, w_T, s_T, group_size)
    torch.cuda.synchronize()
    t_ours = (time.perf_counter() - t0) / runs * 1000

    mem_fp16 = W.nelement() * 2
    mem_4bit = packed.nelement() * 4 + s.nelement() * 2

    del W, q, s, packed, w_T, s_T, W_deq_T, X
    torch.cuda.empty_cache()

    return {
        "M": M, "K": K, "N": N,
        "torch_ms": round(t_torch, 3),
        "ours_ms": round(t_ours, 3),
        "speedup": round(t_torch / t_ours, 2),
        "mem_fp16_MB": round(mem_fp16 / 1e6, 2),
        "mem_4bit_MB": round(mem_4bit / 1e6, 2),
        "compression": round(mem_fp16 / mem_4bit, 2),
    }


def run_kernel_benchmarks(
    test_sizes: Optional[List] = None,
    group_size: int = 128,
) -> List[Dict]:
    """Run kernel benchmarks across multiple matrix sizes."""
    if test_sizes is None:
        test_sizes = [
            (1, 2048, 2048), (1, 2048, 8192), (1, 8192, 2048),
            (32, 2048, 2048), (32, 2048, 8192), (128, 2048, 2048),
        ]

    print("=" * 90)
    print("KERNEL BENCHMARK: matmul_4bit vs torch.matmul")
    print("=" * 90)

    results = []
    for M, K, N in test_sizes:
        r = benchmark_kernel(M, K, N, group_size)
        results.append(r)
        tag = ">>>" if r["speedup"] >= 1.0 else "   "
        print(
            f"{tag} M={M:>4} K={K:>5} N={N:>5} | "
            f"torch={r['torch_ms']:>7.3f}ms  ours={r['ours_ms']:>7.3f}ms  "
            f"speedup={r['speedup']:>5.2f}x | "
            f"{r['mem_fp16_MB']:.1f}MB -> {r['mem_4bit_MB']:.1f}MB ({r['compression']:.1f}x)"
        )

    print()
    return results


def measure_latency(
    model,
    tokenizer,
    prompt: str = "The meaning of life is",
    max_new_tokens: int = 128,
    warmup: int = 2,
    runs: int = 5,
) -> tuple:
    """
    Measure inference speed in tokens per second.

    Returns:
        (mean_tokens_per_sec, std_tokens_per_sec)
    """
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    for _ in range(warmup):
        with torch.no_grad():
            model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)

    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        ntok = out.shape[1] - inp["input_ids"].shape[1]
        times.append(ntok / (time.perf_counter() - t0))

    return float(np.mean(times)), float(np.std(times))


def compute_perplexity(
    model,
    tokenizer,
    max_samples: int = 50,
    max_length: int = 512,
) -> float:
    """
    Compute perplexity on WikiText-2 test set.

    Returns:
        Perplexity score (lower is better).
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 50][:max_samples]

    total_loss, total_tok = 0.0, 0
    for text in texts:
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(DEVICE)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            loss = model(ids, labels=ids).loss
        if loss is not None and not torch.isnan(loss):
            n = ids.shape[1] - 1
            total_loss += loss.item() * n
            total_tok += n

    return float(np.exp(total_loss / total_tok)) if total_tok > 0 else float("inf")


def run_model_benchmarks(
    model,
    tokenizer,
    label: str = "model",
) -> Dict:
    """
    Run full model-level benchmarks: VRAM, speed, and perplexity.

    Returns:
        Dict with vram, tok_sec, tok_std, and ppl.
    """
    torch.cuda.reset_peak_memory_stats()
    dummy = tokenizer("hello", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**dummy)
    vram = torch.cuda.max_memory_allocated() / 1e9

    tok, tok_std = measure_latency(model, tokenizer)
    print(f"  [{label}] VRAM: {vram:.2f} GB | Tokens/sec: {tok:.1f} +/- {tok_std:.1f}")

    print(f"  [{label}] Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer)
    print(f"  [{label}] Perplexity: {ppl:.2f}")

    return {
        "vram": round(vram, 2),
        "tok_sec": round(tok, 1),
        "tok_std": round(tok_std, 1),
        "ppl": round(ppl, 2),
    }