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
main_ternary.py — End-to-end pipeline for ternary quantized LLM inference.

Runs:
  1. Quantization + kernel self-tests
  2. Model loading (TinyLlama-1.1B)
  3. FP16 baseline generation + benchmarks
  4. Ternary quantization + generation + benchmarks
  5. 4-way comparison: FP16 / 4-bit / ternary / bitsandbytes NF4
  6. Chart generation and JSON export
"""

import json
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ternary (this phase)
from quantize_ternary import test_ternary_quantization
from kernels_ternary import test_ternary_kernel
from model_ternary import quantize_model_ternary

# Shared infrastructure from the 4-bit phase (reused unchanged)
from benchmark import measure_latency, compute_perplexity
from generate import generate_text
from visualize import plot_model_benchmarks

# Optional: include the 4-bit pipeline as a baseline
try:
    from model import quantize_model as quantize_model_4bit
    HAVE_4BIT = True
except ImportError:
    HAVE_4BIT = False

# ── Configuration ──────────────────────────────────────────────
DEVICE = "cuda"
GROUP_SIZE = 128
QUANT_METHOD = "mse"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TEST_PROMPTS = [
    "The theory of relativity explains that",
    "In machine learning, gradient descent works by",
    "def fibonacci(n):\n    ",
    "The three main benefits of quantization for LLMs are",
]


def benchmark_one_config(model, tokenizer, label: str, dummy_input):
    """Measure VRAM, speed, and perplexity for a single configuration."""
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(**dummy_input)
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


def main():
    # ── Environment info ───────────────────────────────────────
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}\n")

    # ── Self-tests ─────────────────────────────────────────────
    test_ternary_quantization(DEVICE, GROUP_SIZE)
    test_ternary_kernel(DEVICE, GROUP_SIZE)

    # ── Load tokenizer and dummy input ─────────────────────────
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy = tokenizer("hello", return_tensors="pt").to(DEVICE)

    results = {}

    # ── FP16 baseline ──────────────────────────────────────────
    print("\n=== FP16 Baseline ===")
    torch.cuda.reset_peak_memory_stats()
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model_fp16.eval()

    # Keep FP16 generations for qualitative comparison
    fp16_generations = {
        p: generate_text(model_fp16, tokenizer, p, max_new_tokens=60)
        for p in TEST_PROMPTS
    }
    print("FP16 sample:", fp16_generations[TEST_PROMPTS[0]][:150])

    results["fp16"] = benchmark_one_config(model_fp16, tokenizer, "FP16", dummy)
    del model_fp16
    torch.cuda.empty_cache()

    # ── 4-bit (previous phase, for comparison) ─────────────────
    if HAVE_4BIT:
        print("\n=== 4-bit Triton (previous phase) ===")
        torch.cuda.reset_peak_memory_stats()
        model_4bit = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
        )
        model_4bit.eval()
        model_4bit = quantize_model_4bit(model_4bit, GROUP_SIZE)
        model_4bit.eval()

        results["ours_4bit"] = benchmark_one_config(model_4bit, tokenizer, "4-bit", dummy)
        del model_4bit
        torch.cuda.empty_cache()

    # ── Ternary (this phase) ───────────────────────────────────
    print("\n=== Ternary (1.58-bit) Triton ===")
    torch.cuda.reset_peak_memory_stats()
    model_tern = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model_tern.eval()
    model_tern = quantize_model_ternary(model_tern, GROUP_SIZE, quant_method=QUANT_METHOD)
    model_tern.eval()

    tern_generations = {
        p: generate_text(model_tern, tokenizer, p, max_new_tokens=60)
        for p in TEST_PROMPTS
    }
    print("Ternary sample:", tern_generations[TEST_PROMPTS[0]][:150])

    results["ours_ternary"] = benchmark_one_config(model_tern, tokenizer, "Ternary", dummy)
    del model_tern
    torch.cuda.empty_cache()

    # ── bitsandbytes NF4 (industry baseline) ───────────────────
    try:
        from transformers import BitsAndBytesConfig

        print("\n=== bitsandbytes NF4 ===")
        torch.cuda.reset_peak_memory_stats()
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_bnb = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_cfg, device_map="cuda"
        )
        model_bnb.eval()
        results["bnb_nf4"] = benchmark_one_config(model_bnb, tokenizer, "NF4", dummy)
        del model_bnb
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  bitsandbytes skipped: {e}")

    # ── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Config':<22} {'VRAM (GB)':>12} {'Tokens/sec':>14} {'Perplexity':>14}")
    print("-" * 75)
    for name, b in results.items():
        print(f"{name:<22} {b['vram']:>12.2f} {b['tok_sec']:>14.1f} {b['ppl']:>14.2f}")
    print("-" * 75)

    fp16 = results["fp16"]
    tern = results["ours_ternary"]
    print(
        f"Ternary vs FP16 — VRAM reduction: {(1 - tern['vram'] / fp16['vram']) * 100:.1f}% | "
        f"Perplexity delta: +{tern['ppl'] - fp16['ppl']:.2f}"
    )

    # ── Chart and export ───────────────────────────────────────
    gpu_name = torch.cuda.get_device_name()
    plot_model_benchmarks(
        results,
        model_name=MODEL_NAME,
        gpu_name=gpu_name,
        save_path="ternary_benchmarks.png",
    )

    report = {
        "phase": "ternary_1.58bit",
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "group_size": GROUP_SIZE,
        "quant_method": QUANT_METHOD,
        "benchmarks": results,
        "sample_generations": {
            "fp16": fp16_generations,
            "ternary": tern_generations,
        },
    }
    with open("ternary_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nSaved: ternary_benchmarks.png, ternary_report.json")
    print("All done!")


if __name__ == "__main__":
    main()
