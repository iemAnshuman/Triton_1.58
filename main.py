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
main.py — Master pipeline orchestrating quantization, benchmarking, and evaluation.

Usage:
    python main.py

Runs the complete pipeline:
  1. Self-tests (quantization + kernel correctness)
  2. Kernel-level benchmarks
  3. Model loading, quantization, and text generation comparison
  4. Model-level benchmarks (VRAM, speed, perplexity)
  5. Chart generation and JSON export
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantize import test_quantization
from kernels import test_kernel
from model import quantize_model
from benchmark import run_kernel_benchmarks, run_model_benchmarks
from generate import compare_generations
from visualize import plot_model_benchmarks, plot_kernel_benchmarks

# ── Configuration ──────────────────────────────────────────────
DEVICE = "cuda"
GROUP_SIZE = 128
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TEST_PROMPTS = [
    "The theory of relativity explains that",
    "In machine learning, gradient descent works by",
    "def fibonacci(n):\n    ",
    "The three main benefits of quantization for LLMs are",
]


def main():
    # ── Environment info ───────────────────────────────────────
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}\n")

    # ── Step 1: Self-tests ─────────────────────────────────────
    test_quantization(DEVICE, GROUP_SIZE)
    test_kernel(DEVICE, GROUP_SIZE)

    # ── Step 2: Kernel benchmarks ──────────────────────────────
    kernel_results = run_kernel_benchmarks(group_size=GROUP_SIZE)

    # ── Step 3: Load model and tokenizer ───────────────────────
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    torch.cuda.reset_peak_memory_stats()
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model_fp16.eval()
    print(f"FP16 loaded. VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")

    # ── Step 4: Quantize model ─────────────────────────────────
    print("Quantizing to 4-bit...")
    torch.cuda.reset_peak_memory_stats()
    model_4bit = quantize_model(model_fp16, GROUP_SIZE)
    model_4bit.eval()
    print(f"4-bit ready. VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")

    # ── Step 5: Side-by-side generation comparison ─────────────
    generation_comparison = compare_generations(
        model_fp16, model_4bit, tokenizer, TEST_PROMPTS
    )

    # ── Step 6: Model-level benchmarks ─────────────────────────
    del model_fp16
    torch.cuda.empty_cache()

    print("\nBenchmarking our 4-bit model...")
    ours_bench = run_model_benchmarks(model_4bit, tokenizer, "Ours (4-bit)")
    del model_4bit
    torch.cuda.empty_cache()

    print("\nBenchmarking FP16 baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
    )
    model_fp16.eval()
    fp16_bench = run_model_benchmarks(model_fp16, tokenizer, "FP16")
    del model_fp16
    torch.cuda.empty_cache()

    full_benchmark = {
        "fp16": fp16_bench,
        "ours_4bit": ours_bench,
    }

    # ── Step 7: bitsandbytes comparison ────────────────────────
    try:
        from transformers import BitsAndBytesConfig

        print("\nBenchmarking bitsandbytes NF4...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_bnb = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_cfg, device_map="cuda"
        )
        model_bnb.eval()
        bnb_bench = run_model_benchmarks(model_bnb, tokenizer, "bitsandbytes NF4")
        full_benchmark["bnb_nf4"] = bnb_bench
        del model_bnb
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  bitsandbytes failed: {e}")

    # ── Step 8: Summary ────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Config':<20} {'VRAM (GB)':>10} {'Tok/sec':>10} {'Perplexity':>12}")
    print("-" * 55)
    for name, bench in full_benchmark.items():
        print(f"{name:<20} {bench['vram']:>10.2f} {bench['tok_sec']:>10.1f} {bench['ppl']:>12.2f}")
    print("-" * 55)
    print(
        f"VRAM saved: {(1 - ours_bench['vram'] / fp16_bench['vram']) * 100:.1f}% | "
        f"Perplexity: +{ours_bench['ppl'] - fp16_bench['ppl']:.2f}"
    )

    # ── Step 9: Charts and export ──────────────────────────────
    gpu_name = torch.cuda.get_device_name()
    plot_model_benchmarks(full_benchmark, MODEL_NAME, gpu_name)
    plot_kernel_benchmarks(kernel_results)

    report_data = {
        "model": MODEL_NAME,
        "gpu": gpu_name,
        "group_size": GROUP_SIZE,
        "kernel_benchmarks": kernel_results,
        "model_benchmarks": full_benchmark,
        "generation_comparisons": [
            {"prompt": p, "fp16": f, "quant4": q}
            for p, f, q in generation_comparison
        ],
    }
    with open("report_data.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print("\nSaved: benchmark_charts.png, kernel_charts.png, report_data.json")
    print("All done!")


if __name__ == "__main__":
    main()