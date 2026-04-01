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
generate.py — Text generation pipeline for FP16 vs 4-bit comparison.

Provides utilities for side-by-side text generation and quality comparison
between the original FP16 model and the 4-bit quantized version.
"""

import torch
from typing import List, Tuple

__all__ = ["generate_text", "compare_generations"]

DEVICE = "cuda"


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    do_sample: bool = False,
) -> str:
    """Generate text from a prompt using greedy decoding."""
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=do_sample)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def compare_generations(
    model_fp16,
    model_4bit,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 60,
) -> List[Tuple[str, str, str]]:
    """
    Generate text from both models for each prompt and compare.

    Returns:
        List of (prompt, fp16_output, 4bit_output) tuples.
    """
    results = []

    print("--- FP16 Outputs ---")
    fp16_outputs = []
    for prompt in prompts:
        text = generate_text(model_fp16, tokenizer, prompt, max_new_tokens)
        fp16_outputs.append(text)
        print(f"[FP16] {text[:180]}\n")

    print("--- 4-Bit Outputs ---")
    q4_outputs = []
    for prompt in prompts:
        text = generate_text(model_4bit, tokenizer, prompt, max_new_tokens)
        q4_outputs.append(text)
        print(f"[4BIT] {text[:180]}\n")

    print("=" * 60)
    print("SIDE BY SIDE")
    print("=" * 60)
    for i, p in enumerate(prompts):
        print(f"\nPrompt: {p}")
        print(f"FP16:  {fp16_outputs[i][:150]}")
        print(f"4-Bit: {q4_outputs[i][:150]}")
        results.append((p, fp16_outputs[i], q4_outputs[i]))

    return results