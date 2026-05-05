# Copyright 2026 Anshuman Agrawal
# Licensed under the Apache License, Version 2.0

"""
Search quality-oriented ternary PTQ configurations.

This script is meant to be run on a CUDA machine. Pure post-training ternary
quantization can be too destructive for small LLMs, so this evaluates both
pure ternary and hybrid variants that keep selected sensitive projections in
FP16 while the remaining Linear layers use the Triton ternary kernel.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEVICE = "cuda"


def compute_perplexity_local(model, tokenizer, max_samples: int, max_length: int) -> float:
    from datasets import load_dataset
    import numpy as np

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


def measure_latency_local(
    model,
    tokenizer,
    max_new_tokens: int,
    runs: int,
    prompt: str = "The meaning of life is",
):
    import numpy as np

    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
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


def build_skip_layers(config_name: str, n_layers: int, keep_edge_layers: int):
    skip = ["embed", "lm_head", "rotary"]

    if "attn_fp16" in config_name:
        skip.append("self_attn")
    if "mlp_fp16" in config_name:
        skip.append("mlp")
    if "output_proj_fp16" in config_name:
        skip.extend(["self_attn.o_proj", "mlp.down_proj"])

    if "edge_fp16" in config_name and keep_edge_layers > 0:
        edge_ids = list(range(keep_edge_layers))
        edge_ids += list(range(max(keep_edge_layers, n_layers - keep_edge_layers), n_layers))
        skip.extend([f"model.layers.{i}." for i in sorted(set(edge_ids))])

    return skip


def evaluate_config(args, tokenizer, config_name: str):
    from model_ternary import quantize_model_ternary

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    n_layers = getattr(model.config, "num_hidden_layers", 0)
    skip_layers = build_skip_layers(config_name, n_layers, args.keep_edge_layers)

    t0 = time.time()
    model = quantize_model_ternary(
        model,
        group_size=args.group_size,
        skip_layers=skip_layers,
        quant_method=args.quant_method,
    )
    model.eval()
    quantize_sec = time.time() - t0

    dummy = tokenizer("hello", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        model(**dummy)
    vram = torch.cuda.max_memory_allocated() / 1e9

    ppl = compute_perplexity_local(
        model,
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )
    tok, tok_std = measure_latency_local(
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        runs=args.speed_runs,
    )

    del model
    torch.cuda.empty_cache()

    return {
        "config": config_name,
        "skip_layers": skip_layers,
        "quant_method": args.quant_method,
        "group_size": args.group_size,
        "ppl": round(float(ppl), 2),
        "vram_gb": round(float(vram), 2),
        "tok_sec": round(float(tok), 2),
        "tok_std": round(float(tok_std), 2),
        "quantize_sec": round(float(quantize_sec), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--quant-method", choices=["mse", "bitnet"], default="mse")
    parser.add_argument("--target-ppl", type=float, default=100.0)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--speed-runs", type=int, default=3)
    parser.add_argument("--keep-edge-layers", type=int, default=2)
    parser.add_argument("--out", default="ternary_quality_search.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Triton ternary evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    configs = [
        "pure_ternary",
        "edge_fp16",
        "output_proj_fp16",
        "output_proj_fp16_edge_fp16",
        "attn_fp16",
        "attn_fp16_edge_fp16",
        "mlp_fp16",
    ]

    results = []
    for config_name in configs:
        print(f"\n=== {config_name} ===")
        result = evaluate_config(args, tokenizer, config_name)
        results.append(result)
        print(json.dumps(result, indent=2))

        Path(args.out).write_text(json.dumps(results, indent=2))
        if result["ppl"] <= args.target_ppl:
            print(f"\nTarget reached: {result['ppl']:.2f} <= {args.target_ppl:.2f}")
            break

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
