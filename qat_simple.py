# Copyright 2026 Anshuman Agrawal
# Licensed under the Apache License, Version 2.0

"""
qat_simple.py — Simple calibration attempt for ternary model.
Uses FP32 parameters, CE loss only (no KD), much lower learning rate.
"""

import sys, json, time, torch, numpy as np
sys.path.insert(0, '/content/Triton_1.58')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from model_ternary import quantize_model_ternary
from benchmark import compute_perplexity
from generate import generate_text

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GROUP_SIZE = 128
DEVICE = "cuda"
N_STEPS = 500
LR = 1e-5  # 20x lower than before
MAX_SEQ_LEN = 256

print(f"GPU: {torch.cuda.get_device_name()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# FP16 baseline
print("\n=== FP16 baseline ===")
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
model_fp16.eval()
fp16_ppl = compute_perplexity(model_fp16, tokenizer)
print(f"  FP16 perplexity: {fp16_ppl:.2f}")
del model_fp16
torch.cuda.empty_cache()

# Ternary model
print("\n=== Ternary quantization ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
model = quantize_model_ternary(model, GROUP_SIZE)
model.eval()
pre_ppl = compute_perplexity(model, tokenizer)
print(f"  Pre-calibration perplexity: {pre_ppl:.2f}")

# Prepare data
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in ds["text"] if len(t.strip()) > 100]
all_ids = []
for text in texts[:2000]:
    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    if enc["input_ids"].shape[1] >= 32:
        all_ids.append(enc["input_ids"].squeeze(0))
print(f"  Calibration sequences: {len(all_ids)}")

# Freeze all, then add FP32 scale+shift wrappers
for p in model.parameters():
    p.requires_grad = False

class ScaleShift(torch.nn.Module):
    def __init__(self, module, features):
        super().__init__()
        self.module = module
        # FP32 parameters to avoid overflow
        self.scale = torch.nn.Parameter(torch.ones(features, dtype=torch.float32, device=DEVICE))
        self.shift = torch.nn.Parameter(torch.zeros(features, dtype=torch.float32, device=DEVICE))

    def forward(self, x):
        out = self.module(x)
        return (out.float() * self.scale + self.shift).half()

cal_params = []
n_wrapped = 0
for name, module in list(model.named_modules()):
    if hasattr(module, 'packed_weight') and hasattr(module, 'out_features'):
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        wrapper = ScaleShift(module, module.out_features)
        setattr(parent, parts[-1], wrapper)
        cal_params.extend([wrapper.scale, wrapper.shift])
        n_wrapped += 1

print(f"  Wrapped {n_wrapped} layers | Trainable: {sum(p.numel() for p in cal_params):,} params")

optimizer = torch.optim.AdamW(cal_params, lr=LR, weight_decay=0.0)
ppl_history = [("step_0", pre_ppl)]

print("\n=== Calibration ===")
model.train()
t0 = time.time()

for step in range(N_STEPS):
    idx = np.random.randint(len(all_ids))
    input_ids = all_ids[idx].unsqueeze(0).to(DEVICE)
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    loss = out.loss

    if torch.isnan(loss) or torch.isinf(loss):
        # Skip bad batches instead of exploding
        print(f"  Step {step+1}: NaN/Inf loss, skipping")
        optimizer.zero_grad()
        continue

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(cal_params, 0.5)
    optimizer.step()

    if (step + 1) % 50 == 0:
        print(f"  Step {step+1}/{N_STEPS} | loss={loss.item():.4f} | time={time.time()-t0:.0f}s")

    if (step + 1) % 100 == 0:
        model.eval()
        mid_ppl = compute_perplexity(model, tokenizer)
        ppl_history.append((f"step_{step+1}", mid_ppl))
        print(f"  >>> Perplexity at step {step+1}: {mid_ppl:.2f}")
        model.train()

# Final eval
model.eval()
post_ppl = compute_perplexity(model, tokenizer)
ppl_history.append(("final", post_ppl))

print("\n--- Post-calibration generations ---")
for prompt in ["The theory of relativity explains that", "def fibonacci(n):\n    "]:
    text = generate_text(model, tokenizer, prompt, max_new_tokens=60)
    print(f"  {text[:150]}")

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"  FP16:                {fp16_ppl:.2f}")
print(f"  Ternary (before):    {pre_ppl:.2f}")
print(f"  Ternary (after):     {post_ppl:.2f}")
print(f"\n  Trajectory:")
for label, p in ppl_history:
    print(f"    {label}: {p:.2f}")

with open("/content/qat_simple_results.json", "w") as f:
    json.dump({
        "fp16_ppl": fp16_ppl, "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "trajectory": ppl_history, "n_steps": N_STEPS, "lr": LR,
        "n_wrapped": n_wrapped, "n_trainable": sum(p.numel() for p in cal_params),
    }, f, indent=2, default=str)
print("\nSaved: /content/qat_simple_results.json")
