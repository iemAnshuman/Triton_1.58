# Copyright 2026 Anshuman Agrawal
# Licensed under the Apache License, Version 2.0

"""
qat_calibration.py — Lightweight calibration for ternary-quantized models.

Applies per-linear output scale/shift adapters to the ternary-quantized model
and fine-tunes them on a small calibration set (WikiText-2) using knowledge
distillation from the FP16 teacher. The ternary weights remain frozen.

This is NOT full quantization-aware training (which requires training from scratch).
It is a post-quantization calibration step that adapts the model's residual layers
to compensate for the quality loss introduced by ternary weight quantization.
"""

import sys, json, time, torch, numpy as np
sys.path.insert(0, '/content/Triton_1.58')

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from model_ternary import quantize_model_ternary
from benchmark import compute_perplexity

# ── Configuration ──────────────────────────────────────────────
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GROUP_SIZE = 128
QUANT_METHOD = "mse"
DEVICE = "cuda"

# Calibration hyperparameters
N_CALIBRATION_STEPS = 500
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
MAX_SEQ_LEN = 256
EVAL_EVERY = 100

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Config: {N_CALIBRATION_STEPS} steps, lr={LEARNING_RATE}, seq_len={MAX_SEQ_LEN}")

# ── Load tokenizer ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Step 1: Measure FP16 baseline perplexity ───────────────────
print("\n=== Step 1: FP16 baseline ===")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
)
model_fp16.eval()
fp16_ppl = compute_perplexity(model_fp16, tokenizer)
print(f"  FP16 perplexity: {fp16_ppl:.2f}")

# ── Step 2: Quantize to ternary and measure pre-calibration ───
print("\n=== Step 2: Ternary quantization (pre-calibration) ===")
model_ternary = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
)
model_ternary = quantize_model_ternary(
    model_ternary,
    GROUP_SIZE,
    quant_method=QUANT_METHOD,
)
model_ternary.eval()
pre_cal_ppl = compute_perplexity(model_ternary, tokenizer)
print(f"  Ternary perplexity (pre-calibration): {pre_cal_ppl:.2f}")

# ── Step 3: Prepare calibration data ──────────────────────────
print("\n=== Step 3: Preparing calibration data ===")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in ds["text"] if len(t.strip()) > 100]

# Tokenize and create fixed-length chunks
all_ids = []
for text in texts[:2000]:  # Use up to 2000 passages
    enc = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")
    if enc["input_ids"].shape[1] >= 32:  # Skip very short sequences
        all_ids.append(enc["input_ids"].squeeze(0))

print(f"  Prepared {len(all_ids)} calibration sequences")

# ── Step 4: Knowledge distillation calibration ────────────────
print("\n=== Step 4: Calibration training ===")
print("  Strategy: minimize cross-entropy with teacher (FP16) logits")

# Freeze everything in ternary model
for param in model_ternary.parameters():
    param.requires_grad = False

# Add small trainable adapter: a learnable bias correction per layer
# This is lighter than LoRA — we add a single learnable scale+shift per linear output
calibration_params = []

class CalibrationWrapper(torch.nn.Module):
    """Wraps a frozen module and adds a learnable scale and bias correction."""
    def __init__(self, module, features):
        super().__init__()
        self.module = module
        self.scale = torch.nn.Parameter(torch.ones(features, dtype=torch.float16, device=DEVICE))
        self.shift = torch.nn.Parameter(torch.zeros(features, dtype=torch.float16, device=DEVICE))

    def forward(self, x):
        out = self.module(x)
        return out * self.scale + self.shift

# Wrap the MLP and attention output projections with calibration adapters
n_wrapped = 0
for name, module in model_ternary.named_modules():
    if hasattr(module, 'out_features') and hasattr(module, 'packed_weight'):
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model_ternary
        for p in parent_name.split("."):
            if p:
                parent = getattr(parent, p)
        wrapper = CalibrationWrapper(module, module.out_features)
        setattr(parent, child_name, wrapper)
        calibration_params.extend([wrapper.scale, wrapper.shift])
        n_wrapped += 1

print(f"  Added calibration wrappers to {n_wrapped} layers")
n_trainable = sum(p.numel() for p in calibration_params)
n_total = sum(p.numel() for p in model_ternary.parameters())
print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({n_trainable/n_total*100:.4f}%)")

# Optimizer
optimizer = torch.optim.AdamW(calibration_params, lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_CALIBRATION_STEPS)

# Training loop with knowledge distillation
model_fp16.eval()
model_ternary.train()

losses = []
ppl_history = [("step_0", pre_cal_ppl)]

t_start = time.time()
for step in range(N_CALIBRATION_STEPS):
    # Sample a random calibration sequence
    idx = np.random.randint(len(all_ids))
    input_ids = all_ids[idx].unsqueeze(0).to(DEVICE)

    # Get teacher (FP16) logits
    with torch.no_grad():
        teacher_out = model_fp16(input_ids)
        teacher_logits = teacher_out.logits[:, :-1, :].float()
        teacher_probs = torch.nn.functional.softmax(teacher_logits / 2.0, dim=-1)  # temperature=2

    # Get student (ternary) logits
    student_out = model_ternary(input_ids)
    student_logits = student_out.logits[:, :-1, :].float()
    student_log_probs = torch.nn.functional.log_softmax(student_logits / 2.0, dim=-1)

    # KL divergence loss (knowledge distillation)
    kd_loss = torch.nn.functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Also add standard cross-entropy loss on actual next tokens
    labels = input_ids[:, 1:].contiguous()
    ce_loss = torch.nn.functional.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
    )

    # Combined loss: KD + CE
    loss = kd_loss * 4.0 + ce_loss  # temperature^2 scaling for KD

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(calibration_params, 1.0)
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if (step + 1) % 50 == 0:
        avg_loss = np.mean(losses[-50:])
        elapsed = time.time() - t_start
        print(f"  Step {step+1}/{N_CALIBRATION_STEPS} | loss={avg_loss:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} | time={elapsed:.0f}s")

    if (step + 1) % EVAL_EVERY == 0:
        model_ternary.eval()
        mid_ppl = compute_perplexity(model_ternary, tokenizer)
        ppl_history.append((f"step_{step+1}", mid_ppl))
        print(f"  >>> Perplexity at step {step+1}: {mid_ppl:.2f}")
        model_ternary.train()

# ── Step 5: Final evaluation ──────────────────────────────────
print("\n=== Step 5: Post-calibration evaluation ===")
model_ternary.eval()
post_cal_ppl = compute_perplexity(model_ternary, tokenizer)
ppl_history.append((f"step_{N_CALIBRATION_STEPS}_final", post_cal_ppl))

# Test generation
from generate import generate_text
test_prompts = [
    "The theory of relativity explains that",
    "def fibonacci(n):\n    ",
]
print("\n--- Post-calibration generations ---")
for prompt in test_prompts:
    text = generate_text(model_ternary, tokenizer, prompt, max_new_tokens=60)
    print(f"  [{prompt[:40]}...] {text[:150]}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*70)
print("CALIBRATION RESULTS")
print("="*70)
print(f"  FP16 baseline perplexity:        {fp16_ppl:.2f}")
print(f"  Ternary PRE-calibration:         {pre_cal_ppl:.2f}")
print(f"  Ternary POST-calibration:        {post_cal_ppl:.2f}")
print(f"  Perplexity recovery:             {pre_cal_ppl:.2f} -> {post_cal_ppl:.2f}")
print(f"  Gap to FP16:                     +{post_cal_ppl - fp16_ppl:.2f}")
print(f"  Calibration steps:               {N_CALIBRATION_STEPS}")
print(f"  Trainable parameters:            {n_trainable:,} ({n_trainable/n_total*100:.4f}%)")
print(f"\n  Perplexity trajectory:")
for label, ppl in ppl_history:
    print(f"    {label}: {ppl:.2f}")

# Save results
results = {
    "model": MODEL_NAME,
    "gpu": torch.cuda.get_device_name(),
    "calibration_config": {
        "n_steps": N_CALIBRATION_STEPS,
        "lr": LEARNING_RATE,
        "quant_method": QUANT_METHOD,
        "batch_size": BATCH_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "n_calibration_sequences": len(all_ids),
        "n_trainable_params": n_trainable,
        "total_params": n_total,
    },
    "perplexity": {
        "fp16": fp16_ppl,
        "ternary_pre": pre_cal_ppl,
        "ternary_post": post_cal_ppl,
        "trajectory": ppl_history,
    },
    "loss_curve": losses,
}

with open("/content/qat_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved: /content/qat_results.json")

del model_fp16
torch.cuda.empty_cache()
