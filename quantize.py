import torch
import torch.nn as nn
import triton
import triton.language as tl
import time
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Triton: {triton.__version__} | PyTorch: {torch.__version__}")

DEVICE = "cuda"
DTYPE = torch.float16
GROUP_SIZE = 128

def quantize_4bit(weight, group_size=128):
    N, K = weight.shape
    assert K % group_size == 0
    weight_groups = weight.reshape(N, K // group_size, group_size).float()
    scales = weight_groups.abs().amax(dim=-1).clamp(min=1e-6)
    normalized = weight_groups / scales.unsqueeze(-1)
    quantized = torch.round(normalized * 7.5 + 8.0).clamp(0, 15).to(torch.int8)
    return quantized.reshape(N, K), scales.to(torch.float16)

def dequantize_4bit(quantized, scales, group_size=128):
    N, K = quantized.shape
    qg = quantized.reshape(N, K // group_size, group_size).to(torch.float16)
    return ((qg - 8.0) / 7.5 * scales.unsqueeze(-1)).reshape(N, K)

def pack_4bit(quantized):
    *batch, K = quantized.shape
    reshaped = quantized.reshape(*batch, K // 8, 8).to(torch.int32)
    packed = reshaped[..., 0]
    for i in range(1, 8):
        packed = packed | (reshaped[..., i] << (i * 4))
    return packed

def unpack_4bit(packed):
    *batch, K8 = packed.shape
    return torch.stack(
        [(packed >> (i * 4)) & 0xF for i in range(8)], dim=-1
    ).to(torch.int8).reshape(*batch, K8 * 8)

W_test = torch.randn(256, 1024, dtype=torch.float16, device=DEVICE)
q, s = quantize_4bit(W_test, GROUP_SIZE)
packed = pack_4bit(q)
unpacked = unpack_4bit(packed)
assert torch.equal(q, unpacked), "Pack/unpack mismatch!"
W_recon = dequantize_4bit(q, s, GROUP_SIZE)
err = (W_test.float() - W_recon.float()).abs().mean().item()
orig_bytes = W_test.nelement() * 2
pack_bytes = packed.nelement() * 4 + s.nelement() * 2
print(f"Quantization test: PASSED | Error: {err:.6f} | "
      f"Compression: {orig_bytes/1024:.0f}KB -> {pack_bytes/1024:.0f}KB "
      f"({pack_bytes/orig_bytes*100:.0f}%)")
del W_test, q, s, packed, unpacked, W_recon

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_4bit_kernel(
    x_ptr, w_packed_ptr, scales_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_om, stride_on,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_pack in range(0, (K + 7) // 8):
        base_k = k_pack * 8

        w_ptrs = w_packed_ptr + k_pack * stride_wk + offs_n * stride_wn
        packed = tl.load(w_ptrs, mask=mask_n, other=0)

        group_idx = base_k // GROUP_SIZE
        s_ptrs = scales_ptr + group_idx * stride_sk + offs_n * stride_sn
        scale = tl.load(s_ptrs, mask=mask_n, other=1.0).to(tl.float32)

        for bit in tl.static_range(8):
            k_idx = base_k + bit

            val = ((packed >> (bit * 4)) & 0xF).to(tl.float32)
            deq = ((val - 8.0) / 7.5) * scale

            x_ptrs = x_ptr + offs_m * stride_xm + k_idx * stride_xk
            x_col = tl.load(x_ptrs, mask=(mask_m & (k_idx < K)), other=0.0).to(tl.float32)

            acc += x_col[:, None] * deq[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float16), mask=(mask_m[:, None] & mask_n[None, :]))

def matmul_4bit(x, w_packed, scales, group_size=128):
    M, K = x.shape
    K_packed, N = w_packed.shape
    assert K_packed == K // 8
    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_4bit_kernel[grid](
        x, w_packed, scales, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        output.stride(0), output.stride(1),
        GROUP_SIZE=group_size,
    )
    return output

print("Testing Triton kernel...")
for M, K, N in [(1, 2048, 2048), (1, 2048, 8192), (32, 2048, 2048), (128, 2048, 8192)]:
    W = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    q, s = quantize_4bit(W, GROUP_SIZE)
    packed = pack_4bit(q)
    w_T = packed.T.contiguous()
    s_T = s.T.contiguous()
    X = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    ref = X @ dequantize_4bit(q, s, GROUP_SIZE).T
    out = matmul_4bit(X, w_T, s_T, GROUP_SIZE)
    diff = (ref.float() - out.float()).abs().max().item()
    status = "OK" if diff < 1.0 else "FAIL"
    print(f"  [{status}] M={M:>4} K={K:>5} N={N:>5} | max_diff={diff:.4f}")
    del W, q, s, packed, w_T, s_T, X, ref, out
torch.cuda.empty_cache()
print("Kernel tests done!\n")

def benchmark_kernel(M, K, N, group_size=128, warmup=5, runs=20):
    W = torch.randn(N, K, dtype=torch.float16, device=DEVICE)
    q, s = quantize_4bit(W, group_size)
    packed = pack_4bit(q)
    w_T = packed.T.contiguous()
    s_T = s.T.contiguous()
    W_deq_T = dequantize_4bit(q, s, group_size).T.contiguous()
    X = torch.randn(M, K, dtype=torch.float16, device=DEVICE)
    
    for _ in range(warmup):
        _ = X @ W_deq_T
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = X @ W_deq_T
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / runs * 1000

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
        'M': M, 'K': K, 'N': N,
        'torch_ms': round(t_torch, 3), 'ours_ms': round(t_ours, 3),
        'speedup': round(t_torch / t_ours, 2),
        'mem_fp16_MB': round(mem_fp16 / 1e6, 2),
        'mem_4bit_MB': round(mem_4bit / 1e6, 2),
        'compression': round(mem_fp16 / mem_4bit, 2),
    }

test_sizes = [
    (1, 2048, 2048), (1, 2048, 8192), (1, 8192, 2048),
    (32, 2048, 2048), (32, 2048, 8192), (128, 2048, 2048),
]
print("=" * 90)
print("KERNEL BENCHMARK: matmul_4bit vs torch.matmul")
print("=" * 90)
benchmark_results = []
for M, K, N in test_sizes:
    r = benchmark_kernel(M, K, N)
    benchmark_results.append(r)
    tag = ">>>" if r['speedup'] >= 1.0 else "   "
    print(f"{tag} M={M:>4} K={K:>5} N={N:>5} | "
          f"torch={r['torch_ms']:>7.3f}ms  ours={r['ours_ms']:>7.3f}ms  "
          f"speedup={r['speedup']:>5.2f}x | "
          f"{r['mem_fp16_MB']:.1f}MB -> {r['mem_4bit_MB']:.1f}MB ({r['compression']:.1f}x)")
print()

class Linear4Bit(nn.Module):
    def __init__(self, in_features, out_features, packed_weight, scales, bias=None, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.register_buffer('packed_weight', packed_weight)
        self.register_buffer('scales', scales)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float16)
        out = matmul_4bit(x_2d, self.packed_weight, self.scales, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

def quantize_linear_layer(linear, group_size=128):
    weight = linear.weight.data.to(torch.float16)
    bias = linear.bias.data.to(torch.float16) if linear.bias is not None else None
    N, K = weight.shape
    q, s = quantize_4bit(weight, group_size)
    packed = pack_4bit(q)
    return Linear4Bit(K, N, packed.T.contiguous(), s.T.contiguous(), bias, group_size)

def quantize_model(model, group_size=128, skip_layers=None):
    if skip_layers is None:
        skip_layers = ['embed', 'lm_head', 'rotary']
    
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(s in name for s in skip_layers):
                print(f"  SKIP: {name}")
                continue
            replacements.append(name)
            
    for name in replacements:
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        old = getattr(parent, parts[-1])
        setattr(parent, parts[-1], quantize_linear_layer(old, group_size))
        del old
        torch.cuda.empty_cache()
        print(f"  DONE: {name}")

    print(f"Quantized {len(replacements)} layers.")
    return model

print("Linear4Bit module ready.\n")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
torch.cuda.reset_peak_memory_stats()
model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="cuda"
)
model_fp16.eval()
fp16_load_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"FP16 loaded. VRAM: {fp16_load_mem:.2f} GB\n")

test_prompts = [
    "The theory of relativity explains that",
    "In machine learning, gradient descent works by",
    "def fibonacci(n):\n    ",
    "The three main benefits of quantization for LLMs are",
]

print("--- FP16 Outputs ---")
fp16_outputs = []
for prompt in test_prompts:
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model_fp16.generate(**inp, max_new_tokens=60, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    fp16_outputs.append(text)
    print(f"[FP16] {text[:180]}\n")

print("Quantizing to 4-bit...")
torch.cuda.reset_peak_memory_stats()
model_4bit = quantize_model(model_fp16, GROUP_SIZE)
model_4bit.eval()
q4_load_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"4-bit ready. VRAM: {q4_load_mem:.2f} GB\n")

print("--- 4-Bit Outputs ---")
q4_outputs = []
for prompt in test_prompts:
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model_4bit.generate(**inp, max_new_tokens=60, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    q4_outputs.append(text)
    print(f"[4BIT] {text[:180]}\n")

print("=" * 60)
print("SIDE BY SIDE")
print("=" * 60)
generation_comparison = []
for i, p in enumerate(test_prompts):
    print(f"\nPrompt: {p}")
    print(f"FP16:  {fp16_outputs[i][:150]}")
    print(f"4-Bit: {q4_outputs[i][:150]}")
    generation_comparison.append((p, fp16_outputs[i], q4_outputs[i]))
del model_fp16
torch.cuda.empty_cache()

def measure_latency(model, tokenizer, prompt="The meaning of life is", max_new_tokens=128, warmup=2, runs=5):
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
        ntok = out.shape[1] - inp['input_ids'].shape[1]
        times.append(ntok / (time.perf_counter() - t0))
    return np.mean(times), np.std(times)

def compute_perplexity(model, tokenizer, max_samples=50, max_length=512):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds['text'] if len(t.strip()) > 50][:max_samples]
    total_loss, total_tok = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(DEVICE)
        ids = enc['input_ids']
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            loss = model(ids, labels=ids).loss
        if loss is not None and not torch.isnan(loss):
            n = ids.shape[1] - 1
            total_loss += loss.item() * n
            total_tok += n
    return np.exp(total_loss / total_tok) if total_tok > 0 else float('inf')

print("Benchmarking our 4-bit model...")
torch.cuda.reset_peak_memory_stats()
dummy = tokenizer("hello", return_tensors="pt").to(DEVICE)
with torch.no_grad(): 
    model_4bit(**dummy)
ours_vram = torch.cuda.max_memory_allocated() / 1e9
ours_tok, ours_std = measure_latency(model_4bit, tokenizer)
print(f"  VRAM: {ours_vram:.2f} GB | Tokens/sec: {ours_tok:.1f} ± {ours_std:.1f}")
print("  Computing perplexity...")
ours_ppl = compute_perplexity(model_4bit, tokenizer)
print(f"  Perplexity: {ours_ppl:.2f}")
del model_4bit
torch.cuda.empty_cache()

print("\nBenchmarking FP16 baseline...")
torch.cuda.reset_peak_memory_stats()
model_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
model_fp16.eval()
with torch.no_grad(): 
    model_fp16(**dummy)
fp16_vram = torch.cuda.max_memory_allocated() / 1e9
fp16_tok, fp16_std = measure_latency(model_fp16, tokenizer)
print(f"  VRAM: {fp16_vram:.2f} GB | Tokens/sec: {fp16_tok:.1f} ± {fp16_std:.1f}")
print("  Computing perplexity...")
fp16_ppl = compute_perplexity(model_fp16, tokenizer)
print(f"  Perplexity: {fp16_ppl:.2f}")
del model_fp16
torch.cuda.empty_cache()

print("\nBenchmarking bitsandbytes NF4...")
try:
    from transformers import BitsAndBytesConfig
    torch.cuda.reset_peak_memory_stats()
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model_bnb = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_cfg, device_map="cuda")
    model_bnb.eval()
    with torch.no_grad(): 
        model_bnb(**dummy)
    bnb_vram = torch.cuda.max_memory_allocated() / 1e9
    bnb_tok, bnb_std = measure_latency(model_bnb, tokenizer)
    print(f"  VRAM: {bnb_vram:.2f} GB | Tokens/sec: {bnb_tok:.1f} ± {bnb_std:.1f}")
    print("  Computing perplexity...")
    bnb_ppl = compute_perplexity(model_bnb, tokenizer)
    print(f"  Perplexity: {bnb_ppl:.2f}")
    del model_bnb
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  bitsandbytes failed: {e}")
    bnb_vram = bnb_tok = bnb_std = bnb_ppl = None

print("\n" + "=" * 65)
print(f"{'Config':<20} {'VRAM (GB)':>10} {'Tok/sec':>10} {'Perplexity':>12}")
print("-" * 55)
print(f"{'FP16 Baseline':<20} {fp16_vram:>10.2f} {fp16_tok:>10.1f} {fp16_ppl:>12.2f}")
print(f"{'Ours (4-bit)':<20} {ours_vram:>10.2f} {ours_tok:>10.1f} {ours_ppl:>12.2f}")
if bnb_vram:
    print(f"{'bitsandbytes NF4':<20} {bnb_vram:>10.2f} {bnb_tok:>10.1f} {bnb_ppl:>12.2f}")
print("-" * 55)
print(f"VRAM saved: {(1-ours_vram/fp16_vram)*100:.1f}% | "
      f"Speed: {ours_tok/fp16_tok:.2f}x | "
      f"Perplexity: +{ours_ppl-fp16_ppl:.2f}")

full_benchmark = {
    'fp16': {'vram': fp16_vram, 'tok_sec': fp16_tok, 'ppl': fp16_ppl},
    'ours_4bit': {'vram': ours_vram, 'tok_sec': ours_tok, 'ppl': ours_ppl},
}
if bnb_vram:
    full_benchmark['bnb_nf4'] = {'vram': bnb_vram, 'tok_sec': bnb_tok, 'ppl': bnb_ppl}
    
plt.rcParams['figure.dpi'] = 150
COLORS = {'fp16': '#4285F4', 'ours_4bit': '#EA4335', 'bnb_nf4': '#FBBC05'}
configs = list(full_benchmark.keys())
labels = {'fp16': 'FP16\nBaseline', 'ours_4bit': 'Ours\n(4-bit Triton)', 'bnb_nf4': 'bitsandbytes\nNF4'}
colors = [COLORS[c] for c in configs]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, metric, title, ylabel in [
    (axes[0], 'vram', 'Memory Usage', 'VRAM (GB)'),
    (axes[1], 'tok_sec', 'Inference Speed', 'Tokens/sec'),
    (axes[2], 'ppl', 'Quality (Perplexity)', 'Perplexity ↓'),
]:
    vals = [full_benchmark[c][metric] for c in configs]
    bars = ax.bar([labels[c] for c in configs], vals, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02, f'{v:.1f}',
                ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim(0, max(vals)*1.3)
plt.suptitle(f'4-Bit Quantized Inference Benchmarks — {MODEL_NAME}\nGPU: {torch.cuda.get_device_name()}',
             fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('benchmark_charts.png', dpi=200, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, m_val, title in [(ax1, 1, 'M=1 (Decode)'), (ax2, 32, 'M=32 (Batch)')]:
    rs = [r for r in benchmark_results if r['M'] == m_val]
    if not rs: 
        continue
    xl = [f"{r['K']}×{r['N']}" for r in rs]
    x = range(len(xl))
    w = 0.35
    ax.bar([i-w/2 for i in x], [r['torch_ms'] for r in rs], w, label='torch (FP16)', color='#4285F4')
    ax.bar([i+w/2 for i in x], [r['ours_ms'] for r in rs], w, label='Ours (4-bit)', color='#EA4335')
    ax.set_xticks(list(x))
    ax.set_xticklabels(xl)
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Kernel Latency — {title}', fontweight='bold')
    ax.legend()
    for i, r in enumerate(rs):
        c = 'green' if r['speedup'] >= 1 else 'red'
        ax.text(i, max(r['torch_ms'], r['ours_ms'])*1.05, f"{r['speedup']}x", ha='center', fontsize=9, color=c, fontweight='bold')

plt.suptitle('Standalone Kernel: matmul_4bit vs torch.matmul', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('kernel_charts.png', dpi=200, bbox_inches='tight')
plt.show()

report_data = {
    'model': MODEL_NAME,
    'gpu': torch.cuda.get_device_name(),
    'group_size': GROUP_SIZE,
    'kernel_benchmarks': benchmark_results,
    'model_benchmarks': {k: v for k, v in full_benchmark.items()},
    'generation_comparisons': [
        {'prompt': p, 'fp16': f, 'quant4': q}
        for p, f, q in generation_comparison
    ],
}

with open('report_data.json', 'w') as f:
    json.dump(report_data, f, indent=2, default=str)

print("Saved: benchmark_charts.png, kernel_charts.png, report_data.json")
print("\nAll done! Download these files for your report and presentation.")