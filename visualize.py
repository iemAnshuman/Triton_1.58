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
visualize.py — Matplotlib chart generation for benchmark reports.

Generates model-level comparison charts and kernel latency charts.
"""

import matplotlib.pyplot as plt
from typing import Dict, List

__all__ = ["plot_model_benchmarks", "plot_kernel_benchmarks"]

COLORS = {
    "fp16": "#4285F4",
    "ours_4bit": "#EA4335",
    "bnb_nf4": "#FBBC05",
}

LABELS = {
    "fp16": "FP16\nBaseline",
    "ours_4bit": "Ours\n(4-bit Triton)",
    "bnb_nf4": "bitsandbytes\nNF4",
}


def plot_model_benchmarks(
    benchmarks: Dict[str, Dict],
    model_name: str = "",
    gpu_name: str = "",
    save_path: str = "benchmark_charts.png",
):
    """
    Generate a 3-panel bar chart comparing VRAM, speed, and perplexity.

    Args:
        benchmarks: Dict mapping config name -> {vram, tok_sec, ppl}.
        model_name: Model identifier for the title.
        gpu_name: GPU name for the title.
        save_path: Output file path.
    """
    plt.rcParams["figure.dpi"] = 150
    configs = list(benchmarks.keys())
    colors = [COLORS.get(c, "#999999") for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title, ylabel in [
        (axes[0], "vram", "Memory Usage", "VRAM (GB)"),
        (axes[1], "tok_sec", "Inference Speed", "Tokens/sec"),
        (axes[2], "ppl", "Quality (Perplexity)", "Perplexity"),
    ]:
        vals = [benchmarks[c][metric] for c in configs]
        bars = ax.bar(
            [LABELS.get(c, c) for c in configs],
            vals,
            color=colors,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{v:.1f}",
                ha="center",
                fontweight="bold",
                fontsize=10,
            )
        ax.set_ylim(0, max(vals) * 1.3)

    suptitle = "4-Bit Quantized Inference Benchmarks"
    if model_name:
        suptitle += f" — {model_name}"
    if gpu_name:
        suptitle += f"\nGPU: {gpu_name}"
    plt.suptitle(suptitle, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_kernel_benchmarks(
    results: List[Dict],
    save_path: str = "kernel_charts.png",
):
    """
    Generate kernel latency comparison charts (M=1 decode vs M=32 batch).

    Args:
        results: List of dicts from benchmark_kernel().
        save_path: Output file path.
    """
    plt.rcParams["figure.dpi"] = 150
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, m_val, title in [(ax1, 1, "M=1 (Decode)"), (ax2, 32, "M=32 (Batch)")]:
        rs = [r for r in results if r["M"] == m_val]
        if not rs:
            continue
        xl = [f"{r['K']}x{r['N']}" for r in rs]
        x = range(len(xl))
        w = 0.35
        ax.bar(
            [i - w / 2 for i in x],
            [r["torch_ms"] for r in rs],
            w,
            label="torch (FP16)",
            color="#4285F4",
        )
        ax.bar(
            [i + w / 2 for i in x],
            [r["ours_ms"] for r in rs],
            w,
            label="Ours (4-bit)",
            color="#EA4335",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(xl)
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"Kernel Latency — {title}", fontweight="bold")
        ax.legend()
        for i, r in enumerate(rs):
            c = "green" if r["speedup"] >= 1 else "red"
            ax.text(
                i,
                max(r["torch_ms"], r["ours_ms"]) * 1.05,
                f"{r['speedup']}x",
                ha="center",
                fontsize=9,
                color=c,
                fontweight="bold",
            )

    plt.suptitle(
        "Standalone Kernel: matmul_4bit vs torch.matmul",
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")