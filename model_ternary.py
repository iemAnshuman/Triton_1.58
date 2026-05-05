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
model_ternary.py — PyTorch model integration for ternary (1.58-bit) inference.

Provides LinearTernary (drop-in replacement for nn.Linear) and model-level
quantization utilities for converting HuggingFace Transformer models to
ternary precision via post-training quantization.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from quantize_ternary import quantize_ternary, pack_ternary
from kernels_ternary import matmul_ternary

__all__ = ["LinearTernary", "quantize_linear_layer_ternary", "quantize_model_ternary"]


class LinearTernary(nn.Module):
    """
    Drop-in replacement for nn.Linear with ternary-packed weights.

    Storage layout:
        packed_weight : INT32 buffer of shape (K // 16, N)
        scales        : FP16 buffer of shape (K // group_size, N) holding gamma
        bias          : Optional FP16 bias of shape (N,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).to(torch.float16)
        out = matmul_ternary(x_2d, self.packed_weight, self.scales, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None} "
            f"[ternary 1.58-bit]"
        )


def quantize_linear_layer_ternary(
    linear: nn.Linear,
    group_size: int = 128,
    quant_method: str = "mse",
) -> LinearTernary:
    """
    Convert a single nn.Linear layer to LinearTernary.

    Args:
        linear: Standard PyTorch linear layer.
        group_size: Quantization group size.
        quant_method: "mse" for PTQ reconstruction error or "bitnet" for
            absmean BitNet-style quantization.

    Returns:
        LinearTernary module.
    """
    weight = linear.weight.data.to(torch.float16)
    bias = linear.bias.data.to(torch.float16) if linear.bias is not None else None
    N, K = weight.shape

    q, g = quantize_ternary(weight, group_size, method=quant_method)
    packed = pack_ternary(q)

    return LinearTernary(
        K, N,
        packed.T.contiguous(),
        g.T.contiguous(),
        bias,
        group_size,
    )


def quantize_model_ternary(
    model: nn.Module,
    group_size: int = 128,
    skip_layers: Optional[List[str]] = None,
    quant_method: str = "mse",
) -> nn.Module:
    """
    Quantize all eligible nn.Linear layers in a model to ternary precision.

    Following the BitNet convention, embedding layers, the LM head, and
    normalization-adjacent layers are skipped (they are kept in FP16).

    Args:
        model: HuggingFace or PyTorch model to quantize in-place.
        group_size: Quantization group size (default 128).
        skip_layers: Substrings; matching layer names are skipped.
                     Default skips embeddings, lm_head, and rotary.
        quant_method: "mse" for PTQ reconstruction error or "bitnet" for
            absmean BitNet-style quantization.

    Returns:
        The quantized model (modified in-place).
    """
    if skip_layers is None:
        skip_layers = ["embed", "lm_head", "rotary"]

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(s in name for s in skip_layers):
                print(f"  SKIP: {name}")
                continue
            replacements.append(name)

    for name in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        old = getattr(parent, parts[-1])
        setattr(
            parent,
            parts[-1],
            quantize_linear_layer_ternary(old, group_size, quant_method),
        )
        del old
        torch.cuda.empty_cache()
        print(f"  DONE: {name}")

    print(f"Quantized {len(replacements)} layers to ternary (method={quant_method}).")
    return model
