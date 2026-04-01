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
model.py — PyTorch model integration for 4-bit quantized inference.

Provides Linear4Bit (drop-in replacement for nn.Linear) and model-level
quantization utilities for converting HuggingFace Transformer models.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from quantize import quantize_4bit, pack_4bit
from kernels import matmul_4bit

__all__ = ["Linear4Bit", "quantize_linear_layer", "quantize_model"]


class Linear4Bit(nn.Module):
    """
    Drop-in replacement for nn.Linear that stores weights in 4-bit packed format
    and uses a custom Triton kernel for forward computation.

    Attributes:
        packed_weight: INT32 buffer of shape (K // 8, N) holding packed 4-bit weights.
        scales: FP16 buffer of shape (K // group_size, N) holding per-group scale factors.
        bias: Optional FP16 bias vector of shape (N,).
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
        out = matmul_4bit(x_2d, self.packed_weight, self.scales, self.group_size)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, bias={self.bias is not None}"
        )


def quantize_linear_layer(linear: nn.Linear, group_size: int = 128) -> Linear4Bit:
    """
    Convert a single nn.Linear layer to Linear4Bit.

    Args:
        linear: Standard PyTorch linear layer.
        group_size: Quantization group size.

    Returns:
        Linear4Bit module with packed weights and scale factors.
    """
    weight = linear.weight.data.to(torch.float16)
    bias = linear.bias.data.to(torch.float16) if linear.bias is not None else None
    N, K = weight.shape

    q, s = quantize_4bit(weight, group_size)
    packed = pack_4bit(q)

    return Linear4Bit(
        K, N,
        packed.T.contiguous(),
        s.T.contiguous(),
        bias,
        group_size,
    )


def quantize_model(
    model: nn.Module,
    group_size: int = 128,
    skip_layers: Optional[List[str]] = None,
) -> nn.Module:
    """
    Quantize all eligible nn.Linear layers in a model to 4-bit.

    Traverses the model's module tree, replaces nn.Linear layers with
    Linear4Bit instances, and frees original FP16 weights immediately
    to conserve VRAM during the conversion process.

    Args:
        model: HuggingFace or PyTorch model to quantize.
        group_size: Quantization group size (default 128).
        skip_layers: List of substrings; layers whose names contain any
                     of these are skipped (default: embed, lm_head, rotary).

    Returns:
        The quantized model (modified in-place).
    """
    if skip_layers is None:
        skip_layers = ["embed", "lm_head", "rotary"]

    # Collect layer names first to avoid modifying dict during iteration
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(s in name for s in skip_layers):
                print(f"  SKIP: {name}")
                continue
            replacements.append(name)

    # Replace layers one at a time, freeing memory after each
    for name in replacements:
        parts = name.split(".")
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