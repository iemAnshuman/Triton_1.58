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
quantize.py — Weight quantization and packing utilities.

Provides symmetric 4-bit quantization with per-group scale factors,
bit packing/unpacking for INT32 storage, and dequantization.
"""

import torch

__all__ = ["quantize_4bit", "dequantize_4bit", "pack_4bit", "unpack_4bit"]


def quantize_4bit(weight: torch.Tensor, group_size: int = 128):
    """
    Quantize an FP16 weight tensor to 4-bit symmetric representation.

    Args:
        weight: FP16 tensor of shape (N, K).
        group_size: Number of weights per quantization group (default 128).

    Returns:
        quantized: INT8 tensor of shape (N, K) with values in [0, 15].
        scales: FP16 tensor of shape (N, K // group_size).
    """
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    weight_groups = weight.reshape(N, K // group_size, group_size).float()
    scales = weight_groups.abs().amax(dim=-1).clamp(min=1e-6)
    normalized = weight_groups / scales.unsqueeze(-1)
    quantized = torch.round(normalized * 7.5 + 8.0).clamp(0, 15).to(torch.int8)

    return quantized.reshape(N, K), scales.to(torch.float16)


def dequantize_4bit(quantized: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    """
    Reconstruct approximate FP16 weights from quantized representation.

    Args:
        quantized: INT8 tensor of shape (N, K) with values in [0, 15].
        scales: FP16 tensor of shape (N, K // group_size).
        group_size: Number of weights per quantization group.

    Returns:
        Reconstructed FP16 tensor of shape (N, K).
    """
    N, K = quantized.shape
    qg = quantized.reshape(N, K // group_size, group_size).to(torch.float16)
    return ((qg - 8.0) / 7.5 * scales.unsqueeze(-1)).reshape(N, K)


def pack_4bit(quantized: torch.Tensor) -> torch.Tensor:
    """
    Pack 8 four-bit values into a single INT32.

    Args:
        quantized: Tensor with values in [0, 15], last dim divisible by 8.

    Returns:
        Packed INT32 tensor with last dim reduced by 8x.
    """
    *batch, K = quantized.shape
    assert K % 8 == 0, f"Last dimension ({K}) must be divisible by 8"

    reshaped = quantized.reshape(*batch, K // 8, 8).to(torch.int32)
    packed = reshaped[..., 0]
    for i in range(1, 8):
        packed = packed | (reshaped[..., i] << (i * 4))
    return packed


def unpack_4bit(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack INT32 values into 8 four-bit values each.

    Args:
        packed: INT32 tensor.

    Returns:
        INT8 tensor with last dim expanded by 8x, values in [0, 15].
    """
    *batch, K8 = packed.shape
    return (
        torch.stack([(packed >> (i * 4)) & 0xF for i in range(8)], dim=-1)
        .to(torch.int8)
        .reshape(*batch, K8 * 8)
    )


def test_quantization(device: str = "cuda", group_size: int = 128):
    """Run a self-test on quantization and packing correctness."""
    W = torch.randn(256, 1024, dtype=torch.float16, device=device)
    q, s = quantize_4bit(W, group_size)
    packed = pack_4bit(q)
    unpacked = unpack_4bit(packed)

    assert torch.equal(q, unpacked), "Pack/unpack mismatch!"

    W_recon = dequantize_4bit(q, s, group_size)
    err = (W.float() - W_recon.float()).abs().mean().item()
    orig_bytes = W.nelement() * 2
    pack_bytes = packed.nelement() * 4 + s.nelement() * 2

    print(
        f"Quantization test: PASSED | Error: {err:.6f} | "
        f"Compression: {orig_bytes/1024:.0f}KB -> {pack_bytes/1024:.0f}KB "
        f"({pack_bytes/orig_bytes*100:.0f}%)"
    )
    return True


if __name__ == "__main__":
    test_quantization()