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
quantize_ternary.py — Ternary (1.58-bit) weight quantization and packing.

Implements absmean ternary quantization following the BitNet b1.58 paradigm
(Ma et al., 2024), but applied as a post-training transform on an existing
FP16 model. Weights are constrained to {-1, 0, +1} with per-group scale
factors. 16 ternary values are packed into a single INT32 using 2 bits each.

Encoding scheme (2 bits per value):
    0  -> 00  (binary 0)
    +1 -> 01  (binary 1)
    -1 -> 10  (binary 2)
    unused: 11
"""

import torch

__all__ = [
    "quantize_ternary",
    "dequantize_ternary",
    "pack_ternary",
    "unpack_ternary",
    "effective_bits_per_weight",
]


def _quantize_bitnet_absmean(weight_groups: torch.Tensor):
    gamma = weight_groups.abs().mean(dim=-1).clamp(min=1e-6)
    normalized = weight_groups / gamma.unsqueeze(-1)
    quantized = torch.round(normalized).clamp(-1, 1).to(torch.int8)
    return quantized, gamma


def _quantize_mse_lloyd(weight_groups: torch.Tensor, iterations: int = 4):
    """
    Ternary PTQ optimized for reconstruction error.

    For fixed q in {-1, 0, +1}, the least-squares scale is
    sum(|w| over nonzero q) / count(nonzero q). For fixed scale, the best
    ternary assignment is sign(w) where |w| > scale / 2, else zero.
    A few Lloyd-style iterations gives a better PTQ approximation than raw
    absmean rounding while preserving the same storage format.
    """
    scale = weight_groups.abs().mean(dim=-1).clamp(min=1e-6)
    for _ in range(iterations):
        threshold = scale.unsqueeze(-1) * 0.5
        keep = weight_groups.abs() > threshold
        count = keep.sum(dim=-1).clamp(min=1)
        scale = (
            torch.where(keep, weight_groups.abs(), torch.zeros_like(weight_groups))
            .sum(dim=-1)
            .div(count)
            .clamp(min=1e-6)
        )

    threshold = scale.unsqueeze(-1) * 0.5
    quantized = torch.where(
        weight_groups.abs() > threshold,
        torch.sign(weight_groups),
        torch.zeros_like(weight_groups),
    ).to(torch.int8)
    return quantized, scale


def quantize_ternary(
    weight: torch.Tensor,
    group_size: int = 128,
    method: str = "mse",
    mse_iterations: int = 4,
):
    """
    Quantize FP16 weights to ternary {-1, 0, +1}.

    Supported methods:
        "bitnet": absmean scaling from BitNet b1.58, applied as PTQ.
        "mse":    Lloyd-style per-group scale/threshold update minimizing
                  local reconstruction error. This is usually better for
                  PTQ checkpoints that were not trained with ternary weights.

    Args:
        weight: FP16 tensor of shape (N, K). K must be divisible by group_size.
        group_size: Number of weights per quantization group. Default 128.
        method: "mse" or "bitnet".
        mse_iterations: Number of scale/threshold updates for method="mse".

    Returns:
        quantized: INT8 tensor of shape (N, K) with values in {-1, 0, 1}.
        scales:    FP16 tensor of shape (N, K // group_size).
    """
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"

    weight_groups = weight.reshape(N, K // group_size, group_size).float()
    if method == "bitnet":
        quantized, scale = _quantize_bitnet_absmean(weight_groups)
    elif method == "mse":
        quantized, scale = _quantize_mse_lloyd(weight_groups, mse_iterations)
    else:
        raise ValueError(f"Unknown ternary quantization method: {method}")

    return quantized.reshape(N, K), scale.to(torch.float16)


def dequantize_ternary(quantized: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    """
    Reconstruct approximate FP16 weights from ternary representation.

    W_reconstructed = q * gamma_group

    Args:
        quantized: INT8 tensor of shape (N, K) with values in {-1, 0, 1}.
        scales: FP16 tensor of shape (N, K // group_size).
        group_size: Number of weights per quantization group.

    Returns:
        Reconstructed FP16 tensor of shape (N, K).
    """
    N, K = quantized.shape
    qg = quantized.reshape(N, K // group_size, group_size).to(torch.float16)
    return (qg * scales.unsqueeze(-1)).reshape(N, K)


def pack_ternary(quantized: torch.Tensor) -> torch.Tensor:
    """
    Pack 16 ternary values into a single INT32 using 2 bits per value.

    Encoding: -1 -> 2 (binary 10), 0 -> 0, +1 -> 1.

    Args:
        quantized: Tensor with values in {-1, 0, 1}. Last dim divisible by 16.

    Returns:
        INT32 tensor with last dim reduced by 16x.
    """
    *batch, K = quantized.shape
    assert K % 16 == 0, f"Last dimension ({K}) must be divisible by 16"

    # Map {-1, 0, 1} -> {2, 0, 1} (unsigned 2-bit codes)
    codes = torch.where(
        quantized < 0,
        torch.tensor(2, dtype=torch.int32, device=quantized.device),
        quantized.to(torch.int32),
    )

    reshaped = codes.reshape(*batch, K // 16, 16)
    packed = reshaped[..., 0]
    for i in range(1, 16):
        packed = packed | (reshaped[..., i] << (i * 2))
    return packed


def unpack_ternary(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack INT32 values into 16 ternary values each.

    Decodes 2-bit codes back to {-1, 0, 1}.

    Args:
        packed: INT32 tensor.

    Returns:
        INT8 tensor with last dim expanded by 16x, values in {-1, 0, 1}.
    """
    *batch, K16 = packed.shape
    extracted = torch.stack(
        [(packed >> (i * 2)) & 0x3 for i in range(16)], dim=-1
    )
    # Decode {0, 1, 2} -> {0, 1, -1}
    decoded = torch.where(
        extracted == 2,
        torch.tensor(-1, dtype=torch.int8, device=packed.device),
        extracted.to(torch.int8),
    )
    return decoded.reshape(*batch, K16 * 16)


def effective_bits_per_weight(quantized: torch.Tensor, scales: torch.Tensor, group_size: int = 128) -> float:
    """
    Compute the effective bits per weight including scale factor overhead.

    Returns:
        Effective bits per weight (theoretical ternary is 1.58).
    """
    n_weights = quantized.numel()
    weight_bits = n_weights * 2  # 2 bits per ternary weight in packed form
    scale_bits = scales.numel() * 16  # FP16 scales
    return (weight_bits + scale_bits) / n_weights


def test_ternary_quantization(device: str = "cuda", group_size: int = 128):
    """Run self-test on ternary quantization and packing correctness."""
    W = torch.randn(256, 1024, dtype=torch.float16, device=device)

    q, g = quantize_ternary(W, group_size, method="mse")
    assert q.min() >= -1 and q.max() <= 1, f"Ternary values out of range: [{q.min()}, {q.max()}]"

    unique_vals = torch.unique(q).tolist()
    assert set(unique_vals).issubset({-1, 0, 1}), f"Non-ternary values found: {unique_vals}"

    packed = pack_ternary(q)
    unpacked = unpack_ternary(packed)
    assert torch.equal(q, unpacked), "Pack/unpack mismatch!"

    W_recon = dequantize_ternary(q, g, group_size)
    err = (W.float() - W_recon.float()).abs().mean().item()

    orig_bytes = W.nelement() * 2
    pack_bytes = packed.nelement() * 4 + g.nelement() * 2
    eff_bits = effective_bits_per_weight(q, g, group_size)

    # Sparsity: fraction of zeros (theoretically ~1/3 for Gaussian weights)
    sparsity = (q == 0).float().mean().item()
    q_absmean, g_absmean = quantize_ternary(W, group_size, method="bitnet")
    W_absmean = dequantize_ternary(q_absmean, g_absmean, group_size)
    absmean_err = (W.float() - W_absmean.float()).abs().mean().item()

    print(
        f"Ternary test: PASSED | unique_vals={unique_vals} | error={err:.6f}\n"
        f"  Compression: {orig_bytes/1024:.0f}KB -> {pack_bytes/1024:.0f}KB "
        f"({orig_bytes/pack_bytes:.2f}x)\n"
        f"  Effective bits/weight: {eff_bits:.3f} (theoretical minimum 1.58)\n"
        f"  Sparsity (fraction of zeros): {sparsity:.2%}\n"
        f"  MSE method MAE improvement vs absmean: {absmean_err:.6f} -> {err:.6f}"
    )
    return True


if __name__ == "__main__":
    test_ternary_quantization()
