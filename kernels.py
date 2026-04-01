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
kernels.py — Custom Triton GPU kernels for quantized matrix multiplication.

Implements a fused dequantization + matrix multiplication kernel that loads
packed 4-bit weights from VRAM, unpacks them in on-chip SRAM, dequantizes
using per-group scale factors, and computes via Tensor Core tl.dot().
"""

import torch
import triton
import triton.language as tl

__all__ = ["matmul_4bit"]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            num_warps=4, num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
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
    BLOCK_K: tl.constexpr,
):
    """
    Fused 4-bit dequantization and matrix multiplication kernel.

    Computes: output = x @ dequantize(w_packed, scales).T

    The kernel processes output tiles of shape (BLOCK_M, BLOCK_N), iterating
    over the K dimension in chunks of BLOCK_K. For each chunk, it:
      1. Loads packed INT32 weights from VRAM into SRAM
      2. Extracts 4-bit values via bitwise shift and mask (8 sub-slices)
      3. Dequantizes using per-group scale factors
      4. Computes partial matmul via tl.dot() (Tensor Core acceleration)
      5. Accumulates in FP32, stores final result as FP16
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k_packed = tl.arange(0, BLOCK_K // 8)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        base_k = k_idx * BLOCK_K
        base_k_packed = base_k // 8

        # --- Load packed weights tile: (BLOCK_K // 8, BLOCK_N) ---
        w_k_inds = base_k_packed + offs_k_packed
        w_ptrs = w_packed_ptr + (
            w_k_inds[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (w_k_inds[:, None] < (K // 8)) & mask_n[None, :]
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0)

        # --- Load per-group scale factors ---
        group_idx = base_k // GROUP_SIZE
        s_ptrs = scales_ptr + (group_idx * stride_sk + offs_n * stride_sn)
        scale = tl.load(s_ptrs, mask=mask_n, other=1.0)

        # --- Unpack, dequantize, and multiply (8 sub-slices per packed INT32) ---
        for bit in tl.static_range(8):
            w_slice = (w_packed >> (bit * 4)) & 0xF
            w_deq = ((w_slice.to(tl.float32) - 8.0) / 7.5) * scale[None, :]

            k_inds = base_k + offs_k_packed * 8 + bit
            x_ptrs = x_ptr + (
                offs_m[:, None] * stride_xm + k_inds[None, :] * stride_xk
            )
            x_mask = mask_m[:, None] & (k_inds[None, :] < K)
            x_slice = tl.load(x_ptrs, mask=x_mask, other=0.0)

            acc += tl.dot(x_slice, w_deq.to(tl.float16), out_dtype=tl.float32)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(tl.float16), mask=(mask_m[:, None] & mask_n[None, :]))


def matmul_4bit(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Perform quantized matrix multiplication: output = x @ dequant(w_packed, scales).

    Args:
        x: FP16 activation tensor of shape (M, K).
        w_packed: Packed INT32 weight tensor of shape (K // 8, N).
        scales: FP16 scale factors of shape (K // group_size, N).
        group_size: Quantization group size (default 128).

    Returns:
        FP16 output tensor of shape (M, N).
    """
    M, K = x.shape
    K_packed, N = w_packed.shape
    assert K_packed == K // 8, f"Shape mismatch: K={K}, packed K dim={K_packed}"

    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

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


def test_kernel(device: str = "cuda", group_size: int = 128):
    """Run correctness tests on the Triton kernel against PyTorch reference."""
    from quantize import quantize_4bit, dequantize_4bit, pack_4bit

    print("Testing Triton kernel...")
    for M, K, N in [(1, 2048, 2048), (1, 2048, 8192), (32, 2048, 2048), (128, 2048, 8192)]:
        W = torch.randn(N, K, dtype=torch.float16, device=device)
        q, s = quantize_4bit(W, group_size)
        packed = pack_4bit(q)
        w_T = packed.T.contiguous()
        s_T = s.T.contiguous()
        X = torch.randn(M, K, dtype=torch.float16, device=device)

        ref = X @ dequantize_4bit(q, s, group_size).T
        out = matmul_4bit(X, w_T, s_T, group_size)
        diff = (ref.float() - out.float()).abs().max().item()
        status = "OK" if diff < 1.0 else "FAIL"
        print(f"  [{status}] M={M:>4} K={K:>5} N={N:>5} | max_diff={diff:.4f}")

        del W, q, s, packed, w_T, s_T, X, ref, out

    torch.cuda.empty_cache()
    print("Kernel tests done!\n")


if __name__ == "__main__":
    test_kernel()