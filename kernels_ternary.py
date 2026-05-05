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
kernels_ternary.py — Custom Triton GPU kernel for ternary (1.58-bit) matmul.

The kernel loads packed INT32 weights from VRAM (16 ternary values per INT32),
unpacks them in on-chip SRAM using bitwise operations, applies per-group scale
factors, and computes the matrix product via tl.dot() on Tensor Core hardware.

Note: on current NVIDIA GPUs, ternary matmul is not natively supported in
hardware, so we dequantize to FP16 {-1.0, 0.0, +1.0} and dispatch to the
Tensor Cores. The 8x memory bandwidth reduction (2 bits vs 16 bits per weight)
is where the practical speedup comes from in a memory-bound decode workload.
"""

import torch
import triton
import triton.language as tl

__all__ = ["matmul_ternary"]


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 256},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 256},
            num_warps=4, num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256},
            num_warps=4, num_stages=3,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_ternary_kernel(
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
    Fused ternary dequantization and matrix multiplication kernel.

    Computes: output = x @ dequantize(w_packed, scales).T

    Pipeline per tile:
      1. Load packed INT32 weight tile (BLOCK_K//16, BLOCK_N) from VRAM
      2. For each of 16 sub-slices:
           a. Extract 2-bit code via (packed >> (bit*2)) & 0x3
           b. Decode code -> {-1.0, 0.0, +1.0} via tl.where
           c. Apply per-group scale factor
           d. Load corresponding activation slice
           e. Accumulate via tl.dot() with Tensor Core hardware
      3. Cast FP32 accumulator to FP16 and store
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # W_packed has K/16 rows (16 ternary values packed per INT32 along K axis)
    offs_k_packed = tl.arange(0, BLOCK_K // 16)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        base_k = k_idx * BLOCK_K
        base_k_packed = base_k // 16

        # --- Load packed weights tile: (BLOCK_K//16, BLOCK_N) ---
        w_k_inds = base_k_packed + offs_k_packed
        w_ptrs = w_packed_ptr + (
            w_k_inds[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )
        w_mask = (w_k_inds[:, None] < (K // 16)) & mask_n[None, :]
        w_packed = tl.load(w_ptrs, mask=w_mask, other=0)

        # --- Load per-group scale factors.
        # BLOCK_K can span multiple quantization groups. Since GROUP_SIZE is a
        # multiple of 16, each packed INT32 row belongs to exactly one group.
        group_idx = (base_k + offs_k_packed * 16) // GROUP_SIZE
        s_ptrs = scales_ptr + (
            group_idx[:, None] * stride_sk + offs_n[None, :] * stride_sn
        )
        scale_mask = (group_idx[:, None] < tl.cdiv(K, GROUP_SIZE)) & mask_n[None, :]
        scale = tl.load(s_ptrs, mask=scale_mask, other=1.0)

        # --- Unpack 16 ternary values per INT32, dequantize, and multiply ---
        for bit in tl.static_range(16):
            # Extract 2-bit code in {0, 1, 2}
            w_code = (w_packed >> (bit * 2)) & 0x3

            # Decode to ternary FP32: 0 -> 0.0, 1 -> 1.0, 2 -> -1.0
            w_tern = tl.where(
                w_code == 2,
                tl.full(w_code.shape, -1.0, tl.float32),
                w_code.to(tl.float32),
            )

            # Apply per-group scale
            w_deq = w_tern * scale

            # Load corresponding activation slice
            k_inds = base_k + offs_k_packed * 16 + bit
            x_ptrs = x_ptr + (
                offs_m[:, None] * stride_xm + k_inds[None, :] * stride_xk
            )
            x_mask = mask_m[:, None] & (k_inds[None, :] < K)
            x_slice = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Tensor Core matmul (FP16 inputs, FP32 accumulator)
            acc += tl.dot(x_slice, w_deq.to(tl.float16), out_dtype=tl.float32)

    # Store output
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(tl.float16), mask=(mask_m[:, None] & mask_n[None, :]))


def matmul_ternary(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Perform ternary quantized matrix multiplication.

    Computes: output = x @ dequantize(w_packed, scales)

    Args:
        x: FP16 activation tensor of shape (M, K).
        w_packed: Packed INT32 weight tensor of shape (K // 16, N).
        scales: FP16 per-group gamma factors of shape (K // group_size, N).
        group_size: Quantization group size (default 128).

    Returns:
        FP16 output tensor of shape (M, N).
    """
    M, K = x.shape
    K_packed, N = w_packed.shape
    assert K_packed == K // 16, (
        f"Shape mismatch: K={K} expects packed K dim {K//16}, got {K_packed}"
    )
    assert group_size % 16 == 0, "group_size must be divisible by 16"

    output = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    matmul_ternary_kernel[grid](
        x, w_packed, scales, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        output.stride(0), output.stride(1),
        GROUP_SIZE=group_size,
    )
    return output


def test_ternary_kernel(device: str = "cuda", group_size: int = 128):
    """Run correctness tests against PyTorch reference."""
    from quantize_ternary import quantize_ternary, dequantize_ternary, pack_ternary

    print("Testing Triton ternary kernel...")
    for M, K, N in [(1, 2048, 2048), (1, 2048, 8192), (32, 2048, 2048), (128, 2048, 8192)]:
        W = torch.randn(N, K, dtype=torch.float16, device=device)
        # Force neighboring groups to have very different scales. This catches
        # kernels that accidentally reuse one scale over a larger BLOCK_K tile.
        W[:, group_size:group_size * 2] *= 8
        q, g = quantize_ternary(W, group_size, method="mse")
        packed = pack_ternary(q)
        w_T = packed.T.contiguous()
        g_T = g.T.contiguous()
        X = torch.randn(M, K, dtype=torch.float16, device=device)

        ref = X @ dequantize_ternary(q, g, group_size).T
        out = matmul_ternary(X, w_T, g_T, group_size)
        diff = (ref.float() - out.float()).abs().max().item()
        status = "OK" if diff < 1.0 else "FAIL"
        print(f"  [{status}] M={M:>4} K={K:>5} N={N:>5} | max_diff={diff:.4f}")

        del W, q, g, packed, w_T, g_T, X, ref, out

    torch.cuda.empty_cache()
    print("Ternary kernel tests done!\n")


if __name__ == "__main__":
    test_ternary_kernel()
