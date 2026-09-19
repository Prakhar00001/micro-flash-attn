import math
from typing import Optional
import torch
import triton
import triton.language as tl

@triton.jit
def _flash_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # Program block indices
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    # Batch and Head offsets
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    # Block range indices
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize memory pointers
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
    v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

    # Online softmax accumulators kept in registers / SRAM
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load Tile Q_i: [BLOCK_M, BLOCK_D]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Causal loop upper bound
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    # Outer loop across K, V sequence blocks
    for start_n in range(0, hi, BLOCK_N):
        curr_offs_n = start_n + offs_n

        # Load K_j and V_j tiles with boundary masking
        k_mask = curr_offs_n[None, :] < N_CTX
        v_mask = curr_offs_n[:, None] < N_CTX

        k = tl.load(k_ptrs + start_n * stride_kn, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=v_mask, other=0.0)

        # S_ij = (Q_i * K_j^T) * scale: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * sm_scale

        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= curr_offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Rescale accumulator for new max
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        # Accumulate: acc += P_ij * V_j
        p_cast = p.to(v.dtype)
        acc = tl.dot(p_cast, v, acc)

        # Update running stats
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Final normalization
    acc = acc / l_i[:, None]

    # Write output back to HBM
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        sm_scale: Optional[float] = None
    ) -> torch.Tensor:
        B, H, N, d = q.shape

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(d)

        # Ensure contiguous inputs
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = torch.empty_like(q)

        # SRAM tiling heuristics
        BLOCK_M = 128 if d <= 64 else 64
        BLOCK_N = 64 if d <= 64 else 32
        BLOCK_D = d

        num_warps = 4 if d <= 64 else 8
        num_stages = 3

        grid = (triton.cdiv(N, BLOCK_M), B * H)

        _flash_fwd_kernel[grid](
            q, k, v, out,
            sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            IS_CAUSAL=is_causal,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return out

def flash_attention_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None
) -> torch.Tensor:
    return FlashAttentionFunction.apply(q, k, v, is_causal, sm_scale)