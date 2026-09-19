import math
from typing import Optional
import torch

def online_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure PyTorch implementation of FlashAttention-1 forward pass using block tiling

    and the 2-pass online softmax algorithm. Validates tiling math on CPU or GPU without Triton.
    """
    B, H, N_q, d = q.shape
    _, _, N_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    out = torch.zeros_like(q, dtype=torch.float32)

    # Iterate over outer blocks of Query (Row-blocks)
    for start_m in range(0, N_q, block_m):
        end_m = min(start_m + block_m, N_q)
        q_tile = q[:, :, start_m:end_m, :]  # [B, H, Br, d]
        Br = end_m - start_m

        # Online softmax statistics initialized per row block
        m_i = torch.full((B, H, Br, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l_i = torch.zeros((B, H, Br, 1), device=q.device, dtype=torch.float32)
        acc = torch.zeros((B, H, Br, d), device=q.device, dtype=torch.float32)

        # Inner loop over Key/Value blocks (Column-blocks)
        for start_n in range(0, N_k, block_n):
            end_n = min(start_n + block_n, N_k)
            k_tile = k[:, :, start_n:end_n, :]  # [B, H, Bc, d]
            v_tile = v[:, :, start_n:end_n, :]  # [B, H, Bc, d]
            Bc = end_n - start_n

            # S_ij = (Q_i * K_j^T) * scale: [B, H, Br, Bc]
            s_ij = torch.matmul(q_tile.to(torch.float32), k_tile.to(torch.float32).transpose(-2, -1)) * scale

            if is_causal:
                # Calculate global positions for causal masking
                row_idx = torch.arange(start_m, end_m, device=q.device)[:, None]
                col_idx = torch.arange(start_n, end_n, device=q.device)[None, :]
                causal_mask = row_idx >= col_idx
                s_ij = torch.where(causal_mask[None, None, :, :], s_ij, float("-inf"))

            # 1. Compute local block max
            m_block, _ = torch.max(s_ij, dim=-1, keepdim=True)
            m_new = torch.maximum(m_i, m_block)

            # 2. Exponentiate with running max subtraction
            p_tilde = torch.exp(s_ij - m_new)
            l_block = torch.sum(p_tilde, dim=-1, keepdim=True)

            # 3. Rescaling factor for previously accumulated statistics
            alpha = torch.exp(m_i - m_new)

            # 4. Update denominator sum and partial output accumulator
            l_i = l_i * alpha + l_block
            acc = acc * alpha + torch.matmul(p_tilde, v_tile.to(torch.float32))

            # 5. Update running maximum
            m_i = m_new

        # Final row-wise normalization
        out[:, :, start_m:end_m, :] = acc / torch.clamp(l_i, min=1e-6)

    return out.to(q.dtype)