import math
from typing import Optional
import torch
import torch.nn.functional as F

def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """Standard multi-head attention that materializes intermediate O(N^2) matrices in HBM.

    Args:
        q: Query tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
        v: Value tensor of shape [batch_size, num_heads, seq_len_k, head_dim]
        is_causal: If True, applies causal lower-triangular masking.
        scale: Scaling factor for dot products. Defaults to 1 / sqrt(head_dim).

    Returns:
        Output tensor of shape [batch_size, num_heads, seq_len_q, head_dim]
    """
    B, H, N_q, d = q.shape
    _, _, N_k, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Materialize full S matrix in HBM: [B, H, N_q, N_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        # Causal mask: query at position i can only attend to key positions <= i
        mask = torch.triu(
            torch.full((N_q, N_k), float("-inf"), device=q.device, dtype=scores.dtype),
            diagonal=1 + (N_k - N_q) if N_k >= N_q else 1
        )
        scores = scores + mask

    # Materialize P (softmax probabilities) in HBM: [B, H, N_q, N_k]
    p = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    # Compute attention output: [B, H, N_q, d]
    out = torch.matmul(p, v)
    return out

def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """PyTorch scaled_dot_product_attention reference (C++ FlashAttention / CUTLASS backend)."""
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal, scale=scale
    )