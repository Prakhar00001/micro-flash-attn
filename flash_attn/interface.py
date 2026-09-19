from typing import Literal, Optional
import torch
from flash_attn.naive import naive_attention, sdpa_attention
from flash_attn.online_softmax import online_softmax_attention
from flash_attn.triton_flash_fwd import flash_attention_custom

BackendType = Literal["triton", "sdpa", "naive", "online_cpu"]

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    backend: BackendType = "triton",
) -> torch.Tensor:
    """Unified entrypoint for computing attention across different implementations.

    Args:
        q, k, v: [Batch, Heads, SeqLen, HeadDim]
        is_causal: Causal autoregressive mask
        scale: Scaling constant (1/sqrt(d))
        backend: "triton" | "sdpa" | "naive" | "online_cpu"
    """
    if backend == "triton":
        if not q.is_cuda:
            raise ValueError("Triton backend requires CUDA tensors.")
        return flash_attention_custom(q, k, v, is_causal=is_causal, sm_scale=scale)
    elif backend == "sdpa":
        return sdpa_attention(q, k, v, is_causal=is_causal, scale=scale)
    elif backend == "naive":
        return naive_attention(q, k, v, is_causal=is_causal, scale=scale)
    elif backend == "online_cpu":
        return online_softmax_attention(q, k, v, is_causal=is_causal, scale=scale)
    else:
        raise ValueError(f"Unknown backend: {backend}")