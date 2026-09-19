import pytest
import torch
from flash_attn.naive import naive_attention, sdpa_attention
from flash_attn.online_softmax import online_softmax_attention
from flash_attn.triton_flash_fwd import flash_attention_custom

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_flash_attention_correctness(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    dtype: torch.dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for Triton tests")

    device = torch.device("cuda")
    torch.manual_seed(42)

    q = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), dtype=dtype, device=device)

    # Ground truths
    out_naive = naive_attention(q, k, v, is_causal=is_causal)
    out_sdpa = sdpa_attention(q, k, v, is_causal=is_causal)

    # Custom Triton implementation
    out_triton = flash_attention_custom(q, k, v, is_causal=is_causal)

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-2

    assert torch.allclose(out_triton, out_naive, atol=atol, rtol=rtol), (
        f"Mismatch vs Naive: max diff = {(out_triton - out_naive).abs().max().item()}"
    )
    assert torch.allclose(out_triton, out_sdpa, atol=atol, rtol=rtol), (
        f"Mismatch vs SDPA: max diff = {(out_triton - out_sdpa).abs().max().item()}"
    )

@pytest.mark.parametrize("seq_len", [64, 128, 256])
@pytest.mark.parametrize("is_causal", [False, True])
def test_online_softmax_cpu_numerical_match(seq_len: int, is_causal: bool):
    torch.manual_seed(42)
    q = torch.randn((1, 2, seq_len, 32), dtype=torch.float32)
    k = torch.randn((1, 2, seq_len, 32), dtype=torch.float32)
    v = torch.randn((1, 2, seq_len, 32), dtype=torch.float32)

    out_naive = naive_attention(q, k, v, is_causal=is_causal)
    out_online = online_softmax_attention(q, k, v, block_m=32, block_n=32, is_causal=is_causal)

    assert torch.allclose(out_online, out_naive, atol=1e-5, rtol=1e-5), (
        f"CPU Online Softmax mismatch: max diff = {(out_online - out_naive).abs().max().item()}"
    )