import pytest
import torch
from flash_attn.naive import naive_attention
from flash_attn.triton_flash_fwd import flash_attention_custom

@pytest.mark.parametrize("seq_len", [1, 63, 127, 341, 513, 1023])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_non_power_of_two_sequence_lengths(seq_len: int, head_dim: int, is_causal: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(42)

    q = torch.randn((2, 4, seq_len, head_dim), dtype=dtype, device=device)
    k = torch.randn((2, 4, seq_len, head_dim), dtype=dtype, device=device)
    v = torch.randn((2, 4, seq_len, head_dim), dtype=dtype, device=device)

    out_naive = naive_attention(q, k, v, is_causal=is_causal)
    out_triton = flash_attention_custom(q, k, v, is_causal=is_causal)

    assert torch.allclose(out_triton, out_naive, atol=1e-2, rtol=1e-2), (
        f"Failed on seq_len={seq_len}: max diff = {(out_triton - out_naive).abs().max().item()}"
    )