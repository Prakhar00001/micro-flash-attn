import torch
from flash_attn.naive import naive_attention
from flash_attn.triton_flash_fwd import flash_attention_custom

def profile_kernel_execution():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    B, H, N, D = 2, 8, 2048, 64
    dtype = torch.float16

    q = torch.randn((B, H, N, D), dtype=dtype, device=device)
    k = torch.randn((B, H, N, D), dtype=dtype, device=device)
    v = torch.randn((B, H, N, D), dtype=dtype, device=device)

    print("=" * 60)
    print(f"Memory Profile: Batch={B}, Heads={H}, SeqLen={N}, HeadDim={D}")
    print("=" * 60)

    # 1. Profile Naive Attention
    torch.cuda.reset_peak_memory_stats()
    _ = naive_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    naive_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 2. Profile Custom FlashAttention
    torch.cuda.reset_peak_memory_stats()
    _ = flash_attention_custom(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    flash_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Naive Attention Peak Allocated VRAM:        {naive_mem:.2f} MB")
    print(f"Custom FlashAttention Peak Allocated VRAM: {flash_mem:.2f} MB")
    print(f"Memory Reduction Factor:                    {naive_mem / flash_mem:.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    profile_kernel_execution()