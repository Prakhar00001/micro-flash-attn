"""Unified Verification & Sanity Runner for micro-flash-attn.

Runs numerical checks against Naive PyTorch and PyTorch SDPA across
data types, sequence lengths, head dimensions, and causal modes.
"""

import math
import sys
import torch
from flash_attn.naive import naive_attention, sdpa_attention
from flash_attn.online_softmax import online_softmax_attention
from flash_attn.triton_flash_fwd import flash_attention_custom


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def test_cpu_online_softmax():
    print_header("TEST 1: CPU Online Softmax Mathematical Equivalence")
    seq_lens = [64, 128, 256, 512]
    all_passed = True

    for N in seq_lens:
        for causal in [False, True]:
            q = torch.randn(2, 4, N, 32, dtype=torch.float32)
            k = torch.randn(2, 4, N, 32, dtype=torch.float32)
            v = torch.randn(2, 4, N, 32, dtype=torch.float32)

            ref = naive_attention(q, k, v, is_causal=causal)
            out = online_softmax_attention(q, k, v, block_m=32, block_n=32, is_causal=causal)

            max_diff = (ref - out).abs().max().item()
            passed = max_diff < 1e-5
            all_passed = all_passed and passed
            status = "PASS" if passed else "FAIL"
            print(f"  [CPU] N={N:<4} | Causal={str(causal):<5} | Max Diff: {max_diff:.2e} -> {status}")

    return all_passed


def test_gpu_numerical_invariance():
    print_header("TEST 2: Triton Kernel Correctness vs Naive & SDPA")
    if not torch.cuda.is_available():
        print("  SKIPPED: CUDA GPU not detected.")
        return True

    device = torch.device("cuda")
    configs = [
        # (B, H, N, D, causal, dtype)
        (2, 8, 128, 64, False, torch.float16),
        (2, 8, 128, 64, True, torch.float16),
        (2, 8, 512, 64, False, torch.float16),
        (2, 8, 512, 64, True, torch.float16),
        (1, 4, 1024, 64, True, torch.float16),
        (1, 4, 2048, 64, True, torch.float16),
        (1, 4, 512, 128, True, torch.float16),
        (2, 4, 512, 64, True, torch.bfloat16),
    ]

    all_passed = True
    for B, H, N, D, causal, dtype in configs:
        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H, N, D, dtype=dtype, device=device)

        ref_naive = naive_attention(q, k, v, is_causal=causal)
        ref_sdpa = sdpa_attention(q, k, v, is_causal=causal)
        out_triton = flash_attention_custom(q, k, v, is_causal=causal)

        diff_naive = (ref_naive - out_triton).abs().max().item()
        diff_sdpa = (ref_sdpa - out_triton).abs().max().item()

        tol = 2e-2 if dtype == torch.bfloat16 else 1e-2
        passed = (diff_naive <= tol) and (diff_sdpa <= tol)
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        dt_str = "BF16" if dtype == torch.bfloat16 else "FP16"
        print(f"  [GPU] B={B} H={H:<2} N={N:<4} D={D:<3} {dt_str} | Causal={str(causal):<5} | "
              f"Diff vs Naive: {diff_naive:.4f} | Diff vs SDPA: {diff_sdpa:.4f} -> {status}")

    return all_passed


def test_edge_cases():
    print_header("TEST 3: Non-Power-of-2 Edge Cases & Boundary Masking")
    if not torch.cuda.is_available():
        print("  SKIPPED: CUDA GPU not detected.")
        return True

    device = torch.device("cuda")
    odd_seq_lens = [1, 17, 63, 127, 341, 513, 1023]
    all_passed = True

    for N in odd_seq_lens:
        q = torch.randn(2, 4, N, 64, dtype=torch.float16, device=device)
        k = torch.randn(2, 4, N, 64, dtype=torch.float16, device=device)
        v = torch.randn(2, 4, N, 64, dtype=torch.float16, device=device)

        ref = naive_attention(q, k, v, is_causal=True)
        out = flash_attention_custom(q, k, v, is_causal=True)

        diff = (ref - out).abs().max().item()
        passed = diff <= 1.5e-2
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [EDGE] Non-power-of-2 N={N:<4} | Causal=True  | Max Diff: {diff:.4f} -> {status}")

    return all_passed


def test_memory_reduction():
    print_header("TEST 4: HBM Traffic & Peak VRAM Verification")
    if not torch.cuda.is_available():
        print("  SKIPPED: CUDA GPU not detected.")
        return True

    device = torch.device("cuda")
    B, H, N, D = 2, 8, 2048, 64
    dtype = torch.float16

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # 1. Naive Attention Peak Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = naive_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    mem_naive = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 2. FlashAttention Peak Memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = flash_attention_custom(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    mem_flash = torch.cuda.max_memory_allocated() / (1024 ** 2)

    ratio = mem_naive / mem_flash
    print(f"  Naive Attention Peak Allocated VRAM:        {mem_naive:.2f} MB")
    print(f"  Custom FlashAttention Peak Allocated VRAM: {mem_flash:.2f} MB")
    print(f"  Memory Footprint Reduction:                 {ratio:.2f}x")

    passed = ratio >= 4.0
    print(f"  O(N^2) HBM Materialization Bypassed:        {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    t1 = test_cpu_online_softmax()
    t2 = test_gpu_numerical_invariance()
    t3 = test_edge_cases()
    t4 = test_memory_reduction()

    print("\n" + "=" * 70)
    if all([t1, t2, t3, t4]):
        print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    else:
        print(" ONE OR MORE VERIFICATION CHECKS FAILED.")
        sys.exit(1)
    print("=" * 70)