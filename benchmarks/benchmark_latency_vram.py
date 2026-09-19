import json
import os
import torch
import triton
from flash_attn.naive import naive_attention, sdpa_attention
from flash_attn.triton_flash_fwd import flash_attention_custom

def calculate_tflops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    latency_ms: float
) -> float:
    """Computes operational throughput in TFLOPS."""
    multiplier = 2.0 if is_causal else 4.0
    flops = multiplier * batch_size * num_heads * (seq_len ** 2) * head_dim
    return (flops / (latency_ms / 1000.0)) / 1e12

def measure_peak_memory_mb(fn, *args, **kwargs) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

def run_suite():
    if not torch.cuda.is_available():
        raise SystemError("CUDA GPU required for benchmarking.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Benchmarking on: {gpu_name}")

    B = 4
    H = 16
    D = 64
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]
    is_causal = True
    dtype = torch.float16

    results = []

    print(f"\n{'SeqLen':<8} | {'Impl':<14} | {'Time (ms)':<10} | {'TFLOPS':<8} | {'Peak VRAM (MB)':<14}")
    print("-" * 64)

    for N in seq_lens:
        q = torch.randn((B, H, N, D), dtype=dtype, device=device)
        k = torch.randn((B, H, N, D), dtype=dtype, device=device)
        v = torch.randn((B, H, N, D), dtype=dtype, device=device)

        implementations = [
            ("CustomFlash", lambda: flash_attention_custom(q, k, v, is_causal=is_causal)),
            ("PyTorchSDPA", lambda: sdpa_attention(q, k, v, is_causal=is_causal)),
        ]

        if N <= 2048:
            implementations.insert(0, ("Naive", lambda: naive_attention(q, k, v, is_causal=is_causal)))

        for name, fn in implementations:
            try:
                ms = triton.testing.do_bench(fn, warmup=25, rep=100)
                peak_vram = measure_peak_memory_mb(fn)
                tflops = calculate_tflops(B, H, N, D, is_causal, ms)

                print(f"{N:<8} | {name:<14} | {ms:<10.3f} | {tflops:<8.2f} | {peak_vram:<14.2f}")

                results.append({
                    "gpu": gpu_name,
                    "batch_size": B,
                    "num_heads": H,
                    "seq_len": N,
                    "head_dim": D,
                    "is_causal": is_causal,
                    "dtype": str(dtype),
                    "implementation": name,
                    "latency_ms": ms,
                    "tflops": tflops,
                    "peak_vram_mb": peak_vram
                })
            except torch.cuda.OutOfMemoryError:
                print(f"{N:<8} | {name:<14} | {'OOM':<10} | {'-':<8} | {'-':<14}")

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_data.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark results written to results/benchmark_data.json")

if __name__ == "__main__":
    run_suite()