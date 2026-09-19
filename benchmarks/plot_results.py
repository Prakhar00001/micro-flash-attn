import json
import os
import matplotlib.pyplot as plt
import pandas as pd

def generate_plots():
    if not os.path.exists("results/benchmark_data.json"):
        print("No benchmark data found. Run benchmark_latency_vram.py first.")
        return

    with open("results/benchmark_data.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    os.makedirs("results/figures", exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Latency Plot
    plt.figure(figsize=(9, 5))
    for impl, group in df.groupby("implementation"):
        plt.plot(group["seq_len"], group["latency_ms"], marker="o", linewidth=2, label=impl)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence Length (N)", fontsize=12)
    plt.ylabel("Latency (ms) - Log Scale", fontsize=12)
    plt.title("Attention Latency Scaling vs Sequence Length", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/figures/latency_scaling.png", dpi=300)
    plt.close()

    # 2. Peak VRAM Plot
    plt.figure(figsize=(9, 5))
    for impl, group in df.groupby("implementation"):
        plt.plot(group["seq_len"], group["peak_vram_mb"], marker="s", linewidth=2, label=impl)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Sequence Length (N)", fontsize=12)
    plt.ylabel("Peak Allocated VRAM (MB) - Log Scale", fontsize=12)
    plt.title("Peak GPU Memory Consumption vs Sequence Length", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/figures/vram_scaling.png", dpi=300)
    plt.close()

    # 3. Achieved TFLOPS Plot
    plt.figure(figsize=(9, 5))
    for impl, group in df.groupby("implementation"):
        plt.plot(group["seq_len"], group["tflops"], marker="^", linewidth=2, label=impl)
    plt.xscale("log", base=2)
    plt.xlabel("Sequence Length (N)", fontsize=12)
    plt.ylabel("Achieved TFLOPS", fontsize=12)
    plt.title("Computational Throughput (TFLOPS) vs Sequence Length", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("results/figures/tflops_throughput.png", dpi=300)
    plt.close()

    print("Figures successfully generated in results/figures/")

if __name__ == "__main__":
    generate_plots()