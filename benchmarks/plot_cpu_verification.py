import os
import matplotlib.pyplot as plt
import numpy as np

def plot_cpu_verification_results():
    # Measured test scores from run_verification.py
    seq_lens = [64, 128, 256, 512]
    diff_non_causal = [5.36e-07, 3.87e-07, 3.87e-07, 4.32e-07]
    diff_causal = [3.58e-07, 4.77e-07, 5.96e-07, 7.15e-07]
    fp32_tolerance = 1.0e-05

    os.makedirs("results/figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    x = np.arange(len(seq_lens))
    width = 0.35

    # Grouped bars
    rects1 = ax.bar(x - width/2, diff_non_causal, width, label='Non-Causal (Bidirectional)', color='#2563eb', alpha=0.9)
    rects2 = ax.bar(x + width/2, diff_causal, width, label='Causal (Autoregressive)', color='#059669', alpha=0.9)

    # Tolerance ceiling threshold line
    ax.axhline(y=fp32_tolerance, color='#dc2626', linestyle='--', linewidth=1.5, label='FP32 Test Tolerance ($1.0\\times10^{-5}$)')

    # Labels and scales
    ax.set_xlabel('Sequence Length ($N$)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Max Absolute Difference vs Naive Attention', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('TEST 1: CPU Online Softmax Numerical Invariance Verification', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in seq_lens], fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 2e-5)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc='upper left')

    # Annotate value bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.2e}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.2e}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    output_path = "results/figures/cpu_verification.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Figure successfully saved to {output_path}")

if __name__ == "__main__":
    plot_cpu_verification_results()