import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
BASE_DIR = "outputs_energy_exp_final"
BENCHMARK = "gsm8k"
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]
COLORS = {"3B": "#488fba", "7B": "#ffcc33", "14B": "#5ebf9e"}

def get_avg_tokens(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            match = re.search(r"output avg_cot_length = ([\d\.]+)", content)
            if match: return float(match.group(1))
    except Exception:
        pass
    return None

def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
    print(f"{'MODEL':<6} | {'RATIO':<5} | {'TOKENS':<12} | {'REDUCTION %':<10}")
    print("-" * 55)

    for i, size in enumerate(MODELS):
        ax = axes[i]
        ratios_val, tokens_val = [], []
        
        base_path = os.path.join(BASE_DIR, size, BENCHMARK, "cr_1.0", "run_log.txt")
        baseline_tokens = get_avg_tokens(base_path)
            
        if baseline_tokens is None:
            print(f"Skipping {size}: Baseline tokens not found.")
            continue

        for ratio in RATIOS:
            path = os.path.join(BASE_DIR, size, BENCHMARK, f"cr_{ratio}", "run_log.txt")
            tok = get_avg_tokens(path)
            
            if tok is not None:
                ratios_val.append(float(ratio))
                tokens_val.append(tok)
                red_pct = ((baseline_tokens - tok) / baseline_tokens) * 100
                print(f"{size:<6} | {ratio:<5} | {tok:<12.1f} | -{red_pct:<9.1f}%")

        if ratios_val:
            ax.plot(ratios_val, tokens_val, marker='o', linewidth=3, 
                    color=COLORS[size], label='Actual Tokens')
            
            ideal_tokens = [baseline_tokens * r for r in ratios_val]
            ax.plot(ratios_val, ideal_tokens, linestyle='--', color='gray', 
                    alpha=0.6, label='Theoretical Ideal')

            ax.fill_between(ratios_val, tokens_val, ideal_tokens, color=COLORS[size], alpha=0.1)

            ax.set_title(f"Model: {size} (Baseline: {baseline_tokens:.0f} tok)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Compression Ratio", fontsize=12)
            ax.set_ylabel("Generated Tokens", fontsize=12)
            ax.set_xlim(1.05, 0.45) 
            ax.grid(True, linestyle=':', alpha=0.7)
            
            for j, val in enumerate(tokens_val):
                ax.annotate(f"{val:.0f}", (ratios_val[j], val), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

            if i == 0:
                ax.legend(loc='lower right')

    plt.tight_layout()
    output_file = "graph_token_reduction_trend.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved: {output_file}")

if __name__ == "__main__":
    main()
