import os
import re
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
BASE_DIRS = ["1st", "2nd", "3rd"]
BENCHMARK = "gsm8k"
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]
COLORS = {"3B": "#488fba", "7B": "#ffcc33", "14B": "#5ebf9e"}
# ==================================================

def get_avg_tokens(file_path):
    """Reads average token number from log."""
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
    # Setup: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    
    print(f"{'MODEL':<6} | {'RATIO':<5} | {'AVG TOKENS':<12} | {'REDUCTION %':<10}")
    print("-" * 55)

    for i, size in enumerate(MODELS):
        ax = axes[i]
        ratios_val = []
        tokens_val = []
        
        # 1. Retrieve Baseline (Ratio 1.0)
        base_runs = []
        for d in BASE_DIRS:
            base_path = os.path.join(d, size, BENCHMARK, "cr_1.0", "run_log.txt")
            val = get_avg_tokens(base_path)
            if val is not None: base_runs.append(val)
            
        if not base_runs:
            print(f"Skipping {size}: Baseline tokens not found.")
            continue
            
        baseline_tokens = np.mean(base_runs)

        # 2. Retrieve data for all ratios
        for ratio in RATIOS:
            tok_runs = []
            for d in BASE_DIRS:
                path = os.path.join(d, size, BENCHMARK, f"cr_{ratio}", "run_log.txt")
                val = get_avg_tokens(path)
                if val is not None: tok_runs.append(val)
            
            if tok_runs:
                avg_tok = np.mean(tok_runs)
                ratios_val.append(float(ratio))
                tokens_val.append(avg_tok)
                
                # Calculate % reduction
                red_pct = ((baseline_tokens - avg_tok) / baseline_tokens) * 100
                print(f"{size:<6} | {ratio:<5} | {avg_tok:<12.1f} | -{red_pct:<9.1f}%")

        # 3. Draw graph
        if ratios_val:
            # Main Line (Actual Data)
            ax.plot(ratios_val, tokens_val, marker='o', linewidth=3, 
                    color=COLORS[size], label='Actual Tokens (Avg)')
            
            # Ideal Line (Theoretical)
            ideal_tokens = [baseline_tokens * r for r in ratios_val]
            ax.plot(ratios_val, ideal_tokens, linestyle='--', color='gray', 
                    alpha=0.6, label='Theoretical Ideal')

            # Filled area between actual and ideal
            ax.fill_between(ratios_val, tokens_val, ideal_tokens, color=COLORS[size], alpha=0.1)

            # Styling
            ax.set_title(f"Model: {size} (Baseline: {baseline_tokens:.0f} tok)", fontsize=14, fontweight='bold')
            ax.set_xlabel("Compression Ratio", fontsize=12)
            ax.set_ylabel("Avg Generated Tokens", fontsize=12)
            ax.set_xlim(1.05, 0.45) # Invert X axis (from 1.0 to 0.5)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            # Add labels to points
            for j, val in enumerate(tokens_val):
                ax.annotate(f"{val:.0f}", (ratios_val[j], val), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

            if i == 0:
                ax.legend(loc='lower right')

    plt.tight_layout()
    output_file = "graph_token_reduction_trend_avg.png"
    plt.savefig(output_file, dpi=300)
    print("\n" + "="*50)
    print(f"Graph saved: {output_file}")

if __name__ == "__main__":
    main()
