import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

# ================= CONFIGURATION =================
BASE_DIRS = ["1st", "2nd", "3rd"]
BENCHMARK = "gsm8k"

# TokenSkip Style Colors
COLORS = {
    "3B": "#488fba",  # Steel Blue
    "7B": "#ffcc33",  # Mustard Yellow
    "14B": "#5ebf9e"  # Teal
}

MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]

def parse_log(file_path):
    """Extracts accuracy and token length from the log."""
    acc = None
    tokens = None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            acc_match = re.search(r"output acc = ([\d\.]+)", content)
            tok_match = re.search(r"output avg_cot_length = ([\d\.]+)", content)
            if acc_match: acc = float(acc_match.group(1))
            if tok_match: tokens = float(tok_match.group(1))
    except FileNotFoundError:
        return None, None
    return acc, tokens

def main():
    fig, ax = plt.subplots(figsize=(10, 7))
    
    legend_handles = []
    
    # Legend entry for "Original"
    star_marker = mlines.Line2D([], [], color='brown', marker='*', linestyle='None',
                              markersize=15, label='Original')
    legend_handles.append(star_marker)

    for size in MODELS:
        all_tokens = []
        all_accs = []
        
        baseline_acc = None
        baseline_tok = None

        color = COLORS[size]
        print(f"Processing {size}...")

        for ratio in RATIOS:
            acc_vals = []
            tok_vals = []
            
            # Average over 3 runs
            for d in BASE_DIRS:
                path = os.path.join(d, size, BENCHMARK, f"cr_{ratio}", "run_log.txt")
                acc, tokens = parse_log(path)
                if acc is not None and tokens is not None:
                    acc_vals.append(acc)
                    tok_vals.append(tokens)
            
            if acc_vals and tok_vals:
                avg_acc = np.mean(acc_vals)
                avg_tok = np.mean(tok_vals)
                
                # Always add to plot data for continuous line
                all_tokens.append(avg_tok)
                all_accs.append(avg_acc)

                # Save baseline coordinates separately for the star and dashed line
                if ratio == "1.0":
                    baseline_acc = avg_acc
                    baseline_tok = avg_tok

        # Sort by X axis (Tokens)
        if all_tokens:
            sorted_pairs = sorted(zip(all_tokens, all_accs))
            x_sorted, y_sorted = zip(*sorted_pairs)

            # 1. Draw the LINE connecting EVERYTHING (including the star)
            line, = ax.plot(x_sorted, y_sorted, color=color, linewidth=3, marker='o', 
                            markersize=10, label=f"TokenSkip ({size})", zorder=5)
            legend_handles.append(line)

            # 2. Overwrite the Baseline point with the STAR
            if baseline_acc is not None:
                # Horizontal dashed line
                ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
                # Star on top of the circle
                ax.scatter(baseline_tok, baseline_acc, color='brown', marker='*', s=350, zorder=10)

    # --- UPDATED STYLING ---
    ax.set_xlabel("Reasoning Tokens (Avg of 3 Runs)", fontsize=16)
    ax.set_ylabel("Accuracy (%) (Avg of 3 Runs)", fontsize=16)
    
    # Y-AXIS SCALE: Every 5 units
    ax.yaxis.set_major_locator(MultipleLocator(5))
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Legend
    ax.legend(handles=legend_handles, loc='lower right', fontsize=14, frameon=True)
    
    plt.tight_layout()
    output_file = "graph_accuracy_tokens_connected_avg.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as: {output_file}")

if __name__ == "__main__":
    main()
