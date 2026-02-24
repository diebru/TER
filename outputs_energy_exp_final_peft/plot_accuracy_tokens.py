import os
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

# ================= CONFIGURATION =================
BASE_DIR = "outputs_energy_exp_final"
BENCHMARK = "gsm8k"

COLORS = {"3B": "#488fba", "7B": "#ffcc33", "14B": "#5ebf9e"}
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]

def parse_log(file_path):
    acc, tokens = None, None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            acc_match = re.search(r"output acc = ([\d\.]+)", content)
            tok_match = re.search(r"output avg_cot_length = ([\d\.]+)", content)
            if acc_match: acc = float(acc_match.group(1))
            if tok_match: tokens = float(tok_match.group(1))
    except FileNotFoundError:
        pass
    return acc, tokens

def main():
    fig, ax = plt.subplots(figsize=(10, 7))
    legend_handles = []
    
    star_marker = mlines.Line2D([], [], color='brown', marker='*', linestyle='None',
                              markersize=15, label='Original')
    legend_handles.append(star_marker)

    for size in MODELS:
        all_tokens = []
        all_accs = []
        baseline_acc, baseline_tok = None, None
        color = COLORS[size]
        print(f"Processing {size}...")

        for ratio in RATIOS:
            path = os.path.join(BASE_DIR, size, BENCHMARK, f"cr_{ratio}", "run_log.txt")
            acc, tokens = parse_log(path)
            
            if acc is not None and tokens is not None:
                all_tokens.append(tokens)
                all_accs.append(acc)

                if ratio == "1.0":
                    baseline_acc = acc
                    baseline_tok = tokens

        if all_tokens:
            sorted_pairs = sorted(zip(all_tokens, all_accs))
            x_sorted, y_sorted = zip(*sorted_pairs)

            line, = ax.plot(x_sorted, y_sorted, color=color, linewidth=3, marker='o', 
                            markersize=10, label=f"TokenSkip ({size})", zorder=5)
            legend_handles.append(line)

            if baseline_acc is not None:
                ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
                ax.scatter(baseline_tok, baseline_acc, color='brown', marker='*', s=350, zorder=10)

    ax.set_xlabel("Reasoning Tokens", fontsize=16)
    ax.set_ylabel("Accuracy (%)", fontsize=16)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(handles=legend_handles, loc='lower right', fontsize=14, frameon=True)
    
    plt.tight_layout()
    output_file = "graph_accuracy_tokens_connected.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as: {output_file}")

if __name__ == "__main__":
    main()
