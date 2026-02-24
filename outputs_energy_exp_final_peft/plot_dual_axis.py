import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ================= CONFIGURATION =================
BASE_DIR = "outputs_energy_exp_final"
BENCHMARK = "gsm8k"
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]

def get_avg_power(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f: data = json.load(f)
        if not data: return None
        values = [float(p['value']) if isinstance(p, dict) else float(p[1]) for p in data]
        if not values: return None
        return np.mean(values) # Qui la media è tra i sample dello stesso file
    except: return None

def main():
    watt_vals, bmc_vals, labels = [], [], []

    for size in MODELS:
        for ratio in RATIOS:
            path_watt = os.path.join(BASE_DIR, size, BENCHMARK, f"cr_{ratio}", "metrics_watt.json")
            path_bmc = os.path.join(BASE_DIR, size, BENCHMARK, f"cr_{ratio}", "metrics_bmc.json")
            w = get_avg_power(path_watt)
            b = get_avg_power(path_bmc)
            
            if w is not None and b is not None:
                watt_vals.append(w)
                bmc_vals.append(b)
                labels.append(f"{size} | {ratio}")

    if not watt_vals:
        print("No data found!")
        return

    corr, _ = stats.pearsonr(watt_vals, bmc_vals)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    x_indexes = np.arange(len(labels))

    color_1 = 'tab:red'
    ax1.set_xlabel('Experiment (Model | Ratio)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg Wattmeter Power [W] (Wall)', color=color_1, fontsize=14, fontweight='bold')
    line1 = ax1.plot(x_indexes, watt_vals, color=color_1, marker='o', linewidth=2, label='Wattmeter (Wall)')
    ax1.tick_params(axis='y', labelcolor=color_1, labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_2 = 'tab:blue'
    ax2.set_ylabel('Avg BMC Power [W] (Internal)', color=color_2, fontsize=14, fontweight='bold')
    line2 = ax2.plot(x_indexes, bmc_vals, color=color_2, marker='x', linestyle='--', linewidth=2, label='BMC (Board)')
    ax2.tick_params(axis='y', labelcolor=color_2, labelsize=12)

    ax1.set_xticks(x_indexes)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

    plt.title(f"Sensor Consistency Check\nPearson Correlation: {corr:.5f} (1.0 = Perfect match)", fontsize=16)
    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    output_file = "graph_consistency_dual_axis.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as: {output_file}")

if __name__ == "__main__":
    main()
