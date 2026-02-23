import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ================= CONFIGURATION =================
BASE_DIRS = ["1st", "2nd", "3rd"]
BENCHMARK = "gsm8k"
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]
# ==================================================

def get_avg_power(file_path):
    """Calculates average power in Watts from the json."""
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if not data: return None
        values = [float(p['value']) if isinstance(p, dict) else float(p[1]) for p in data]
        if not values: return None
        return np.mean(values)
    except:
        return None

def main():
    # Lists for data
    watt_vals = [] # Left Axis (Red)
    bmc_vals = []  # Right Axis (Blue)
    labels = []    # X Axis

    # 1. Data Extraction
    for size in MODELS:
        for ratio in RATIOS:
            w_runs = []
            b_runs = []
            
            for d in BASE_DIRS:
                path_watt = os.path.join(d, size, BENCHMARK, f"cr_{ratio}", "metrics_watt.json")
                path_bmc = os.path.join(d, size, BENCHMARK, f"cr_{ratio}", "metrics_bmc.json")
                w = get_avg_power(path_watt)
                b = get_avg_power(path_bmc)
                if w is not None: w_runs.append(w)
                if b is not None: b_runs.append(b)
            
            if w_runs and b_runs:
                watt_vals.append(np.mean(w_runs))
                bmc_vals.append(np.mean(b_runs))
                # X axis label: e.g., "14B | 0.5"
                labels.append(f"{size} | {ratio}")

    if not watt_vals:
        print("No data found!")
        return

    # 2. Calculate Correlation for the Title
    corr, _ = stats.pearsonr(watt_vals, bmc_vals)

    # 3. Create Dual Axis Chart
    fig, ax1 = plt.subplots(figsize=(14, 7))

    x_indexes = np.arange(len(labels))

    # --- LEFT AXIS (Wattmeter - RED) ---
    color_1 = 'tab:red'
    ax1.set_xlabel('Experiment (Model | Ratio)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg Wattmeter Power [W] (Wall)', color=color_1, fontsize=14, fontweight='bold')
    # Plot solid line with circles
    line1 = ax1.plot(x_indexes, watt_vals, color=color_1, marker='o', linewidth=2, label='Wattmeter (Wall)')
    ax1.tick_params(axis='y', labelcolor=color_1, labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- RIGHT AXIS (BMC - BLUE) ---
    ax2 = ax1.twinx()  # Create second axis sharing the same X
    color_2 = 'tab:blue'
    ax2.set_ylabel('Avg BMC Power [W] (Internal)', color=color_2, fontsize=14, fontweight='bold')
    # Plot dashed line with X
    line2 = ax2.plot(x_indexes, bmc_vals, color=color_2, marker='x', linestyle='--', linewidth=2, label='BMC (Board)')
    ax2.tick_params(axis='y', labelcolor=color_2, labelsize=12)

    # --- X AXIS FORMATTING ---
    ax1.set_xticks(x_indexes)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

    # --- TITLE AND LEGEND ---
    plt.title(f"Sensor Consistency Check (3-Run Avg)\nPearson Correlation: {corr:.5f} (1.0 = Perfect match)", fontsize=16)
    
    # Combined Legend
    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    output_file = "graph_consistency_dual_axis_avg.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved as: {output_file}")

if __name__ == "__main__":
    main()
