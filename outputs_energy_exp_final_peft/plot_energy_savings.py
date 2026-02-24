import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ================= CONFIGURATION =================
BASE_DIR = "outputs_energy_exp_final"
BENCHMARK = "gsm8k"
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]
COLORS = {"3B": "#488fba", "7B": "#ffcc33", "14B": "#5ebf9e"}

def parse_timestamp(ts):
    if isinstance(ts, (int, float)): return float(ts)
    if isinstance(ts, str):
        try: return datetime.fromisoformat(ts).timestamp()
        except ValueError: return None
    return None

def calculate_joules(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f: data = json.load(f)
        if not data: return None
        ts_list, val_list = [], []
        for p in data:
            raw_t = p['timestamp'] if isinstance(p, dict) else p[0]
            val = p['value'] if isinstance(p, dict) else p[1]
            t_float = parse_timestamp(raw_t)
            if t_float:
                ts_list.append(t_float)
                val_list.append(float(val))
        if len(ts_list) < 2: return None
        ts, val = np.array(ts_list), np.array(val_list)
        idx = np.argsort(ts)
        return np.trapz(val[idx], ts[idx])
    except: return None

def main():
    fig, ax = plt.subplots(figsize=(10, 7))

    print(f"{'MODEL':<6} | {'RATIO':<5} | {'JOULES':<12} | {'SAVING %':<10}")
    print("-" * 55)

    for size in MODELS:
        ratios_plot, savings_plot = [], []
        
        path_base = os.path.join(BASE_DIR, size, BENCHMARK, "cr_1.0", "metrics_watt.json")
        baseline_joules = calculate_joules(path_base)
            
        if baseline_joules is None:
            print(f"Skipping {size}: Baseline missing.")
            continue

        for ratio in RATIOS: 
            path = os.path.join(BASE_DIR, size, BENCHMARK, f"cr_{ratio}", "metrics_watt.json")
            joules = calculate_joules(path)

            if joules is not None:
                saving_pct = ((baseline_joules - joules) / baseline_joules) * 100
                ratios_plot.append(float(ratio))
                savings_plot.append(saving_pct)
                print(f"{size:<6} | {ratio:<5} | {joules:<12.0f} | {saving_pct:<10.2f}%")

        if ratios_plot:
            sorted_pairs = sorted(zip(ratios_plot, savings_plot), key=lambda x: x[0], reverse=True)
            r_sorted, s_sorted = zip(*sorted_pairs)
            ax.plot(r_sorted, s_sorted, marker='o', linewidth=3, color=COLORS[size], label=f"Model {size}")

    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, label="Baseline (0%)")
    ax.set_xlabel("Compression Ratio", fontsize=14)
    ax.set_ylabel("Energy Saving (%)", fontsize=14)
    ax.set_title("Energy Efficiency vs Compression Ratio", fontsize=16)
    ax.set_xlim(1.02, 0.48)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12)
    
    output_file = "graph_energy_savings.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved: {output_file}")

if __name__ == "__main__":
    main()
