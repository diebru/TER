import os
import json
import re
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

def get_accuracy(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f:
            match = re.search(r"output acc = ([\d\.]+)", f.read())
            if match: 
                val = float(match.group(1))
                return val * 100 if val <= 1.0 else val
    except: pass
    return None

def get_total_energy(file_path):
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f: data = json.load(f)
        if not data: return None
        ts_list, val_list = [], []
        for p in data:
            raw_t = p.get('timestamp') if isinstance(p, dict) else p[0]
            val = p.get('value') if isinstance(p, dict) else p[1]
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
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, size in enumerate(MODELS):
        ax = axes[i]
        energies, accuracies, labels = [], [], []
        
        for ratio in RATIOS:
            dir_path = os.path.join(BASE_DIR, size, BENCHMARK, f"cr_{ratio}")
            acc = get_accuracy(os.path.join(dir_path, "run_log.txt"))
            joules = get_total_energy(os.path.join(dir_path, "metrics_watt.json"))
            
            if acc is not None and joules is not None:
                energies.append(joules / 1000.0)
                accuracies.append(acc)
                labels.append(ratio)

        if energies:
            color = COLORS[size]
            idx_baseline = labels.index("1.0") if "1.0" in labels else -1
            rest_energies, rest_accuracies = [], []

            for j in range(len(labels)):
                if j == idx_baseline:
                    ax.scatter([energies[j]], [accuracies[j]], color=color, marker='*', s=400, edgecolors='k', zorder=15)
                else:
                    rest_energies.append(energies[j])
                    rest_accuracies.append(accuracies[j])

            if rest_energies:
                ax.plot(rest_energies, rest_accuracies, color=color, linestyle='--', alpha=0.5, linewidth=2)
                marker = 'D' if size == '14B' else 'o'
                ax.scatter(rest_energies, rest_accuracies, color=color, s=150, label=f"{size}", marker=marker, edgecolors='k', zorder=10)
            
            for j, txt in enumerate(labels):
                y_offset = 12
                if size == "14B" and j % 2 == 1: y_offset = -18
                elif txt == "0.9": y_offset = -15
                ax.annotate(f"r{txt}", (energies[j], accuracies[j]), 
                           xytext=(0, y_offset), textcoords='offset points',
                           ha='center', fontsize=9, fontweight='bold', color='black')

            # INGRANDIMENTO SCALA (Local Padding)
            e_min, e_max = min(energies), max(energies)
            a_min, a_max = min(accuracies), max(accuracies)
            e_pad = (e_max - e_min) * 0.15 if e_max != e_min else e_max * 0.1
            a_pad = (a_max - a_min) * 0.15 if a_max != a_min else a_max * 0.1
            
            ax.set_xlim(e_min - e_pad, e_max + e_pad)
            ax.set_ylim(a_min - a_pad, a_max + a_pad)
        
        ax.set_title(f"Model: {size}", fontsize=16, fontweight='bold', color='#333333')
        ax.set_xlabel("Total Energy (kJ) [Lower is Better]", fontsize=12)
        if i == 0:
            ax.set_ylabel("Accuracy (%) [Higher is Better]", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_file = "graph_tradeoff_unified.png"
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved: {output_file}")

if __name__ == "__main__":
    main()
