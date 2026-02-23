import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ================= CONFIGURATION =================
BASE_DIRS = ["1st", "2nd", "3rd"]
BENCHMARK = "gsm8k"
MODELS = ["3B", "7B", "14B"]
RATIOS = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]
COLORS = {"3B": "#488fba", "7B": "#ffcc33", "14B": "#5ebf9e"}
# ==================================================

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
                # Smart fix: normalize to 0-100 if necessary
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
    # --- PHASE 1: GLOBAL DATA COLLECTION ---
    # We must read EVERYTHING first to calculate common scales
    all_data = {} 
    
    global_min_eng, global_max_eng = float('inf'), float('-inf')
    global_min_acc, global_max_acc = float('inf'), float('-inf')

    print("Reading data across 3 runs to calculate global scales...")
    
    for size in MODELS:
        all_data[size] = {'energies': [], 'accs': [], 'ratios': []}
        
        for ratio in RATIOS:
            acc_vals = []
            eng_vals = []
            
            for d in BASE_DIRS:
                dir_path = os.path.join(d, size, BENCHMARK, f"cr_{ratio}")
                acc = get_accuracy(os.path.join(dir_path, "run_log.txt"))
                joules = get_total_energy(os.path.join(dir_path, "metrics_watt.json"))
                
                if acc is not None and joules is not None:
                    acc_vals.append(acc)
                    eng_vals.append(joules / 1000.0) # kj
            
            if acc_vals and eng_vals:
                avg_acc = np.mean(acc_vals)
                avg_eng = np.mean(eng_vals)
                
                all_data[size]['energies'].append(avg_eng)
                all_data[size]['accs'].append(avg_acc)
                all_data[size]['ratios'].append(ratio)
                
                # Update global min/max
                if avg_eng < global_min_eng: global_min_eng = avg_eng
                if avg_eng > global_max_eng: global_max_eng = avg_eng
                if avg_acc < global_min_acc: global_min_acc = avg_acc
                if avg_acc > global_max_acc: global_max_acc = avg_acc

    # Calculate Margins (10% Padding)
    eng_padding = (global_max_eng - global_min_eng) * 0.1
    acc_padding = (global_max_acc - global_min_acc) * 0.1
    
    # Final axis limits
    xlims = (global_min_eng - eng_padding, global_max_eng + eng_padding)
    ylims = (global_min_acc - acc_padding, global_max_acc + acc_padding)

    print(f"Global Scales Set -> Energy: {xlims[0]:.1f}-{xlims[1]:.1f} kJ | Acc: {ylims[0]:.1f}-{ylims[1]:.1f}%")

    # --- PHASE 2: PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, size in enumerate(MODELS):
        ax = axes[i]
        data = all_data[size]
        
        if data['energies']:
            color = COLORS[size]
            energies = data['energies']
            accuracies = data['accs']
            labels = data['ratios']

            # Identify the index for the baseline ratio "1.0"
            idx_baseline = labels.index("1.0") if "1.0" in labels else -1

            rest_energies = []
            rest_accuracies = []

            for j in range(len(labels)):
                if j == idx_baseline:
                    # Draw a distinct star for the 1.0 point
                    ax.scatter([energies[j]], [accuracies[j]], color=color, marker='*', s=400, edgecolors='k', zorder=15)
                else:
                    rest_energies.append(energies[j])
                    rest_accuracies.append(accuracies[j])

            # Draw the dashed line and standard markers ONLY for the remaining points (0.9 to 0.5)
            if rest_energies:
                ax.plot(rest_energies, rest_accuracies, color=color, linestyle='--', alpha=0.5, linewidth=2)
                marker = 'D' if size == '14B' else 'o'
                ax.scatter(rest_energies, rest_accuracies, color=color, s=150, label=f"{size}", marker=marker, edgecolors='k', zorder=10)
            
            # Ratio Labels (apply labels to all points, including the star)
            for j, txt in enumerate(labels):
                y_offset = 12
                if size == "14B" and j % 2 == 1: y_offset = -18
                elif txt == "0.9": y_offset = -15
                
                ax.annotate(f"r{txt}", (energies[j], accuracies[j]), 
                           xytext=(0, y_offset), textcoords='offset points',
                           ha='center', fontsize=9, fontweight='bold', color='black')

        # --- APPLY UNIFIED SCALES ---
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        # Styling
        ax.set_title(f"Model: {size} (3-Run Avg)", fontsize=16, fontweight='bold', color='#333333')
        ax.set_xlabel("Avg Energy (kJ) [Lower is Better]", fontsize=12)
        
        # Show Y label only on the first plot
        if i == 0:
            ax.set_ylabel("Avg Accuracy (%) [Higher is Better]", fontsize=12)
        
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_file = "graph_tradeoff_unified_avg.png"
    plt.savefig(output_file, dpi=300)
    print("\n" + "="*55)
    print(f"Unified scale graph (Avg) saved: {output_file}")
    print("="*55)

if __name__ == "__main__":
    main()
