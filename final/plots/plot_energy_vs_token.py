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

def get_avg_reasoning_tokens(file_path):
    """Extracts the average length of reasoning tokens (cot_length) from the log."""
    if not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Search for the exact line: "output avg_cot_length = 225.89841"
            match = re.search(r"output avg_cot_length = ([\d\.]+)", content)
            if match: 
                return float(match.group(1))
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
    all_data = {} 
    
    global_min_eng, global_max_eng = float('inf'), float('-inf')
    global_min_tok, global_max_tok = float('inf'), float('-inf')

    print("Reading data across 3 runs to calculate global scales...")
    
    for size in MODELS:
        all_data[size] = {'energies': [], 'reasoning_tokens': [], 'ratios': []}
        
        for ratio in RATIOS:
            tok_vals = []
            eng_vals = []
            
            # Average over 3 runs
            for d in BASE_DIRS:
                dir_path = os.path.join(d, size, BENCHMARK, f"cr_{ratio}")
                avg_tok = get_avg_reasoning_tokens(os.path.join(dir_path, "run_log.txt"))
                joules = get_total_energy(os.path.join(dir_path, "metrics_watt.json"))
                
                if avg_tok is not None and joules is not None:
                    tok_vals.append(avg_tok)
                    eng_vals.append(joules / 1000.0) # Convert to kJ
            
            if tok_vals and eng_vals:
                final_avg_tok = np.mean(tok_vals)
                final_avg_eng = np.mean(eng_vals)
                
                all_data[size]['energies'].append(final_avg_eng)
                all_data[size]['reasoning_tokens'].append(final_avg_tok)
                all_data[size]['ratios'].append(ratio)
                
                # Update global min/max
                if final_avg_eng < global_min_eng: global_min_eng = final_avg_eng
                if final_avg_eng > global_max_eng: global_max_eng = final_avg_eng
                if final_avg_tok < global_min_tok: global_min_tok = final_avg_tok
                if final_avg_tok > global_max_tok: global_max_tok = final_avg_tok

    # Handle cases with no data
    if global_min_eng == float('inf'):
        print("No data found to create the graph. Please check paths and files.")
        return

    # Calculate Margins (10% padding so points don't touch borders)
    eng_padding = (global_max_eng - global_min_eng) * 0.1
    tok_padding = (global_max_tok - global_min_tok) * 0.1
    
    # Final axis limits
    xlims = (global_min_tok - tok_padding, global_max_tok + tok_padding)
    ylims = (global_min_eng - eng_padding, global_max_eng + eng_padding)

    print(f"Global Scales Set -> Avg CoT Length: {xlims[0]:.1f}-{xlims[1]:.1f} | Energy: {ylims[0]:.1f}-{ylims[1]:.1f} kJ")

    # --- PHASE 2: PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, size in enumerate(MODELS):
        ax = axes[i]
        data = all_data[size]
        
        if data['energies']:
            color = COLORS[size]
            energies = data['energies']
            reasoning_tokens = data['reasoning_tokens']
            labels = data['ratios']

            line_energies = []
            line_tokens = []

            # Separate ratio 1.0 from the rest of the points
            for j, ratio in enumerate(labels):
                if ratio == "1.0":
                    # Draw only the point (Star) for ratio 1.0 (without line connecting it)
                    ax.scatter(reasoning_tokens[j], energies[j], color=color, s=400, marker='*', edgecolors='k', zorder=15)
                else:
                    # Accumulate the remaining points to draw the line
                    line_tokens.append(reasoning_tokens[j])
                    line_energies.append(energies[j])

            # Draw the line and regular points for the other ratios (0.9 to 0.5)
            if line_tokens:
                # Plot Line
                ax.plot(line_tokens, line_energies, color=color, linestyle='--', alpha=0.5, linewidth=2)
                # Plot Regular Points
                marker = 'D' if size == '14B' else 'o'
                ax.scatter(line_tokens, line_energies, color=color, s=150, label=f"{size}", marker=marker, edgecolors='k', zorder=10)
            
            # Ratio Labels (apply to all plotted points, including the star)
            for j, txt in enumerate(labels):
                y_offset = 12
                if size == "14B" and j % 2 == 1: y_offset = -18
                elif txt == "0.9": y_offset = -15
                
                ax.annotate(f"r{txt}", (reasoning_tokens[j], energies[j]), 
                           xytext=(0, y_offset), textcoords='offset points',
                           ha='center', fontsize=9, fontweight='bold', color='black')

        # --- APPLY UNIFIED SCALES ---
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        
        # Styling
        ax.set_title(f"Model: {size} (3-Run Avg)", fontsize=16, fontweight='bold', color='#333333')
        ax.set_xlabel("Average CoT (Reasoning) Length", fontsize=12)
        
        # Show Y label only on the first plot for cleanliness
        if i == 0:
            ax.set_ylabel("Avg Energy (kJ)", fontsize=12)
        
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_file = "graph_reasoning_vs_energy_avg.png"
    plt.savefig(output_file, dpi=300)
    print("\n" + "="*55)
    print(f"Reasoning Tokens vs Energy graph saved: {output_file}")
    print("="*55)

if __name__ == "__main__":
    main()
