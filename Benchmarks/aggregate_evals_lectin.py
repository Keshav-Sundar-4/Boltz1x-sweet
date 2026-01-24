import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import sys
from pathlib import Path

# --- CONFIGURATION ---
# Metric Definitions:
# col: CSV Column Name
# thresh: Success Threshold
# less: True if "Lower is Better", False if "Higher is Better"
# label: X-Axis Label
# --- CONFIGURATION ---
METRIC_CONFIG = [
    {"col": "LRMS", "thresh": 2.0, "less": True, "label": "LRMS < 2.0"},
    {"col": "DockQC", "thresh": 0.80, "less": False, "label": "DockQC > 0.80"},
    {"col": "Pocket_LRMS", "thresh": 2.0, "less": True, "label": "Pocket LRMS < 2.0"},
    {"col": "Internal_RMSD", "thresh": 1.0, "less": True, "label": "Sugar-RMSD < 1.0"},
]

# Color Scheme
COLORS = {
    "Boltz-1x oracle": "#188F52",   # Green (Base Oracle)
    "Boltz-1x top-1": "#86E935",    # Light Green (Base Top-1)
    "Boltz-1xg oracle": "#004D80",  # Blue (Glycan Oracle)
    "Boltz-1xg top-1": "#55C2FF",   # Light Blue (Glycan Top-1)
}

def bootstrap_ci(series, n_boot=1000, alpha=0.05):
    """
    Calculates the mean and 95% CI using bootstrapping.
    Returns: mean, lower_bound, upper_bound
    """
    data = np.array(series)
    # Filter out NaNs if any (shouldn't be, but for safety)
    data = data[~np.isnan(data)]
    
    if len(data) < 2:
        val = np.mean(data) if len(data) > 0 else 0.0
        return val, val, val
    
    n = len(data)
    # Vectorized resampling
    indices = np.random.randint(0, n, (n_boot, n))
    resampled_data = data[indices]
    boot_means = np.mean(resampled_data, axis=1)
    
    mean_val = np.mean(data)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return mean_val, lower, upper

def safe_parse_val(val):
    """
    Robust parser:
    - If it's a float/int, return it.
    - If it's a string looking like a list, parse and return MEAN.
    - If it's a string looking like a float, cast it.
    """
    try:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val = val.strip()
            # If it's a list string "[1.0, 2.0]", parse and take mean
            if val.startswith('[') and val.endswith(']'):
                lst = ast.literal_eval(val)
                if lst: return np.mean(lst)
                else: return np.nan
            return float(val)
        return np.nan
    except:
        return np.nan

def get_model_index(model_name):
    """Extracts integer model index from string (e.g., 'Target_model_0' -> 0)."""
    try:
        parts = str(model_name).split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 999 
    except:
        return 999

def process_dataframe(df, tool_name):
    """
    Aggregates metrics per target (Oracle and Top-1) and calculates binary success.
    Handles multiple ligands per target correctly.
    """
    # 1. Clean Column Names
    df.columns = df.columns.str.strip()
    
    # 2. Parse Numeric Values
    cols_to_parse = list(set([m["col"] for m in METRIC_CONFIG]))
    for col in cols_to_parse:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_val)

    results = []
    targets = df["Target"].unique()

    for target in targets:
        t_df = df[df["Target"] == target]
        
        # Build dictionary {model_idx: [row_ligand_1, row_ligand_2, ...]}
        # We assume the rows in the CSV are ordered consistently by ligand
        models = {}
        for _, row in t_df.iterrows():
            idx = get_model_index(row["Model"])
            if idx not in models:
                models[idx] = []
            models[idx].append(row)

        if not models: continue

        # Determine number of ligands for this target (based on max rows found for any model)
        # We take the max to be safe, though ideally all models have same count
        num_ligands = max(len(rows) for rows in models.values())

        # --- Calculate Success for each Metric ---
        for metric_info in METRIC_CONFIG:
            col = metric_info["col"]
            thresh = metric_info["thresh"]
            is_less = metric_info["less"]
            label = metric_info["label"]

            # If column missing, skip
            if col not in t_df.columns:
                continue
            
            # Iterate over every ligand instance
            for i in range(num_ligands):
                
                # 1. TOP-1 (Model 0 or seed-1)
                # ----------------------------
                success_top1 = 0.0
                if 0 in models and i < len(models[0]):
                    val = models[0][i][col]
                    if not np.isnan(val):
                        if is_less:
                            if val < thresh: success_top1 = 1.0
                        else:
                            if val >= thresh: success_top1 = 1.0 # Note: usually inclusive
                
                results.append({
                    "tool": f"{tool_name} top-1",
                    "metric": label,
                    "value": success_top1
                })

                # 2. ORACLE (Best of Available Models for this specific ligand)
                # -------------------------------------------------------------
                # Collect valid values from all models for ligand i
                valid_vals = []
                for m_idx in models:
                    if i < len(models[m_idx]):
                        v = models[m_idx][i][col]
                        if not np.isnan(v):
                            valid_vals.append(v)
                
                success_oracle = 0.0
                if valid_vals:
                    if is_less:
                        best_val = min(valid_vals)
                        if best_val < thresh: success_oracle = 1.0
                    else:
                        best_val = max(valid_vals)
                        if best_val >= thresh: success_oracle = 1.0
                
                results.append({
                    "tool": f"{tool_name} oracle",
                    "metric": label,
                    "value": success_oracle
                })

    return pd.DataFrame(results)

def plot_performance(df, filename):
    """Generates the comparison bar chart with clipped error bars."""
    
    # 1. Bootstrap Statistics
    boot_stats = df.groupby(["tool", "metric"])["value"].apply(bootstrap_ci)
    boot_stats = boot_stats.apply(pd.Series)
    boot_stats.columns = ["mean", "lower", "upper"]
    
    # 2. Arrange Data for Plotting
    desired_tools = [
        "Boltz-1x oracle", "Boltz-1x top-1",
        "Boltz-1xg oracle", "Boltz-1xg top-1"
    ]
    
    # Use exact order from CONFIG
    metric_order = [m["label"] for m in METRIC_CONFIG]
    
    plot_data = boot_stats["mean"].unstack("tool").reindex(metric_order)
    lower_data = boot_stats["lower"].unstack("tool").reindex(metric_order)
    upper_data = boot_stats["upper"].unstack("tool").reindex(metric_order)
    
    # Ensure columns exist (fill 0 if missing)
    for t in desired_tools:
        if t not in plot_data.columns: plot_data[t] = 0.0
        if t not in lower_data.columns: lower_data[t] = 0.0
        if t not in upper_data.columns: upper_data[t] = 0.0

    plot_data = plot_data[desired_tools]
    lower_data = lower_data[desired_tools]
    upper_data = upper_data[desired_tools]
    
    mean_vals = plot_data.values

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(plot_data.index))
    
    bar_spacing = 0.04
    total_width = 0.85
    width = (total_width - (len(desired_tools) - 1) * bar_spacing) / len(desired_tools)
    
    for i, tool in enumerate(desired_tools):
        offsets = x - (total_width - width) / 2 + i * (width + bar_spacing)
        
        tool_means = plot_data[tool].values
        
        # --- Error Bar Clipping Logic ---
        # Calculate raw error lengths
        # lower_err = mean - lower_bound
        # upper_err = upper_bound - mean
        
        # We must clamp bounds to [0, 1] first
        safe_lowers = np.maximum(lower_data.values[:, i], 0.0)
        safe_uppers = np.minimum(upper_data.values[:, i], 1.0)
        
        err_lower = tool_means - safe_lowers
        err_upper = safe_uppers - tool_means
        
        # Safety for tiny floating point errors causing negative lengths
        err_lower = np.maximum(err_lower, 0)
        err_upper = np.maximum(err_upper, 0)
        
        tool_yerr = np.vstack([err_lower, err_upper])
        
        ax.bar(
            offsets,
            tool_means,
            width=width,
            color=COLORS.get(tool, "#333333"),
            label=tool,
            yerr=tool_yerr,
            capsize=3,
            error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"}
        )

    # 4. Styling
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, fontsize=10, weight="bold")
    ax.set_ylabel("Success Rate (Ratio)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Metrics (Top-1 & Oracle)", fontsize=14, pad=20)
    
    # Legend
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), 
              ncols=4, frameon=False, fontsize=10)
    
    # Horizontal grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Graph saved to {filename}")

def main():
    # Input Files
    base_file = "lectinz_base_evals.csv"
    glycan_file = "lectinz_glycan_evals.csv"
    
    print("--- Aggregate Evals Script ---")
    
    # 1. Load Data
    if not Path(base_file).exists() or not Path(glycan_file).exists():
        print(f"Error: Missing input files.\nExpected: {base_file}, {glycan_file}")
        return

    try:
        df_base = pd.read_csv(base_file)
        df_glycan = pd.read_csv(glycan_file)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # 2. Process
    print(f"Processing {base_file} as Boltz-1x...")
    res_base = process_dataframe(df_base, "Boltz-1x")
    
    print(f"Processing {glycan_file} as Boltz-1xg...")
    res_glycan = process_dataframe(df_glycan, "Boltz-1xg")
    
    final_df = pd.concat([res_base, res_glycan], ignore_index=True)
    
    # 3. Plot
    if not final_df.empty:
        out_name = "performance_comparison.pdf"
        print(f"Generating plot: {out_name}")
        plot_performance(final_df, out_name)
        print("Done.")
    else:
        print("No valid results computed.")

if __name__ == "__main__":
    main()
