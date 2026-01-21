import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
METRICS = ["lddt", "bb_lddt", "tm_score", "rmsd"]

def compute_af3_metrics(preds, evals, name):
    metrics = {}
    top_model = None
    top_confidence = -1000
    
    for model_id in range(5):
        # Path: seed-1_sample-{id}/summary_confidences.json
        confidence_file = Path(preds) / f"seed-1_sample-{model_id}" / "summary_confidences.json"
        
        if confidence_file.exists():
            with confidence_file.open("r") as f:
                confidence_data = json.load(f)
                confidence = confidence_data.get("ranking_score", -1000)
                if confidence > top_confidence:
                    top_model = model_id
                    top_confidence = confidence
        else:
            if top_model is None: top_model = 0

        # Load Eval
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        if eval_file.exists():
            with eval_file.open("r") as f:
                eval_data = json.load(f)
                
                # Standard Metrics
                for metric_name in METRICS:
                    if metric_name in eval_data and eval_data[metric_name] is not None:
                        metrics.setdefault(metric_name, []).append(eval_data[metric_name])
                
                # RMSD < 7.5 Calculation (RNA specific)
                if "rmsd" in eval_data and eval_data["rmsd"] is not None:
                    val = eval_data["rmsd"]
                    # 1.0 if Success, 0.0 if Failure
                    score = 1.0 if val < 7.5 else 0.0
                    metrics.setdefault("rmsd<7.5", []).append(score)

    if not metrics: return {}

    if top_model is None: top_model = 0
    
    # Oracle logic: Min for raw RMSD, Max for everything else (including rmsd<7.5)
    oracle = {k: min(v) if "rmsd" in k and "<" not in k else max(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] if top_model < len(v) else v[0] for k, v in metrics.items()}

    results = {}
    for k in metrics:
        if k.startswith("len_"): continue
        results[k] = {"oracle": oracle[k], "top1": top1[k]}
    return results

def compute_chai_metrics(preds, evals, name):
    metrics = {}
    top_model = None
    top_confidence = 0
    
    for model_id in range(5):
        # Path: scores.model_idx_{id}.npz
        confidence_file = Path(preds) / f"scores.model_idx_{model_id}.npz"
        
        if confidence_file.exists():
            try:
                confidence_data = np.load(confidence_file)
                confidence = confidence_data["aggregate_score"].item()
                if confidence > top_confidence:
                    top_model = model_id
                    top_confidence = confidence
            except: pass
        else:
             if top_model is None: top_model = 0

        # Load Eval
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        if eval_file.exists():
            with eval_file.open("r") as f:
                eval_data = json.load(f)
                
                # Standard Metrics
                for metric_name in METRICS:
                    if metric_name in eval_data and eval_data[metric_name] is not None:
                        metrics.setdefault(metric_name, []).append(eval_data[metric_name])

                # RMSD < 7.5 Calculation
                if "rmsd" in eval_data and eval_data["rmsd"] is not None:
                    val = eval_data["rmsd"]
                    score = 1.0 if val < 7.5 else 0.0
                    metrics.setdefault("rmsd<7.5", []).append(score)

    if not metrics: return {}
    if top_model is None: top_model = 0
    
    oracle = {k: min(v) if "rmsd" in k and "<" not in k else max(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] if top_model < len(v) else v[0] for k, v in metrics.items()}

    results = {}
    for k in metrics:
        results[k] = {"oracle": oracle[k], "top1": top1[k]}
    return results

def compute_boltz_metrics(preds, evals, name):
    metrics = {}
    top_model = None
    top_confidence = 0
    
    target_folder = Path(preds) 
    
    for model_id in range(5):
        candidates = [
            target_folder / f"confidence_{name}_model_{model_id}.json",
            target_folder / f"confidence_{name.lower()}_model_{model_id}.json",
            target_folder / f"confidence_{name.upper()}_model_{model_id}.json",
            target_folder / f"confidence_{name.capitalize()}_model_{model_id}.json"
        ]
        
        confidence_file = None
        for c in candidates:
            if c.exists():
                confidence_file = c
                break
        
        if confidence_file:
            try:
                with confidence_file.open("r") as f:
                    confidence_data = json.load(f)
                    confidence = confidence_data.get("confidence_score", 0)
                    if confidence > top_confidence:
                        top_model = model_id
                        top_confidence = confidence
            except: pass
        else:
            if top_model is None: top_model = 0

        # Load Eval
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        if eval_file.exists():
            with eval_file.open("r") as f:
                eval_data = json.load(f)
                
                # Standard Metrics
                for metric_name in METRICS:
                    if metric_name in eval_data and eval_data[metric_name] is not None:
                        metrics.setdefault(metric_name, []).append(eval_data[metric_name])

                # RMSD < 7.5 Calculation
                if "rmsd" in eval_data and eval_data["rmsd"] is not None:
                    val = eval_data["rmsd"]
                    score = 1.0 if val < 7.5 else 0.0
                    metrics.setdefault("rmsd<7.5", []).append(score)

    if not metrics: return {}
    if top_model is None: top_model = 0
    
    oracle = {k: min(v) if "rmsd" in k and "<" not in k else max(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] if top_model < len(v) else v[0] for k, v in metrics.items()}

    results = {}
    for k in metrics:
        results[k] = {"oracle": oracle[k], "top1": top1[k]}
    return results

def bootstrap_ci(series, n_boot=1000, alpha=0.05):
    if len(series) < 2: return np.mean(series), np.mean(series), np.mean(series)
    n = len(series)
    boot_means = []
    for _ in range(n_boot):
        sample = series.sample(n, replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    mean_val = np.mean(series)
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return mean_val, lower, upper

def plot_data(desired_metrics, df, dataset, filename):
    desired_tools = [
        "AF3 oracle", "AF3 top-1",
        "Chai-1 oracle", "Chai-1 top-1",
        "Boltz-1 oracle", "Boltz-1 top-1",
        "Boltz-1xg oracle", "Boltz-1xg top-1"
    ]
    
    filtered_df = df[df["tool"].isin(desired_tools) & df["metric"].isin(desired_metrics)]

    if filtered_df.empty:
        print("No data found to plot.")
        return

    boot_stats = filtered_df.groupby(["tool", "metric"])["value"].apply(bootstrap_ci)
    boot_stats = boot_stats.apply(pd.Series)
    boot_stats.columns = ["mean", "lower", "upper"]

    plot_data = boot_stats["mean"].unstack("tool").reindex(desired_metrics)
    lower_data = boot_stats["lower"].unstack("tool").reindex(desired_metrics)
    upper_data = boot_stats["upper"].unstack("tool").reindex(desired_metrics)

    # Filter keys to ensure they exist in the data
    valid_tools = [t for t in desired_tools if t in plot_data.columns]
    plot_data = plot_data[valid_tools]
    lower_data = lower_data[valid_tools]
    upper_data = upper_data[valid_tools]

    renaming = {
        "lddt": "Mean LDDT",
        "bb_lddt": "Backbone LDDT",
        "tm_score": "TM-Score",
        "rmsd<7.5": "RMSD < 7.5Å" 
    }
    plot_data = plot_data.rename(index=renaming)
    lower_data = lower_data.rename(index=renaming)
    upper_data = upper_data.rename(index=renaming)
    mean_vals = plot_data.values

    # Colors
    tool_colors = [
        "#994C00",  # AF3 oracle
        "#FFB55A",  # AF3 top-1
        "#931652",  # Chai-1 oracle
        "#FC8AD9",  # Chai-1 top-1
        "#188F52",  # Boltz-1 oracle (Green)
        "#86E935",  # Boltz-1 top-1 (Light Green)
        "#004D80",  # Boltz-1xg oracle (Blue)
        "#55C2FF",  # Boltz-1xg top-1 (Light Blue)
    ]
    
    # Slice colors for valid tools
    current_colors = tool_colors[:len(valid_tools)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_data.index))
    
    bar_spacing = 0.02
    total_width = 0.8
    width = (total_width - (len(valid_tools) - 1) * bar_spacing) / len(valid_tools)

    for i, tool in enumerate(valid_tools):
        offsets = x - (total_width - width) / 2 + i * (width + bar_spacing)
        
        tool_means = plot_data[tool].values
        tool_yerr_lower = mean_vals[:, i] - lower_data.values[:, i]
        tool_yerr_upper = upper_data.values[:, i] - mean_vals[:, i]
        tool_yerr = np.vstack([tool_yerr_lower, tool_yerr_upper])

        ax.bar(
            offsets,
            tool_means,
            width=width,
            color=current_colors[i],
            label=tool,
            yerr=tool_yerr,
            capsize=3,
            error_kw={"elinewidth": 1}
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title(f"Performances on {dataset} with 95% CI (Bootstrap)")
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), 
              ncols=4, frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Graph saved to {filename}")

def main():
    base_folder = Path("casp15_benchmark_RNA") 
    
    tasks = {
        "af3": {
            "preds": base_folder / "af3_RNA",
            "evals": base_folder / "evals_af3",
            "func": compute_af3_metrics,
            "label": "AF3"
        },
        "chai": {
            "preds": base_folder / "chai_RNA",
            "evals": base_folder / "evals_chai",
            "func": compute_chai_metrics,
            "label": "Chai-1"
        },
        "boltz": {
            "preds": base_folder / "casp15_base_RNA_results",
            "evals": base_folder / "evals_base",
            "func": compute_boltz_metrics,
            "label": "Boltz-1"
        },
        "boltzx": {
            "preds": base_folder / "casp15_glycan_RNA_results",
            "evals": base_folder / "evals_glycan",
            "func": compute_boltz_metrics,
            "label": "Boltz-1xg"
        }
    }
    
    def get_targets(p):
        if not p.exists(): return set()
        return {x.name for x in p.iterdir() if x.is_dir() and not x.name.startswith(".")}

    target_sets = [get_targets(t["preds"]) for t in tasks.values()]
    target_sets = [s for s in target_sets if s]
    
    if not target_sets:
        print("No targets found.")
        return

    common_lower = set.intersection(*[{t.lower() for t in s} for s in target_sets])
    
    ref_set = target_sets[0]
    real_names = {}
    for t in ref_set:
        if t.lower() in common_lower:
            real_names[t.lower()] = t
    
    print(f"Common targets: {len(real_names)}")
    
    results_list = []
    
    for t_lower in tqdm(real_names):
        name = real_names[t_lower]
        
        for task_key, config in tasks.items():
            try:
                res = config["func"](config["preds"] / name, config["evals"], name)
                
                for m, v in res.items():
                    results_list.append({"tool": f"{config['label']} oracle", "target": name, "metric": m, "value": v["oracle"]})
                    results_list.append({"tool": f"{config['label']} top-1", "target": name, "metric": m, "value": v["top1"]})
            except Exception as e:
                pass

    if not results_list:
        print("No results compiled. Check paths.")
        return

    df = pd.DataFrame(results_list)
    df.to_csv("comparison_results.csv", index=False)
    
    # Plotting: lddt, bb_lddt, tm_score, and the new rmsd<7.5
    desired_metrics = ["lddt", "bb_lddt", "tm_score", "rmsd<7.5"]
    plot_data(desired_metrics, df, "CASP15 RNA", "performances_casp15_rna.pdf")

if __name__ == "__main__":
    main()
