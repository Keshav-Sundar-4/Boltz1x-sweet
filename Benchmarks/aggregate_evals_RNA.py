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
                for metric_name in METRICS:
                    if metric_name in eval_data:
                        metrics.setdefault(metric_name, []).append(eval_data[metric_name])

                if "dockq" in eval_data and eval_data["dockq"] is not None:
                    valid_dockq = [float(v > 0.23) for v in eval_data["dockq"] if v is not None]
                    if valid_dockq:
                        metrics.setdefault("dockq_>0.23", []).append(np.mean(valid_dockq))
                        metrics.setdefault("len_dockq_", []).append(len(valid_dockq))

        eval_file_lig = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        if eval_file_lig.exists():
            with eval_file_lig.open("r") as f:
                eval_data = json.load(f)
                if "lddt_pli" in eval_data:
                    scores = [x["score"] for x in eval_data["lddt_pli"].get("assigned_scores", [])]
                    for _ in eval_data["lddt_pli"].get("model_ligand_unassigned_reason", {}).items():
                        scores.append(0)
                    if scores:
                        metrics.setdefault("lddt_pli", []).append(np.mean(scores))
                        metrics.setdefault("len_lddt_pli", []).append(len(scores))

                if "rmsd" in eval_data:
                    scores = [x["score"] for x in eval_data["rmsd"].get("assigned_scores", [])]
                    for _ in eval_data["rmsd"].get("model_ligand_unassigned_reason", {}).items():
                        scores.append(100)
                    if scores:
                        metrics.setdefault("rmsd<2", []).append(np.mean([x < 2.0 for x in scores]))
                        metrics.setdefault("len_rmsd", []).append(len(scores))

    if not metrics: return {}

    if top_model is None: top_model = 0
    oracle = {k: min(v) if "rmsd" in k and "rmsd<" not in k else max(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] if top_model < len(v) else v[0] for k, v in metrics.items()}

    results = {}
    for k in metrics:
        if k.startswith("len_"): continue
        # Find corresponding length metric
        length_key = "len_lddt_pli" if k == "lddt_pli" else "len_rmsd" if "rmsd" in k and "<" in k else "len_dockq_" if "dockq" in k else None
        l = metrics[length_key][0] if length_key and length_key in metrics else 1
        
        results[k] = {"oracle": oracle[k], "top1": top1[k], "len": l}
    return results

def compute_chai_metrics(preds, evals, name):
    metrics = {}
    top_model = None
    top_confidence = 0
    
    for model_id in range(5):
        # Path: pred.model_idx_{id}.npz or similar. Original used .npz
        # Assuming folder/scores.model_idx_{id}.npz as per original script logic
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
                for metric_name in METRICS:
                    if metric_name in eval_data:
                        metrics.setdefault(metric_name, []).append(eval_data[metric_name])

                if "dockq" in eval_data and eval_data["dockq"] is not None:
                    valid_dockq = [float(v > 0.23) for v in eval_data["dockq"] if v is not None]
                    if valid_dockq:
                        metrics.setdefault("dockq_>0.23", []).append(np.mean(valid_dockq))
                        metrics.setdefault("len_dockq_", []).append(len(valid_dockq))

        eval_file_lig = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        if eval_file_lig.exists():
            with eval_file_lig.open("r") as f:
                eval_data = json.load(f)
                # reuse same logic as AF3 for ligand
                if "lddt_pli" in eval_data:
                    scores = [x["score"] for x in eval_data["lddt_pli"].get("assigned_scores", [])]
                    for _ in eval_data["lddt_pli"].get("model_ligand_unassigned_reason", {}).items():
                        scores.append(0)
                    if scores:
                        metrics.setdefault("lddt_pli", []).append(np.mean(scores))
                        metrics.setdefault("len_lddt_pli", []).append(len(scores))

                if "rmsd" in eval_data:
                    scores = [x["score"] for x in eval_data["rmsd"].get("assigned_scores", [])]
                    for _ in eval_data["rmsd"].get("model_ligand_unassigned_reason", {}).items():
                        scores.append(100)
                    if scores:
                        metrics.setdefault("rmsd<2", []).append(np.mean([x < 2.0 for x in scores]))
                        metrics.setdefault("len_rmsd", []).append(len(scores))

    if not metrics: return {}
    if top_model is None: top_model = 0
    oracle = {k: min(v) if "rmsd" in k and "rmsd<" not in k else max(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] if top_model < len(v) else v[0] for k, v in metrics.items()}

    results = {}
    for k in metrics:
        if k.startswith("len_"): continue
        length_key = "len_lddt_pli" if k == "lddt_pli" else "len_rmsd" if "rmsd" in k and "<" in k else "len_dockq_" if "dockq" in k else None
        l = metrics[length_key][0] if length_key and length_key in metrics else 1
        results[k] = {"oracle": oracle[k], "top1": top1[k], "len": l}
    return results

def compute_boltz_metrics(preds, evals, name):
    metrics = {}
    top_model = None
    top_confidence = 0
    
    # preds is the folder containing target folders (e.g. data/T1187/)
    # We need to look inside data/T1187/confidence...
    
    # Find the specific folder name (case insensitive matching might be needed elsewhere, 
    # but here preds is the direct target folder path from the loop)
    target_folder = Path(preds) 
    
    for model_id in range(5):
        # Updated Boltz Structure: confidence file is inside the target folder
        # Naming: confidence_{target}_model_{id}.json
        # We try a few naming variations
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
                for metric_name in METRICS:
                    if metric_name in eval_data:
                        metrics.setdefault(metric_name, []).append(eval_data[metric_name])

                if "dockq" in eval_data and eval_data["dockq"] is not None:
                    valid_dockq = [float(v > 0.23) for v in eval_data["dockq"] if v is not None]
                    if valid_dockq:
                        metrics.setdefault("dockq_>0.23", []).append(np.mean(valid_dockq))
                        metrics.setdefault("len_dockq_", []).append(len(valid_dockq))

        eval_file_lig = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        if eval_file_lig.exists():
            with eval_file_lig.open("r") as f:
                eval_data = json.load(f)
                if "lddt_pli" in eval_data:
                    scores = [x["score"] for x in eval_data["lddt_pli"].get("assigned_scores", [])]
                    for _ in eval_data["lddt_pli"].get("model_ligand_unassigned_reason", {}).items():
                        scores.append(0)
                    if scores:
                        metrics.setdefault("lddt_pli", []).append(np.mean(scores))
                        metrics.setdefault("len_lddt_pli", []).append(len(scores))

                if "rmsd" in eval_data:
                    scores = [x["score"] for x in eval_data["rmsd"].get("assigned_scores", [])]
                    for _ in eval_data["rmsd"].get("model_ligand_unassigned_reason", {}).items():
                        scores.append(100)
                    if scores:
                        metrics.setdefault("rmsd<2", []).append(np.mean([x < 2.0 for x in scores]))
                        metrics.setdefault("len_rmsd", []).append(len(scores))

    if not metrics: return {}
    if top_model is None: top_model = 0
    oracle = {k: min(v) if "rmsd" in k and "rmsd<" not in k else max(v) for k, v in metrics.items()}
    top1 = {k: v[top_model] if top_model < len(v) else v[0] for k, v in metrics.items()}

    results = {}
    for k in metrics:
        if k.startswith("len_"): continue
        length_key = "len_lddt_pli" if k == "lddt_pli" else "len_rmsd" if "rmsd" in k and "<" in k else "len_dockq_" if "dockq" in k else None
        l = metrics[length_key][0] if length_key and length_key in metrics else 1
        results[k] = {"oracle": oracle[k], "top1": top1[k], "len": l}
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
    # Tool Naming: Green = Boltz-1, Blue = Boltz-1xg
    desired_tools = [
        "AF3 oracle", "AF3 top-1",
        "Chai-1 oracle", "Chai-1 top-1",
        "Boltz-1 oracle", "Boltz-1 top-1",
        "Boltz-1xg oracle", "Boltz-1xg top-1"
    ]
    
    filtered_df = df[df["tool"].isin(desired_tools) & df["metric"].isin(desired_metrics)]

    boot_stats = filtered_df.groupby(["tool", "metric"])["value"].apply(bootstrap_ci)
    boot_stats = boot_stats.apply(pd.Series)
    boot_stats.columns = ["mean", "lower", "upper"]

    plot_data = boot_stats["mean"].unstack("tool").reindex(desired_metrics)
    lower_data = boot_stats["lower"].unstack("tool").reindex(desired_metrics)
    upper_data = boot_stats["upper"].unstack("tool").reindex(desired_metrics)

    # Reorder columns
    plot_data = plot_data[desired_tools]
    lower_data = lower_data[desired_tools]
    upper_data = upper_data[desired_tools]

    # Rename metrics for plotting
    renaming = {
        "lddt_pli": "Mean LDDT-PLI",
        "rmsd<2": "L-RMSD < 2A",
        "lddt": "Mean LDDT",
        "dockq_>0.23": "DockQ > 0.23",
        "physical validity": "Physical Validity"
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

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_data.index))
    
    bar_spacing = 0.02
    total_width = 0.8
    width = (total_width - (len(desired_tools) - 1) * bar_spacing) / len(desired_tools)

    for i, tool in enumerate(desired_tools):
        offsets = x - (total_width - width) / 2 + i * (width + bar_spacing)
        
        tool_means = plot_data[tool].values
        tool_yerr_lower = mean_vals[:, i] - lower_data.values[:, i]
        tool_yerr_upper = upper_data.values[:, i] - mean_vals[:, i]
        tool_yerr = np.vstack([tool_yerr_lower, tool_yerr_upper])

        ax.bar(
            offsets,
            tool_means,
            width=width,
            color=tool_colors[i],
            label=tool,
            yerr=tool_yerr,
            capsize=3,
            error_kw={"elinewidth": 1}
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title(f"Performances on {dataset} with 95% CI (Bootstrap)")
    
    # Legend setup
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), 
              ncols=4, frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Graph saved to {filename}")


def main():
    # --- UPDATE THESE PATHS ---
    base_folder = Path("casp15_benchmark_RNA") # Root based on screenshot
    
    # Structure: Outputs (Predictions) and Evals
    # Mapped exactly to your screenshot structure
    paths = {
        "chai":   {"preds": base_folder / "chai_RNA",                  "evals": base_folder / "evals_chai"},
        "af3":    {"preds": base_folder / "af3_RNA",                   "evals": base_folder / "evals_af3"},
        "boltz":  {"preds": base_folder / "casp15_base_RNA_results",   "evals": base_folder / "evals_base"},
        "boltzx": {"preds": base_folder / "casp15_glycan_RNA_results", "evals": base_folder / "evals_glycan"}
    }
        
    # Gather common targets
    def get_targets(p, fmt):
        if not Path(p).exists(): return set()
        if fmt == "boltz":
            # Folder-based target names
            return {x.name for x in Path(p).iterdir() if x.is_dir() and not x.name.startswith(".")}
        else:
            return {x.name for x in Path(p).iterdir() if x.is_dir() and not x.name.startswith(".")}

    # Get sets of targets
    targets_af3 = get_targets(paths["af3"]["preds"], "af3")
    targets_chai = get_targets(paths["chai"]["preds"], "chai")
    targets_boltz = get_targets(paths["boltz"]["preds"], "boltz")
    targets_boltzx = get_targets(paths["boltzx"]["preds"], "boltz")
    
    # Find Intersection (Case insensitive handling might be needed here if folder naming varies)
    # Assuming lower case standardization for intersection
    common = set(x.lower() for x in targets_af3) & set(x.lower() for x in targets_chai) & \
             set(x.lower() for x in targets_boltz) & set(x.lower() for x in targets_boltzx)
    
    # Map back to real folder names
    real_names = {}
    for t in targets_af3: real_names[t.lower()] = t # Prefer one source for name
    
    print(f"Common targets: {len(common)}")
    
    results_list = []
    
    for t_lower in tqdm(common):
        name = real_names[t_lower]
        
        # 1. AF3
        try:
            res = compute_af3_metrics(Path(paths["af3"]["preds"]) / name, paths["af3"]["evals"], name)
            for m, v in res.items():
                results_list.append({"tool": "AF3 oracle", "target": name, "metric": m, "value": v["oracle"]})
                results_list.append({"tool": "AF3 top-1", "target": name, "metric": m, "value": v["top1"]})
        except Exception as e: print(f"Err AF3 {name}: {e}")

        # 2. Chai
        try:
            res = compute_chai_metrics(Path(paths["chai"]["preds"]) / name, paths["chai"]["evals"], name)
            for m, v in res.items():
                results_list.append({"tool": "Chai-1 oracle", "target": name, "metric": m, "value": v["oracle"]})
                results_list.append({"tool": "Chai-1 top-1", "target": name, "metric": m, "value": v["top1"]})
        except Exception as e: print(f"Err Chai {name}: {e}")

        # 3. Boltz-1 (Base)
        try:
            # Note: For Boltz, preds path is the root folder containing target subfolders
            res = compute_boltz_metrics(Path(paths["boltz"]["preds"]) / name, paths["boltz"]["evals"], name)
            for m, v in res.items():
                results_list.append({"tool": "Boltz-1 oracle", "target": name, "metric": m, "value": v["oracle"]})
                results_list.append({"tool": "Boltz-1 top-1", "target": name, "metric": m, "value": v["top1"]})
        except Exception as e: print(f"Err Boltz {name}: {e}")

        # 4. Boltz-1xg (Glycan)
        try:
            res = compute_boltz_metrics(Path(paths["boltzx"]["preds"]) / name, paths["boltzx"]["evals"], name)
            for m, v in res.items():
                results_list.append({"tool": "Boltz-1xg oracle", "target": name, "metric": m, "value": v["oracle"]})
                results_list.append({"tool": "Boltz-1xg top-1", "target": name, "metric": m, "value": v["top1"]})
        except Exception as e: print(f"Err BoltzX {name}: {e}")

    if not results_list:
        print("No results compiled. Check paths.")
        return

    df = pd.DataFrame(results_list)
    df.to_csv("comparison_results.csv", index=False)
    
    desired_metrics = ["lddt", "dockq_>0.23", "lddt_pli", "rmsd<2"]
    plot_data(desired_metrics, df, "CASP15", "performances_casp15.pdf")

if __name__ == "__main__":
    main()
