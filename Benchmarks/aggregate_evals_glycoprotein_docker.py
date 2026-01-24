import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

METRICS = ["lddt", "bb_lddt", "tm_score", "rmsd"]

def compute_boltz_metrics(preds, evals, name):
    metrics = {}
    top_model = None
    top_confidence = -1000  # Initialized lower to capture any valid score
    
    for model_id in range(5):
        # Load confidence file
        # Expects structure: TargetFolder/confidence_TargetName_model_0.json
        confidence_file = (
            Path(preds) / f"confidence_{Path(preds).name}_model_{model_id}.json"
        )
        
        # If confidence file missing, skip model
        if not confidence_file.exists():
            continue

        with confidence_file.open("r") as f:
            confidence_data = json.load(f)
            confidence = confidence_data["confidence_score"]
            if confidence > top_confidence:
                top_model = model_id
                top_confidence = confidence

        # Load eval file
        eval_file = Path(evals) / f"{name}_model_{model_id}.json"
        
        if not eval_file.exists():
            continue

        with eval_file.open("r") as f:
            eval_data = json.load(f)
            for metric_name in METRICS:
                if metric_name in eval_data:
                    metrics.setdefault(metric_name, []).append(eval_data[metric_name])
            
            if "dockq" in eval_data and eval_data["dockq"] is not None:
                metrics.setdefault("dockq_>0.23", []).append(
                    np.mean([float(v > 0.23) for v in eval_data["dockq"] if v is not None])
                )
                metrics.setdefault("dockq_>0.49", []).append(
                    np.mean([float(v > 0.49) for v in eval_data["dockq"] if v is not None])
                )
                metrics.setdefault("len_dockq_", []).append(
                    len([v for v in eval_data["dockq"] if v is not None])
                )

        eval_file = Path(evals) / f"{name}_model_{model_id}_ligand.json"
        if eval_file.exists():
            with eval_file.open("r") as f:
                eval_data = json.load(f)
                if "lddt_pli" in eval_data:
                    lddt_plis = [x["score"] for x in eval_data["lddt_pli"]["assigned_scores"]]
                    for _ in eval_data["lddt_pli"]["model_ligand_unassigned_reason"].items():
                        lddt_plis.append(0)
                    if lddt_plis:
                        lddt_pli = np.mean(lddt_plis)
                        metrics.setdefault("lddt_pli", []).append(lddt_pli)
                        metrics.setdefault("len_lddt_pli", []).append(len(lddt_plis))
                
                if "rmsd" in eval_data:
                    rmsds = [x["score"] for x in eval_data["rmsd"]["assigned_scores"]]
                    for _ in eval_data["rmsd"]["model_ligand_unassigned_reason"].items():
                        rmsds.append(100)
                    if rmsds:
                        rmsd2 = np.mean([x < 2.0 for x in rmsds])
                        rmsd5 = np.mean([x < 5.0 for x in rmsds])
                        metrics.setdefault("rmsd<2", []).append(rmsd2)
                        metrics.setdefault("rmsd<5", []).append(rmsd5)
                        metrics.setdefault("len_rmsd", []).append(len(rmsds))
    
    if not metrics or top_model is None:
        return None

    # Get oracle
    oracle = {k: min(v) if k == "rmsd" else max(v) for k, v in metrics.items()}
    avg = {k: sum(v) / len(v) for k, v in metrics.items()}
    # Ensure top_model index exists in list (it corresponds to the order appended)
    # Note: metrics lists correspond to the order files were processed (0..4). 
    # If a model was skipped, indices shift. 
    # However, standard workflow processes 0-4 sequentially. 
    # top_model is the index 0-4. If 0-4 are all present, direct indexing works.
    top1 = {k: v[top_model] for k, v in metrics.items() if len(v) > top_model}

    results = {}
    for metric_name in metrics:
        if metric_name.startswith("len_"):
            continue
        
        # Length handling logic
        if metric_name == "lddt_pli":
            l = metrics.get("len_lddt_pli", [0])[0]
        elif metric_name == "rmsd<2" or metric_name == "rmsd<5":
            l = metrics.get("len_rmsd", [0])[0]
        elif metric_name == "dockq_>0.23" or metric_name == "dockq_>0.49":
            l = metrics.get("len_dockq_", [0])[0]
        else:
            l = 1
            
        if metric_name in top1:
             results[metric_name] = {
                "oracle": oracle[metric_name],
                "average": avg[metric_name],
                "top1": top1[metric_name],
                "len": l,
            }
    return results

def eval_models(
    smiles_preds,
    smiles_evals,
    iupac_preds,
    iupac_evals,
):
    # Load folders
    smiles_preds_names = {x.name.lower(): x for x in Path(smiles_preds).iterdir() if x.is_dir()}
    iupac_preds_names = {x.name.lower(): x for x in Path(iupac_preds).iterdir() if x.is_dir()}

    print("SMILES (Boltz-1x) preds:", len(smiles_preds_names))
    print("IUPAC (Boltz-1xg) preds:", len(iupac_preds_names))

    common = set(smiles_preds_names.keys()) & set(iupac_preds_names.keys())
    print("Common Targets:", len(common))

    results = []
    
    for name in tqdm(common):
        # 1. SMILES (Boltz-1x) - Green Bar
        try:
            smiles_res = compute_boltz_metrics(
                smiles_preds_names[name],
                smiles_evals,
                name,
            )
        except Exception as e:
            print(f"Error evaluating SMILES {name}: {e}")
            smiles_res = None

        # 2. IUPAC (Boltz-1xg) - Blue Bar
        try:
            iupac_res = compute_boltz_metrics(
                iupac_preds_names[name],
                iupac_evals,
                name,
            )
        except Exception as e:
            print(f"Error evaluating IUPAC {name}: {e}")
            iupac_res = None

        if smiles_res and iupac_res:
            for metric_name in smiles_res:
                if metric_name in iupac_res:
                    # Consistency check on lengths (optional but good)
                    if smiles_res[metric_name]["len"] == iupac_res[metric_name]["len"]:
                        
                        # GREEN BAR DATA (Boltz-1x)
                        results.append({
                            "tool": "Boltz-1x oracle",
                            "target": name,
                            "metric": metric_name,
                            "value": smiles_res[metric_name]["oracle"],
                        })
                        results.append({
                            "tool": "Boltz-1x top-1",
                            "target": name,
                            "metric": metric_name,
                            "value": smiles_res[metric_name]["top1"],
                        })

                        # BLUE BAR DATA (Boltz-1xg)
                        results.append({
                            "tool": "Boltz-1xg oracle",
                            "target": name,
                            "metric": metric_name,
                            "value": iupac_res[metric_name]["oracle"],
                        })
                        results.append({
                            "tool": "Boltz-1xg top-1",
                            "target": name,
                            "metric": metric_name,
                            "value": iupac_res[metric_name]["top1"],
                        })

    df = pd.DataFrame(results)
    return df

def bootstrap_ci(series, n_boot=1000, alpha=0.05):
    """
    Compute 95% bootstrap confidence intervals for the mean of 'series'.
    """
    if len(series) == 0:
        return np.nan, np.nan, np.nan
        
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

def plot_data(desired_tools, desired_metrics, df, dataset, filename):
    filtered_df = df[
        df["tool"].isin(desired_tools) & df["metric"].isin(desired_metrics)
    ]
    
    if filtered_df.empty:
        print("No data to plot!")
        return

    # Apply bootstrap
    boot_stats = filtered_df.groupby(["tool", "metric"])["value"].apply(bootstrap_ci)
    boot_stats = boot_stats.apply(pd.Series)
    boot_stats.columns = ["mean", "lower", "upper"]
    
    # Unstack
    plot_data = boot_stats["mean"].unstack("tool")
    plot_data = plot_data.reindex(desired_metrics)
    
    lower_data = boot_stats["lower"].unstack("tool")
    lower_data = lower_data.reindex(desired_metrics)
    
    upper_data = boot_stats["upper"].unstack("tool")
    upper_data = upper_data.reindex(desired_metrics)
    
    # Specific Order Requested
    tool_order = [
        "Boltz-1x oracle",  # Green
        "Boltz-1x top-1",   # Light Green
        "Boltz-1xg oracle", # Blue
        "Boltz-1xg top-1",  # Light Blue
    ]
    
    plot_data = plot_data[tool_order]
    lower_data = lower_data[tool_order]
    upper_data = upper_data[tool_order]
    
    # Rename metrics
    renaming = {
        "lddt_pli": "Mean LDDT-PLI",
        "rmsd<2": "L-RMSD < 2A",
        "lddt": "Mean LDDT",
        "dockq_>0.23": "DockQ > 0.23",
    }
    plot_data = plot_data.rename(index=renaming)
    lower_data = lower_data.rename(index=renaming)
    upper_data = upper_data.rename(index=renaming)
    
    mean_vals = plot_data.values
    
    # Colors Mapping
    # Green for Boltz-1x (SMILES), Blue for Boltz-1xg (IUPAC)
    tool_colors = [
        "#188F52",  # Boltz-1x oracle (Green)
        "#86E935",  # Boltz-1x top-1 (Light Green)
        "#004D80",  # Boltz-1xg oracle (Blue)
        "#55C2FF",  # Boltz-1xg top-1 (Light Blue)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_data.index))
    bar_spacing = 0.015
    total_width = 0.7
    width = (total_width - (len(tool_order) - 1) * bar_spacing) / len(tool_order)
    
    for i, tool in enumerate(tool_order):
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
            capsize=2,
            error_kw={"elinewidth": 0.75},
        )
        
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data.index, rotation=0)
    ax.set_ylabel("Value")
    ax.set_title(f"Performances on {dataset} with 95% CI (Bootstrap)")
    plt.tight_layout()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.85), ncols=4, frameon=False)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()

def main():
    # PATHS - Assuming script is run from Downloads
    # Predictions (Contain confidence JSONs)
    smiles_preds_dir = "Glycoprotein_Results_SMILES"
    iupac_preds_dir = "Glycoprotein_Results_IUPAC"
    
    # Evaluations (Contain computed metrics JSONs)
    smiles_evals_dir = "eval_results_SMILES"
    iupac_evals_dir = "eval_results_IUPAC"
    
    # Run Comparison
    df = eval_models(
        smiles_preds_dir,
        smiles_evals_dir,
        iupac_preds_dir,
        iupac_evals_dir,
    )
    
    if df.empty:
        print("DataFrame is empty. Check input paths and filenames.")
        return

    # Save CSV
    df.to_csv("results_glycoprotein_benchmark.csv", index=False)
    
    desired_tools = [
        "Boltz-1x oracle",
        "Boltz-1x top-1",
        "Boltz-1xg oracle",
        "Boltz-1xg top-1",
    ]
    
    # Standard metrics
    desired_metrics = ["lddt", "dockq_>0.23", "lddt_pli", "rmsd<2"]
    
    plot_data(
        desired_tools, 
        desired_metrics, 
        df, 
        "Glycoprotein Benchmark", 
        "Glycoprotein_Benchmark.pdf"
    )

if __name__ == "__main__":
    main()
