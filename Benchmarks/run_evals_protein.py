import argparse
import concurrent.futures
import subprocess
import sys
import os
from pathlib import Path
from tqdm import tqdm

# --- DOCKER CONFIGURATION ---
IMAGE_V28 = "registry.scicore.unibas.ch/schwede/openstructure:2.8.0"
IMAGE_LATEST = "registry.scicore.unibas.ch/schwede/openstructure:latest"

OST_COMPARE_STRUCTURE = (
    "docker run --platform linux/amd64 -u $(id -u):$(id -g) --rm --volume \"{host_mount}:/data\" {image_name} "
    "compare-structures "
    "-m \"/data/{model_rel}\" "
    "-r \"/data/{ref_rel}\" "
    "--fault-tolerant "
    "--min-pep-length 4 "
    "--min-nuc-length 4 "
    "-o \"/data/{output_rel}\" "
    "--lddt --bb-lddt --qs-score --dockq "
    "--ics --ips --rigid-scores --patch-scores --tm-score"
)

OST_COMPARE_LIGAND = (
    "docker run --platform linux/amd64 -u $(id -u):$(id -g) --rm --volume \"{host_mount}:/data\" {image_name} "
    "compare-ligand-structures "
    "-m \"/data/{model_rel}\" "
    "-r \"/data/{ref_rel}\" "
    "--fault-tolerant "
    "--lddt-pli --rmsd "
    "--substructure-match "
    "-o \"/data/{output_rel}\""
)

def get_docker_image():
    """Checks if v2.8.0 exists locally; if not, uses latest."""
    print("Checking for OpenStructure Docker image...")
    try:
        subprocess.run(
            ["docker", "image", "inspect", IMAGE_V28], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )
        print(f"✅ Found preferred image: {IMAGE_V28}")
        return IMAGE_V28
    except subprocess.CalledProcessError:
        print(f"⚠️  Image {IMAGE_V28} not found.")
        print(f"🔄 Falling back to: {IMAGE_LATEST}")
        return IMAGE_LATEST

def evaluate_structure(
    name: str,
    pred: Path,
    reference: Path,
    outdir: str,
    host_mount: Path,
    image: str,
) -> None:
    """Evaluate the structure."""
    
    # 1. Evaluate Polymer (Protein)
    out_path = Path(outdir) / f"{name}.json"
    
    # Check if already exists to skip
    if out_path.exists():
        # print(f"Skipping {name}, exists.")
        pass
    else:
        # CALCULATE RELATIVE PATHS (CASE-INSENSITIVE FIX)
        try:
            # We try strict resolution first
            model_rel = pred.resolve().relative_to(host_mount.resolve())
            ref_rel = reference.resolve().relative_to(host_mount.resolve())
            output_rel = out_path.resolve().relative_to(host_mount.resolve())
        except ValueError:
            # Fallback: If casing mismatch, we force alignment
            def rel_path_ignore_case(path, start):
                return os.path.relpath(str(path.resolve()), str(start.resolve()))

            model_rel = rel_path_ignore_case(pred, host_mount)
            ref_rel = rel_path_ignore_case(reference, host_mount)
            output_rel = rel_path_ignore_case(out_path, host_mount)

        # Final Safety Check
        if str(model_rel).startswith("..") or str(ref_rel).startswith("..") or str(output_rel).startswith(".."):
            print(f"❌ PATH ERROR: Files are outside the mounted folder!")
            print(f"   Mount: {host_mount}")
            return

        # Construct command
        cmd = OST_COMPARE_STRUCTURE.format(
            host_mount=str(host_mount.resolve()),
            model_rel=str(model_rel),
            ref_rel=str(ref_rel),
            output_rel=str(output_rel),
            image_name=image
        )

        # Run Docker
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            capture_output=True, 
            text=True 
        )

        # CHECK FOR FAILURE
        if not out_path.exists():
            print(f"\n❌ FAILURE processing {name}")
            print(f"DOCKER ERROR LOG:\n{result.stderr}")
    
    # 2. Evaluate Ligand
    out_path_lig = Path(outdir) / f"{name}_ligand.json"
    if out_path_lig.exists():
        return
    
    # Recalculate output relative for ligand
    try:
        output_lig_rel = out_path_lig.resolve().relative_to(host_mount.resolve())
    except ValueError:
        output_lig_rel = os.path.relpath(str(out_path_lig.resolve()), str(host_mount.resolve()))

    # Re-calculate inputs (needed if scope changed, usually same)
    # Using previous model_rel/ref_rel is fine
    
    cmd_lig = OST_COMPARE_LIGAND.format(
        host_mount=str(host_mount.resolve()),
        model_rel=str(model_rel),
        ref_rel=str(ref_rel),
        output_rel=str(output_lig_rel),
        image_name=image
    )

    subprocess.run(
        cmd_lig,
        shell=True,
        check=False,
        capture_output=True,
        text=True
    )

def main(args):
    # 1. Determine Docker Image
    active_image = get_docker_image()

    # 2. Identify Targets and Prediction Files based on Format
    print(f"Scanning directories in: {args.data} for format: {args.format}")
    
    # Get all subdirectories (Targets)
    target_folders = [f for f in args.data.iterdir() if f.is_dir() and not f.name.startswith(".")]
    
    tasks = []

    for folder in target_folders:
        target_name = folder.name
        
        # Determine Reference File
        # Handle casing for reference search (e.g. T1187 vs t1187)
        candidates = [
            args.pdb / f"{target_name}.cif",
            args.pdb / f"{target_name.upper()}.cif",
            args.pdb / f"{target_name.lower()}.cif",
            args.pdb / f"{target_name}.cif.gz",
            args.pdb / f"{target_name.lower()}.cif.gz"
        ]
        
        ref_path = None
        for c in candidates:
            if c.exists():
                ref_path = c
                break
        
        if not ref_path:
            # print(f"⚠️ Reference not found for {target_name}, skipping.")
            continue

        # Look for 5 models
        for model_id in range(5):
            pred_path = None
            output_name = f"{target_name}_model_{model_id}"

            if args.format == "af3":
                # AF3: seed-1_sample-{id}/model.cif
                p = folder / f"seed-1_sample-{model_id}" / "model.cif"
                if p.exists(): pred_path = p
            
            elif args.format == "chai":
                # Chai: pred.model_idx_{id}.cif
                p = folder / f"pred.model_idx_{model_id}.cif"
                if p.exists(): pred_path = p
            
            elif args.format == "boltz":
                # Boltz (New Structure): T1187/T1187_model_{id}.cif
                # Handle casing for model file name
                # Standard Boltz casing usually follows target name or capitalized target name
                possible_names = [
                    f"{target_name}_model_{model_id}.cif",
                    f"{target_name.lower()}_model_{model_id}.cif",
                    f"{target_name.upper()}_model_{model_id}.cif",
                    f"{target_name[0].upper()}{target_name[1:]}_model_{model_id}.cif"
                ]
                for pname in possible_names:
                    if (folder / pname).exists():
                        pred_path = folder / pname
                        break

            if pred_path:
                tasks.append({
                    "name": output_name,
                    "pred": pred_path,
                    "ref": ref_path
                })

    if not tasks:
        print("❌ ERROR: No matching prediction files found!")
        sys.exit(1)
        
    print(f"✅ Found {len(tasks)} predictions to evaluate.")

    # 3. Create Output Directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # 4. Automatic Mount Point Detection
    data_abs = args.data.resolve()
    pdb_abs = args.pdb.resolve()
    out_abs = args.outdir.resolve()

    if args.mount:
        host_mount = args.mount.resolve()
    else:
        # Calculate common ancestor
        try:
            common = os.path.commonpath([str(data_abs), str(pdb_abs), str(out_abs)])
            host_mount = Path(common).resolve()
        except Exception as e:
            print(f"⚠️ Could not auto-detect common path ({e}). Defaulting to current working directory.")
            host_mount = Path.cwd().resolve()

    print(f"Mounting Host Path: {host_mount}")
    
    print("\n🚀 Starting Evaluations...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(
                evaluate_structure,
                name=task["name"],
                pred=task["pred"],
                reference=task["ref"],
                outdir=str(args.outdir),
                host_mount=host_mount,
                image=active_image,
            )
            futures.append(future)

        # Wait for completion
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    print("\n✅ Done! Check your output folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Folder containing prediction target subfolders")
    parser.add_argument("pdb", type=Path, help="Folder containing ground truth .cif files")
    parser.add_argument("outdir", type=Path, help="Folder to save results")
    parser.add_argument("--format", type=str, default="boltz", choices=["af3", "chai", "boltz"])
    parser.add_argument("--testset", type=str, default="casp")
    parser.add_argument("--mount", type=Path, help="Optional: Force a specific mount point")
    parser.add_argument("--executable", type=str, default="/bin/bash")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    main(args)
