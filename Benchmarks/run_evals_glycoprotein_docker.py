import argparse
import concurrent.futures
import subprocess
import os
import sys
from pathlib import Path
from tqdm import tqdm

# CONSTANTS
IMAGE_NAME = "registry.scicore.unibas.ch/schwede/openstructure:latest"

# DOCKER TEMPLATES
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

def evaluate_structure(
    name: str,
    pred: Path,
    reference: Path,
    outdir: Path,
    host_mount: Path,
    executable: str = "/bin/bash",
) -> None:
    """Evaluate the structure using robust relative paths."""
    
    # Calculate paths relative to the host_mount
    # Since host_mount is auto-calculated from these files, this should always work.
    try:
        model_rel = pred.resolve().relative_to(host_mount)
        ref_rel = reference.resolve().relative_to(host_mount)
        
        # Output paths
        out_prot_path = outdir / f"{name}.json"
        out_lig_path = outdir / f"{name}_ligand.json"
        
        out_prot_rel = out_prot_path.resolve().relative_to(host_mount)
        out_lig_rel = out_lig_path.resolve().relative_to(host_mount)
        
    except ValueError as e:
        print(f"Skipping {name}: Path error. {e}")
        return

    # 1. Evaluate Polymer Metrics
    if out_prot_path.exists():
        pass 
    else:
        cmd = OST_COMPARE_STRUCTURE.format(
            image_name=IMAGE_NAME,
            host_mount=str(host_mount),
            model_rel=str(model_rel),
            ref_rel=str(ref_rel),
            output_rel=str(out_prot_rel)
        )
        
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            executable=executable,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"ERROR on {name} (Protein): {result.stderr.decode('utf-8')}")

    # 2. Evaluate Ligand Metrics
    if out_lig_path.exists():
        pass
    else:
        cmd = OST_COMPARE_LIGAND.format(
            image_name=IMAGE_NAME,
            host_mount=str(host_mount),
            model_rel=str(model_rel),
            ref_rel=str(ref_rel),
            output_rel=str(out_lig_rel)
        )

        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            executable=executable,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"ERROR on {name} (Ligand): {result.stderr.decode('utf-8')}")


def main(args):
    # Resolve all inputs to absolute, canonical paths (Resolves casing issues)
    data_dir = args.data.resolve()
    pdb_dir = args.pdb.resolve()
    out_dir = args.outdir.resolve()

    # ---------------------------
    # AUTOMATIC MOUNT DETECTION
    # ---------------------------
    # Instead of asking the user, we find the common parent folder of all inputs.
    # This guarantees the mount point is valid and matches the casing of the files.
    try:
        # os.path.commonpath returns the longest common sub-path
        common = os.path.commonpath([str(data_dir), str(pdb_dir), str(out_dir)])
        host_mount = Path(common).resolve()
    except Exception as e:
        print(f"Error determining mount point: {e}")
        return

    print(f"Auto-detected Mount Point: {host_mount}")

    # ---------------------------
    # FILE DISCOVERY
    # ---------------------------
    files = list(data_dir.iterdir())
    names = {f.stem.lower(): f for f in files if not f.name.startswith(".")}
    
    print(f"Found {len(names)} targets in {data_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as executor:
        futures = []
        for name, folder in names.items():
            for model_id in range(5):
                # ---------------------------
                # BOLTZ / FORMAT LOGIC
                # ---------------------------
                if args.format == "boltz":
                    if folder.is_dir():
                        name_file = (
                            f"{name[0].upper()}{name[1:]}"
                            if args.testset == "casp"
                            else name.lower()
                        )
                        pred_path = folder / f"{name_file}_model_{model_id}.cif"
                    else:
                        if model_id > 0: continue
                        pred_path = folder
                
                elif args.format == "af3":
                    pred_path = folder / f"seed-1_sample-{model_id}" / "model.cif"
                elif args.format == "chai":
                    pred_path = folder / f"pred.model_idx_{model_id}.cif"

                # ---------------------------
                # TARGET MATCHING LOGIC
                # ---------------------------
                if args.testset == "casp":
                    ref_path = pdb_dir / f"{name[0].upper()}{name[1:]}.cif"
                elif args.testset == "test":
                    ref_path = pdb_dir / f"{name.lower()}.cif.gz"

                # Check existence
                if not pred_path.exists():
                    continue

                if not ref_path.exists():
                    if model_id == 0: 
                        print(f"Warning: Target file not found: {ref_path}")
                    continue

                future = executor.submit(
                    evaluate_structure,
                    name=f"{name}_model_{model_id}",
                    pred=pred_path,
                    reference=ref_path,
                    outdir=out_dir,
                    host_mount=host_mount,
                    executable=args.executable,
                )
                futures.append(future)

        with tqdm(total=len(futures)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--format", type=str, default="af3")
    parser.add_argument("--testset", type=str, default="casp")
    # Removed --mount argument to prevent user error; logic is now automatic
    parser.add_argument("--executable", type=str, default="/bin/bash")
    parser.add_argument("--max-workers", type=int, default=32)
    args = parser.parse_args()
    main(args)
