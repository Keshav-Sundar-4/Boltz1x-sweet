#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark.py: Central Nervous System Control Script

This script orchestrates the generation of YAML sequence files for a benchmark dataset.
It performs two main tasks:
1. Iterates through all PDB files in the 'dockqc_dataset_cleaned' directory,
   calling 'benchmark_generator.py' for each to create a corresponding YAML file.
   The MSA ID is set to the PDB filename without its extension.
2. Parses the 'Benchmark_Sequences.txt' CASP file, and for each individual
   sequence entry, calls 'benchmark_generator.py' to generate a separate,
   uniquely named YAML file.

All output YAML files are saved to the 'benchmark_yamls' directory.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

def parse_casp_into_entries(filepath: Path) -> Dict[str, str]:
    """
    Parses a CASP-formatted text file into a dictionary of individual entries.

    Args:
        filepath: The path to the CASP text file.

    Returns:
        A dictionary where keys are sequence IDs (e.g., "R1107") and
        values are the string content of that entry (header + sequence).
    """
    entries = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('>'):
            header = line
            seq_id = header.split()[0][1:]
            # The sequence is expected on the very next line
            if i + 1 < len(lines):
                sequence = lines[i+1].strip()
                if sequence: # Ensure the sequence line is not empty
                    entries[seq_id] = f"{header}\n{sequence}\n"
                    i += 2 # Move past header and sequence
                else:
                    i += 1 # Move past header only
            else:
                i += 1 # End of file
        else:
            i += 1 # Not a header, move to the next line
    return entries


def main():
    """Main execution function."""
    try:
        # Define paths relative to the script's location for robustness.
        script_dir = Path(__file__).resolve().parent
        
        pdb_dir = script_dir / "dockqc_dataset_cleaned"
        casp_file = script_dir / "Benchmark_Sequences.txt"
        output_dir = script_dir / "benchmark_yamls"
        generator_script = script_dir / "benchmark_generator.py"

        # --- Pre-flight Checks ---
        if not generator_script.exists():
            print(f"Error: The generator script '{generator_script.name}' was not found.")
            return
        if not pdb_dir.is_dir():
            print(f"Error: PDB dataset directory '{pdb_dir.name}' not found.")
            return
        if not casp_file.exists():
            print(f"Warning: CASP file '{casp_file.name}' not found. Will skip CASP processing.")

        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        print(f"Output will be saved to: {output_dir}")

        # --- Task 1: Process all PDB files ---
        pdb_files = sorted(list(pdb_dir.glob("*.pdb")))
        if not pdb_files:
            print(f"Warning: No .pdb files found in '{pdb_dir.name}'.")
        else:
            print(f"\nFound {len(pdb_files)} PDB files to process...")
            for i, pdb_path in enumerate(pdb_files):
                msa_id = pdb_path.stem  # e.g., "XXXX" from "XXXX.pdb"
                output_path = output_dir / f"{msa_id}.yaml"
                
                print(f"[{i+1}/{len(pdb_files)}] PDB: {pdb_path.name} -> {output_path.name}")
                
                command = [
                    "python3", str(generator_script),
                    "--pdb_file", str(pdb_path),
                    "--output_file", str(output_path),
                    "--msa_id", msa_id
                ]
                subprocess.run(command, check=True, capture_output=True, text=True)

        # --- Task 2: Process the CASP text file entry by entry ---
        if casp_file.exists():
            casp_entries = parse_casp_into_entries(casp_file)
            if not casp_entries:
                print(f"Warning: No valid entries found in CASP file '{casp_file.name}'.")
            else:
                print(f"\nFound {len(casp_entries)} CASP entries to process...")
                
                # Use a context manager for a temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    for i, (seq_id, content) in enumerate(casp_entries.items()):
                        output_path = output_dir / f"{seq_id}.yaml"
                        
                        # Create a temporary file for the single entry
                        temp_entry_file = temp_path / f"{seq_id}.txt"
                        with open(temp_entry_file, 'w') as f_temp:
                            f_temp.write(content)
                        
                        print(f"[{i+1}/{len(casp_entries)}] CASP: {seq_id} -> {output_path.name}")

                        command = [
                            "python3", str(generator_script),
                            "--casp", str(temp_entry_file),
                            "--output_file", str(output_path)
                        ]
                        subprocess.run(command, check=True, capture_output=True, text=True)

        print("\nBenchmark generation complete.")

    except FileNotFoundError:
        print("Error: 'python3' command not found. Please ensure Python 3 is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print("\n--- An error occurred during generation ---")
        print(f"Command failed: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        print("------------------------------------------")
        print("Processing halted due to an error.")

if __name__ == "__main__":
    main()
