## Processing Instructions:
1. 

## File Descriptions:
- pdb_api_call.py: A script that obtains all PDB files from the PDB directory that are both <9 angstroms in resolution and contain 1 or more CCD codes that are considered sugars
- phase1_cleaner.py: This script is used to clean the folder of raw PDB files obtained from pdb_api_call.py. The script discards all atoms other than those belonging to protein or glycan residues. The script checks for atomic clashes (heavy atoms closer than 0.5A to one another) and removes such files. The script also handles models with multiple conformations by choosing one of the provided conformations. Furthermore, if a glycan residue has less than 6 heavy atoms only the glycan is scrubbed from the file. The script then outputs a clean PDB file with the valid protein/glycan atoms as well as SEQRES information to correctly inform the model of any unresolved residues or unnatural amino acids. 
-  
