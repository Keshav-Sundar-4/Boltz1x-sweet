## Processing Instructions:
1. Use pdb_api_call.py to obtain a folder of PDB files

## File Descriptions:
- pdb_api_call.py: A script that obtains all PDB files from the PDB directory that are both <9 angstroms in resolution and contain 1 or more CCD codes that are considered sugars

- phase1_cleaner.py: This script is used to clean the folder of raw PDB files obtained from pdb_api_call.py. The script discards all atoms other than those belonging to protein or glycan residues. The script checks for atomic clashes (heavy atoms closer than 0.5A to one another) and removes such files. The script also handles models with multiple conformations by choosing one of the provided conformations. Furthermore, if a glycan residue has less than 6 heavy atoms only the glycan is scrubbed from the file. The script then outputs a clean PDB file with the valid protein/glycan atoms as well as SEQRES information to correctly inform the model of any unresolved residues or unnatural amino acids. 

-  free_glycan_data.py: A cleaning script specialized for free-floating glycans. It works with PDB files and begins by stripping hydrogen atoms from the file. It generates a connectivty file from scratch and deals with any PDB files with mutliple conformations or poses. No steroehemical cleaning etc. is done.

- lectinz_clean.py: This script is used to manage and clean MD-derived PDB files. Such files often defy common PDB conventions and thus require special handling. Hydrogens are scrubbed, and atom names are normalized. Moreover, such files often break up a single monosaccharide into distinct residues, so a connectivity derived alhorithm is used to generate unified residue naming schemes. The corrected file is then converted into standard PDB format. 

-  preprocess_glycans.py: This script converts .pdb files into .npz files. The general flow is that it reads .pdb files and parses the atoms, groups residues/chains, and then converts the necessary features into arrays. Necessary features include atom names, elements, residue names, chain IDs, bond connectivity, glycan semantics, protein sequences, etc.
      - Glycans are often 'chained' as part of the protein. The script generates connectivity via sci-py's cKDTree algorithm and identifies protein and glycan molecules, creating chains based on                    connectivity. It generates chain ID's from scratch based on these re-chained entities. 
      - Glycosylation sites are specially detected and placed into the connections array
      - The script obtains glycan anomeric configuration using a 3d approach. Anomeric carbons of sugars are identified based on their residue name. The dihedral calculation is found from 4 atoms,                  which consist of [Anomeric_Oxygen, Anomeric_Carbon, Ring Oxygen, Ring_Neighboring_Carbon]. If the dihedral is between -95 and 95 degrees then the anomericity is classified as alpha. If the                  dihedral is outside of this range it is classified as beta.
      - Glycosylation Filtering is also applied due to inconsistencies in glycosylation structure prediction. If the [CG, ND2, C1] angle is outside of 110-130 degrees, the glycan is filtered out. If                the [OD2, CG, ND2, C1] dihedral angle is not within the -20 < X < 20 range the glycan is filtered out. (Note that filtering the glycan out is equivalent to removing the glycosylation site. The              protein sample is maintained as there are often many glycosylation sites in one structure. To maximize data only the glycan is removed. 
      - Occasionally, there are PDB artifacts where the oxygen involved in a glycosidic bond is not the only oxygen bound to the acceptor monosaccharide's carbon. In this case we remove the extra oxygen
      - For certain MD files, the atom names are not consistent with standard PDB atom names. A name mapping is used to modify the file in these edge cases
 
-  validation_dataset_creation.py: This script generates a validation text file containing of 3.5% of the processes in a given file. 
