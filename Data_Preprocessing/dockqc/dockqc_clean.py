import os
import shutil
from collections import defaultdict
import warnings

# Suppress PDBConstructionWarning
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

# --- Configuration ---

# 1. Define standard amino acid three-letter codes
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
}

# 2. Define the comprehensive list of allowed glycan three-letter codes.
ALLOWED_GLYCANS = {
    "05L", "07E", "0HX", "0LP", "0MK", "0NZ", "0UB", "0WK", "0XY", "0YT", "12E", "145", "147", "149", "14T", "15L", "16F", "16G", "16O", "17T", "18D", "18O", "1CF", "1GL", "1GN", "1S3", "1S4", "1SD", "1X4", "20S", "20X",
    "22O", "22S", "23V", "24S", "25E", "26O", "27C", "289", "291", "293", "2DG", "2DR", "2F8", "2FG", "2FL", "2GL", "2GS", "2H5", "2M5", "2M8", "2WP", "32O", "34V", "38J", "3DO", "3FM", "3HD", "3J3", "3J4", "3LJ", "3MG",
    "3MK", "3R3", "3S6", "3YW", "42D", "445", "44S", "46Z", "475", "491", "49A", "49S", "49T", "49V", "4AM", "4CQ", "4GL", "4GP", "4JA", "4N2", "4NN", "4QY", "4R1", "4SG", "4U0", "4U1", "4U2", "4UZ", "4V5", "50A",
    "510", "51N", "56N", "57S", "5DI", "5GF", "5GO", "5KQ", "5KV", "5L2", "5L3", "5LS", "5LT", "5N6", "5QP", "5TH", "5TJ", "5TK", "5TM", "604", "61J", "62I", "64K", "66O", "6BG", "6C2", "6GB", "6GP", "6GR",
    "6K3", "6KH", "6KL", "6KS", "6KU", "6KW", "6LS", "6LW", "6MJ", "6MN", "6PY", "6PZ", "6S2", "6UD", "6Y6", "6YR", "6ZC", "73E", "79J", "7CV", "7D1", "7GP", "7JZ", "7K2", "7K3", "7NU", "83Y", "89Y", "8B7", "8B9", "8EX", "8GA",
    "8GG", "8GP", "8LM", "8LR", "8OQ", "8PK", "8S0", "95Z", "96O", "9AM", "9C1", "9CD", "9GP", "9KJ", "9MR", "9OK", "9PG", "9QG", "9QZ", "9S7", "9SG", "9SJ", "9SM", "9SP", "9T1", "9T7", "9VP", "9WJ", "9WN", "9WZ", "9YW",
    "A0K", "A1Q", "A2G", "A5C", "A6P", "AAL", "ABD", "ABE", "ABF", "ABL", "AC1", "ACR", "ACX", "ADA", "AF1", "AFD", "AFO", "AFP", "AFR", "AGL", "AGR", "AH2", "AH8", "AHG", "AHM", "AHR", "AIG", "ALL", "ALX", "AMG", "AMN", "AMU",
    "AMV", "ANA", "AOG", "AQA", "ARA", "ARB", "ARI", "ARW", "ASC", "ASG", "ASO", "AXP", "AXR", "AY9", "AZC", "B0D", "B16", "B1H", "B1N", "B6D", "B7G", "B8D", "B9D", "BBK", "BBV", "BCD", "BCW", "BDF", "BDG", "BDP", "BDR", "BDZ",
    "BEM", "BFN", "BG6", "BG8", "BGC", "BGL", "BGN", "BGP", "BGS", "BHG", "BM3", "BM7", "BMA", "BMX", "BND", "BNG", "BNX", "BXY", "BZD",
    "C3B", "C3G", "C3X", "C4B", "C4W", "C5X", "CBF", "CBI", "CBK", "CDR", "CE5", "CE6", "CE8", "CEG", "CEX", "CEY", "CEZ", "CGF", "CJB", "CKB", "CKP", "CNP", "CR1", "CR6", "CRA", "CT3", "CTO", "CTR", "CTT",
    "D0N", "D1M", "D5E", "D6G", "DAF", "DAG", "DAN", "DDA", "DDL", "DEG", "DEL", "DFR", "DFX", "DGO", "DGS", "DJB", "DJE", "DK4", "DKX", "DKZ", "DL6", "DLD", "DLF", "DLG", "DO8", "DOM", "DPC", "DQR", "DR2", "DR3", "DR5",
    "DRI", "DSR", "DT6", "DVC", "DYM", "E3M", "E5G", "EAG", "EBG", "EBQ", "EEN", "EEQ", "EGA", "EMP", "EMZ", "EPG", "EQP", "EQV", "ERE", "ERI", "ETT", "F1P", "F1X", "F55", "F58", "F6P", "FBP", "FCA", "FCB", "FCT", "FDP",
    "FDQ", "FFC", "FFX", "FIF", "FK9", "FKD", "FMF", "FMO", "FNG", "FNY", "FRU", "FSA", "FSI", "FSM", "FSR", "FSW", "FUB", "FUC", "FUF", "FUL", "FUY", "FVQ", "FX1", "FYJ", "G0S", "G16", "G1P", "G20", "G28", "G2F",
    "G3F", "G4D", "G4S", "G6D", "G6P", "G6S", "G7P", "G8Z", "GAA", "GAC", "GAD", "GAF", "GAL", "GAT", "GBH", "GC1", "GC4", "GC9", "GCB", "GCD", "GCN", "GCO", "GCS", "GCT", "GCU", "GCV", "GCW", "GDA", "GDL",
    "GE1", "GE3", "GFP", "GIV", "GL0", "GL1", "GL2", "GL4", "GL5", "GL6", "GL7", "GL9", "GLA", "GLC", "GLD", "GLF", "GLG", "GLO", "GLP", "GLS", "GLT", "GM0", "GMB", "GMH", "GMT", "GMZ", "GN1", "GN4", "GNS", "GNX",
    "GP0", "GP1", "GP4", "GPH", "GPK", "GPM", "GPO", "GPQ", "GPU", "GPV", "GPW", "GQ1", "GRF", "GRX", "GS1", "GS9", "GTK", "GTM", "GTR", "GU0", "GU1", "GU2", "GU3", "GU4", "GU5", "GU6", "GU8", "GU9", "GUF", "GUL", "GUP",
    "GUZ", "GXL", "GYE", "GYG", "GYP", "GYU", "GYV", "GZL", "H1M", "H1S", "H2P", "H53", "H6Q", "H6Z", "HBZ", "HD4", "HNV", "HNW", "HSG", "HSH", "HSJ", "HSQ", "HSX", "HSY", "HTG", "HTM", "I57", "IAB", "IDC", "IDF", "IDG", "IDR",
    "IDS", "IDU", "IDX", "IDY", "IEM", "IN1", "IPT", "ISD", "ISL", "ISX", "IXD", "J5B", "JFZ", "JHM", "JLT", "JS2", "JV4", "JVA", "JVS", "JZR", "K5B", "K99", "KBA", "KBG", "KD5", "KDA", "KDB", "KDD", "KDE", "KDF", "KDM", "KDN",
    "KDO", "KDR", "KFN", "KG1", "KGM", "KHP", "KME", "KO1", "KO2", "KOT", "KTU",
    "L1L", "L6S", "LAH", "LAK", "LAO", "LAT", "LB2", "LBS", "LBT", "LCN", "LDY", "LEC", "LFR", "LGC", "LGU", "LKA", "LKS", "LNV", "LOG", "LOX", "LRH", "LVO", "LVZ", "LXB", "LXC", "LXZ", "LZ0", "M1F", "M1P", "M2F", "M3N", "M55", "M6D",
    "M6P", "M7B", "M7P", "M8C", "MA1", "MA2", "MA3", "MA8", "MAF", "MAG", "MAL", "MAN", "MAT", "MAV", "MAW", "MBE", "MBF", "MBG", "MCU", "MDA", "MDP", "MFB", "MFU", "MG5", "MGC", "MGL", "MGS", "MJJ", "MLB", "MLR", "MMA", "MN0",
    "MNA", "MQG", "MQT", "MRH", "MRP", "MSX", "MTT", "MUB", "MUR", "MVP", "MXY", "MXZ", "MYG", "N1L", "N9S", "NA1", "NAA", "NAG", "NBG", "NBX", "NBY", "NDG", "NFG", "NG1", "NG6", "NGA", "NGC", "NGE", "NGK", "NGR", "NGS", "NGY", "NGZ", "NHF",
    "NLC", "NM6", "NM9", "NNG", "NPF", "NSQ", "NT1", "NTF", "NTO", "NTP", "NXD", "NYT",
    "O1G", "OAK", "OEL", "OI7", "OPM", "OSU", "OTG", "OTN", "OTU", "OX2", "P53", "P6P", "PA1", "PAV", "PDX", "PH5", "PKM", "PNA", "PNG", "PNJ", "PNW", "PPC", "PRP", "PSG", "PSV", "PUF", "PZU", "QIF", "QKH", "QPS", "R1P", "R1X", "R2B", "R2G",
    "RAE", "RAF", "RAM", "RAO", "RCD", "RER", "RF5", "RGG", "RHA", "RHC", "RI2", "RIB", "RIP", "RM4", "RP3", "RP5", "RP6", "RR7", "RRJ", "RRY", "RST", "RTG", "RTV", "RUG", "RUU", "RV7", "RVG", "RVM", "RWI", "RY7", "RZM", "S7P", "S81",
    "SA0", "SCG", "SCR", "SDY", "SEJ", "SF6", "SF9",
    "SFJ", "SFU", "SG4", "SG5", "SG6", "SG7", "SGA", "SGC", "SGD", "SGN", "SHB", "SHD", "SHG", "SIA", "SID", "SIO", "SIZ", "SLB", "SLM", "SLT", "SMD", "SN5", "SNG", "SOE", "SOG",
    "SOR", "SR1", "SSG", "STZ", "SUC", "SUP", "SUS", "SWE", "SZZ", "T68", "T6P", "T6T", "TA6", "TCB", "TCG", "TDG", "TEU", "TF0", "TFU", "TGA", "TGK", "TGR", "TGY", "TH1", "TMR",
    "TMX", "TNX", "TOA", "TOC", "TQY", "TRE", "TRV", "TS8", "TT7", "TTV", "TTZ", "TU4", "TUG", "TUJ", "TUP", "TUR", "TVD", "TVG", "TVM", "TVS", "TVV", "TVY", "TW7", "TWA", "TWD", "TWG", "TWJ", "TWY", "TXB", "TYV",
    "U1Y", "U2A", "U2D", "U63", "U8V", "U97", "U9A", "U9D", "U9G", "U9J", "U9M", "UAP", "UCD", "UDC", "UEA", "V3M", "V3P", "V71", "VG1", "VTB", "W9T", "WIA", "WOO", "WUN", "X0X", "X1P", "X1X", "X2F", "X6X", "XDX", "XGP",
    "XIL", "XLF", "XLS", "XMM", "XXM", "XXR", "XXX", "XYF", "XYL", "XYP", "XYS", "XYT", "XYZ", "YIO", "YJM", "YKR", "YO5", "YX0", "YX1", "YYB", "YYH", "YYJ", "YYK", "YYM", "YYQ", "YZ0", "Z0F", "Z15", "Z16", "Z2D", "Z2T", "Z3K", "Z3L", "Z3Q", "Z3U",
    "Z4K", "Z4R", "Z4S", "Z4U", "Z4V", "Z4W", "Z4Y", "Z57", "Z5J", "Z5L", "Z61", "Z6H", "Z6J", "Z6W", "Z8H", "Z8T", "Z9D", "Z9E", "Z9H", "Z9K", "Z9L", "Z9M", "Z9N", "Z9W", "ZB0", "ZB1", "ZB2", "ZB3", "ZCD", "ZCZ", "ZD0", "ZDC", "ZDO", "ZEE", "ZEL", "ZGE", "ZMR"
}


# 3. Define distance thresholds in Angstroms
COMPLEX_CONTACT_DISTANCE = 4.0
REMOVAL_CONTACT_DISTANCE = 4.0
GLYCAN_CONTACT_DISTANCE = 5.0

# --- Helper Functions and Classes ---

def is_standard_residue(residue):
    """Checks if a residue is a standard amino acid."""
    return residue.get_resname().strip() in STANDARD_AMINO_ACIDS

def is_glycan(residue):
    """Checks if a residue is in our list of allowed glycans."""
    return residue.get_resname().strip() in ALLOWED_GLYCANS

def separate_glycans_and_split_by_connectivity(structure):
    """
    Identifies glycan residues on protein chains, groups them by physical 
    connectivity (distance < 2.0A), and assigns each connected group 
    to a unique new Chain ID.
    
    New IDs are assigned A-Z (excluding N, Y), then a-z, then 0-9, avoiding
    any IDs already present in the structure (e.g., protein chains).
    """
    model = structure[0]
    
    # 1. Setup available Chain IDs
    # Track IDs currently used in the model (e.g. Protein chains 'A', 'B')
    existing_ids = set(chain.id for chain in model)
    
    # Define priority pool: A-Z, a-z, 0-9.
    # We deliberately exclude 'N' (No) and 'Y' (Yes) to prevent YAML boolean interpretation issues.
    priority_chars = "ABCDEFGHIJKLMOPQRSTUVWXZabcdefghijklmnopqrstuvwxyz0123456789"
    
    def get_next_chain_id():
        for char in priority_chars:
            if char not in existing_ids:
                existing_ids.add(char) # Reserve it immediately so the next group doesn't take it
                return char
        return None

    # Iterate over a copy of chains so we can modify the model safely
    for chain in list(model.get_chains()):
        
        # Identify all glycan residues in this chain
        glycan_residues = [res for res in chain if is_glycan(res)]
        
        if not glycan_residues:
            continue

        # --- Connectivity Clustering (Union-Find) ---
        
        # Map each residue to an index 0..N
        res_list = glycan_residues
        n = len(res_list)
        parent = list(range(n))

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Collect all atoms for NeighborSearch
        all_glycan_atoms = []
        atom_to_res_idx = {}
        for idx, res in enumerate(res_list):
            for atom in res:
                all_glycan_atoms.append(atom)
                atom_to_res_idx[atom] = idx

        # Perform Neighbor Search to find connections
        # 2.0A is a standard threshold to detect covalent bonds between sugars
        ns = NeighborSearch(all_glycan_atoms)
        
        for atom in all_glycan_atoms:
            # Find neighbors within 2.0 Angstroms
            neighbors = ns.search(atom.coord, 2.0, 'A')
            current_idx = atom_to_res_idx[atom]
            
            for neighbor in neighbors:
                neighbor_idx = atom_to_res_idx[neighbor]
                # If atoms belong to different residues, union the residues
                if current_idx != neighbor_idx:
                    union(current_idx, neighbor_idx)

        # Group residues by their root parent
        groups = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups[root].append(res_list[i])

        # --- Re-chaining ---

        # If the original chain was PURE glycan and only has 1 group, 
        # we don't need to do anything (it's already a single valid chain).
        has_protein = any(is_standard_residue(res) for res in chain)
        if not has_protein and len(groups) == 1:
            continue

        # Otherwise, move each group to a new chain
        for root_id, residue_group in groups.items():
            new_chain_id = get_next_chain_id()
            
            if new_chain_id is None:
                print(f"  WARNING: Ran out of chain IDs. Cannot separate glycan cluster from chain {chain.id}.")
                break
            
            new_chain = Chain(new_chain_id)
            
            for res in residue_group:
                # Remove from old chain
                chain.detach_child(res.id)
                # Add to new chain
                new_chain.add(res)
            
            model.add(new_chain)
            print(f"  INFO: Moved glycan cluster ({len(residue_group)} residues) from {chain.id} to new chain {new_chain_id}.")

def find_complexes(protein_chains, distance_threshold):
    """
    Identifies complexes of protein chains based on a distance threshold.
    Two chains are considered connected if any of their atoms are within
    the threshold.
    """
    if not protein_chains: return []
    all_atoms = [atom for chain in protein_chains for atom in chain.get_atoms()]
    ns = NeighborSearch(all_atoms)
    adj = defaultdict(list)
    chain_list = list(protein_chains)
    
    # Check for contacts between chains
    for i in range(len(chain_list)):
        for j in range(i + 1, len(chain_list)):
            chain1, chain2 = chain_list[i], chain_list[j]
            found_contact = False
            for atom1 in chain1.get_atoms():
                close_atoms = ns.search(atom1.coord, distance_threshold, 'A')
                # Check if any close atom belongs to chain2
                if any(atom.get_parent().get_parent().id == chain2.id for atom in close_atoms):
                    adj[chain1.id].append(chain2.id)
                    adj[chain2.id].append(chain1.id)
                    found_contact = True
                    break
            if found_contact:
                continue

    # Identify connected components (complexes)
    complexes, visited = [], set()
    for chain in chain_list:
        if chain.id not in visited:
            current_complex_ids, q = set(), [chain.id]
            visited.add(chain.id)
            while q:
                current_chain_id = q.pop(0)
                current_complex_ids.add(current_chain_id)
                for neighbor_id in adj[current_chain_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        q.append(neighbor_id)
            complex_chains = [c for c in chain_list if c.id in current_complex_ids]
            complexes.append(complex_chains)
    return complexes

class FinalStructureSelector(Select):
    def __init__(self, chains_to_keep_ids):
        self.chains_to_keep_ids = chains_to_keep_ids
    def accept_chain(self, chain):
        return chain.id in self.chains_to_keep_ids

# --- Main Processing Function ---

def process_pdb_file(pdb_path, output_dir):
    filename = os.path.basename(pdb_path)
    print(f"\n--- Processing {filename} ---")
    
    parser = PDBParser()
    try:
        structure = parser.get_structure(filename.split('.')[0], pdb_path)
    except Exception as e:
        print(f"  ERROR: Could not parse {filename}. Reason: {e}")
        return

    # Updated to use the correct connectivity-based splitting
    separate_glycans_and_split_by_connectivity(structure)
    
    model = structure[0]
    protein_chains = []
    
    # Flag to determine if the file should be skipped entirely.
    file_should_be_skipped = False

    # PART 1: Initial check for file-deleting condition and cleaning of other residues.
    for chain in model:
        residues = list(chain.get_residues())
        
        for i, res in enumerate(residues):
            if is_standard_residue(res):
                continue # This is a standard amino acid, move to the next residue.
            
            if is_glycan(res):
                continue # This is a recognized glycan, move to the next residue.

            # If we reach here, the residue is non-standard and non-glycan.
            # Now, check the specific condition for skipping the file.
            is_middle = (i > 0 and i < len(residues) - 1)
            if is_middle:
                prev_res = residues[i-1]
                next_res = residues[i+1]
                # THIS IS THE TRIGGER CONDITION
                if is_standard_residue(prev_res) and is_standard_residue(next_res):
                    print(f"  DELETE: Found modified amino acid '{res.get_resname()}' "
                          f"between standard residues in chain {chain.id}.")
                    file_should_be_skipped = True
                    break # Exit the residue loop immediately.
            
            # If the condition above was not met, this is just a regular unwanted
            # residue (like a terminal ligand or ion), so we clean it.
            chain.detach_child(res.id)
        
        if file_should_be_skipped:
            break # Exit the chain loop as well.

    # If the flag was set, abort all further processing for this file.
    if file_should_be_skipped:
        print(f"  ACTION: File {filename} will be excluded from the cleaned dataset.")
        return # This exits the function, and the file is never saved.

    
    # If the script continues, the file is valid. Now we find the protein chains.
    for chain in model:
        if any(is_standard_residue(res) for res in chain.get_residues()):
             protein_chains.append(chain)

    if not protein_chains:
        print("  INFO: No standard protein chains found after cleaning. Skipping file.")
        return

    # PART 2: Find complexes and handle multiple complexes
    print(f"  INFO: Found {len(protein_chains)} protein chains. Identifying complexes...")
    complexes = find_complexes(protein_chains, COMPLEX_CONTACT_DISTANCE)
    print(f"  INFO: Identified {len(complexes)} complex(es).")

    chains_in_kept_complex = []
    
    if len(complexes) > 1:
        complexes.sort(key=len, reverse=True)
        kept_complex = complexes[0]
        removed_complexes = complexes[1:]
        chains_in_kept_complex = kept_complex
        chains_to_remove_from_complexes = {chain.id for comp in removed_complexes for chain in comp}
        print(f"  ACTION: More than one complex found. Keeping the largest ({len(kept_complex)} chains).")
        atoms_of_removed_complexes = [atom for comp in removed_complexes for chain in comp for atom in chain.get_atoms()]
        if atoms_of_removed_complexes:
            ns_removed = NeighborSearch(atoms_of_removed_complexes)
            for chain in model:
                if chain.id in chains_to_remove_from_complexes: continue
                for atom in chain.get_atoms():
                    if ns_removed.search(atom.coord, REMOVAL_CONTACT_DISTANCE, 'A'):
                        print(f"  ACTION: Chain {chain.id} is within {REMOVAL_CONTACT_DISTANCE}Å of a removed complex. Removing entire chain.")
                        chains_to_remove_from_complexes.add(chain.id)
                        break
        for chain_id in list(chains_to_remove_from_complexes):
            if chain_id in model: model.detach_child(chain_id)
    elif len(complexes) == 1:
        chains_in_kept_complex = complexes[0]
    else:
        print("  INFO: No interacting protein complexes found. Skipping file.")
        return
        
    # PART 3: Final cleaning of the kept complex and associated glycans
    for chain in chains_in_kept_complex:
        for res in list(chain.get_residues()):
            if not is_standard_residue(res) and not is_glycan(res):
                chain.detach_child(res.id)

    final_chains_to_keep_ids = {chain.id for chain in chains_in_kept_complex}
    atoms_in_kept_complex = [atom for chain in chains_in_kept_complex for atom in chain.get_atoms()]
    
    if not atoms_in_kept_complex:
        print("  INFO: Kept complex has no atoms after cleaning. Skipping file.")
        return
        
    ns_kept = NeighborSearch(atoms_in_kept_complex)
    
    for chain in model:
        if chain.id in final_chains_to_keep_ids: continue
        residues_in_chain = list(chain.get_residues())
        if residues_in_chain and all(is_glycan(res) for res in residues_in_chain):
            for atom in chain.get_atoms():
                if ns_kept.search(atom.coord, GLYCAN_CONTACT_DISTANCE, 'A'):
                    print(f"  INFO: Keeping glycan chain {chain.id} as it is within {GLYCAN_CONTACT_DISTANCE}Å of the complex.")
                    final_chains_to_keep_ids.add(chain.id)
                    break
                    
    # PART 4: Save the final, cleaned structure
    io = PDBIO()
    io.set_structure(structure)
    output_path = os.path.join(output_dir, filename)
    io.save(output_path, FinalStructureSelector(final_chains_to_keep_ids))
    print(f"  SUCCESS: Cleaned file saved to {output_path}")

# --- Script Execution ---

if __name__ == "__main__":
    input_directory = 'dockqc_dataset'
    output_directory = 'dockqc_dataset_cleaned'

    if not os.path.isdir(input_directory):
        print(f"Error: Input directory '{input_directory}' not found.")
        print("Please make sure this script is in the 'downloads' folder and 'dockqc_dataset' exists.")
    else:
        if os.path.exists(output_directory):
            print(f"Output directory '{output_directory}' already exists. Overwriting files within.")
        else:
            os.makedirs(output_directory)
            print(f"Created output directory: '{output_directory}'")

        for filename in os.listdir(input_directory):
            if filename.endswith('.pdb') or filename.endswith('.ent'):
                file_path = os.path.join(input_directory, filename)
                process_pdb_file(file_path, output_directory)

    print("\n--- All files processed. ---")
