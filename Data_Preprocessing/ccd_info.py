# Stereochemistry.json creator script

# ccd_info.py

import pickle
import sys
from pathlib import Path
import json
from typing import Union, List, Dict
import re

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolTransforms
from tqdm import tqdm

# Disable RDKit logging to keep the output clean and progress bar uninterrupted
rdBase.DisableLog('rdApp.*')

# The provided set of allowed glycan CCD codes
ALLOWED_GLYCANS = {
    "05L", "07E", "0HX", "0LP", "0MK", "0NZ", "0UB", "0WK", "0XY", "0YT", "12E", "145", "147", "149", "14T", "15L", "16F", "16G", "16O", "17T", "18D", "18O", "1CF", "1GL", "1GN", "1S3", "1S4", "1SD", "1X4", "20S", "20X",
    "22O", "22S", "23V", "24S", "25E", "26O", "27C", "289", "291", "293", "2DG", "2DR", "2F8", "2FG", "2FL", "2GL", "2GS", "2H5", "2M5", "2M8", "2WP", "32O", "34V", "38J", "3DO", "3FM", "3HD", "3J3", "3J4", "3LJ", "3MG",
    "3MK", "3R3", "3S6", "3YW", "42D", "445", "44S", "46Z", "475", "491", "49A", "49S", "49T", "49V", "4AM", "4CQ", "4GL", "4GP", "4JA", "4N2", "4NN", "4QY", "4R1", "4SG", "4U0", "4U1", "4U2", "4UZ", "4V5", "50A",
    "510", "51N", "56N", "57S", "5DI", "5GF", "5GO", "5KQ", "5KV", "5L2", "5L3", "5LS", "5LT", "5N6", "5QP", "5TH", "5TJ", "5TK", "5TM", "604", "61J", "62I", "64K", "66O", "6BG", "6C2", "6GB", "6GP", "6GR",
    "6K3", "6KH", "6KL", "6KS", "6KU", "6KW", "6LS", "6LW", "6MJ", "6MN", "6PY", "6PZ", "6S2", "6UD", "6Y6", "6YR", "6ZC", "73E", "79J", "7CV", "7D1", "7GP", "7JZ", "7K2", "7K3", "7NU", "83Y", "89Y", "8B7", "8B9", "8EX", "8GA",
    "8GG", "8GP", "8LM", "8LR", "8OQ", "8PK", "8S0", "95Z", "96O", "9AM", "9C1", "9CD", "9GP", "9KJ", "9MR", "9OK", "9PG", "9QG", "9QZ", "9S7", "9SG", "9SJ", "9SM", "9SP", "9T1", "9T7", "9VP", "9WJ", "9WN", "9WZ", "9YW",
    "A0K", "A1Q", "A2G", "A5C", "A6P", "AAL", "ABD", "ABE", "ABF", "ABL", "AC1", "ACR", "ADA", "AF1", "AFD", "AFO", "AFP", "AFR", "AGL", "AGR", "AH2", "AH8", "AHG", "AHM", "AHR", "AIG", "ALL", "ALX", "AMG", "AMN", "AMU",
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

def find_main_ring(mol: Chem.Mol) -> Union[list[int], None, str]:
    """
    Finds the largest non-aromatic ring in the molecule.
    """
    try:
        Chem.SanitizeMol(mol)
        rings = Chem.GetSymmSSSR(mol)
    except Exception:
        return None

    non_aromatic_rings = [
        list(ring_indices) for ring_indices in rings
        if not any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_indices)
    ]

    if not non_aromatic_rings:
        return None

    max_size = max(len(r) for r in non_aromatic_rings)
    largest_rings = [r for r in non_aromatic_rings if len(r) == max_size]

    if len(largest_rings) == 1:
        return largest_rings[0]
    else:
        return "MULTIPLE_RINGS"

def order_ring_atoms(mol: Chem.Mol, ring_indices: list[int]) -> Union[list[int], None]:
    """Orders the atoms in a ring by connectivity using a simple walk."""
    if not ring_indices or len(ring_indices) < 3:
        return None

    ring_set = set(ring_indices)
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring_set] for i in ring_indices}
    
    ordered_ring = [ring_indices[0]]
    prev_atom_idx = -1
    curr_atom_idx = ring_indices[0]

    while len(ordered_ring) < len(ring_indices):
        found_next = False
        for neighbor_idx in adj.get(curr_atom_idx, []):
            if neighbor_idx != prev_atom_idx:
                ordered_ring.append(neighbor_idx)
                prev_atom_idx = curr_atom_idx
                curr_atom_idx = neighbor_idx
                found_next = True
                break
        if not found_next:
            return None
            
    if ordered_ring[0] not in adj.get(ordered_ring[-1], []):
        return None 

    return ordered_ring

def get_atom_name(atom: Chem.Atom) -> str:
    """Safely gets the PDB atom name from the atom's properties."""
    props = atom.GetPropsAsDict()
    if 'name' in props:
        name_val = props['name']
        if isinstance(name_val, str) and name_val.strip():
            return name_val.strip()

    try:
        info = atom.GetMonomerInfo()
        if info:
            name = info.GetName().strip()
            if name:
                return name
    except Exception:
        pass

    if '_AtomName' in props:
        return str(props['_AtomName']).strip()
    if 'molFileAlias' in props:
        return str(props['molFileAlias']).strip()

    return f"UNKNOWN_ATOM_{atom.GetIdx()}"

def analyze_stereochemistry(mol: Chem.Mol) -> Union[List[list], None]:
    """
    Analyzes a sugar molecule to find the stereochemistry of its ring atoms.
    This robust version uses a universal C5-based reference and determines substituents
    by finding the heaviest exocyclic atom based on atomic number, not by name or a
    limited priority list.
    """
    # --- Nested Helper Functions ---
    def compute_ring_normal(conf: Chem.Conformer, ordered_ring: list[int]) -> np.ndarray:
        coords = np.array([np.array(conf.GetAtomPosition(i)) for i in ordered_ring], dtype=float)
        centroid = coords.mean(axis=0)
        X = coords - centroid
        cov = X.T @ X
        w, v = np.linalg.eigh(cov)
        n = v[:, np.argmin(w)]
        a = coords[1] - coords[0]
        b = coords[2] - coords[1]
        ref = np.cross(a, b)
        if np.dot(ref, n) < 0: n = -n
        return n / (np.linalg.norm(n) + 1e-12)

    def signed_side(n: np.ndarray, vec: np.ndarray) -> float:
        return float(np.dot(n, vec))

    def find_heaviest_exocyclic_neighbor(m: Chem.Mol, target_idx: int, ring_set: set[int]) -> Union[int, None]:
        """Finds the neighbor of target_idx with the highest atomic number that is not in the ring."""
        target_atom = m.GetAtomWithIdx(target_idx)
        exocyclic_neighbors = [n for n in target_atom.GetNeighbors() if n.GetIdx() not in ring_set]
        if not exocyclic_neighbors:
            return None
        
        heaviest_neighbor = max(exocyclic_neighbors, key=lambda atom: atom.GetAtomicNum())
        return heaviest_neighbor.GetIdx()

    # --- Main Logic ---
    mol = Chem.Mol(mol)
    Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True)

    main_ring_indices = find_main_ring(mol)
    if not isinstance(main_ring_indices, list): return None

    ordered_ring_indices = order_ring_atoms(mol, main_ring_indices)
    if not ordered_ring_indices: return None
    ring_set = set(ordered_ring_indices)
    
    # 1. Find Anomeric Carbon (C1 or C2)
    anomeric_carbon_idx = None
    for idx in ordered_ring_indices:
        if get_atom_name(mol.GetAtomWithIdx(idx)).upper() == 'C1':
            anomeric_carbon_idx = idx
            break
    if anomeric_carbon_idx is None:
        for idx in ordered_ring_indices:
            if get_atom_name(mol.GetAtomWithIdx(idx)).upper() == 'C2':
                anomeric_carbon_idx = idx
                break
    if anomeric_carbon_idx is None: return None

    # 2. Find Anomeric Substituent (Heaviest Exocyclic Neighbor)
    anomeric_substituent_idx = find_heaviest_exocyclic_neighbor(mol, anomeric_carbon_idx, ring_set)
    if anomeric_substituent_idx is None: return None

    # 3. Find C5 Reference Atom (Heaviest Exocyclic Neighbor of C5)
    c5_idx = next((idx for idx in ordered_ring_indices if get_atom_name(mol.GetAtomWithIdx(idx)).upper() == 'C5'), -1)
    if c5_idx == -1: return None
    
    c5_substituent_idx = find_heaviest_exocyclic_neighbor(mol, c5_idx, ring_set)
    if c5_substituent_idx is None: return None

    # 4. Perform Geometric Calculation
    try:
        conf = mol.GetConformer()
    except ValueError:
        return None
    
    n = compute_ring_normal(conf, ordered_ring_indices)

    # Anomeric vector projection
    a_pos = np.array(conf.GetAtomPosition(anomeric_carbon_idx), dtype=float)
    exo_pos = np.array(conf.GetAtomPosition(anomeric_substituent_idx), dtype=float)
    sign_glyco = signed_side(n, exo_pos - a_pos)
    if abs(sign_glyco) < 1e-3: return None # Ambiguous

    # Reference vector projection
    c5_pos = np.array(conf.GetAtomPosition(c5_idx), dtype=float)
    substituent_pos = np.array(conf.GetAtomPosition(c5_substituent_idx), dtype=float)
    sign_ref = signed_side(n, substituent_pos - c5_pos)
    if abs(sign_ref) < 1e-3: return None # Ambiguous

    ab_config = 'a' if (sign_glyco * sign_ref) < 0.0 else 'b'

    # 5. Collect and return results
    stereo_results = []
    for atom_idx in ordered_ring_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_name = get_atom_name(atom)
        rs_config = '?'
        if atom.HasProp('_CIPCode'):
            cip_code = atom.GetProp('_CIPCode')
            if cip_code in ('R', 'S'):
                rs_config = cip_code

        if atom_idx == anomeric_carbon_idx:
            stereo_results.append([atom_name, ab_config, rs_config])
        elif rs_config != '?':
            stereo_results.append([atom_name, rs_config])

    return stereo_results if stereo_results else None

def main():
    """
    Main function to analyze sugar stereochemistry from a CCD pickle file.
    This version manually constructs the JSON output for maximum performance and
    to achieve a specific, human-readable format. The anomeric carbon entry and
    its flipped configuration are placed on the same line.

    Modified: exclude any sugars whose stereochemistry lists contain '?' in any position.
    """
    CCD_PATH = Path('/Users/keshavsundar/Downloads/ccd.pkl')
    OUTPUT_JSON_FILE = Path('stereochemistry.json')

    if not CCD_PATH.exists():
        print(f"Error: CCD file not found at '{CCD_PATH}'.", file=sys.stderr)
        return

    with open(CCD_PATH, 'rb') as f:
        ccd_mols = pickle.load(f)

    codes_to_process = sorted(list(ALLOWED_GLYCANS.intersection(ccd_mols.keys())))
    
    final_data = {}

    # --- Step 1: Process all sugars and prepare the data structure in memory ---
    for ccd_code in tqdm(codes_to_process, desc="Analyzing Sugar Stereochemistry"):
        mol_orig = ccd_mols.get(ccd_code)
        if not mol_orig or mol_orig.GetNumAtoms() < 3:
            continue
        
        try:
            mol_orig.GetConformer()
        except ValueError:
            continue

        original_stereo_list = analyze_stereochemistry(mol_orig)
        
        if original_stereo_list:
            # This list will hold all stereo info for one sugar.
            # The anomeric pair will be the first two items.
            combined_stereo_list = []
            
            # Find the anomeric item (has 3 elements) to process it first.
            anomeric_item = None
            anomeric_idx = -1
            for i, item in enumerate(original_stereo_list):
                if len(item) == 3:
                    anomeric_item = item
                    anomeric_idx = i
                    break
            
            if anomeric_item:
                # Add the original anomeric entry
                combined_stereo_list.append(anomeric_item)
                
                # Add the flipped anomeric entry right after it
                atom_name, original_ab, original_rs = anomeric_item
                flipped_ab = 'b' if original_ab == 'a' else 'a'
                flipped_rs = 'S' if original_rs == 'R' else ('R' if original_rs == 'S' else '?')
                combined_stereo_list.append([atom_name, flipped_ab, flipped_rs])

                # Add the rest of the non-anomeric items
                for i, item in enumerate(original_stereo_list):
                    if i != anomeric_idx:
                        combined_stereo_list.append(item)

                # --- NEW FILTER: Skip sugars that contain any '?' anywhere in their stereochemistry entries ---
                has_unknown = any(
                    (isinstance(elem, str) and '?' in elem)
                    for entry in combined_stereo_list
                    for elem in entry
                )
                if has_unknown:
                    continue  # exclude this sugar entirely from the JSON output

                final_data[ccd_code] = combined_stereo_list

    # --- Step 2: Write the prepared data to a file with custom formatting ---
    with open(OUTPUT_JSON_FILE, 'w') as out_f:
        out_f.write('{\n') # Start of the JSON object

        # Use an iterator to elegantly handle commas between entries
        data_iterator = iter(final_data.items())
        
        # Write the first entry without a leading comma
        try:
            first_ccd_code, first_stereo_list = next(data_iterator)
            
            # Write key and start of list
            out_f.write(f'  "{first_ccd_code}": [\n')
            
            # Write the special, combined anomeric line
            anomer_original = json.dumps(first_stereo_list[0])
            anomer_flipped = json.dumps(first_stereo_list[1])
            out_f.write(f'    {anomer_original} , {anomer_flipped}')
            
            # Write the rest of the items, each on a new line
            for item in first_stereo_list[2:]:
                out_f.write(',\n')
                out_f.write(f'    {json.dumps(item)}')
            
            out_f.write('\n  ]') # Close the list

        except StopIteration:
            # This handles the case where final_data is empty
            pass

        # Write all subsequent entries, each prefixed with a comma
        for ccd_code, stereo_list in data_iterator:
            out_f.write(',\n') # Comma separating from the previous entry
            
            out_f.write(f'  "{ccd_code}": [\n')
            
            anomer_original = json.dumps(stereo_list[0])
            anomer_flipped = json.dumps(stereo_list[1])
            out_f.write(f'    {anomer_original} , {anomer_flipped}')
            
            for item in stereo_list[2:]:
                out_f.write(',\n')
                out_f.write(f'    {json.dumps(item)}')
            
            out_f.write('\n  ]')

        out_f.write('\n}\n') # End of the JSON object

    print(f"\nAnalysis complete.")
    print(f"Results for {len(final_data)} valid CCDs saved to '{OUTPUT_JSON_FILE}'")
        
if __name__ == '__main__':
    main()
