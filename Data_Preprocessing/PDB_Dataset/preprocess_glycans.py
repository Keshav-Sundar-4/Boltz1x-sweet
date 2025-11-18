# pre_process_glycan.py

import argparse
import json
import multiprocessing
import pickle
import traceback
import re
import os
import time
import sys
import site
from dataclasses import asdict, dataclass, replace
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple, Mapping, Set
from scipy.spatial import cKDTree
import random
from contextlib import redirect_stdout, redirect_stderr
import string

import numpy as np
import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, Conformer

import boltz
from boltz.data import const
from boltz.data.feature.featurizer import MONO_TYPE_MAP
from boltz.data.types import (
    Atom, Bond, Residue, Chain, Connection, Interface, Structure, StructureInfo,
    Record, Target, ChainInfo, InterfaceInfo, GlycosylationSite
    # We don't need MSA types here
)
# Import necessary helper functions (adapt paths if needed)
from boltz.data.parse.schema import (
    get_conformer, convert_atom_name, parse_ccd_residue,
    stitch_monosaccharides, relax_glycan_structure
)
# Need ParsedAtom, ParsedBond, ParsedResidue, ParsedChain for intermediate representation
#from boltz.data.parse.mmcif import ParsedAtom, ParsedBond, ParsedResidue, ParsedChain


from tqdm import tqdm

@dataclass(frozen=True, slots=True)
class ParsedAtom:
    """A parsed atom object."""
    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int

@dataclass(frozen=True, slots=True)
class ParsedBond:
    """A parsed bond object."""
    atom_1: int
    atom_2: int
    type: int

@dataclass(frozen=True, slots=True)
class ParsedResidue:
    """A parsed residue object."""
    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool

@dataclass(frozen=True, slots=True)
class ParsedChain:
    """A parsed chain object."""
    name: str
    entity: str
    type: str
    residues: list[ParsedResidue]
    sequence: list[str]

# Define connection types BEFORE they are used in ParsedGlycoproteinData
@dataclass(frozen=True, slots=True)
class GlycoConnection:
    """Represents a detected glycosidic linkage."""
    parent_chain_id: str
    child_chain_id: str
    parent_res_id: int
    child_res_id: int
    parent_acceptor_atom_name: str # e.g., "O4"
    child_donor_atom_name: str     # e.g., "C1"
    anomeric: Optional[str]        # 'a', 'b', or None

@dataclass(frozen=True, slots=True)
class GlycosylationSiteConnection:
    """Represents a detected protein-glycan linkage."""
    protein_chain_id: str
    protein_res_id: int
    protein_atom_name: str
    glycan_chain_id: str
    glycan_res_id: int
    glycan_atom_name: str

# Now, define the main data container, which can safely reference the classes above.
@dataclass
class ParsedGlycoproteinData:
    """Intermediate storage for parsed glycoprotein data."""
    pdb_id: str
    chains: Dict[str, ParsedChain]
    glycosidic_connections: List[GlycoConnection]
    glycosylation_sites: List[GlycosylationSiteConnection]

@dataclass(frozen=True, slots=True)
class PDBFile:
    """Represents a raw PDB input file."""
    id: str
    path: Path
    cluster_num: int
    frame_num: int

@dataclass(frozen=True)
class AnomericAtom:
    """Represents an atom for anomeric configuration analysis."""
    idx: int
    name: str
    res_name: str
    coords: np.ndarray
    element: str



worker_ccd_data = None # Global variable placeholder for worker processes

standard_amino_acids = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
}

ATOMIC_NUMBERS = { 'H': 1, 'HE': 2, 'LI': 3, 'BE': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'NE': 10, 'NA': 11, 'MG': 12, 'AL': 13, 'SI': 14, 'P': 15, 'S': 16, 'CL': 17, 'AR': 18, 'K': 19, 'CA': 20, 'SC': 21, 'TI': 22, 'V': 23, 'CR': 24, 'MN': 25, 'FE': 26, 'CO': 27, 'NI': 28, 'CU': 29, 'ZN': 30, 'GA': 31, 'GE': 32, 'AS': 33, 'SE': 34, 'BR': 35, 'KR': 36, 'RB': 37, 'SR': 38, 'Y': 39, 'ZR': 40, 'NB': 41, 'MO': 42, 'TC': 43, 'RU': 44, 'RH': 45, 'PD': 46, 'AG': 47, 'CD': 48, 'IN': 49, 'SN': 50, 'SB': 51, 'TE': 52, 'I': 53, 'XE': 54, 'CS': 55, 'BA': 56, 'LA': 57, 'CE': 58, 'PR': 59, 'ND': 60, 'PM': 61, 'SM': 62, 'EU': 63, 'GD': 64, 'TB': 65, 'DY': 66, 'HO': 67, 'ER': 68, 'TM': 69, 'YB': 70, 'LU': 71, 'HF': 72, 'TA': 73, 'W': 74, 'RE': 75, 'OS': 76, 'IR': 77, 'PT': 78, 'AU': 79, 'HG': 80, 'TL': 81, 'PB': 82, 'BI': 83, 'PO': 84, 'AT': 85, 'RN': 86 }


# --- Constants ---
GLYCOSIDIC_BOND_THRESHOLD = 2.0 # Angstrom distance to detect potential glycosidic linkages
BOND_TYPE_SINGLE = const.bond_type_ids.get("SINGLE", 1) # Default to 1 if not found

class NumpyJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that converts NumPy data types to native Python types,
    making them serializable.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

# These dataclasses are temporary, for use by the IUPAC generation logic.
@dataclass(frozen=True)
class AnomericTestAtom:
    idx: int; name: str; res_name: str; chain_id: str; res_num: int; element: str; coords: np.ndarray

@dataclass(frozen=True)
class AnomericTestGlycoConnection:
    parent_res_key: Tuple[str, int]; child_res_key: Tuple[str, int]; parent_acceptor_atom: AnomericTestAtom
    child_donor_atom: AnomericTestAtom; anomeric_config: Optional[str]

# --- Utility Functions ---
def get_glycan_sequence_key(
    pdb_file: PDBFile,
) -> Tuple[str, Optional[str]]:
    """
    (REVISED & FIXED) Generates a sequence key for clustering based on the
    pre-separated chains in the input file.

    This strategy now implements two paths:
    1. If a file contains ANY protein chains, it is assigned a unique
       cluster key to ensure it is processed uniquely as a glycoprotein complex.
    2. If a file contains ONLY glycan chains, it is clustered using the
       4-bin glycan size strategy based on the total number of monosaccharides.
    """
    try:
        # --- FIX START ---
        # The file must be opened and its lines read before being passed to the parser.
        with open(pdb_file.path, 'r', encoding='latin-1') as f:
            pdb_lines = f.readlines()
        
        # Now, pass the list of lines, not the path object.
        atoms_by_residue, chain_types = parse_pdb_atoms_by_residue(pdb_lines)
        # --- FIX END ---

        if not atoms_by_residue:
            return pdb_file.id, None

        has_protein_component = any(typ == "PROTEIN" for typ in chain_types.values())

        if has_protein_component:
            # Path 1: It's a glycoprotein complex. Assign a unique cluster key.
            sequence_key = f"PROTEIN_COMPLEX_{pdb_file.id}"
            return pdb_file.id, sequence_key
        else:
            # Path 2: It's a glycan-only file. Cluster by size.
            glycan_residue_count = sum(
                1 for atoms in atoms_by_residue.values()
                if atoms and atoms[0]['residue_name'] in MONO_TYPE_MAP and MONO_TYPE_MAP[atoms[0]['residue_name']] != "OTHER"
            )

            if glycan_residue_count > 0:
                if glycan_residue_count == 1:
                    sequence_key = "GLYCAN_1-mer"
                elif glycan_residue_count == 2:
                    sequence_key = "GLYCAN_2-mer"
                elif 3 <= glycan_residue_count <= 6:
                    sequence_key = "GLYCAN_3-6-mer"
                else:  # glycan_residue_count >= 7
                    sequence_key = "GLYCAN_7+-mer"
                return pdb_file.id, sequence_key
            else:
                return pdb_file.id, None

    except Exception:
        tb_str = traceback.format_exc()
        print(
            f"--- TOP-LEVEL EXCEPTION in get_glycan_sequence_key for {pdb_file.id} ---\n{tb_str}\n--------------------",
            file=sys.stderr,
            flush=True
        )
        return pdb_file.id, None

def _generate_chain_ids(exclude=None):
    """
    Efficiently generates a sequence of unique chain IDs, skipping any
    that are in the 'exclude' set (e.g., existing protein chains).
    """
    if exclude is None:
        exclude = set()

    # 1. Uppercase letters (A-Z)
    for char in string.ascii_uppercase:
        if char not in exclude: yield char
        
    # 2. Lowercase letters (a-z)
    for char in string.ascii_lowercase:
        if char not in exclude: yield char
        
    # 3. Digits (0-9)
    for char in string.digits:
        if char not in exclude: yield char
    
    # 4. Two-character combinations (for > 62 chains)
    two_char_pool = string.ascii_uppercase + string.digits
    for char1 in two_char_pool:
        for char2 in two_char_pool:
            combo = char1 + char2
            if combo not in exclude: yield combo

def _rechain_glycan_components(pdb_lines: List[str], inter_residue_threshold: float = 2.0) -> List[str]:
    """
    (PORTED LOGIC) Separates glycan trees into new chains based on residue-level
    proximity, ensuring new Chain IDs do not conflict with existing protein chains.
    """
    if not pdb_lines:
        return []

    # 1. Group atoms by residue, identify glycans, and find used protein chains
    atoms_by_residue = defaultdict(list)
    protein_chain_ids = set()

    for line in pdb_lines:
        res_name = line[17:20].strip()
        chain_id = line[21]
        
        # Use MONO_TYPE_MAP as the definition of a glycan for consistency
        is_glycan = res_name in MONO_TYPE_MAP and MONO_TYPE_MAP[res_name] != "OTHER"
        
        if not is_glycan and res_name in standard_amino_acids:
            protein_chain_ids.add(chain_id)

        # Key: (chain, res_num, insertion_code)
        residue_key = (line[21], line[22:26].strip(), line[26])
        atoms_by_residue[residue_key].append({
            'line': line,
            'coords': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            'is_glycan': is_glycan
        })
    
    glycan_residue_keys = [key for key, atoms in atoms_by_residue.items() if atoms and atoms[0]['is_glycan']]

    if not glycan_residue_keys:
        return pdb_lines # No glycans to rechain, return original lines

    # 2. Build inter-residue graph for GLYCANS ONLY
    flat_glycan_atoms = []
    atom_idx_to_res_key = []
    for res_key in glycan_residue_keys:
        for atom in atoms_by_residue[res_key]:
            # Only consider heavy atoms for linkage detection
            if not atom['line'][12:16].strip().startswith('H'):
                flat_glycan_atoms.append(atom['coords'])
                atom_idx_to_res_key.append(res_key)
    
    adj_residues = defaultdict(set)
    if flat_glycan_atoms:
        tree = cKDTree(np.array(flat_glycan_atoms))
        pairs = tree.query_pairs(r=inter_residue_threshold)
        for i, j in pairs:
            res_key1 = atom_idx_to_res_key[i]
            res_key2 = atom_idx_to_res_key[j]
            if res_key1 != res_key2:
                adj_residues[res_key1].add(res_key2)
                adj_residues[res_key2].add(res_key1)

    # 3. Find connected components and assign new, unique chain IDs
    visited_residues = set()
    chain_id_gen = _generate_chain_ids(exclude=protein_chain_ids)
    final_lines = []
    processed_glycan_residue_keys = set()

    for res_key in glycan_residue_keys:
        if res_key not in visited_residues:
            component_residue_keys = set()
            q = deque([res_key])
            visited_residues.add(res_key)
            
            while q:
                current_res_key = q.popleft()
                component_residue_keys.add(current_res_key)
                for neighbor_res_key in adj_residues.get(current_res_key, []):
                    if neighbor_res_key not in visited_residues:
                        visited_residues.add(neighbor_res_key)
                        q.append(neighbor_res_key)
            
            new_chain_id = next(chain_id_gen)

            for key in component_residue_keys:
                processed_glycan_residue_keys.add(key)
                for atom in atoms_by_residue[key]:
                    line = atom['line']
                    # PDB format handles 1 and 2 character chain IDs differently
                    if len(new_chain_id) == 1:
                        new_line = f"{line[:21]}{new_chain_id}{line[22:]}"
                    else: # Two-character chain ID
                        new_line = f"{line[:20]}{new_chain_id.ljust(2)}{line[22:]}"
                    final_lines.append(new_line)

    # 4. Append all original non-glycan lines
    for res_key, atoms_data in atoms_by_residue.items():
        if res_key not in processed_glycan_residue_keys:
            for atom in atoms_data:
                final_lines.append(atom['line'])
                
    return final_lines

def parse_pdb_atoms_by_residue(pdb_lines: List[str]) -> Tuple[Dict[Tuple[str, int], List[Dict]], Dict[str, str]]:
    """
    (MODIFIED) Parses a list of PDB lines, grouping atoms by their 
    (chain_id, residue_number) key. It also determines the type of each chain
    (PROTEIN or GLYCAN).

    Args:
        pdb_lines: A list of strings, where each string is a line from a PDB file.

    Returns:
        A tuple containing:
        - A dictionary mapping residue keys to lists of atom dictionaries.
        - A dictionary mapping chain IDs to their determined type ('PROTEIN' or 'GLYCAN').
    """
    atoms_by_residue = defaultdict(list)
    chain_has_protein = defaultdict(bool)
    chain_has_glycan = defaultdict(bool)

    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')):
            try:
                res_name = line[17:20].strip().upper()
                # Correctly parse chain IDs that may now be two characters
                chain_id = line[20:22].strip() or "A" # Default to 'A' if chain is blank
                res_num = int(line[22:26].strip())
                res_key = (chain_id, res_num)

                atom_dict = {
                    'atom_number':   int(line[6:11].strip()),
                    'atom_name':     line[12:16].strip(),
                    'residue_name':  res_name,
                    'chain_id':      chain_id,
                    'residue_num':   res_num,
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': (line[76:78].strip() or line[12:16].strip()[0]).upper()
                }
                atoms_by_residue[res_key].append(atom_dict)

                if res_name in standard_amino_acids:
                    chain_has_protein[chain_id] = True
                elif res_name in MONO_TYPE_MAP and MONO_TYPE_MAP[res_name] != "OTHER":
                    chain_has_glycan[chain_id] = True

            except (ValueError, IndexError):
                continue

    chain_types = {}
    all_chains = set(chain_has_protein.keys()) | set(chain_has_glycan.keys())
    for chain_id in all_chains:
        if chain_has_protein[chain_id]:
            chain_types[chain_id] = "PROTEIN"
        elif chain_has_glycan[chain_id]:
            chain_types[chain_id] = "GLYCAN"

    return dict(atoms_by_residue), chain_types

def detect_all_connections(
    atoms_by_residue: Dict[Tuple[str, int], List[Dict]],
    pdb_id: str = "UNKNOWN"
) -> Tuple[List[GlycoConnection], List[GlycosylationSiteConnection], Dict[str, Dict[str, int]], List[Tuple[str, str, int, str]], List[Tuple[str, str, int]]]:
    """
    (CORRECTED) Detects linkages and correctly separates glycosylation sites from
    glycosidic bonds.
    """
    threshold = 2.0
    glycosidic_conns_temp: List[GlycoConnection] = []
    glycosylation_sites_temp: List[GlycosylationSiteConnection] = []
    site_stats = defaultdict(lambda: defaultdict(int))
    anomalous_sites = []
    no_ring_errors = []

    flat_atoms = []
    all_coords = []
    is_standard_aa = {
        key: atoms_by_residue[key][0]['residue_name'] in standard_amino_acids
        for key in atoms_by_residue if atoms_by_residue[key]
    }

    for res_key, atoms in atoms_by_residue.items():
        if not atoms: continue
        is_aa = is_standard_aa.get(res_key, False)
        for atom_dict in atoms:
            if atom_dict['atom_name'].startswith('H'): continue
            coords = np.array([atom_dict['x'], atom_dict['y'], atom_dict['z']])
            all_coords.append(coords)
            flat_atoms.append({'res_key': res_key, 'is_aa': is_aa, 'atom_dict': atom_dict})

    if len(flat_atoms) < 2: return [], [], {}, [], []

    tree = cKDTree(np.array(all_coords))
    nearby_pairs = tree.query_pairs(r=threshold, output_type='set')

    for i, j in nearby_pairs:
        atom1_info, atom2_info = flat_atoms[i], flat_atoms[j]
        res1_key, is_aa1 = atom1_info['res_key'], atom1_info['is_aa']
        res2_key, is_aa2 = atom2_info['res_key'], atom2_info['is_aa']

        if res1_key == res2_key: continue

        # Case 1: Protein-Glycan linkage (Glycosylation Site)
        if is_aa1 != is_aa2:
            protein_info, glycan_info = (atom1_info, atom2_info) if is_aa1 else (atom2_info, atom1_info)
            protein_atom_dict = protein_info['atom_dict']
            glycan_atom_dict = glycan_info['atom_dict']
            
            anom_config_site = determine_anomeric_config_universal(
                child_residue_dicts=atoms_by_residue[glycan_info['res_key']],
                acceptor_atom_dict=protein_atom_dict,
                donor_atom_dict=glycan_atom_dict
            )
            
            protein_res_name = protein_atom_dict['residue_name']
            protein_res_id = protein_info['res_key'][1]
            site_stats[protein_res_name][anom_config_site] += 1

            if 'no ring' in anom_config_site:
                no_ring_errors.append((pdb_id, protein_res_name, protein_res_id))
            else:
                is_alpha_asn = (protein_res_name == "ASN" and anom_config_site == 'a')
                is_beta_thr = (protein_res_name == "THR" and anom_config_site == 'b')
                is_beta_ser = (protein_res_name == "SER" and anom_config_site == 'b')
                if is_alpha_asn or is_beta_thr or is_beta_ser:
                    config_str = 'alpha' if anom_config_site == 'a' else 'beta'
                    anomalous_sites.append((pdb_id, protein_res_name, protein_res_id, config_str))
            
            # --- THE BUG WAS HERE ---
            # The following line incorrectly added protein-glycan links to the sugar-sugar bond list.
            # It has been REMOVED.
            
            # This is the CORRECT list for this type of connection.
            glycosylation_sites_temp.append(GlycosylationSiteConnection(
                protein_chain_id=protein_info['res_key'][0], protein_res_id=protein_info['res_key'][1],
                protein_atom_name=protein_atom_dict['atom_name'],
                glycan_chain_id=glycan_info['res_key'][0], glycan_res_id=glycan_info['res_key'][1],
                glycan_atom_name=glycan_atom_dict['atom_name'],
            ))

        # Case 2: Glycan-Glycan linkage (Glycosidic Bond)
        elif not is_aa1 and not is_aa2:
            a1_dict, a2_dict = atom1_info['atom_dict'], atom2_info['atom_dict']
            parent_key, child_key, parent_atom, child_atom = (None, None, None, None)
            
            config1 = determine_anomeric_config_universal(atoms_by_residue[res2_key], a1_dict, a2_dict)
            if 'error' not in config1:
                parent_key, child_key = res1_key, res2_key
                parent_atom, child_atom = a1_dict, a2_dict
                anom = config1
            else:
                config2 = determine_anomeric_config_universal(atoms_by_residue[res1_key], a2_dict, a1_dict)
                if 'error' not in config2:
                    parent_key, child_key = res2_key, res1_key
                    parent_atom, child_atom = a2_dict, a1_dict
                    anom = config2

            if parent_key:
                glycosidic_conns_temp.append(GlycoConnection(
                    parent_chain_id=parent_key[0], child_chain_id=child_key[0],
                    parent_res_id=parent_key[1], child_res_id=child_key[1],
                    parent_acceptor_atom_name=parent_atom['atom_name'],
                    child_donor_atom_name=child_atom['atom_name'], anomeric=anom
                ))

    # Remove duplicate sugar-sugar connections
    unique_glycosidic_conns: List[GlycoConnection] = []
    seen_glyco_pairs: Set[Tuple[Tuple[str, int], Tuple[str, int]]] = set()
    for conn in glycosidic_conns_temp:
        pair = tuple(sorted(((conn.parent_chain_id, conn.parent_res_id), (conn.child_chain_id, conn.child_res_id))))
        if pair not in seen_glyco_pairs:
            unique_glycosidic_conns.append(conn)
            seen_glyco_pairs.add(pair)
            
    return unique_glycosidic_conns, glycosylation_sites_temp, dict(site_stats), anomalous_sites, no_ring_errors

def find_atom_by_name(atoms: List[Dict], name: str) -> Optional[Dict]:
    """Finds the first atom matching the name (case-insensitive)."""
    for atom in atoms:
        if atom['atom_name'].upper() == name.upper():
            return atom
    return None


def build_residue_graph(atoms: List[AnomericAtom]) -> Dict[int, List[int]]:
    """Builds a covalent bond graph for a list of atoms using a distance threshold."""
    COVALENT_BOND_THRESHOLD_INTERNAL = 2.0  # CHANGED AS REQUESTED
    graph = defaultdict(list)
    if len(atoms) < 2:
        return graph
    coords = np.array([a.coords for a in atoms])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=COVALENT_BOND_THRESHOLD_INTERNAL)
    for i, j in pairs:
        graph[atoms[i].idx].append(atoms[j].idx)
        graph[atoms[j].idx].append(atoms[i].idx)
    return graph

def find_and_order_ring(graph: Dict[int, List[int]], start_node_idx: int, atoms_map: Dict[int, AnomericAtom]) -> Optional[List[int]]:
    """
    Finds and canonically orders a 5 or 6-membered sugar ring containing the start node.
    This version validates that the ring has the correct chemical composition
    (n-1 carbons and 1 heteroatom).
    """
    from collections import deque

    for ring_size in [6, 5]:
        q = deque([(start_node_idx, [start_node_idx])])
        visited_paths = {frozenset([start_node_idx])}

        while q:
            curr_idx, path = q.popleft()

            if len(path) == ring_size:
                if start_node_idx in graph.get(curr_idx, []):
                    ring_atoms = [atoms_map[i] for i in path]
                    num_carbons = sum(1 for atom in ring_atoms if atom.element == 'C')
                    num_heteroatoms = ring_size - num_carbons

                    is_valid_sugar_ring = (ring_size == 6 and num_carbons == 5 and num_heteroatoms == 1) or \
                                          (ring_size == 5 and num_carbons == 4 and num_heteroatoms == 1)

                    if not is_valid_sugar_ring:
                        continue # Ring found, but it's not a valid sugar ring.

                    # Ring is valid, now order it
                    c1_neighbors_in_ring = [idx for idx in graph.get(start_node_idx, []) if idx in path]
                    c2_candidate = -1
                    # Find the neighbor that is NOT the ring oxygen to start the walk
                    for neighbor_idx in c1_neighbors_in_ring:
                        if atoms_map[neighbor_idx].element != 'O':
                             c2_candidate = neighbor_idx
                             break
                    
                    if c2_candidate == -1:
                        continue # Could not determine walk direction

                    ordered_ring = [start_node_idx, c2_candidate]
                    prev_node, curr_node = start_node_idx, c2_candidate
                    while len(ordered_ring) < ring_size:
                        found_next = False
                        for neighbor in graph.get(curr_node, []):
                            if neighbor in path and neighbor != prev_node:
                                ordered_ring.append(neighbor)
                                prev_node, curr_node = curr_node, neighbor
                                found_next = True
                                break
                        if not found_next:
                             ordered_ring = [] # Should not happen in a valid ring
                             break
                    
                    if ordered_ring:
                        return ordered_ring

                continue

            for neighbor_idx in graph.get(curr_idx, []):
                # This check prevents walking backwards immediately
                if len(path) > 1 and neighbor_idx == path[-2]:
                    continue
                
                new_path_set = frozenset(path + [neighbor_idx])
                if new_path_set not in visited_paths:
                    q.append((neighbor_idx, path + [neighbor_idx]))
                    visited_paths.add(new_path_set)
    return None

def _compute_ring_normal(ordered_ring_indices: List[int], atoms_map: Dict[int, AnomericAtom]) -> Optional[np.ndarray]:
    """
    Compute a stable ring-plane normal via PCA of ring atom positions.
    Orientation is made deterministic using a cross product reference.
    """
    try:
        coords = np.array([atoms_map[i].coords for i in ordered_ring_indices], dtype=float)
        centroid = coords.mean(axis=0)
        X = coords - centroid
        cov = X.T @ X
        w, v = np.linalg.eigh(cov)
        n = v[:, np.argmin(w)]
        # Fix orientation using first/last edges as reference
        a = coords[1] - coords[0]
        b = coords[-2] - coords[0]
        ref = np.cross(a, b)
        if np.dot(ref, n) < 0:
            n = -n
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            return None
        return n / norm
    except Exception:
        return None

def _signed_projection(n: np.ndarray, vec: np.ndarray) -> float:
    """Signed projection of vec onto unit normal n."""
    return float(np.dot(n, vec))

def _find_heaviest_exocyclic_substituent(
    target_atom: AnomericAtom,
    ring_indices: Set[int],
    graph: Dict[int, List[int]],
    atoms_map: Dict[int, AnomericAtom]
) -> Optional[AnomericAtom]:
    """
    Finds the heaviest exocyclic substituent attached to a target atom based on atomic number.
    """
    exocyclic_neighbors = [idx for idx in graph.get(target_atom.idx, []) if idx not in ring_indices]
    if not exocyclic_neighbors:
        return None

    best_substituent = None
    max_atomic_num = -1

    for nbr_idx in exocyclic_neighbors:
        neighbor_atom = atoms_map.get(nbr_idx)
        if not neighbor_atom: continue
        
        atomic_num = ATOMIC_NUMBERS.get(neighbor_atom.element.upper(), 0)
        if atomic_num > max_atomic_num:
            max_atomic_num = atomic_num
            best_substituent = neighbor_atom
    
    return best_substituent

def _find_c5_reference_substituent(
    ordered_ring_indices: List[int],
    ring_set: Set[int],
    graph: Dict[int, List[int]],
    atoms_map: Dict[int, AnomericAtom]
) -> Optional[Tuple[AnomericAtom, AnomericAtom]]:
    """
    (CORRECTED) Finds the reference vector based strictly on the C5 atom and its 
    heaviest exocyclic substituent. Returns a tuple of (C5_atom, substituent_atom) 
    or None if C5 is not found.
    """
    # Step 1: Invariantly search for the C5 atom within the ordered ring.
    c5_atom = next((atoms_map[idx] for idx in ordered_ring_indices if atoms_map[idx].name.upper() == 'C5'), None)
    
    # Step 2: If C5 is not found, fail immediately. Do not fall back to C4.
    if c5_atom is None:
        return None
        
    # Step 3: Find the heaviest exocyclic substituent attached to the C5 atom.
    heaviest_substituent = _find_heaviest_exocyclic_substituent(c5_atom, ring_set, graph, atoms_map)
    
    # Step 4: If no such substituent exists, fail.
    if heaviest_substituent is None:
        return None
        
    # Step 5: Return the valid (C5 atom, substituent atom) pair.
    return (c5_atom, heaviest_substituent)

def determine_anomeric_config_universal(
    child_residue_dicts: List[Dict],
    acceptor_atom_dict: Dict,
    donor_atom_dict: Dict
) -> str:
    """
    (REVISED) Unified α/β assignment using a C5-based reference rule and robust
    ring validation.
    """
    # Adapter to convert dicts to AnomericAtom objects
    child_residue_atoms: List[AnomericAtom] = [
        AnomericAtom(
            idx=a['atom_number'], name=a['atom_name'], res_name=a['residue_name'],
            coords=np.array([a['x'], a['y'], a['z']]), element=a['element']
        ) for a in child_residue_dicts
    ]
    acceptor_atom = AnomericAtom(
        idx=acceptor_atom_dict['atom_number'], name=acceptor_atom_dict['atom_name'], res_name=acceptor_atom_dict['residue_name'],
        coords=np.array([acceptor_atom_dict['x'], acceptor_atom_dict['y'], acceptor_atom_dict['z']]), element=acceptor_atom_dict['element']
    )
    donor_atom = AnomericAtom(
        idx=donor_atom_dict['atom_number'], name=donor_atom_dict['atom_name'], res_name=donor_atom_dict['residue_name'],
        coords=np.array([donor_atom_dict['x'], donor_atom_dict['y'], donor_atom_dict['z']]), element=donor_atom_dict['element']
    )

    try:
        graph = build_residue_graph(child_residue_atoms)
        atoms_map = {a.idx: a for a in child_residue_atoms}
        
        ordered_ring_indices = find_and_order_ring(graph, donor_atom.idx, atoms_map)
        if not ordered_ring_indices or len(ordered_ring_indices) not in [5, 6]:
            return "error (no ring)"
        
        if donor_atom.idx not in ordered_ring_indices:
            return "error (donor not in ring)"

        ring_set = set(ordered_ring_indices)
        n = _compute_ring_normal(ordered_ring_indices, atoms_map)
        if n is None: 
            return "error (collinear plane)"

        # Anomeric vector
        donor_pos, acceptor_pos = donor_atom.coords, acceptor_atom.coords
        sign_glyco = _signed_projection(n, acceptor_pos - donor_pos)
        if abs(sign_glyco) < 1e-3: return "error (ambiguous plane)"

        # Reference vector (C5/C4 based)
        ref_pair = _find_c5_reference_substituent(ordered_ring_indices, ring_set, graph, atoms_map)
        if ref_pair is None:
            return "error (no C5/C4 reference)"
        
        ref_base_atom, ref_substituent_atom = ref_pair
        ref_base_pos, ref_substituent_pos = ref_base_atom.coords, ref_substituent_atom.coords
        sign_ref = _signed_projection(n, ref_substituent_pos - ref_base_pos)
        if abs(sign_ref) < 1e-3: return "error (ambiguous plane)"

        return "a" if (sign_glyco * sign_ref) < 0.0 else "b"
        
    except Exception:
        return "error (exception)"

def _determine_root_anomeric_config(
    atoms_by_residue: Dict[Tuple[str, int], List[Dict]],
    existing_connections: List[GlycoConnection]
) -> List[GlycoConnection]:
    """
    (CORRECTED) Determines the anomeric configuration for root or lone monosaccharides
    using a strict, element-agnostic, and topologically-aware method.

    Logic:
    1. First, search for a 'C1' atom.
    2. If C1 is found and is validated to be endocyclic (part of the sugar ring),
       the algorithm COMMITS to C1 as the anomeric carbon.
       a. It then searches for C1's heaviest exocyclic substituent.
       b. If a substituent is found, the anomeric config is calculated.
       c. If no substituent is found, the config is None for this residue.
       d. The process for this residue STOPS. It will NOT fall back to check C2.
    3. ONLY IF C1 was not found OR was found to be exocyclic, the algorithm
       proceeds to perform the same check for the 'C2' atom.
    """
    all_glycan_residues = {
        key for key, atoms in atoms_by_residue.items()
        if atoms and atoms[0]['residue_name'] not in standard_amino_acids
    }
    
    child_residues = {
        (conn.child_chain_id, conn.child_res_id) for conn in existing_connections
    }
    
    root_and_lone_residues = all_glycan_residues - child_residues
    
    pseudo_connections = []
    
    for res_key in root_and_lone_residues:
        chain_id, res_id = res_key
        residue_dicts = atoms_by_residue[res_key]
        
        # Convert atom dictionaries to AnomericAtom objects for graph/ring functions
        anomeric_atoms: List[AnomericAtom] = [
            AnomericAtom(
                idx=atom_dict['atom_number'], name=atom_dict['atom_name'],
                res_name=atom_dict['residue_name'],
                coords=np.array([atom_dict['x'], atom_dict['y'], atom_dict['z']]),
                element=atom_dict['element']
            ) for atom_dict in residue_dicts
        ]
        
        if not anomeric_atoms:
            continue

        atoms_map: Dict[int, AnomericAtom] = {a.idx: a for a in anomeric_atoms}
        graph = build_residue_graph(anomeric_atoms)
        
        donor_atom_dict, acceptor_atom_dict = None, None
        anomeric_center_committed = False

        # --- Step 1: Try C1 first ---
        c1_atom_dict = find_atom_by_name(residue_dicts, 'C1')
        if c1_atom_dict:
            c1_anomeric_atom = atoms_map.get(c1_atom_dict['atom_number'])
            if c1_anomeric_atom:
                ordered_ring_indices = find_and_order_ring(graph, c1_anomeric_atom.idx, atoms_map)
                
                # Check if C1 is endocyclic
                if ordered_ring_indices and c1_anomeric_atom.idx in ordered_ring_indices:
                    anomeric_center_committed = True # We commit to C1
                    ring_set = set(ordered_ring_indices)
                    
                    # Find the heaviest exocyclic substituent, NOT specifically 'O1'
                    heaviest_substituent = _find_heaviest_exocyclic_substituent(c1_anomeric_atom, ring_set, graph, atoms_map)
                    
                    if heaviest_substituent:
                        donor_atom_dict = c1_atom_dict
                        # Find the original dictionary for the acceptor atom
                        acceptor_atom_dict = next(a for a in residue_dicts if a['atom_number'] == heaviest_substituent.idx)

        # --- Step 2: ONLY if C1 was not a valid center, try C2 ---
        if not anomeric_center_committed:
            c2_atom_dict = find_atom_by_name(residue_dicts, 'C2')
            if c2_atom_dict:
                c2_anomeric_atom = atoms_map.get(c2_atom_dict['atom_number'])
                if c2_anomeric_atom:
                    ordered_ring_indices = find_and_order_ring(graph, c2_anomeric_atom.idx, atoms_map)
                    
                    # Check if C2 is endocyclic
                    if ordered_ring_indices and c2_anomeric_atom.idx in ordered_ring_indices:
                        ring_set = set(ordered_ring_indices)
                        
                        # Find the heaviest exocyclic substituent
                        heaviest_substituent = _find_heaviest_exocyclic_substituent(c2_anomeric_atom, ring_set, graph, atoms_map)
                        
                        if heaviest_substituent:
                            donor_atom_dict = c2_atom_dict
                            acceptor_atom_dict = next(a for a in residue_dicts if a['atom_number'] == heaviest_substituent.idx)

        # --- Step 3: If a valid anomeric C-substituent pair was found, calculate the configuration ---
        if donor_atom_dict and acceptor_atom_dict:
            anom_config = determine_anomeric_config_universal(
                child_residue_dicts=residue_dicts,
                acceptor_atom_dict=acceptor_atom_dict,
                donor_atom_dict=donor_atom_dict
            )
            
            # Create a "pseudo-connection" to store this information
            pseudo_connections.append(GlycoConnection(
                parent_chain_id=chain_id,
                child_chain_id=chain_id,
                parent_res_id=res_id,
                child_res_id=res_id,
                parent_acceptor_atom_name=acceptor_atom_dict['atom_name'],
                child_donor_atom_name=donor_atom_dict['atom_name'],
                anomeric=anom_config
            ))
            
    return pseudo_connections

def parse_seqres(filepath: Path) -> Dict[str, List[str]]:
    """
    Parses SEQRES records from a PDB file to get the full polymer sequence.

    Args:
        filepath: Path to the PDB file.

    Returns:
        A dictionary mapping chain ID to a list of 3-letter residue codes.
    """
    seqres_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("SEQRES"):
                try:
                    chain_id = line[11]
                    res_names = line[19:].strip().split()
                    seqres_data[chain_id].extend(res_names)
                except IndexError:
                    continue
    return dict(seqres_data)

def parse_glycoprotein_pdb(
    pdb_file: PDBFile,
    ccd: Mapping[str, Mol]
) -> Tuple[Optional[ParsedGlycoproteinData], Optional[Dict], Optional[Dict], Optional[List], Optional[List]]:
    """
    (MODIFIED) Parses a PDB file by first applying glycan rechaining, then
    correctly flagging specifically glycosylated amino acids as non-standard.
    """
    pdb_id = pdb_file.id
    try:
        name_map = {
            "A2G": {"C2N": "C7", "CME": "C8", "O2N": "O7"}, "NAG": {"C2N": "C7", "CME": "C8", "O2N": "O7"},
            "NGA": {"C2N": "C7", "CME": "C8", "O2N": "O7"}, "NDG": {"C2N": "C7", "CME": "C8", "O2N": "O7"},
            "SIA": {"C5N": "C10", "CME": "C11", "O5N": "O10"},
            "NGC": {"C5N": "C10", "CME": "C11", "O5N": "O10", "OHG": "O11"},
            "TOA": {'O3': 'N3', 'O6A': 'O1', 'O6B': 'O6'}
        }
        inverse_name_map = {res: {v: k for k, v in nmap.items()} for res, nmap in name_map.items()}

        with open(pdb_file.path, 'r') as f:
            raw_atom_lines = [line for line in f if line.startswith(('ATOM', 'HETATM'))]
        
        if not raw_atom_lines:
            return None, {"type": "Parsing Error", "pdb_id": pdb_id, "message": "File contains no ATOM/HETATM records."}, {}, [], []
            
        rechained_atom_lines = _rechain_glycan_components(raw_atom_lines)
        atoms_by_residue, chain_types = parse_pdb_atoms_by_residue(rechained_atom_lines)
        seqres_data = parse_seqres(pdb_file.path)

        glycosidic_conns, glycosylation_sites, site_stats, anomalous_sites, no_ring_errors = detect_all_connections(
            atoms_by_residue, pdb_id=pdb_id
        )
        
        # --- FIX START ---
        # Create a fast lookup set of protein residues that are glycosylated.
        # The key is (chain_id, residue_number).
        glycosylated_protein_res_keys = {
            (site.protein_chain_id, site.protein_res_id) for site in glycosylation_sites
        }
        # --- FIX END ---

        root_pseudo_conns = _determine_root_anomeric_config(atoms_by_residue, glycosidic_conns)
        glycosidic_conns.extend(root_pseudo_conns)
        
        chains_to_residues: Dict[str, List[ParsedResidue]] = defaultdict(list)

        for (chain_id, res_num_pdb), pdb_atoms in atoms_by_residue.items():
            if not pdb_atoms: continue
            res_name = pdb_atoms[0]['residue_name']
            
            temp_res_idx = len(chains_to_residues[chain_id])
            
            parsed_res = None
            if res_name in standard_amino_acids:
                parsed_res = parse_protein_residue(res_name, res_num_pdb, temp_res_idx, pdb_atoms, ccd)
                
                # --- FIX START ---
                # Check if this standard amino acid is actually glycosylated. If so,
                # create a new ParsedResidue object with the is_standard flag set to False.
                res_key = (chain_id, res_num_pdb)
                if res_key in glycosylated_protein_res_keys:
                    parsed_res = replace(parsed_res, is_standard=False)
                # --- FIX END ---

            elif res_name in ccd: # Handles glycans and unnatural amino acids
                parsed_res = parse_glycan_residue(res_name, res_num_pdb, temp_res_idx, pdb_atoms, ccd, name_map, inverse_name_map)
            
            if parsed_res:
                chains_to_residues[chain_id].append(parsed_res)
        
        for chain_id, mol_type in chain_types.items():
            if mol_type == "PROTEIN" and chain_id in seqres_data:
                full_sequence = seqres_data[chain_id]
                observed_res_nums = {r.orig_idx for r in chains_to_residues[chain_id]}
                for res_idx, res_name in enumerate(full_sequence):
                    res_num_pdb = res_idx + 1
                    if res_num_pdb not in observed_res_nums and res_name in standard_amino_acids:
                        parsed_res = parse_protein_residue(res_name, res_num_pdb, res_idx, None, ccd)
                        if parsed_res:
                            chains_to_residues[chain_id].append(parsed_res)

        parsed_chains: Dict[str, ParsedChain] = {}
        for chain_id, res_list in chains_to_residues.items():
            if not res_list: continue
            res_list.sort(key=lambda r: r.orig_idx)
            
            for i, res in enumerate(res_list):
                res_list[i] = replace(res, idx=i)
            
            chain_type_str = chain_types.get(chain_id, "GLYCAN")
            chain_type_id = const.chain_type_ids.get(chain_type_str, const.chain_type_ids["NONPOLYMER"])

            parsed_chains[chain_id] = ParsedChain(
                name=chain_id, entity="", type=chain_type_id,
                residues=res_list, sequence=[r.name for r in res_list]
            )

        return ParsedGlycoproteinData(
            pdb_id=pdb_id, chains=parsed_chains,
            glycosidic_connections=glycosidic_conns,
            glycosylation_sites=glycosylation_sites
        ), None, site_stats, anomalous_sites, no_ring_errors

    except Exception as e:
        error_message = f"{type(e).__name__}: {e}"
        return None, {"type": "Processing Exception", "pdb_id": pdb_id, "message": error_message, "traceback": traceback.format_exc()}, {}, [], []

def _parse_unresolved_ccd_residue(
    name: str,
    ccd: Mapping[str, Mol],
    res_idx: int,
    res_num_pdb: int,
) -> Optional[ParsedResidue]:
    """
    Creates a ParsedResidue for an unresolved (missing) non-standard residue,
    using only the CCD as a template. All atoms will be marked as not present.
    """
    if name not in ccd:
        return None

    ref_mol = ccd[name]
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    
    unk_chirality = const.chirality_type_ids.get(const.unk_chirality_type, 0)
    
    parsed_atoms_list = [
        ParsedAtom(
            name=atom.GetProp("name"),
            element=atom.GetAtomicNum(),
            charge=atom.GetFormalCharge(),
            coords=(0.0, 0.0, 0.0),
            conformer=(0.0, 0.0, 0.0), # Conformer also 0 as it's unresolved
            is_present=False, # CRITICAL: Mark as not present
            chirality=const.chirality_type_ids.get(str(atom.GetChiralTag()), unk_chirality),
        ) for atom in ref_mol.GetAtoms()
    ]

    parsed_bonds_list = [
        ParsedBond(
            atom_1=bond.GetBeginAtomIdx(), atom_2=bond.GetEndAtomIdx(),
            type=const.bond_type_ids.get(bond.GetBondType().name, const.bond_type_ids[const.unk_bond_type])
        ) for bond in ref_mol.GetBonds()
    ]
    
    return ParsedResidue(
        name=name, type=const.token_ids.get("UNK"), atoms=parsed_atoms_list,
        bonds=parsed_bonds_list, idx=res_idx, orig_idx=res_num_pdb,
        atom_center=0, atom_disto=min(1, len(parsed_atoms_list)-1), 
        is_standard=False, is_present=False
    )

# --- This is a new helper function for parsing glycan residues (extracted from old logic) ---
def parse_glycan_residue(
    res_name: str,
    res_num_pdb: int,
    res_idx: int,
    pdb_atoms: List[Dict[str, Any]],
    ccd: Mapping[str, Mol],
    name_map: Dict,
    inverse_name_map: Dict,
) -> ParsedResidue:
    """
    (MODIFIED) Parses a glycan residue, using idealized CCD coordinates for the
    'conformer' field to provide a good starting structure for stitching.
    """
    ref_mol_no_h = AllChem.RemoveHs(ccd[res_name], sanitize=False)
    for atom in ref_mol_no_h.GetAtoms():
        if not atom.HasProp("name"):
            atom.SetProp("name", f"{atom.GetSymbol()}{atom.GetIdx()+1}")

    # --- CHANGE START ---
    # Get the reference conformer from the CCD molecule
    try:
        ref_conformer = get_conformer(ref_mol_no_h)
    except ValueError:
        print(f"Warning: No CCD conformer found for glycan residue '{res_name}'. Using zero vectors for conformer.", file=sys.stderr)
        ref_conformer = None
    # --- CHANGE END ---

    parsed_atoms_list: List[ParsedAtom] = []
    pdb_atom_map = {a['atom_name'].upper(): a for a in pdb_atoms}
    unk_chirality = const.chirality_type_ids.get(const.unk_chirality_type, 0)

    for ref_idx, ref_atom in enumerate(ref_mol_no_h.GetAtoms()):
        ref_atom_name_ccd = ref_atom.GetProp("name")
        ref_atom_name_ccd_upper = ref_atom_name_ccd.upper()
        pdb_atom_dict = pdb_atom_map.get(ref_atom_name_ccd_upper)
        if pdb_atom_dict is None:
            unmapped_pdb_name = inverse_name_map.get(res_name, {}).get(ref_atom_name_ccd_upper)
            if unmapped_pdb_name:
                pdb_atom_dict = pdb_atom_map.get(unmapped_pdb_name)

        # Get ground truth coordinates from the PDB (for loss calculation)
        coords = tuple(pdb_atom_dict.get(c, 0.0) for c in 'xyz') if pdb_atom_dict else (0.0, 0.0, 0.0)

        # --- CHANGE START ---
        # Get idealized conformer coordinates from CCD (for initial structure guess)
        if ref_conformer:
            ref_coords_rdkit = ref_conformer.GetAtomPosition(ref_atom.GetIdx())
            conformer_coords = (ref_coords_rdkit.x, ref_coords_rdkit.y, ref_coords_rdkit.z)
        else:
            conformer_coords = (0.0, 0.0, 0.0)
        # --- CHANGE END ---

        parsed_atoms_list.append(ParsedAtom(
            name=ref_atom_name_ccd, element=ref_atom.GetAtomicNum(), charge=ref_atom.GetFormalCharge(),
            coords=coords,
            conformer=conformer_coords, # Use the idealized CCD conformer
            is_present=bool(pdb_atom_dict),
            chirality=const.chirality_type_ids.get(ref_atom.GetChiralTag(), unk_chirality),
        ))

    parsed_bonds_list = [
        ParsedBond(
            atom_1=bond.GetBeginAtomIdx(), atom_2=bond.GetEndAtomIdx(),
            type=const.bond_type_ids.get(bond.GetBondType().name, const.bond_type_ids[const.unk_bond_type])
        ) for bond in ref_mol_no_h.GetBonds()
    ]

    center_idx = next((i for i, pa in enumerate(parsed_atoms_list) if pa.name.upper() == 'C1'), 0)
    disto_idx = next((i for i, pa in enumerate(parsed_atoms_list) if pa.name.upper() in ["C4'", "C4"]), center_idx if len(parsed_atoms_list) <= 1 else 1)

    return ParsedResidue(
        name=res_name, type=const.token_ids.get(res_name, const.token_ids["UNK"]), idx=res_idx,
        atoms=parsed_atoms_list, bonds=parsed_bonds_list, orig_idx=res_num_pdb,
        atom_center=center_idx, atom_disto=disto_idx, is_standard=False, is_present=True
    )

def parse_protein_residue(
    res_name: str,
    res_num_pdb: int,
    res_idx: int,
    pdb_atoms: Optional[List[Dict[str, Any]]], # Can be None for unresolved
    ccd: Mapping[str, Mol],
) -> ParsedResidue:
    """
    (CORRECTED) Parses a standard amino acid residue, ensuring that its internal
    bonds are also parsed and stored. This is critical for cases like glycosylated
    amino acids that are tokenized at the atom level.
    """
    is_present = pdb_atoms is not None
    pdb_atom_map = {a['atom_name'].upper(): a for a in pdb_atoms} if is_present else {}

    if res_name not in ccd:
        if res_name == "MSE": res_name = "MET"
        else: raise ValueError(f"Standard residue '{res_name}' not found in CCD.")

    ref_mol = ccd[res_name]
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    
    try:
        ref_conformer = get_conformer(ref_mol)
    except ValueError:
        print(f"Warning: No CCD conformer for '{res_name}'. Using zero vectors.", file=sys.stderr)
        ref_conformer = None

    ref_atom_names_ordered = const.ref_atoms.get(res_name, [])
    # Create a map from atom name to its index IN THE FINAL `parsed_atoms_list`
    ref_name_to_final_idx = {name: i for i, name in enumerate(ref_atom_names_ordered)}
    
    ref_name_to_atom_obj = {atom.GetProp("name"): atom for atom in ref_mol.GetAtoms()}
    
    parsed_atoms_list: List[ParsedAtom] = []
    unk_chirality = const.chirality_type_ids.get(const.unk_chirality_type, 0)

    for atom_name in ref_atom_names_ordered:
        pdb_atom_dict = pdb_atom_map.get(atom_name.upper())
        ref_atom_obj = ref_name_to_atom_obj.get(atom_name)

        if ref_atom_obj is None: continue

        atom_is_present = bool(pdb_atom_dict)
        coords = (
            (pdb_atom_dict['x'], pdb_atom_dict['y'], pdb_atom_dict['z'])
            if atom_is_present else (0.0, 0.0, 0.0)
        )
        conformer_coords = (0.0, 0.0, 0.0)
        if ref_conformer:
            ref_coords_rdkit = ref_conformer.GetAtomPosition(ref_atom_obj.GetIdx())
            conformer_coords = (ref_coords_rdkit.x, ref_coords_rdkit.y, ref_coords_rdkit.z)

        parsed_atoms_list.append(ParsedAtom(
            name=atom_name, element=ref_atom_obj.GetAtomicNum(), charge=ref_atom_obj.GetFormalCharge(),
            coords=coords, conformer=conformer_coords, is_present=atom_is_present,
            chirality=unk_chirality,
        ))

    # --- THIS IS THE NEW, CORRECTED BOND LOGIC ---
    parsed_bonds_list: List[ParsedBond] = []
    unk_bond_type = const.bond_type_ids.get(const.unk_bond_type, 1)
    for bond in ref_mol.GetBonds():
        begin_atom_name = bond.GetBeginAtom().GetProp("name")
        end_atom_name = bond.GetEndAtom().GetProp("name")

        # Check if both atoms of the bond are part of our defined atom set
        if begin_atom_name in ref_name_to_final_idx and end_atom_name in ref_name_to_final_idx:
            atom_1_idx = ref_name_to_final_idx[begin_atom_name]
            atom_2_idx = ref_name_to_final_idx[end_atom_name]
            bond_type = const.bond_type_ids.get(bond.GetBondType().name, unk_bond_type)
            parsed_bonds_list.append(ParsedBond(atom_1=atom_1_idx, atom_2=atom_2_idx, type=bond_type))
    # --- END OF NEW LOGIC ---

    center_idx = const.res_to_center_atom_id.get(res_name, 0)
    disto_idx = const.res_to_disto_atom_id.get(res_name, 1)

    return ParsedResidue(
        name=res_name, type=const.token_ids.get(res_name, const.token_ids["UNK"]),
        idx=res_idx, atoms=parsed_atoms_list, bonds=parsed_bonds_list, orig_idx=res_num_pdb,
        atom_center=center_idx, atom_disto=disto_idx, is_standard=True,
        is_present=is_present,
    )

def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """
    (Required) Convert an atom name to a standard numerical format.
    This function must be added to the script.
    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)

def _find_glycan_components(
    all_glycan_residues: List[Tuple[str, ParsedResidue]],
    glycosidic_connections: List[GlycoConnection],
) -> List[List[Tuple[str, int]]]:
    """
    Finds connected components of glycans using a graph traversal.

    Each component represents a single, complete, independent glycan tree.

    Args:
        all_glycan_residues: A list of tuples (original_chain_name, ParsedResidue).
        glycosidic_connections: A list of detected covalent bonds between monosaccharides.

    Returns:
        A list of components, where each component is a list of unique residue keys
        (original_chain_name, original_residue_id).
    """
    if not all_glycan_residues:
        return []

    # Create a map of residue key -> ParsedResidue for easy lookup
    res_map = {(chain_name, res.orig_idx): res for chain_name, res in all_glycan_residues}
    all_res_keys = set(res_map.keys())

    # Build the adjacency list for the graph
    adj = defaultdict(list)
    for conn in glycosidic_connections:
        parent_key = (conn.parent_chain_id, conn.parent_res_id)
        child_key = (conn.child_chain_id, conn.child_res_id)
        # Ensure we only consider edges between known glycan residues
        if parent_key in all_res_keys and child_key in all_res_keys:
            adj[parent_key].append(child_key)
            adj[child_key].append(parent_key)

    # Find connected components using BFS
    visited = set()
    components = []
    for res_key in all_res_keys:
        if res_key not in visited:
            new_component = []
            q = deque([res_key])
            visited.add(res_key)
            while q:
                current_key = q.popleft()
                new_component.append(current_key)
                for neighbor_key in adj.get(current_key, []):
                    if neighbor_key not in visited:
                        visited.add(neighbor_key)
                        q.append(neighbor_key)
            components.append(new_component)

    return components

def assemble_glycoprotein_structure(
    parsed_data: ParsedGlycoproteinData,
    cluster_id: int,
) -> Tuple[Dict[str, Any], Record]:
    """
    (CORRECTED) Assembles all parsed data into final, dense numpy arrays,
    correctly differentiating between true glycans and other non-standard residues
    (like glycosylated amino acids) during the initial sorting phase.
    """
    pdb_id = parsed_data.pdb_id
    atom_rows, res_rows, chain_rows_tuples = [], [], []
    bond_rows, connection_rows = [], []
    glycosylation_site_tuples = []
    final_glycan_feature_map, final_atom_to_mono_idx_map = {}, {}

    atom_map: Dict[Tuple[str, int, str], int] = {}
    res_map: Dict[Tuple[str, int], Tuple[int, int]] = {}
    chain_map: Dict[str, int] = {}

    atom_offset, res_offset, chain_offset = 0, 0, 0

    protein_residues_by_chain = defaultdict(list)
    all_glycan_residues_with_orig_chain = []
    res_lookup_map = {}

    for chain_name, chain in parsed_data.chains.items():
        for res in chain.residues:
            res_key = (chain_name, res.orig_idx if res.orig_idx is not None else res.idx)
            res_lookup_map[res_key] = res
            
            is_true_glycan = res.name in MONO_TYPE_MAP and MONO_TYPE_MAP[res.name] != "OTHER"

            if res.is_standard or not is_true_glycan:
                # Path 1: Standard AAs AND non-standard, non-glycan residues (e.g., glycosylated ASN)
                # are grouped with protein components.
                protein_residues_by_chain[chain_name].append(res)
            else:
                # Path 2: Only true monosaccharides are sent to the glycan processing list.
                all_glycan_residues_with_orig_chain.append((chain_name, res))
    # --- END OF CORRECTION ---

    # --- Step 1: Process PROTEIN chains (and other atomized components) ---
    for chain_name in sorted(protein_residues_by_chain.keys()):
        chain_residues = protein_residues_by_chain[chain_name]
        chain_residues.sort(key=lambda r: r.orig_idx)

        chain_map[chain_name] = chain_offset
        
        num_atoms_in_chain = sum(len(r.atoms) for r in chain_residues)
        chain_rows_tuples.append((
            chain_name, const.chain_type_ids["PROTEIN"], chain_offset, chain_offset, chain_offset,
            atom_offset, num_atoms_in_chain, res_offset, len(chain_residues), 0
        ))
        
        for res in chain_residues:
            res_key = (chain_name, res.orig_idx if res.orig_idx is not None else res.idx)
            res_map[res_key] = (res_offset, atom_offset)
            res_rows.append((
                res.name, res.type, res.idx, atom_offset, len(res.atoms),
                atom_offset + res.atom_center, atom_offset + res.atom_disto,
                res.is_standard, res.is_present
            ))
            
            for local_atom_idx, atom in enumerate(res.atoms):
                atom_map[res_key + (atom.name,)] = atom_offset + local_atom_idx
                atom_rows.append((convert_atom_name(atom.name), atom.element, atom.charge, atom.coords, atom.conformer, atom.is_present, atom.chirality))
            
            for bond in res.bonds:
                bond_rows.append((min(bond.atom_1, bond.atom_2) + atom_offset, max(bond.atom_1, bond.atom_2) + atom_offset, bond.type))
            
            atom_offset += len(res.atoms)
            res_offset += 1

        for prev_res, next_res in zip(chain_residues, chain_residues[1:]):
            if not (prev_res.is_present and next_res.is_present): continue
            prev_key = (chain_name, prev_res.orig_idx if prev_res.orig_idx is not None else prev_res.idx)
            next_key = (chain_name, next_res.orig_idx if next_res.orig_idx is not None else next_res.idx)
            c_atom_key, n_atom_key = prev_key + ('C',), next_key + ('N',)

            if c_atom_key in atom_map and n_atom_key in atom_map:
                atom_idx_c, atom_idx_n = atom_map[c_atom_key], atom_map[n_atom_key]
                res_idx_prev, _ = res_map[prev_key]
                res_idx_next, _ = res_map[next_key]
                connection_rows.append((chain_offset, chain_offset, res_idx_prev, res_idx_next, atom_idx_c, atom_idx_n))
        chain_offset += 1

    # --- Step 2: Process GLYCAN components ---
    glycan_components = _find_glycan_components(all_glycan_residues_with_orig_chain, parsed_data.glycosidic_connections)
    orig_res_key_to_new_mono_info = {}

    for component_res_keys in glycan_components:
        final_glycan_chain_id = chain_offset
        agg_chain_name = f"_GLYCAN_{final_glycan_chain_id}_"
        chain_map[agg_chain_name] = final_glycan_chain_id

        comp_res_with_chains = [(key[0], res_lookup_map[key]) for key in component_res_keys]
        
        agg_chain, gly_feat_map, atom_mono_map, res_key_mono_map = _aggregate_glycan_component(
            pdb_id=pdb_id, component_residues_with_chains=comp_res_with_chains,
            all_connections=parsed_data.glycosidic_connections, final_glycan_chain_idx=final_glycan_chain_id,
            final_glycan_chain_name=agg_chain_name
        )
        
        for res_key, mono_idx in res_key_mono_map.items():
            orig_res_key_to_new_mono_info[res_key] = (final_glycan_chain_id, mono_idx)
            res_map[res_key] = (res_offset, atom_offset)

        final_glycan_feature_map.update(gly_feat_map)
        final_atom_to_mono_idx_map.update(atom_mono_map)
        
        if agg_chain and agg_chain.residues:
            glycan_res = agg_chain.residues[0]
            new_chain_tuple = (
                agg_chain.name, agg_chain.type, final_glycan_chain_id, final_glycan_chain_id, final_glycan_chain_id,
                atom_offset, len(glycan_res.atoms), res_offset, 1, 0
            )
            chain_rows_tuples.append(new_chain_tuple)
            
            res_rows.append((glycan_res.name, glycan_res.type, 0, atom_offset, len(glycan_res.atoms),
                             atom_offset + glycan_res.atom_center, atom_offset + glycan_res.atom_disto, False, True))
            
            for local_atom_idx, atom in enumerate(glycan_res.atoms):
                atom_map[(agg_chain_name, 0, atom.name)] = atom_offset + local_atom_idx
                atom_rows.append((convert_atom_name(atom.name), atom.element, atom.charge, atom.coords, atom.conformer, atom.is_present, atom.chirality))
            
            for bond in glycan_res.bonds:
                bond_rows.append((min(bond.atom_1, bond.atom_2) + atom_offset, max(bond.atom_1, bond.atom_2) + atom_offset, bond.type))
            
            atom_offset += len(glycan_res.atoms)
            res_offset += 1
            chain_offset += 1

    chains = np.array(chain_rows_tuples, dtype=Chain)

    # --- Step 3: Create final glycosylation site entries ---
    for i, conn in enumerate(parsed_data.glycosylation_sites):
        prot_res_key = (conn.protein_chain_id, conn.protein_res_id)
        glycan_orig_res_key = (conn.glycan_chain_id, conn.glycan_res_id)
        
        if prot_res_key in res_map and glycan_orig_res_key in orig_res_key_to_new_mono_info:
            prot_res_idx, _ = res_map[prot_res_key]
            glycan_chain_idx, mono_idx = orig_res_key_to_new_mono_info[glycan_orig_res_key]
            
            glycan_res_idx = chains[glycan_chain_idx]['res_idx']
            prot_chain_idx = chain_map[conn.protein_chain_id]
            
            prot_atom_key = prot_res_key + (conn.protein_atom_name,)
            glycan_agg_chain_name = chains[glycan_chain_idx]['name'].strip()
            glycan_atom_key = (glycan_agg_chain_name, 0, conn.glycan_atom_name)

            if prot_atom_key in atom_map and glycan_atom_key in atom_map:
                atom_idx_prot = atom_map[prot_atom_key]
                atom_idx_glycan = atom_map[glycan_atom_key]
                connection_rows.append((prot_chain_idx, glycan_chain_idx, prot_res_idx, glycan_res_idx, atom_idx_prot, atom_idx_glycan))
            
            prot_chain_res_start_idx = chains[prot_chain_idx]['res_idx']
            glycosylation_site_tuples.append((
                prot_chain_idx, prot_res_idx - prot_chain_res_start_idx, conn.protein_atom_name,
                glycan_chain_idx, mono_idx, conn.glycan_atom_name
            ))
    
    # --- Step 4: Create final numpy arrays ---
    atoms = np.array(atom_rows, dtype=Atom)
    bonds = np.array(sorted(list(set(bond_rows))), dtype=Bond) if bond_rows else np.array([], dtype=Bond)
    residues = np.array(res_rows, dtype=Residue)
    connections = np.array(connection_rows, dtype=Connection) if connection_rows else np.array([], dtype=Connection)
    glycosylation_sites_arr = np.array(glycosylation_site_tuples, dtype=GlycosylationSite) if glycosylation_site_tuples else None

    npz_data = {
        'atoms': atoms, 'bonds': bonds, 'residues': residues, 'chains': chains,
        'connections': connections, 'interfaces': np.array([], dtype=Interface),
        'mask': np.ones(len(chains), dtype=bool), 'glycosylation_sites': glycosylation_sites_arr,
        'glycan_feature_map': final_glycan_feature_map, 'atom_to_mono_idx_map': final_atom_to_mono_idx_map,
    }
    
    chain_info_list = [ChainInfo(chain_id=i, chain_name=c['name'].strip(), mol_type=c['mol_type'], num_residues=c['res_num'], valid=True, entity_id=c['entity_id'], msa_id='', cluster_id=cluster_id) for i, c in enumerate(chains)]
    record = Record(id=pdb_id, structure=StructureInfo(num_chains=len(chains)), chains=chain_info_list, interfaces=[], inference_options=None)
    
    return npz_data, record

def _aggregate_glycan_component(
    pdb_id: str,
    component_residues_with_chains: List[Tuple[str, ParsedResidue]],
    all_connections: List[GlycoConnection],
    final_glycan_chain_idx: int,
    final_glycan_chain_name: str,
) -> Tuple[ParsedChain, Dict, Dict, Dict]:
    """
    (REVISED) Aggregates a glycan component. The 'conformer' coordinates for all
    atoms in this component are explicitly set to (0,0,0) to maintain the
    glycan-specific heuristic. `coords` (from PDB) are preserved.
    """

    if not component_residues_with_chains:
        return None, {}, {}, {}

    res_map = {(chain_name, res.orig_idx): res for chain_name, res in component_residues_with_chains}
    component_res_keys = set(res_map.keys())
    connections_this_component = [
        conn for conn in all_connections
        if (conn.parent_chain_id, conn.parent_res_id) in component_res_keys and
           (conn.child_chain_id, conn.child_res_id) in component_res_keys
    ]
    master_atoms, master_bonds = [], []
    atom_to_mono_idx_list = []
    residue_key_to_mono_idx = {}
    glycan_feature_map = {}
    atom_offsets_map = {}
    atom_offset = 0
    sorted_res_keys = sorted(list(component_res_keys))

    for mono_idx, res_key in enumerate(sorted_res_keys):
        res = res_map[res_key]
        atom_offsets_map[res_key] = atom_offset
        residue_key_to_mono_idx[res_key] = mono_idx

        # --- CRITICAL FIX: Enforce (0,0,0) conformer for glycan components ---
        for atom in res.atoms:
            # Create a new atom with the conformer field zeroed out
            zeroed_conformer_atom = replace(atom, conformer=(0.0, 0.0, 0.0))
            master_atoms.append(zeroed_conformer_atom)
        
        atom_to_mono_idx_list.extend([mono_idx] * len(res.atoms))

        for bond in res.bonds:
            master_bonds.append(ParsedBond(bond.atom_1 + atom_offset, bond.atom_2 + atom_offset, bond.type))
        
        atom_offset += len(res.atoms)
        
        conn_as_child = next((c for c in connections_this_component if (c.child_chain_id, c.child_res_id) == res_key), None)
        glycan_feature_map[(final_glycan_chain_idx, mono_idx)] = boltz.data.parse.schema.MonosaccharideFeatures(
            asym_id=final_glycan_chain_idx, ccd_code=res.name, source_glycan_idx=0,
            anomeric_config=conn_as_child.anomeric if conn_as_child else None
        )

    # ... (The rest of the function for adding inter-residue bonds is correct) ...
    # (The final creation of ParsedResidue and ParsedChain is also correct)
    for conn in connections_this_component:
        parent_key = (conn.parent_chain_id, conn.parent_res_id)
        child_key = (conn.child_chain_id, conn.child_res_id)
        try:
            parent_res, child_res = res_map[parent_key], res_map[child_key]
            acceptor_idx = next(i for i, a in enumerate(parent_res.atoms) if a.name.upper() == conn.parent_acceptor_atom_name.upper())
            donor_idx = next(i for i, a in enumerate(child_res.atoms) if a.name.upper() == conn.child_donor_atom_name.upper())
            global_acceptor_idx = acceptor_idx + atom_offsets_map[parent_key]
            global_donor_idx = donor_idx + atom_offsets_map[child_key]
            master_bonds.append(ParsedBond(global_acceptor_idx, global_donor_idx, BOND_TYPE_SINGLE))
        except StopIteration:
            continue
            
    final_residue = ParsedResidue("GLYCAN", const.token_ids["UNK"], 0, master_atoms, master_bonds, 0, 0, min(1, len(master_atoms)-1), False, True)
    final_chain = ParsedChain(final_glycan_chain_name, "", const.chain_type_ids["NONPOLYMER"], [final_residue], ["GLYCAN"])
    atom_to_mono_idx_map = {final_glycan_chain_idx: np.array(atom_to_mono_idx_list, dtype=np.int32)}

    return final_chain, glycan_feature_map, atom_to_mono_idx_map, residue_key_to_mono_idx

# --- Main Processing Functions --#
def finalize(outdir: Path) -> None:
    """
    Aggregates all individual record .json files into a single, RANDOMLY SHUFFLED
    manifest.json. This is critical for ensuring that downstream dataloaders
    which iterate sequentially still produce representative training batches.
    """
    records_dir = outdir / "records"
    if not records_dir.is_dir():
        print(f"Warning: Records directory not found: {records_dir}")
        return

    final_manifest_entries = []
    record_files = list(records_dir.glob("*.json"))
    
    if not record_files:
        print("No record files found to aggregate.")
        return

    print(f"Aggregating {len(record_files)} record files...")
    for record_path in tqdm(record_files, desc="Creating manifest"):
        try:
            with record_path.open("r") as f:
                record_data = json.load(f)
                final_manifest_entries.append(record_data)
        except Exception as e:
            print(f"Warning: Failed to parse record file {record_path}. Skipping. Error: {e}")
            continue

    print("Randomly shuffling manifest entries...")
    random.shuffle(final_manifest_entries)

    outpath = outdir / "manifest.json"
    print(f"Saving shuffled manifest with {len(final_manifest_entries)} entries to: {outpath}")
    with outpath.open("w") as f:
        json.dump(final_manifest_entries, f, indent=2)
    print("Manifest saved successfully.")

# --- The main `process_pdb_file` function must be rewritten to use the new logic ---
def process_pdb_file(
    task: Tuple[PDBFile, int], 
    outdir: Path
) -> Tuple[bool, Optional[Dict], bool, Optional[Dict], Optional[List], Optional[List]]:
    """
    (REVISED TO RETURN ERRORS) Processes a single PDB file and returns a
    6-element tuple including stats, anomalies, and "no ring" errors.
    """
    pdb_file, cluster_id = task
    
    global worker_ccd_data
    if worker_ccd_data is None:
        error_info = {"type": "Worker Error", "pdb_id": "N/A", "message": "Worker CCD data not loaded."}
        return False, error_info, False, {}, [], []

    pdb_id = pdb_file.id
    struct_path = outdir / "structures" / f"{pdb_id}.npz"
    record_path = outdir / "records" / f"{pdb_id}.json"

    struct_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Unpack the new no_ring_errors list
        parsed_data, error, site_stats, anomalous_sites, no_ring_errors = parse_glycoprotein_pdb(pdb_file, worker_ccd_data)
        
        if error:
            return False, error, False, {}, [], []
        if not parsed_data:
            return False, {"type": "Unknown Parse Error", "pdb_id": pdb_id, "message": "Parsing returned no data."}, False, {}, [], []

        npz_data, record = assemble_glycoprotein_structure(parsed_data, cluster_id)
        
        has_site = (
            npz_data.get('glycosylation_sites') is not None and
            npz_data['glycosylation_sites'].size > 0
        )
        
        np.savez_compressed(struct_path, **npz_data, allow_pickle=True)
        
        with open(record_path, 'w') as f:
            json.dump(asdict(record), f, indent=4, cls=NumpyJSONEncoder)

        # Return the no_ring_errors list
        return True, None, has_site, site_stats, anomalous_sites, no_ring_errors

    except Exception as e:
        tb_str = traceback.format_exc()
        error_type = type(e).__name__
        error_msg = str(e)
        
        print(f"\n--- FAILURE processing {pdb_id} ---\n", file=sys.stderr, flush=True)
        print(f"  REASON: [{error_type}] {error_msg}", file=sys.stderr, flush=True)
        print(f"  TRACEBACK:\n{tb_str}", file=sys.stderr, flush=True)
        print(f"--- END {pdb_id} ---\n", file=sys.stderr, flush=True)

        return False, {
            "type": "Processing Exception",
            "pdb_id": pdb_id,
            "message": f"{error_type}: {error_msg}",
            "traceback": tb_str
        }, False, {}, [], []

# --- Main Execution ---
def main(args):
    """
    (REVISED FOR DETAILED ANOMALY/ERROR SUMMARY) Main loop that aggregates
    statistics and prints final summaries with specific examples for both
    biochemically anomalous sites and "no ring" calculation errors.
    """
    script_overall_start_time = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "structures").mkdir(parents=True, exist_ok=True)
    (args.outdir / "records").mkdir(parents=True, exist_ok=True)

    if not args.ccd_path.is_file():
        print(f"Error: CCD file not found at {args.ccd_path}")
        return 1

    all_pdb_paths = list(args.datadir.rglob("*.pdb"))
    if not all_pdb_paths:
        print("No PDB files found. Exiting.")
        return 0
    
    initial_tasks = [PDBFile(id=p.stem, path=p, cluster_num=0, frame_num=0) for p in all_pdb_paths]
    print(f"Found {len(initial_tasks)} PDB files to process.")

    use_parallel = args.num_processes > 1 and len(initial_tasks) > 1
    ccd_path_str = str(args.ccd_path.resolve())
    
    print("\n--- Starting Pass 1: Generating canonical sequence keys ---")
    id_to_sequence = {}
    if use_parallel:
        with multiprocessing.Pool(args.num_processes, initializer=worker_init, initargs=(ccd_path_str,)) as pool:
            key_results = list(tqdm(pool.imap_unordered(get_glycan_sequence_key, initial_tasks), total=len(initial_tasks), desc="Generating Sequences"))
    else:
        worker_init(ccd_path_str)
        key_results = [get_glycan_sequence_key(task) for task in tqdm(initial_tasks, desc="Generating Sequences (Serial)")]
    
    sequencing_failures = 0
    for pdb_id, sequence_key in key_results:
        if sequence_key is not None:
            id_to_sequence[pdb_id] = sequence_key
        else:
            id_to_sequence[pdb_id] = "_SEQ_FAIL_CLUSTER_"
            sequencing_failures += 1

    print("\n--- Clustering structures based on sequence keys ---")
    sequence_to_ids = defaultdict(list)
    for pdb_id, sequence_key in id_to_sequence.items():
        sequence_to_ids[sequence_key].append(pdb_id)

    id_to_cluster = {pdb_id: cluster_id for cluster_id, (_, pdb_ids) in enumerate(sequence_to_ids.items()) for pdb_id in pdb_ids}
    
    print("\n--- CLUSTERING SUMMARY ---")
    print(f"Total files scanned: {len(initial_tasks)}")
    print(f"Successfully sequenced: {len(id_to_sequence) - sequencing_failures}")
    print(f"Failed to sequence (grouped): {sequencing_failures}")
    print(f"Number of unique sequences (clusters): {len(sequence_to_ids)}")
    
    print("\n--- Starting Pass 2: Processing and saving files with cluster IDs ---")
    final_processing_tasks = [(task, id_to_cluster.get(task.id)) for task in initial_tasks]

    results = []
    if final_processing_tasks:
        if use_parallel:
            process_func = partial(process_pdb_file, outdir=args.outdir)
            with multiprocessing.Pool(args.num_processes, initializer=worker_init, initargs=(ccd_path_str,)) as pool:
                results = list(tqdm(pool.imap_unordered(process_func, final_processing_tasks), total=len(final_processing_tasks), desc="Final Processing"))
        else:
            worker_init(ccd_path_str)
            results = [process_pdb_file(task, args.outdir) for task in tqdm(final_processing_tasks, desc="Final Processing (Serial)")]
    
    success_count = 0
    failed_files = []
    global_site_stats = defaultdict(lambda: defaultdict(int))
    global_anomalous_sites = []
    global_no_ring_errors = []

    # Unpack the 6-element tuple from results
    for success, error_info, has_site, site_stats_for_file, anomalous_sites_for_file, no_ring_errors_for_file in results:
        if success:
            success_count += 1
            if site_stats_for_file:
                for res_name, counts in site_stats_for_file.items():
                    for config, count in counts.items():
                        global_site_stats[res_name][config] += count
            if anomalous_sites_for_file:
                global_anomalous_sites.extend(anomalous_sites_for_file)
            if no_ring_errors_for_file:
                global_no_ring_errors.extend(no_ring_errors_for_file)
        else:
            failed_files.append(error_info)

    print("\n--- Aggregating Records into Manifest ---")
    finalize(args.outdir)

    print("\n\n--- GLOBAL Glycosylation Site Anomeric Configuration Summary ---")
    if not global_site_stats:
        print("No glycosylation sites were detected across all successfully processed files.")
    else:
        print("Summary of all detected protein-glycan linkages across the dataset:")
        for res_name, counts in sorted(global_site_stats.items()):
            alpha_count = counts.get('a', 0)
            beta_count = counts.get('b', 0)
            summary_line = f"For {res_name}: found {alpha_count} alpha and {beta_count} beta configurations."
            other_configs = {k: v for k, v in counts.items() if k not in ['a', 'b']}
            if other_configs:
                other_parts = ", ".join([f"{v} '{k}'" for k, v in other_configs.items()])
                summary_line += f" (Additionally found: {other_parts})"
            print(summary_line)
    print("----------------------------------------------------------------")

    # --- New, more detailed summary block for anomalous sites ---
    print("\n\n--- Summary of Biochemically Anomalous Glycosylation Sites ---")
    if not global_anomalous_sites:
        print("No anomalous linkages (alpha-ASN, beta-THR, beta-SER) were found.")
    else:
        # Separate examples by type
        alpha_asn_examples = [s for s in global_anomalous_sites if s[1] == 'ASN']
        beta_ser_examples = [s for s in global_anomalous_sites if s[1] == 'SER']
        beta_thr_examples = [s for s in global_anomalous_sites if s[1] == 'THR']

        print(f"Found {len(alpha_asn_examples)} alpha-ASN, {len(beta_ser_examples)} beta-SER, and {len(beta_thr_examples)} beta-THR linkages.")
        
        if alpha_asn_examples:
            print("\nDisplaying up to 3 examples of alpha-ASN linkages:")
            for i, (pdb_id, res_name, res_num, config) in enumerate(alpha_asn_examples[:3]):
                print(f"  - PDB ID: {pdb_id}, Residue: {res_name}{res_num}, Detected Config: {config}")
        
        if beta_ser_examples:
            print("\nDisplaying up to 3 examples of beta-SER linkages:")
            for i, (pdb_id, res_name, res_num, config) in enumerate(beta_ser_examples[:3]):
                print(f"  - PDB ID: {pdb_id}, Residue: {res_name}{res_num}, Detected Config: {config}")

        if beta_thr_examples:
            print("\nDisplaying up to 3 examples of beta-THR linkages:")
            for i, (pdb_id, res_name, res_num, config) in enumerate(beta_thr_examples[:3]):
                print(f"  - PDB ID: {pdb_id}, Residue: {res_name}{res_num}, Detected Config: {config}")
    print("----------------------------------------------------------------")

    # --- New summary block for "no ring" errors ---
    print("\n\n--- Summary of 'No Ring' Calculation Errors ---")
    if not global_no_ring_errors:
        print("No 'no ring' errors were encountered during processing.")
    else:
        print(f"Found {len(global_no_ring_errors)} sites where a ring could not be determined. Displaying up to 5 examples:")
        for i, (pdb_id, res_name, res_num) in enumerate(global_no_ring_errors[:5]):
            print(f"  - PDB ID: {pdb_id}, Glycosylated Residue: {res_name}{res_num}")
        if len(global_no_ring_errors) > 5:
             print(f"... and {len(global_no_ring_errors) - 5} more.")
    print("----------------------------------------------------------------")


    print("\n\n--- SCRIPT COMPLETE ---")
    print(f"Total files processed and saved successfully: {success_count}")
    print(f"Total files that were skipped or failed: {len(failed_files)}")
    
    if failed_files:
        print("\n\n--- FAILURE SUMMARY ---")
        errors_by_reason = defaultdict(list)
        for err in failed_files:
            if err:
                reason = f"[{err.get('type', 'Unknown')}] {err.get('message', 'No details provided.')}"
                errors_by_reason[reason].append(err.get('pdb_id', 'N/A'))
        for reason, pdb_ids in sorted(errors_by_reason.items(), key=lambda item: len(item[1]), reverse=True):
            print("-" * 70)
            print(f"REASON: {reason} ({len(pdb_ids)} files)")
            display_limit = 10
            if len(pdb_ids) > display_limit:
                print(f"  Examples: {', '.join(pdb_ids[:display_limit])}, ...")
            else:
                print(f"  Affected Files: {', '.join(pdb_ids)}")
        print("-" * 70)

    print(f"\nTotal script execution time: {time.time() - script_overall_start_time:.2f} seconds.")
    return 0

def worker_init(ccd_path_str: str):
    """
    Initializer for each worker process. It now raises an exception on failure
    instead of calling sys.exit(), allowing the main process to catch it.
    """
    global worker_ccd_data
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    try:
        rdkit.Chem.SetDefaultPickleProperties(rdkit.Chem.PropertyPickleOptions.AllProps)
        with open(ccd_path_str, "rb") as f:
            worker_ccd_data = pickle.load(f)
        
        if not worker_ccd_data or not isinstance(worker_ccd_data, dict):
            # This is a critical failure. Raise an exception to terminate the pool.
            raise RuntimeError("Worker failed to load or received empty/invalid CCD data.")
            
    except Exception as e:
        # Re-raise the exception with more context. This will be caught by the main process.
        raise RuntimeError(f"A worker process failed to initialize: {e}") from e




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process specialized glycan PDB dataset.")
    parser.add_argument(
        "--datadir",
        type=Path,
        required=True,
        help="Directory containing the input PDB files.",
    )
    parser.add_argument(
        "--ccd-path",
        type=Path,
        required=True,
        help="Path to the pickled CCD dictionary (ccd.pkl).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default="glycan_processed",
        help="The output directory for processed .npz files.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 2), # Default to half the cores
        help="Number of parallel processes to use.",
    )
    args = parser.parse_args()

    # Basic validation
    if not args.datadir.is_dir():
        print(f"Error: Input data directory not found: {args.datadir}")
        exit(1)
    elif not args.ccd_path.is_file():
         print(f"Error: CCD file not found: {args.ccd_path}")
         exit(1)
    else:
        # Add traceback import if not already present at top level
        import traceback
        exit_code = main(args)
        exit(exit_code)
