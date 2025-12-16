#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDB/CASP to Sequence YAML Converter

This script reads either a PDB file or a CASP-formatted text file and generates
a structured YAML file containing molecular sequences.

PDB Mode:
- Parses protein, glycan, and RNA sequences from PDB files.
- Detects glycosylation sites.
- Generates IUPAC strings for glycans.

CASP Mode:
- Parses protein and RNA sequences from a simple text format.
- Headers start with '>' and the sequence is on the next line.
- The MSA name for proteins is derived from the header ID.
"""

import sys
import os
import argparse
from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import yaml
from yaml import Dumper
from scipy.spatial import cKDTree

# --- Constants and Data Mappings ---
ALLOWED_GLYCANS = set(["05L", "07E", "0HX", "0LP", "0MK", "0NZ", "0UB", "0WK", "0XY", "0YT", "12E", "145", "147", "149", "14T", "15L", "16F", "16G", "16O", "17T", "18D", "18O", "1CF", "1GL", "1GN", "1S3", "1S4", "1SD", "1X4", "20S", "20X", "22O", "22S", "23V", "24S", "25E", "26O", "27C", "289", "291", "293", "2DG", "2DR", "2F8", "2FG", "2FL", "2GL", "2GS", "2H5", "2M5", "2M8", "2WP", "32O", "34V", "38J", "3DO", "3FM", "3HD", "3J3", "3J4", "3LJ", "3MG", "3MK", "3R3", "3S6", "3YW", "42D", "445", "44S", "46Z", "475", "491", "49A", "49S", "49T", "49V", "4AM", "4CQ", "4GL", "4GP", "4JA", "4N2", "4NN", "4QY", "4R1", "4SG", "4U0", "4U1", "4U2", "4UZ", "4V5", "50A", "510", "51N", "56N", "57S", "5DI", "5GF", "5GO", "5KQ", "5KV", "5L2", "5L3", "5LS", "5LT", "5N6", "5QP", "5TH", "5TJ", "5TK", "5TM", "604", "61J", "62I", "64K", "66O", "6BG", "6C2", "6GB", "6GP", "6GR", "6K3", "6KH", "6KL", "6KS", "6KU", "6KW", "6LS", "6LW", "6MJ", "6MN", "6PY", "6PZ", "6S2", "6UD", "6Y6", "6YR", "6ZC", "73E", "79J", "7CV", "7D1", "7GP", "7JZ", "7K2", "7K3", "7NU", "83Y", "89Y", "8B7", "8B9", "8EX", "8GA", "8GG", "8GP", "8LM", "8LR", "8OQ", "8PK", "8S0", "95Z", "96O", "9AM", "9C1", "9CD", "9GP", "9KJ", "9MR", "9OK", "9PG", "9QG", "9QZ", "9S7", "9SG", "9SJ", "9SM", "9SP", "9T1", "9T7", "9VP", "9WJ", "9WN", "9WZ", "9YW", "A0K", "A1Q", "A2G", "A5C", "A6P", "AAL", "ABD", "ABE", "ABF", "ABL", "AC1", "ACR", "ACX", "ADA", "AF1", "AFD", "AFO", "AFP", "AFR", "AGL", "AGR", "AH2", "AH8", "AHG", "AHM", "AHR", "AIG", "ALL", "ALX", "AMG", "AMN", "AMU", "AMV", "ANA", "AOG", "AQA", "ARA", "ARB", "ARI", "ARW", "ASC", "ASG", "ASO", "AXP", "AXR", "AY9", "AZC", "B0D", "B16", "B1H", "B1N", "B6D", "B7G", "B8D", "B9D", "BBK", "BBV", "BCD", "BCW", "BDF", "BDG", "BDP", "BDR", "BDZ", "BEM", "BFN", "BG6", "BG8", "BGC", "BGL", "BGN", "BGP", "BGS", "BHG", "BM3", "BM7", "BMA", "BMX", "BND", "BNG", "BNX", "BXY", "BZD", "C3B", "C3G", "C3X", "C4B", "C4W", "C5X", "CBF", "CBI", "CBK", "CDR", "CE5", "CE6", "CE8", "CEG", "CEX", "CEY", "CEZ", "CGF", "CJB", "CKB", "CKP", "CNP", "CR1", "CR6", "CRA", "CT3", "CTO", "CTR", "CTT", "D0N", "D1M", "D5E", "D6G", "DAF", "DAG", "DAN", "DDA", "DDL", "DEG", "DEL", "DFR", "DFX", "DGO", "DGS", "DJB", "DJE", "DK4", "DKX", "DKZ", "DL6", "DLD", "DLF", "DLG", "DO8", "DOM", "DPC", "DQR", "DR2", "DR3", "DR5", "DRI", "DSR", "DT6", "DVC", "DYM", "E3M", "E5G", "EAG", "EBG", "EBQ", "EEN", "EEQ", "EGA", "EMP", "EMZ", "EPG", "EQP", "EQV", "ERE", "ERI", "ETT", "F1P", "F1X", "F55", "F58", "F6P", "FBP", "FCA", "FCB", "FCT", "FDP", "FDQ", "FFC", "FFX", "FIF", "FK9", "FKD", "FMF", "FMO", "FNG", "FNY", "FRU", "FSA", "FSI", "FSM", "FSR", "FSW", "FUB", "FUC", "FUF", "FUL", "FUY", "FVQ", "FX1", "FYJ", "G0S", "G16", "G1P", "G20", "G28", "G2F", "G3F", "G4D", "G4S", "G6D", "G6P", "G6S", "G7P", "G8Z", "GAA", "GAC", "GAD", "GAF", "GAL", "GAT", "GBH", "GC1", "GC4", "GC9", "GCB", "GCD", "GCN", "GCO", "GCS", "GCT", "GCU", "GCV", "GCW", "GDA", "GDL", "GE1", "GE3", "GFP", "GIV", "GL0", "GL1", "GL2", "GL4", "GL5", "GL6", "GL7", "GL9", "GLA", "GLC", "GLD", "GLF", "GLG", "GLO", "GLP", "GLS", "GLT", "GM0", "GMB", "GMH", "GMT", "GMZ", "GN1", "GN4", "GNS", "GNX", "GP0", "GP1", "GP4", "GPH", "GPK", "GPM", "GPO", "GPQ", "GPU", "GPV", "GPW", "GQ1", "GRF", "GRX", "GS1", "GS9", "GTK", "GTM", "GTR", "GU0", "GU1", "GU2", "GU3", "GU4", "GU5", "GU6", "GU8", "GU9", "GUF", "GUL", "GUP", "GUZ", "GXL", "GYE", "GYG", "GYP", "GYU", "GYV", "GZL", "H1M", "H1S", "H2P", "H53", "H6Q", "H6Z", "HBZ", "HD4", "HNV", "HNW", "HSG", "HSH", "HSJ", "HSQ", "HSX", "HSY", "HTG", "HTM", "I57", "IAB", "IDC", "IDF", "IDG", "IDR", "IDS", "IDU", "IDX", "IDY", "IEM", "IN1", "IPT", "ISD", "ISL", "ISX", "IXD", "J5B", "JFZ", "JHM", "JLT", "JS2", "JV4", "JVA", "JVS", "JZR", "K5B", "K99", "KBA", "KBG", "KD5", "KDA", "KDB", "KDD", "KDE", "KDF", "KDM", "KDN", "KDO", "KDR", "KFN", "KG1", "KGM", "KME", "KO1", "KO2", "KOT", "KTU", "L1L", "L6S", "LAH", "LAK", "LAO", "LAT", "LB2", "LBS", "LBT", "LCN", "LDY", "LEC", "LFR", "LGC", "LGU", "LKA", "LKS", "LNV", "LOG", "LOX", "LRH", "LVO", "LVZ", "LXB", "LXC", "LXZ", "LZ0", "M1F", "M1P", "M2F", "M3N", "M55", "M6D", "M6P", "M7B", "M7P", "M8C", "MA1", "MA2", "MA3", "MA8", "MAF", "MAG", "MAL", "MAN", "MAT", "MAV", "MAW", "MBE", "MBF", "MBG", "MCU", "MDA", "MDP", "MFB", "MFU", "MG5", "MGC", "MGL", "MGS", "MJJ", "MLB", "MLR", "MMA", "MN0", "MNA", "MQG", "MQT", "MRH", "MRP", "MSX", "MTT", "MUB", "MUR", "MVP", "MXY", "MXZ", "MYG", "N1L", "N9S", "NA1", "NAA", "NAG", "NBG", "NBX", "NBY", "NDG", "NFG", "NG1", "NG6", "NGA", "NGC", "NGE", "NGK", "NGR", "NGS", "NGY", "NGZ", "NHF", "NLC", "NM6", "NM9", "NNG", "NPF", "NSQ", "NT1", "NTF", "NTO", "NTP", "NXD", "NYT", "O1G", "OAK", "OEL", "OI7", "OPM", "OSU", "OTG", "OTN", "OTU", "OX2", "P53", "P6P", "PA1", "PAV", "PDX", "PH5", "PKM", "PNA", "PNG", "PNJ", "PNW", "PPC", "PRP", "PSG", "PSV", "PUF", "PZU", "QIF", "QKH", "QPS", "R1P", "R1X", "R2B", "R2G", "RAE", "RAF", "RAM", "RAO", "RCD", "RER", "RF5", "RGG", "RHA", "RHC", "RI2", "RIB", "RIP", "RM4", "RP3", "RP5", "RP6", "RR7", "RRJ", "RRY", "RST", "RTG", "RTV", "RUG", "RUU", "RV7", "RVG", "RVM", "RWI", "RY7", "RZM", "S7P", "S81", "SA0", "SCG", "SCR", "SDY", "SEJ", "SF6", "SF9", "SFJ", "SFU", "SG4", "SG5", "SG6", "SG7", "SGA", "SGC", "SGD", "SGN", "SHB", "SHD", "SHG", "SIA", "SID", "SIO", "SIZ", "SLB", "SLM", "SLT", "SMD", "SN5", "SNG", "SOE", "SOG", "SOR", "SR1", "SSG", "STZ", "SUC", "SUP", "SUS", "SWE", "SZZ", "T68", "T6P", "T6T", "TA6", "TCB", "TCG", "TDG", "TEU", "TF0", "TFU", "TGA", "TGK", "TGR", "TGY", "TH1", "TMR", "TMX", "TNX", "TOA", "TOC", "TQY", "TRE", "TRV", "TS8", "TT7", "TTV", "TTZ", "TU4", "TUG", "TUJ", "TUP", "TUR", "TVD", "TVG", "TVM", "TVS", "TVV", "TVY", "TW7", "TWA", "TWD", "TWG", "TWJ", "TWY", "TXB", "TYV", "U1Y", "U2A", "U2D", "U63", "U8V", "U97", "U9A", "U9D", "U9G", "U9J", "U9M", "UAP", "UCD", "UDC", "UEA", "V3M", "V3P", "V71", "VG1", "VTB", "W9T", "WIA", "WOO", "WUN", "X0X", "X1P", "X1X", "X2F", "X6X", "XDX", "XGP", "XIL", "XLF", "XLS", "XMM", "XXM", "XXR", "XXX", "XYF", "XYL", "XYP", "XYS", "XYT", "XYZ", "YIO", "YJM", "YKR", "YO5", "YX0", "YX1", "YYB", "YYH", "YYJ", "YYK", "YYM", "YYQ", "YZ0", "Z0F", "Z15", "Z16", "Z2D", "Z2T", "Z3K", "Z3L", "Z3Q", "Z3U", "Z4K", "Z4R", "Z4S", "Z4U", "Z4V", "Z4W", "Z4Y", "Z57", "Z5J", "Z5L", "Z61", "Z6H", "Z6J", "Z6W", "Z8H", "Z8T", "Z9D", "Z9E", "Z9H", "Z9K", "Z9L", "Z9M", "Z9N", "Z9W", "ZB0", "ZB1", "ZB2", "ZB3", "ZCD", "ZCZ", "ZD0", "ZDC", "ZDO", "ZEE", "ZEL", "ZGE", "ZMR"])
THREE_TO_ONE = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
RNA_NUCLEOTIDES = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}

GLYCOSIDIC_BOND_THRESHOLD = 2.0
COVALENT_BOND_THRESHOLD = 1.75
GLYCOSYLATION_BOND_THRESHOLD = 1.6

# --- Dataclasses ---
@dataclass(frozen=True)
class Atom:
    idx: int; record_type: str; name: str; res_name: str; chain_id: str
    res_num: int; element: str; coords: np.ndarray

@dataclass(frozen=True)
class GlycoConnection:
    parent_res_key: Tuple[str, int]; child_res_key: Tuple[str, int]
    parent_acceptor_atom: Atom; child_donor_atom: Atom; anomeric_config: Optional[str]

# --- Custom YAML Dumper ---
class SingleQuotedStr(str): pass
class FlowList(list): pass

def represent_single_quoted_str(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")

def represent_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

class MyDumper(Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

MyDumper.add_representer(SingleQuotedStr, represent_single_quoted_str)
MyDumper.add_representer(FlowList, represent_flow_list)

def get_atom_mass(element: str) -> float:
    """Returns heuristic mass for priority sorting: O > N > C > others."""
    table = {'O': 16.0, 'N': 14.0, 'C': 12.0, 'S': 32.0, 'P': 31.0, 'H': 1.0}
    return table.get(element.upper(), 0.0)

def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    Calculate the dihedral angle (in degrees) defined by 4 points.
    Range: -180 to 180.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-6:
        return 0.0
    b2_u = b2 / norm_b2

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    x = np.dot(n1, n2)
    y = np.dot(np.cross(n1, n2), b2_u)

    return np.degrees(np.arctan2(y, x))

class GlycanProcessor:
    def __init__(self, glycan_atoms: List[Atom]):
        self.atoms_map = {a.idx: a for a in glycan_atoms}
        self.atoms_by_residue = self._group_atoms_by_residue(glycan_atoms)

    def _group_atoms_by_residue(self, atoms: List[Atom]) -> Dict[Tuple[str, int], List[Atom]]:
        grouped = defaultdict(list)
        for atom in atoms: 
            grouped[(atom.chain_id, atom.res_num)].append(atom)
        return dict(grouped)

    def build_residue_graph(self, atoms: List[Atom]) -> Dict[int, List[int]]:
        graph = defaultdict(list)
        if len(atoms) < 2: return graph
        coords = np.array([a.coords for a in atoms])
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=COVALENT_BOND_THRESHOLD)
        for i, j in pairs: 
            graph[atoms[i].idx].append(atoms[j].idx)
            graph[atoms[j].idx].append(atoms[i].idx)
        return graph

    def find_cycle_indices(self, graph: Dict[int, List[int]]) -> Set[int]:
        """
        Finds a single cycle in the graph using DFS.
        Returns a set of atom indices comprising the ring.
        Matches logic from preprocess_glycans.py.
        """
        visited = set()
        for start_node in sorted(graph.keys()):
            if start_node in visited: continue
            stack = [(start_node, -1, [start_node])]
            while stack:
                curr, parent, path = stack.pop()
                if curr not in visited: visited.add(curr)
                for neighbor in sorted(graph[curr]):
                    if neighbor == parent: continue
                    if neighbor in path:
                        cycle_start_index = path.index(neighbor)
                        cycle_path = path[cycle_start_index:]
                        if len(cycle_path) >= 3:
                            return set(cycle_path)
                    else:
                        stack.append((neighbor, curr, path + [neighbor]))
        return set()

    def determine_config_dihedral(self, child_residue_atoms: List[Atom], acceptor_atom: Atom, donor_atom: Atom) -> Optional[str]:
        """
        Determines Alpha/Beta configuration using the dihedral angle quartet method.
        (Atom 1: Acceptor, Atom 2: Anomeric C, Atom 3: Ring O, Atom 4: Endocyclic Neighbor)
        """
        try:
            atoms_map = {a.idx: a for a in child_residue_atoms}
            
            # 1. Find Ring
            graph = self.build_residue_graph(child_residue_atoms)
            ring_indices = self.find_cycle_indices(graph)
            if not ring_indices: return None

            # 2. Find Ring Non-Carbon Atom (Atom 3)
            ring_non_carbons = [idx for idx in ring_indices if atoms_map[idx].element.upper() != 'C']
            if not ring_non_carbons: return None
            atom_3_idx = ring_non_carbons[0]
            atom_3 = atoms_map[atom_3_idx]

            # 3. Find Anomeric Carbon (Atom 2) - C1 or C2 in ring
            c1_candidates = [idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C1']
            c2_candidates = [idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C2']
            
            if c1_candidates: atom_2_idx = c1_candidates[0]
            elif c2_candidates: atom_2_idx = c2_candidates[0]
            else: return None # No standard anomeric carbon found
            
            atom_2 = atoms_map[atom_2_idx]

            # 4. Find Other Endocyclic Atom (Atom 4)
            neighbors_of_3 = graph[atom_3_idx]
            atom_4_idx = -1
            for n_idx in neighbors_of_3:
                if n_idx in ring_indices and n_idx != atom_2_idx:
                    atom_4_idx = n_idx
                    break
            if atom_4_idx == -1: return None
            atom_4 = atoms_map[atom_4_idx]

            # Atom 1 is the acceptor/substituent
            atom_1 = acceptor_atom

            # 5. Calculate Dihedral
            angle = calculate_dihedral(atom_1.coords, atom_2.coords, atom_3.coords, atom_4.coords)

            # 6. Determine Config
            if -95.0 <= angle <= 95.0:
                return 'a'
            elif (angle > 95.0 and angle < 225.0) or (angle < -95.0 and angle > -225.0):
                return 'b'
            else:
                # Edge case close to 180
                if abs(abs(angle) - 180.0) < 1e-3: return 'b'
                return None # Out of bounds/Error
        except Exception:
            return None

    def _determine_root_config(self, atoms: List[Atom]) -> Optional[str]:
        """
        Determines the anomeric configuration for a root or lone monosaccharide.
        Finds the local exocyclic substituent (e.g. O1) to act as the 'acceptor' 
        for the dihedral calculation.
        """
        atoms_map = {a.idx: a for a in atoms}
        graph = self.build_residue_graph(atoms)
        ring_indices = self.find_cycle_indices(graph)
        if not ring_indices: return None

        # Identify Anomeric Carbon
        c1_candidates = [idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C1']
        c2_candidates = [idx for idx in ring_indices if atoms_map[idx].name.strip().upper() == 'C2']
        
        anomeric_idx = None
        if c1_candidates: anomeric_idx = c1_candidates[0]
        elif c2_candidates: anomeric_idx = c2_candidates[0]
        
        if anomeric_idx is None: return None

        anomeric_atom = atoms_map[anomeric_idx]

        # Find heaviest exocyclic substituent (should be O1, O2, etc.)
        neighbors = graph[anomeric_idx]
        exocyclic_candidates = []
        for n_idx in neighbors:
            if n_idx not in ring_indices:
                exocyclic_candidates.append(atoms_map[n_idx])
        
        if not exocyclic_candidates: return None
        
        # Sort by mass (O > C > H)
        exocyclic_candidates.sort(key=lambda a: get_atom_mass(a.element), reverse=True)
        best_substituent = exocyclic_candidates[0]

        # Use the standard dihedral logic with the local substituent
        return self.determine_config_dihedral(atoms, best_substituent, anomeric_atom)

    def detect_and_analyze_connections(self) -> List[GlycoConnection]:
        connections, seen_bonds = [], set()
        carbons = [a for a in self.atoms_map.values() if a.element == 'C']
        oxygens = [a for a in self.atoms_map.values() if a.element == 'O']
        
        if not carbons or not oxygens: return []
        
        oxygen_tree = cKDTree(np.array([o.coords for o in oxygens]))
        carbon_coords = np.array([c.coords for c in carbons])
        nearby_pairs = oxygen_tree.query_ball_point(carbon_coords, r=GLYCOSIDIC_BOND_THRESHOLD)
        
        for c_idx, o_indices in enumerate(nearby_pairs):
            if not o_indices: continue
            carbon_atom = carbons[c_idx]
            child_key = (carbon_atom.chain_id, carbon_atom.res_num)
            
            for o_idx in o_indices:
                oxygen_atom = oxygens[o_idx]
                parent_key = (oxygen_atom.chain_id, oxygen_atom.res_num)
                
                if parent_key == child_key: continue
                
                bond_key = tuple(sorted((carbon_atom.idx, oxygen_atom.idx)))
                if bond_key in seen_bonds: continue
                
                child_atoms = self.atoms_by_residue.get(child_key)
                if not child_atoms: continue
                
                # USE DIHEDRAL LOGIC
                config = self.determine_config_dihedral(child_atoms, oxygen_atom, carbon_atom)
                connections.append(GlycoConnection(parent_key, child_key, oxygen_atom, carbon_atom, config))
                seen_bonds.add(bond_key)
        return connections

    def _parse_index_from_atom(self, atom_name: str, prefix: str) -> Optional[int]:
        s = atom_name.strip().upper()
        digits = ''.join(ch for ch in atom_name[len(prefix):] if ch.isdigit())
        try: return int(digits) if s.startswith(prefix) and digits else None
        except ValueError: return None

    def _build_graph_for_iupac(self, connections: List[GlycoConnection]):
        children_of, parent_of, nodes = defaultdict(list), {}, set()
        for c in connections: 
            children_of[c.parent_res_key].append((c.child_res_key, c))
            parent_of[c.child_res_key] = (c.parent_res_key, c)
            nodes.add(c.parent_res_key)
            nodes.add(c.child_res_key)
        return children_of, parent_of, nodes

    def _longest_path_len(self, node, children_of, memo) -> int:
        if node in memo: return memo[node]
        memo[node] = 1 + max([self._longest_path_len(child, children_of, memo) for child, _ in children_of.get(node, [])] or [0])
        return memo[node]

    def _render_link(self, child_key, edge_conn, memo_depth, children_of):
        # Child in a link is NOT a root, pass False
        child_seg = self._render_subtree(child_key, children_of, memo_depth, is_root=False)
        cfg = (edge_conn.anomeric_config or '?').lower()
        d_str = str(self._parse_index_from_atom(edge_conn.child_donor_atom.name, 'C') or '?')
        a_str = str(self._parse_index_from_atom(edge_conn.parent_acceptor_atom.name, 'O') or '?')
        return f"{child_seg}({cfg}{d_str}-{a_str})"

    def _render_subtree(self, node_key, children_of, memo_depth, is_root=False) -> str:
        children = children_of.get(node_key, [])
        atoms = self.atoms_by_residue[node_key]
        res_name = atoms[0].res_name
        
        # Determine suffix ONLY if it is a root/lone residue
        suffix = ""
        if is_root:
            root_config = self._determine_root_config(atoms)
            if root_config in ['a', 'b']:
                suffix = f"({root_config})"
        
        if not children: 
            return res_name + suffix
        
        child_depths = sorted([(self._longest_path_len(ck, children_of, memo_depth), ck) for ck, _ in children], reverse=True)
        main_child_key = child_depths[0][1]
        
        main_edge = next(edge for ck, edge in children if ck == main_child_key)
        main_seg = self._render_link(main_child_key, main_edge, memo_depth, children_of)
        
        branch_segs = sorted([self._render_link(ck, edge, memo_depth, children_of) for (_, ck), edge in zip(child_depths[1:], [e for c, e in children if c != main_child_key])])
        
        return main_seg + "".join(f"[{s}]" for s in branch_segs) + res_name + suffix

    def generate_iupac_strings(self, connections: List[GlycoConnection]) -> Dict[Tuple[str, int], str]:
        children_of, parent_of, _ = self._build_graph_for_iupac(connections)
        all_res_keys = sorted(self.atoms_by_residue.keys(), key=lambda k: (k[0], k[1]))
        
        # Roots are nodes with no parent
        roots = sorted([k for k in all_res_keys if k not in parent_of], key=lambda k: (k[0], k[1]))
        
        memo_depth, iupac_map = {}, {}
        for r_key in roots: 
            # These are global roots, so pass is_root=True
            iupac_map[r_key] = self._render_subtree(r_key, children_of, memo_depth, is_root=True)
        return iupac_map

    def run(self) -> Dict[Tuple[str, int], str]:
        connections = self.detect_and_analyze_connections()
        
        # Fallback for single monosaccharides with no connections
        if not connections and len(self.atoms_by_residue) > 0:
             iupac_map = {}
             for res_key, atoms in self.atoms_by_residue.items():
                 # Treat lone residue as a root
                 iupac_map[res_key] = self._render_subtree(res_key, {}, {}, is_root=True)
             return iupac_map
             
        return self.generate_iupac_strings(connections)

# --- CASP File Processing ---
def parse_casp_file(filepath: str) -> List[Dict[str, str]]:
    """Parses a CASP-formatted text file into a list of sequence dictionaries."""
    sequences = []
    with open(filepath, 'r') as f:
        current_header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("###"):
                continue
            if line.startswith('>'):
                current_header = line
            elif current_header:
                seq_id = current_header.split()[0][1:]
                sequence = line
                is_rna = all(c.upper() in RNA_NUCLEOTIDES for c in sequence)
                seq_type = 'rna' if is_rna else 'protein'
                sequences.append({'id': seq_id, 'sequence': sequence, 'type': seq_type})
                current_header = None
    return sequences

def generate_yaml_from_casp(casp_data: List[Dict[str, str]], output_path: str):
    """Generates the final YAML file from parsed CASP data."""
    output_data = {'version': 1, 'sequences': []}
    for item in sorted(casp_data, key=lambda x: x['id']):
        if item['type'] == 'protein':
            # Construct the full, invariant MSA path as a standard string
            msa_path = f"work/keshavsundar/work_sundar/pdb_glycan_test/sugar_benchmark_msa/hhblits_full_{item['id']}.a3m"
            entry = {'id': item['id'], 'sequence': item['sequence'], 'msa': msa_path}
            output_data['sequences'].append({'protein': entry})
        elif item['type'] == 'rna':
            entry = {'id': item['id'], 'sequence': item['sequence']}
            output_data['sequences'].append({'rna': entry})
    
    # Dump to a string first to allow for post-processing
    yaml_string = yaml.dump(output_data, Dumper=MyDumper, sort_keys=False, default_flow_style=False, indent=2)
    
    # Add a newline ONLY after 'version: 1'
    yaml_string = yaml_string.replace('version: 1\n', 'version: 1\n\n')

    with open(output_path, 'w') as f:
        f.write(yaml_string)

# --- PDB File Processing ---
def parse_pdb_file(filepath: str) -> List[Atom]:
    """Parses a PDB file, returning a list of unique Atom objects."""
    atoms, seen_keys = [], set()
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if not (line.startswith("ATOM") or line.startswith("HETATM")): continue
            alt_loc = line[16]
            if alt_loc not in (' ', 'A'): continue
            try:
                chain_id, res_num, atom_name = line[21:22].strip() or "A", int(line[22:26].strip()), line[12:16].strip()
                atom_key = (chain_id, res_num, atom_name)
                if atom_key in seen_keys: continue
                seen_keys.add(atom_key)
                atoms.append(Atom(
                    idx=i, record_type=line[0:6].strip(), name=atom_name, res_name=line[17:20].strip(),
                    chain_id=chain_id, res_num=res_num,
                    element=(line[76:78].strip() or line[12:14].strip()).upper(),
                    coords=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                ))
            except (ValueError, IndexError): continue
    return atoms

def process_pdb_data(all_atoms: List[Atom]):
    """Processes atoms from a PDB file to extract sequence and site information."""
    protein_atoms, glycan_atoms, rna_atoms = [], [], []
    for atom in all_atoms:
        if atom.res_name in THREE_TO_ONE: protein_atoms.append(atom)
        elif atom.res_name in ALLOWED_GLYCANS: glycan_atoms.append(atom)
        elif atom.res_name in RNA_NUCLEOTIDES: rna_atoms.append(atom)

    protein_chains = defaultdict(list); [protein_chains[a.chain_id].append(a) for a in protein_atoms]
    glycan_chains = defaultdict(list); [glycan_chains[a.chain_id].append(a) for a in glycan_atoms]
    rna_chains = defaultdict(list); [rna_chains[a.chain_id].append(a) for a in rna_atoms]
    
    proteins = {cid: {'sequence': "".join(THREE_TO_ONE.get(r_name, '') for _, r_name in sorted(list(set((a.res_num, a.res_name) for a in atoms))))} for cid, atoms in protein_chains.items() if cid != 'N'}
    rnas = {cid: {'sequence': "".join(RNA_NUCLEOTIDES.get(r_name, '') for _, r_name in sorted(list(set((a.res_num, a.res_name) for a in atoms))))} for cid, atoms in rna_chains.items() if cid != 'N'}
    glycans = {cid: {'iupac': list(iupac_map.values())[0]} for cid, atoms in glycan_chains.items() if cid != 'N' and (iupac_map := GlycanProcessor(atoms).run())}

    glycosylation_sites = []
    valid_protein_atoms, valid_glycan_atoms = [a for a in protein_atoms if a.chain_id != 'N'], [a for a in glycan_atoms if a.chain_id != 'N']
    if valid_protein_atoms and valid_glycan_atoms:
        prot_tree, glyc_tree = cKDTree([a.coords for a in valid_protein_atoms]), cKDTree([a.coords for a in valid_glycan_atoms])
        pairs, seen_sites = prot_tree.query_ball_tree(glyc_tree, r=GLYCOSYLATION_BOND_THRESHOLD), set()
        for p_idx, g_indices in enumerate(pairs):
            for g_idx in g_indices:
                p_atom, g_atom = valid_protein_atoms[p_idx], valid_glycan_atoms[g_idx]
                site_key = (p_atom.chain_id, p_atom.res_num, g_atom.chain_id)
                if site_key in seen_sites: continue
                seen_sites.add(site_key)
                glycosylation_sites.append({'site': {'protein': FlowList([p_atom.chain_id, p_atom.res_num, p_atom.name]), 'glycan': FlowList([g_atom.chain_id, 0, g_atom.name])}})
    return proteins, rnas, glycans, glycosylation_sites

def generate_yaml_from_pdb(proteins, rnas, glycans, glycosylation_sites, output_path, msa_id=None):
    """Generates the final YAML file from the processed PDB data."""
    output_data = {'version': 1, 'sequences': []}
    for chain_id, data in sorted(proteins.items()):
        # Default MSA entry is 'empty'
        msa_entry = 'empty'
        if msa_id:
            # Construct the full, invariant MSA path as a standard string
            msa_path = f"work/keshavsundar/work_sundar/pdb_glycan_test/sugar_benchmark_msa/hhblits_full_{msa_id}.a3m"
            #msa_entry = 'msa_path'
            msa_entry = 'empty'
            
        output_data['sequences'].append({'protein': {'id': chain_id, 'sequence': data['sequence'], 'msa': msa_entry}})
    
    rna_groups = defaultdict(list); [rna_groups[data['sequence']].append(cid) for cid, data in rnas.items()]
    for seq, ids in sorted(rna_groups.items()):
        output_data['sequences'].append({'rna': {'id': ids[0] if len(ids) == 1 else sorted(ids), 'sequence': seq}})
        
    for chain_id, data in sorted(glycans.items()):
        # Glycan IUPAC strings correctly keep the single quote wrapper
        output_data['sequences'].append({'glycan': {'id': chain_id, 'iupac': SingleQuotedStr(data['iupac'])}})

    if glycosylation_sites:
        output_data['glycosylation'] = sorted(glycosylation_sites, key=lambda x: (x['site']['protein'][0], x['site']['protein'][1], x['site']['glycan'][0]))
        
    # Dump to a string first to allow for post-processing
    yaml_string = yaml.dump(output_data, Dumper=MyDumper, sort_keys=False, default_flow_style=False, indent=2)

    # Add a newline after 'version: 1'
    yaml_string = yaml_string.replace('version: 1\n', 'version: 1\n\n')

    # If the glycosylation key exists, add a newline before it to separate it from sequences
    if 'glycosylation' in output_data:
        yaml_string = yaml_string.replace('\nglycosylation:', '\n\nglycosylation:')

    with open(output_path, 'w') as f:
        f.write(yaml_string)
        
def main():
    parser = argparse.ArgumentParser(description="Create a YAML sequence file from a PDB or CASP text file.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdb_file", type=str, help="Path to the input PDB file.")
    group.add_argument("--casp", type=str, dest="casp_file", help="Path to the input CASP format text file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the output YAML file.")
    parser.add_argument("--msa_id", type=str, default=None, help="The unique ID for the MSA file path (PDB mode only).")
    args = parser.parse_args()

    if args.casp_file:
        if not os.path.exists(args.casp_file):
            sys.exit(f"Error: CASP file not found at '{args.casp_file}'")
        casp_data = parse_casp_file(args.casp_file)
        generate_yaml_from_casp(casp_data, args.output_file)
    
    elif args.pdb_file:
        if not os.path.exists(args.pdb_file):
            sys.exit(f"Error: PDB file not found at '{args.pdb_file}'")
        all_atoms = parse_pdb_file(args.pdb_file)
        if not all_atoms:
            # Create an empty file for compatibility with the control script
            with open(args.output_file, 'w') as f:
                yaml.dump({'version': 1, 'sequences': []}, f)
            print(f"Warning: No ATOM/HETATM records in {os.path.basename(args.pdb_file)}. Wrote empty YAML.", file=sys.stderr)
            return

        proteins, rnas, glycans, glycosylation_sites = process_pdb_data(all_atoms)
        generate_yaml_from_pdb(proteins, rnas, glycans, glycosylation_sites, args.output_file, msa_id=args.msa_id)

if __name__ == "__main__":
    main()
