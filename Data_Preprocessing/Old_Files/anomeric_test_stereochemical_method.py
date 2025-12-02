#!/usr/bin/env python3
#
# anomeric_test_c5_invariant.py
#
# This script analyzes glycan connectivity in a PDB file and determines
# anomeric configuration using a single, precise plane-based cis/trans rule.
#
# Unified α/β definition (MODIFIED):
#   - Build the ring plane and its normal.
#   - Use vector from anomeric carbon (child donor C) to glycosidic oxygen (parent acceptor O)
#     as the "anomeric substituent" direction.
#   - Reference substituent (Invariant Rule):
#       * Invariantly find the ring atom named 'C5'.
#       * Find its heaviest exocyclic substituent (Priority: O > N > C).
#       * Use the vector from C5 to this substituent as the reference.
#   - α if the signed projections onto the ring normal have opposite signs (trans).
#   - β if the signs are the same (cis).
#
# Dependencies: numpy, scipy
#

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

# --- Constants ---
GLYCOSIDIC_BOND_THRESHOLD = 2.0
COVALENT_BOND_THRESHOLD = 1.75

# --- Helpers to identify residues we must ignore ---
NON_GLYCAN_RESIDUES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYV","VAL",
    "MSE","SEC","PYL", "ASX","GLX","HID","HIE","HIP",
    "HOH", "WAT", "ROH", "SO3"
}

def is_non_glycan(res_name: str) -> bool:
    return res_name.upper() in NON_GLYCAN_RESIDUES

# --- Dataclasses ---
@dataclass(frozen=True)
class Atom:
    idx: int
    name: str
    res_name: str
    chain_id: str
    res_num: int
    element: str
    coords: np.ndarray

@dataclass(frozen=True)
class GlycoConnection:
    parent_res_key: Tuple[str, int]
    child_res_key: Tuple[str, int]
    parent_acceptor_atom: Atom   # glycosidic oxygen (on parent residue)
    child_donor_atom: Atom       # anomeric carbon (on child residue)
    config: Optional[str]        # 'a' / 'b' or error string

# --- Graph and Topology Functions ---
def build_residue_graph(atoms: List[Atom]) -> Dict[int, List[int]]:
    graph = defaultdict(list)
    coords = np.array([a.coords for a in atoms])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=COVALENT_BOND_THRESHOLD)
    for i, j in pairs:
        graph[atoms[i].idx].append(atoms[j].idx)
        graph[atoms[j].idx].append(atoms[i].idx)
    return graph

def find_and_order_ring(graph: Dict[int, List[int]], start_node_idx: int, atoms_map: Dict[int, Atom]) -> Optional[List[int]]:
    """
    Try to find a 6- or 5-membered ring that contains start_node_idx (the anomeric C).
    Return an ordered list of indices around the ring, ending with the ring oxygen.
    Heuristic mirrors the original approach to remain robust on PDB CCD sugars.
    """
    for ring_size in [6, 5]:
        q = [(start_node_idx, [start_node_idx])]
        while q:
            curr_idx, path = q.pop(0)
            if len(path) == ring_size:
                if start_node_idx in graph.get(curr_idx, []):
                    ring_neighbors = [idx for idx in graph.get(start_node_idx, []) if idx in path]
                    c2_candidate = -1
                    best_dist_to_o = float('inf')
                    for neighbor_idx in ring_neighbors:
                        try:
                            path_from_neighbor = path[path.index(neighbor_idx):]
                            dist_to_o = next((i for i, atom_idx in enumerate(path_from_neighbor) if atoms_map[atom_idx].element == 'O'), float('inf'))
                            if dist_to_o < best_dist_to_o:
                                best_dist_to_o = dist_to_o
                                c2_candidate = neighbor_idx
                        except ValueError:
                            continue
                    if c2_candidate == -1:
                        continue
                    ordered_ring = [start_node_idx, c2_candidate]
                    prev_node, curr_node = start_node_idx, c2_candidate
                    while len(ordered_ring) < ring_size:
                        for neighbor in graph[curr_node]:
                            if neighbor in path and neighbor != prev_node:
                                ordered_ring.append(neighbor)
                                prev_node, curr_node = curr_node, neighbor
                                break
                    return ordered_ring
                continue
            for neighbor_idx in graph.get(curr_idx, []):
                if neighbor_idx not in path:
                    q.append((neighbor_idx, path + [neighbor_idx]))
    return None

# --- Unified α/β Determination (Plane-Based, precise) ---

def _compute_ring_normal(ordered_ring_indices: List[int], atoms_map: Dict[int, Atom]) -> Optional[np.ndarray]:
    try:
        coords = np.array([atoms_map[i].coords for i in ordered_ring_indices], dtype=float)
        centroid = coords.mean(axis=0)
        X = coords - centroid
        cov = X.T @ X
        w, v = np.linalg.eigh(cov)
        n = v[:, np.argmin(w)]
        # Fix orientation for determinism using first edges
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
    return float(np.dot(n, vec))

# --- MODIFIED LOGIC START ---
# The original logic for finding the reference substituent has been replaced
# with a single, invariant method based on the C5 atom and its heaviest substituent.

def _find_c5_reference_substituent(ordered_ring_indices: List[int],
                                   ring_set: set,
                                   graph: Dict[int, List[int]],
                                   atoms_map: Dict[int, Atom]) -> Optional[Tuple[int, int]]:
    """
    Finds the reference vector for anomeric determination.
    Rule: Invariantly use the C5 atom and its heaviest exocyclic substituent.
    Returns a tuple of (c5_atom_index, substituent_atom_index) or None if not found.
    """
    # 1. Find the C5 atom within the ring by its PDB name.
    c5_idx = -1
    for idx in ordered_ring_indices:
        if atoms_map[idx].name.upper() == 'C5':
            c5_idx = idx
            break
    if c5_idx == -1:
        return None # C5 atom not found in the ring

    # 2. Find all exocyclic neighbors of C5.
    exocyclic_neighbors = [nbr for nbr in graph.get(c5_idx, []) if nbr not in ring_set]
    if not exocyclic_neighbors:
        return None # C5 has no exocyclic substituents

    # 3. Find the heaviest substituent based on element priority.
    PRIORITY = {'O': 3, 'N': 2, 'C': 1}
    best_substituent_idx = -1
    max_priority = -1
    for nbr_idx in exocyclic_neighbors:
        element = atoms_map[nbr_idx].element.upper()
        priority = PRIORITY.get(element, 0)
        if priority > max_priority:
            max_priority = priority
            best_substituent_idx = nbr_idx

    if best_substituent_idx == -1:
        return None # No O, N, or C substituent found

    return c5_idx, best_substituent_idx

# --- MODIFIED LOGIC END ---

def determine_anomeric_config(child_residue_atoms: List[Atom], acceptor_atom: Atom, donor_atom: Atom) -> Optional[str]:
    """
    Determine α/β by ring-plane cis/trans with a unified rule:
    - sign_glyco = proj of (O_gly - C_anomeric) on ring normal
    - sign_ref   = proj of (Heaviest_Substituent_on_C5 - C5) on ring normal
    - α if sign_glyco * sign_ref < 0 (trans), β otherwise (cis)
    """
    try:
        graph = build_residue_graph(child_residue_atoms)
        atoms_map = {a.idx: a for a in child_residue_atoms}

        # 1) Find ordered ring around the anomeric carbon (donor_atom.idx)
        ordered_ring_indices = find_and_order_ring(graph, donor_atom.idx, atoms_map)
        if not ordered_ring_indices:
            return "error (no ring)"
        ring_size = len(ordered_ring_indices)
        if ring_size not in [5, 6]:
            return "error (bad ring size)"
        ring_set = set(ordered_ring_indices)

        # 2) Ring plane normal
        n = _compute_ring_normal(ordered_ring_indices, atoms_map)
        if n is None:
            return "error (collinear plane)"

        # 3) Glycosidic substituent direction (anomeric C -> glycosidic O on parent)
        a_pos = atoms_map[donor_atom.idx].coords
        og_pos = acceptor_atom.coords
        sign_glyco = _signed_projection(n, og_pos - a_pos)
        if abs(sign_glyco) < 1e-3:
            return "error (ambiguous plane)"

        # 4) Reference substituent: Invariantly use C5 and its heaviest substituent
        ref_pair = _find_c5_reference_substituent(ordered_ring_indices, ring_set, graph, atoms_map)
        if ref_pair is None:
            return "error (no C5 reference)"
        
        c5_idx, substituent_idx = ref_pair
        c5_pos = atoms_map[c5_idx].coords
        substituent_pos = atoms_map[substituent_idx].coords
        sign_ref = _signed_projection(n, substituent_pos - c5_pos)
        if abs(sign_ref) < 1e-3:
            return "error (ambiguous plane)"

        # 5) α/β classification: trans -> α, cis -> β
        return 'a' if (sign_glyco * sign_ref) < 0.0 else 'b'

    except Exception:
        return "error (exception)"

# --- IUPAC and Reporting Functions (single-source config) ---
def _parse_index(name: str, prefix: str) -> Optional[int]:
    s = name.strip().upper()
    if not s.startswith(prefix):
        return None
    digits = ''.join(c for c in s[len(prefix):] if c.isdigit())
    return int(digits) if digits else None

def _build_graph(conns: List[GlycoConnection]):
    children_of, parent_of, nodes = defaultdict(list), {}, set()
    for c in conns:
        p, ch = c.parent_res_key, c.child_res_key
        children_of[p].append((ch, c))
        parent_of[ch] = (p, c)
        nodes.add(p); nodes.add(ch)
    return children_of, parent_of, nodes

def _longest_path(node, children, memo):
    if node in memo:
        return memo[node]
    memo[node] = 1 + max([_longest_path(c, children, memo) for c, _ in children.get(node, [])] or [0])
    return memo[node]

def _render_link(child_key, edge: GlycoConnection, atoms, memo, children):
    child_seg = _render_subtree(child_key, children, atoms, memo)
    cfg = (edge.config or '?').lower()
    d_str = str(_parse_index(edge.child_donor_atom.name, 'C') or '?')
    a_str = str(_parse_index(edge.parent_acceptor_atom.name, 'O') or '?')
    return f"{child_seg}({cfg}{d_str}-{a_str})"

def _render_subtree(node, children, atoms, memo):
    child_list = children.get(node, [])
    if not child_list:
        return atoms[node][0].res_name
    child_depths = sorted([(_longest_path(ck, children, memo), ck) for ck, _ in child_list], reverse=True)
    main_child = child_depths[0][1]
    main_edge = next(edge for ck, edge in child_list if ck == main_child)
    main_seg = _render_link(main_child, main_edge, atoms, memo, children)
    branch_segs = sorted([
        _render_link(ck, e, atoms, memo, children)
        for _, ck in child_depths[1:] for c, e in child_list if c == ck
    ])
    return main_seg + "".join(f"[{s}]" for s in branch_segs) + atoms[node][0].res_name

def generate_iupac_names(atoms_by_residue: Dict, connections: List[GlycoConnection]) -> List[str]:
    children_of, parent_of, nodes = _build_graph(connections)
    roots = sorted([n for n in nodes if n not in parent_of], key=lambda k: (k[0], k[1]))
    all_glycan_keys = [k for k, a in atoms_by_residue.items() if a and not is_non_glycan(a[0].res_name)]
    isolated = sorted([k for k in all_glycan_keys if k not in nodes], key=lambda k: (k[0], k[1]))
    memo_depth = {}
    names = [_render_subtree(r, children_of, atoms_by_residue, memo_depth) for r in roots]
    names.extend([atoms_by_residue[iso][0].res_name for iso in isolated])
    return names

# --- Main PDB Parsing and Connection Detection ---
def parse_pdb_atoms(filepath: str) -> Tuple[Dict[int, Atom], Dict[Tuple, List[Atom]]]:
    all_atoms_map, atoms_by_residue = {}, defaultdict(list)
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    element = (line[76:78].strip() or line[12:16].strip()[0]).upper()
                    if element in ('H', 'D'):
                        continue
                    atom = Atom(
                        idx=i,
                        name=line[12:16].strip(),
                        res_name=line[17:20].strip(),
                        chain_id=line[21:22].strip() or "A",
                        res_num=int(line[22:26].strip()),
                        element=element,
                        coords=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    )
                    all_atoms_map[i] = atom
                    atoms_by_residue[(atom.chain_id, atom.res_num)].append(atom)
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        print(f"Error: File not found '{filepath}'", file=sys.stderr)
        sys.exit(1)
    return all_atoms_map, dict(atoms_by_residue)

def detect_and_analyze_connections(all_atoms: Dict[int, Atom], atoms_by_residue: Dict[Tuple[str, int], List[Atom]]) -> List[GlycoConnection]:
    connections, seen_bonds = [], set()

    carbons = [a for a in all_atoms.values() if a.element == 'C']
    oxygens = [a for a in all_atoms.values() if a.element == 'O']
    if not carbons or not oxygens:
        return []

    oxygen_tree = cKDTree(np.array([o.coords for o in oxygens]))
    nearby_pairs = oxygen_tree.query_ball_point(np.array([c.coords for c in carbons]), r=GLYCOSIDIC_BOND_THRESHOLD)

    for c_idx, o_indices in enumerate(nearby_pairs):
        if not o_indices:
            continue
        carbon_atom = carbons[c_idx]
        child_key = (carbon_atom.chain_id, carbon_atom.res_num)
        child_atoms = atoms_by_residue.get(child_key)
        if not child_atoms or is_non_glycan(child_atoms[0].res_name):
            continue

        for o_idx in o_indices:
            oxygen_atom = oxygens[o_idx]
            parent_key = (oxygen_atom.chain_id, oxygen_atom.res_num)
            if parent_key == child_key:
                continue
            parent_atoms = atoms_by_residue.get(parent_key)
            if not parent_atoms or is_non_glycan(parent_atoms[0].res_name):
                continue

            bond_key = tuple(sorted((carbon_atom.idx, oxygen_atom.idx)))
            if bond_key in seen_bonds:
                continue

            config = determine_anomeric_config(child_atoms, oxygen_atom, carbon_atom)

            connections.append(GlycoConnection(
                parent_res_key=parent_key,
                child_res_key=child_key,
                parent_acceptor_atom=oxygen_atom,
                child_donor_atom=carbon_atom,
                config=config
            ))
            seen_bonds.add(bond_key)
    return connections

def print_connection_report(connections: List[GlycoConnection], atoms_by_residue: Dict[Tuple[str, int], List[Atom]]):
    print("-" * 80)
    print(" " * 26 + "Anomeric Configuration Report")
    print("-" * 80)
    if not connections:
        print("No glycosidic connections detected.")
        return
    print(f"{'Child Residue':<22} -> {'Parent Residue':<22} | {'Config'}")
    print("-" * 80)
    for conn in connections:
        child_info = atoms_by_residue[conn.child_res_key][0]
        parent_info = atoms_by_residue[conn.parent_res_key][0]
        child_str = f"{child_info.res_name} {child_info.res_num} ({child_info.chain_id})"
        parent_str = f"{parent_info.res_name} {parent_info.res_num} ({parent_info.chain_id})"
        print(f"{child_str:<22} -> {parent_str:<22} | {conn.config}")

def main():
    parser = argparse.ArgumentParser(description="Determine glycan anomeric configurations using a unified plane-based method.")
    parser.add_argument("pdb_file", type=str, help="Path to the input PDB file.")
    args = parser.parse_args()

    all_atoms, atoms_by_residue = parse_pdb_atoms(args.pdb_file)
    if not all_atoms:
        print("No atoms found.", file=sys.stderr)
        return

    connections = detect_and_analyze_connections(all_atoms, atoms_by_residue)

    print_connection_report(connections, atoms_by_residue)

    iupac_names = generate_iupac_names(atoms_by_residue, connections)
    print("\n" + "=" * 80 + "\nIUPAC-Condensed Sequence\n" + "=" * 80)
    if iupac_names:
        for i, name in enumerate(iupac_names, 1):
            print(f"  [{i}] {name}")
    else:
        print("(none detected)")

if __name__ == "__main__":
    main()
