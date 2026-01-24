import argparse
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import copy
import csv

import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy import stats

from Bio.PDB import PDBParser, MMCIFParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D

# Suppress warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
warnings.simplefilter('ignore', UserWarning)

# --- CONSTANTS ---
BOND_DIST = 2.0        
CONTACT_CUTOFF = 5.0    # [cite: 811]
RING_BOND_CUTOFF = 2.0 
GLYCAN_WEIGHT = 1.0     # Placeholder for weighted alignment

# DockQC Scaling Factors [cite: 876, 877]
D_RRMS = 2.5
D_LRMS = 5.0

STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEC', 'PYL', 'HOH', 'WAT'
}

# --- HELPER CLASS: Glycan ---
class Glycan:
    def __init__(self, coor, atom_names, BOND_CUTOFF=1.85):
        self.coor = np.array(coor)
        self.atom_names = atom_names
        self.BOND_CUTOFF = BOND_CUTOFF
        self.adj_mat = []
        self.edges = []
        self.ring_atom = []
        self.ring_atom_plus = []
        self.calc_adjacency()

        ope = []
        for jj in range(len(self.coor)):
            o = self.calc_ring(jj)
            ope.append(o)
            self.calc_adjacency()

        ring = []
        if len(ope) > 0:
            ring.append(ope[0])
            for jj in range(1, len(ope)):
                if isinstance(ope[jj], bool): continue
                skip = False
                for kk in range(len(ring)):
                    if len(ring[kk]) > 0 and len(ope[jj]) > 0 and ring[kk][0] == ope[jj][0]:
                        skip = True
                if skip: continue
                ring.append(ope[jj])

        valid_rings = [r for r in ring if not isinstance(r, bool) and len(r) > 0]
        self.ring_atom = valid_rings
        self.ring_atom_name, self.ring_com = self.get_ring_atom_name()

    def calc_adjacency(self):
        if len(self.coor) == 0:
            self.adj_mat = np.array([])
            self.edges = []
            return
        dm = distance_matrix(self.coor, self.coor)
        adj_mat = dm < self.BOND_CUTOFF
        np.fill_diagonal(adj_mat, 0)
        edge_list = []
        for ii in range(len(adj_mat)):
            edge_list.append([])
            for jj in range(len(adj_mat)):
                if adj_mat[ii, jj]: edge_list[ii].append(jj)
        self.adj_mat = adj_mat
        self.edges = edge_list

    def visit(self, n, edge_list, visited, st):
        if n == st and visited[st] == True: return [n]
        visited[n] = True
        arr = []
        if n >= len(edge_list): return False
        for e in edge_list[n]:
            try: edge_list[e].remove(n)
            except ValueError: continue
            r = self.visit(e, edge_list, visited, st)
            if isinstance(r, list):
                arr.append(n)
                for j in r: arr.append(j)
        if not arr: return False
        return arr

    def calc_ring(self, i):
        ring = self.visit(i, copy.deepcopy(self.edges), np.zeros(len(self.coor), dtype=bool), i)
        ind = 0
        while isinstance(ring, bool):
            ring = self.visit(ind, copy.deepcopy(self.edges), np.zeros(len(self.coor), dtype=bool), ind)
            ind += 1
            if ind >= len(self.coor): break
        if isinstance(ring, list): return np.unique(ring).astype(int)
        return False

    def get_ring_atom_name(self):
        r = []
        com = []
        for jj in self.ring_atom:
            r.append([])
            current_com = np.array([0., 0., 0.])
            for kk in jj:
                r[-1].append(self.atom_names[kk])
                current_com += self.coor[kk]
            if len(jj) > 0: current_com /= len(jj)
            com.append(current_com)
        return r, np.array(com)

# --- HELPER CLASS: LigandGraph ---
class LigandGraph:
    def __init__(self, atoms):
        self.atoms = atoms
        self.coords = np.array([a.coord for a in atoms])
        self.center = np.mean(self.coords, axis=0) if len(self.coords) > 0 else np.array([0,0,0])
        self.size = len(atoms)
        self.atom_names = [a.name for a in atoms]
        self.rdkit_mol = self._to_rdkit()
        self.glycan_obj = Glycan(self.coords, self.atom_names, BOND_CUTOFF=RING_BOND_CUTOFF)

    def _to_rdkit(self):
        mol = Chem.RWMol()
        conf = Chem.Conformer(len(self.atoms))
        for i, atom in enumerate(self.atoms):
            elem = atom.element.capitalize()
            if not elem or len(elem) > 2: elem = "C"
            rd_atom = Chem.Atom(elem)
            mol.AddAtom(rd_atom)
            conf.SetAtomPosition(i, Point3D(float(atom.coord[0]), float(atom.coord[1]), float(atom.coord[2])))
        
        dists = distance_matrix(self.coords, self.coords)
        rows, cols = np.where((dists < BOND_DIST) & (dists > 0))
        bonds = set()
        for r, c in zip(rows, cols):
            if r < c: bonds.add((int(r), int(c)))
        for r, c in bonds: mol.AddBond(r, c, Chem.BondType.SINGLE)
            
        mol_obj = mol.GetMol()
        mol_obj.AddConformer(conf)
        try:
            mol_obj.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol_obj) 
        except: pass
        return mol_obj

# --- CORE FUNCTIONS ---
def load_structure(path, name):
    """
    Loads a structure from PDB or CIF file.
    Robust to case-sensitivity in file extensions.
    """
    if path.suffix.lower() == '.cif':
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
        
    try: 
        return parser.get_structure(name, str(path))
    except Exception: 
        return None

def extract_components(structure):
    protein_atoms = [] 
    protein_ca = []
    het_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in STANDARD_AA:
                    for atom in residue:
                        if atom.element.upper() != 'H':
                            protein_atoms.append(atom)
                            if atom.name == 'CA': protein_ca.append(atom)
                else:
                    for atom in residue:
                        if atom.element.upper() != 'H': het_atoms.append(atom)
        break 
    ligands = []
    if het_atoms:
        coords = np.array([a.coord for a in het_atoms])
        dists = distance_matrix(coords, coords)
        G = nx.Graph()
        for i in range(len(het_atoms)): G.add_node(i)
        rows, cols = np.where((dists < BOND_DIST) & (dists > 0))
        for r, c in zip(rows, cols): G.add_edge(r, c)
        for component_indices in nx.connected_components(G):
            comp_atoms = [het_atoms[i] for i in component_indices]
            if len(comp_atoms) >= 3: ligands.append(LigandGraph(comp_atoms))
    return protein_ca, protein_atoms, ligands

# --- METRICS & ALIGNMENT ---

def get_residue_id(atom, residue_offset=0):
    p = atom.get_parent()
    original_id = p.id[1]
    return (p.get_parent().id, original_id + residue_offset, p.get_resname())

def calculate_residue_offset(ref_ca, pred_ca):
    ref_res = [(a.get_parent().id[1], a.get_parent().get_resname()) for a in ref_ca]
    pred_res = [(a.get_parent().id[1], a.get_parent().get_resname()) for a in pred_ca]
    ref_set = set(ref_res)
    best_offset = 0
    max_overlap = -1
    for offset in range(-2000, 2000):
        shifted_matches = 0
        for (r_num, r_name) in pred_res:
            if (r_num + offset, r_name) in ref_set:
                shifted_matches += 1
        if shifted_matches > max_overlap:
            max_overlap = shifted_matches
            best_offset = offset
        if max_overlap == len(pred_res):
            break
    return best_offset

def calculate_global_rmsd(ref_prot_atoms, pred_prot_atoms, residue_offset):
    """
    Global structure RMSD over all matched non-hydrogen PROTEIN atoms.

    Matching key: (chain_id, residue_number, residue_name, atom_name)
    - Reference uses residue_number as-is.
    - Prediction uses (residue_number + residue_offset) to account for indexing shifts.
    - Requires that pred coordinates are already in the same global frame as ref (i.e., after Superimposer.apply()).

    Returns:
        float RMSD in Å, or np.nan if insufficient matches.
    """
    # Build reference atom lookup
    ref_map = {}
    for a in ref_prot_atoms:
        if a.element.upper() == 'H':
            continue
        chain_id = a.get_parent().get_parent().id  # Chain ID
        res = a.get_parent()
        resnum = res.id[1]
        resname = res.get_resname()
        atom_name = a.name
        ref_map[(chain_id, resnum, resname, atom_name)] = np.array(a.coord, dtype=float)

    # Build matched coordinate lists
    ref_coords = []
    pred_coords = []
    for a in pred_prot_atoms:
        if a.element.upper() == 'H':
            continue
        chain_id = a.get_parent().get_parent().id
        res = a.get_parent()
        resnum = res.id[1] + residue_offset
        resname = res.get_resname()
        atom_name = a.name
        key = (chain_id, resnum, resname, atom_name)
        if key in ref_map:
            ref_coords.append(ref_map[key])
            pred_coords.append(np.array(a.coord, dtype=float))

    if len(ref_coords) < 3:
        return np.nan

    ref_coords = np.vstack(ref_coords)
    pred_coords = np.vstack(pred_coords)
    diffs = ref_coords - pred_coords
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))


def calculate_lddt(ref_coords, pred_coords, inclusion_radius=15.0, thresholds=[0.5, 1.0, 2.0, 4.0]):
    """
    Calculates LDDT score following AlphaFold's methodology.
    Inputs are N x 3 coordinate arrays.
    """
    if len(ref_coords) != len(pred_coords) or len(ref_coords) == 0:
        return 0.0

    # Distance matrices
    ref_dists = distance_matrix(ref_coords, ref_coords)
    pred_dists = distance_matrix(pred_coords, pred_coords)

    n_atoms = len(ref_coords)
    total_preserved = 0
    total_considered = 0

    # Iterate over atoms to check neighbors
    for i in range(n_atoms):
        # Find neighbors in Ref within inclusion radius (exclude self)
        # AlphaFold considers distances preserved if they are within thresholds
        neighbors = np.where((ref_dists[i] < inclusion_radius) & (ref_dists[i] > 0))[0]
        
        if len(neighbors) == 0:
            continue
            
        d_ref = ref_dists[i, neighbors]
        d_pred = pred_dists[i, neighbors]
        
        diffs = np.abs(d_ref - d_pred)
        
        # Calculate score for this atom
        atom_score = 0
        for th in thresholds:
            atom_score += np.sum(diffs < th)
        
        # Normalize by 4 (number of thresholds)
        total_preserved += (atom_score / 4.0)
        total_considered += len(neighbors)

    if total_considered == 0:
        return 0.0
        
    return total_preserved / total_considered

def calculate_fnat_full(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, pred_offset):
    d_ref = distance_matrix(ref_lig.coords, np.array([a.coord for a in ref_prot_atoms]))
    min_d_ref = np.min(d_ref, axis=0) 
    ref_contact_mask = min_d_ref < CONTACT_CUTOFF
    ref_contact_residues = set()
    for i, is_contact in enumerate(ref_contact_mask):
        if is_contact:
            ref_contact_residues.add(get_residue_id(ref_prot_atoms[i], 0))

    if not ref_contact_residues:
        return 0.0

    d_pred = distance_matrix(pred_lig.coords, np.array([a.coord for a in pred_prot_atoms]))
    min_d_pred = np.min(d_pred, axis=0)
    pred_contact_mask = min_d_pred < CONTACT_CUTOFF
    pred_contact_residues = set()
    for i, is_contact in enumerate(pred_contact_mask):
        if is_contact:
            pred_contact_residues.add(get_residue_id(pred_prot_atoms[i], pred_offset))

    tp = len(ref_contact_residues.intersection(pred_contact_residues))
    fn = len(ref_contact_residues)
    return tp / fn

def calculate_fnat_res(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, pred_offset):
    ref_rings = ref_lig.glycan_obj.ring_atom
    pred_rings = pred_lig.glycan_obj.ring_atom

    if not ref_rings or not pred_rings:
        return 0.0

    ref_ring_contacts = []
    prot_coords_ref = np.array([a.coord for a in ref_prot_atoms])
    for atom_indices in ref_rings:
        ring_coords = ref_lig.coords[atom_indices]
        d_mat = distance_matrix(ring_coords, prot_coords_ref)
        min_d = np.min(d_mat, axis=0)
        contact_indices = np.where(min_d < CONTACT_CUTOFF)[0]
        contacts = set()
        for p_idx in contact_indices:
            contacts.add(get_residue_id(ref_prot_atoms[p_idx], 0))
        ref_ring_contacts.append(list(contacts))

    pred_ring_contacts = []
    prot_coords_pred = np.array([a.coord for a in pred_prot_atoms])
    for atom_indices in pred_rings:
        ring_coords = pred_lig.coords[atom_indices]
        d_mat = distance_matrix(ring_coords, prot_coords_pred)
        min_d = np.min(d_mat, axis=0)
        contact_indices = np.where(min_d < CONTACT_CUTOFF)[0]
        contacts = set()
        for p_idx in contact_indices:
            contacts.add(get_residue_id(pred_prot_atoms[p_idx], pred_offset))
        pred_ring_contacts.append(list(contacts))

    f_matrix = np.zeros((len(ref_ring_contacts), len(pred_ring_contacts)))
    n_matrix = np.zeros((len(ref_ring_contacts), len(pred_ring_contacts))) 

    for r, r_contacts in enumerate(ref_ring_contacts):
        for p, p_contacts in enumerate(pred_ring_contacts):
            shared = 0
            for c in r_contacts:
                if c in p_contacts: shared += 1
            f_matrix[r, p] = shared
            n_matrix[r, p] = len(r_contacts)

    rolling_f = 0.0
    rolling_n = 0.0
    f_mtx = f_matrix.copy()
    n_mtx = n_matrix.copy()
    
    while True:
        max_val = np.max(f_mtx)
        if max_val < 1 and np.sum(n_mtx) == 0: break
        flat_idx = np.argmax(f_mtx)
        r, c = divmod(flat_idx, f_mtx.shape[1])
        rolling_f += f_mtx[r, c]
        rolling_n += n_mtx[r, c]
        f_mtx[r, :] = -1
        f_mtx[:, c] = -1
        n_mtx[r, :] = 0
        n_mtx[:, c] = 0
        if np.max(f_mtx) < 0: break

    while np.sum(n_mtx) > 0:
        flat_idx = np.argmax(n_mtx)
        r, c = divmod(flat_idx, n_mtx.shape[1])
        rolling_n += n_mtx[r, c]
        n_mtx[r, :] = 0
        n_mtx[:, c] = 0

    if rolling_n == 0: return 0.0
    return rolling_f / rolling_n

def calculate_rmsd_from_mols(ref_mol, pred_mol):
    try:
        mcs = rdFMCS.FindMCS([ref_mol, pred_mol], timeout=2, 
                             matchValences=False, ringMatchesRingOnly=True, 
                             completeRingsOnly=True)
        if not mcs.smartsString: return 100.0

        pattern = Chem.MolFromSmarts(mcs.smartsString)
        ref_match = ref_mol.GetSubstructMatch(pattern)
        pred_match = pred_mol.GetSubstructMatch(pattern)
        
        if not ref_match or not pred_match: return 100.0

        atom_map = list(zip(pred_match, ref_match))
        rms = Chem.rdMolAlign.CalcRMS(pred_mol, ref_mol, map=[atom_map])
        return rms
    except Exception:
        return 100.0

def calculate_lrms(ref_lig, pred_lig):
    """Global Ligand RMSD (Position dependent)."""
    return calculate_rmsd_from_mols(ref_lig.rdkit_mol, pred_lig.rdkit_mol)

def calculate_rirms(ref_lig, pred_lig):
    ref_coms = ref_lig.glycan_obj.ring_com
    pred_coms = pred_lig.glycan_obj.ring_com
    
    if len(ref_coms) == 0 or len(pred_coms) == 0: 
        return 10.0

    cost_mtx = np.zeros((len(ref_coms), len(pred_coms)))
    for r, rc in enumerate(ref_coms):
        for p, pc in enumerate(pred_coms):
            cost_mtx[r, p] = np.sum((rc - pc)**2)

    sq_diffs = []
    mtx = cost_mtx.copy()
    max_pairs = min(mtx.shape)
    
    for _ in range(max_pairs):
        min_val = np.min(mtx)
        if min_val == np.inf: break
        flat_idx = np.argmin(mtx)
        r, c = divmod(flat_idx, mtx.shape[1])
        sq_diffs.append(min_val)
        mtx[r, :] = np.inf
        mtx[:, c] = np.inf

    if not sq_diffs: return 10.0
    return np.sqrt(np.sum(sq_diffs) / len(sq_diffs))

def calculate_dockqc(fnat_res, fnat_full, lrms, rirms):
    fnat_avg = 0.5 * (fnat_res + fnat_full)
    lrms_scaled = 1.0 / (1.0 + (lrms / D_LRMS)**2)
    rrms_scaled = 1.0 / (1.0 + (rirms / D_RRMS)**2)
    return (fnat_avg + rrms_scaled + lrms_scaled) / 3.0

def get_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2: return np.mean(a), 0.0
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

# --- MAIN PROCESSING ---
def process_target(target_folder, pdb_dir, args):
    target_name = Path(target_folder).name
    
    # --- LOAD REFERENCE ---
    ref_files = sorted([f for f in Path(pdb_dir).iterdir() 
                        if f.is_file() and f.name.lower().startswith(target_name.lower()) 
                        and f.suffix.lower() in ['.cif', '.pdb']])
    if not ref_files: return []
    
    ref_struct = load_structure(ref_files[0], "REF")
    if not ref_struct: return []
    
    ref_ca, ref_prot_atoms, ref_ligands = extract_components(ref_struct)
    if not ref_ligands: return []

    # --- IDENTIFY PREDICTIONS ---
    if args.format == 'af3':
        pred_files = list(Path(target_folder).glob("seed-1_sample-*/model.cif"))
    elif args.format == 'boltz':
        pred_files = list(Path(target_folder).glob(f"*model_*.cif"))
    elif args.format == 'chai':
        pred_files = list(Path(target_folder).glob("pred.model_*.cif"))
    else:
        pred_files = []
        
    results = []
    
    for pred_path in pred_files:
        model_id = pred_path.stem
        pred_struct = load_structure(pred_path, "PRED")
        if not pred_struct: continue
        
        pred_ca, pred_prot_atoms, pred_ligands = extract_components(pred_struct)
        if not pred_ligands: continue
        
        # --- OFFSET CALCULATION ---
        residue_offset = calculate_residue_offset(ref_ca, pred_ca)
        
        # Identify matching CA atoms
        ref_ids = { (a.get_parent().id[1], a.get_parent().get_resname()): i for i, a in enumerate(ref_ca) }
        common_pairs = []
        for i, p_atom in enumerate(pred_ca):
            p_res = p_atom.get_parent()
            key = (p_res.id[1] + residue_offset, p_res.get_resname())
            if key in ref_ids:
                common_pairs.append((ref_ids[key], i))
        
        if len(common_pairs) < 3: continue

        ref_atoms_align = [ref_ca[i] for i, j in common_pairs]
        pred_atoms_align = [pred_ca[j] for i, j in common_pairs]

        # ==============================================================================
        # SCHEME 1: GLOBAL ALIGNMENT (Metrics 1 & 4)
        # ==============================================================================
        # We align the entire PRED structure to REF for global reporting.
        # This is the "Base State" for the prediction object.
        
        sup_global = Superimposer()
        sup_global.set_atoms(ref_atoms_align, pred_atoms_align)
        sup_global.apply(pred_struct.get_atoms()) # Modifies PRED coordinates in-place!

        # Metric 4: Global RMSD (calculated on the now globally-aligned structure)
        global_rmsd = calculate_global_rmsd(ref_prot_atoms, pred_prot_atoms, residue_offset)

        # Metric 1: Global LDDT (Invariant to alignment, but good to calculate here)
        ref_ca_coords = np.array([a.coord for a in ref_atoms_align])
        pred_ca_coords = np.array([a.coord for a in pred_atoms_align])
        global_lddt = calculate_lddt(ref_ca_coords, pred_ca_coords)

        # Update Pred Ligand Objects to reflect the Global Alignment
        for pl in pred_ligands:
            pl.coords = np.array([a.coord for a in pl.atoms])
            pl.center = np.mean(pl.coords, axis=0)
            pl.glycan_obj = Glycan(pl.coords, pl.atom_names, BOND_CUTOFF=RING_BOND_CUTOFF)
            pl.rdkit_mol = pl._to_rdkit()
            
        # --- LIGAND MATCHING (Using Global Alignment) ---
        # We match based on the global position, as per your description.
        cost_mtx = np.full((len(ref_ligands), len(pred_ligands)), np.inf)
        for r, ref in enumerate(ref_ligands):
            for p, pred in enumerate(pred_ligands):
                # Secondary Filter: Exact atom count match (per your request)
                if ref.size == pred.size: 
                    dist = np.linalg.norm(ref.center - pred.center)
                    cost_mtx[r, p] = dist
                else:
                    cost_mtx[r, p] = np.inf

        matches = []
        curr_mtx = cost_mtx.copy()
        for _ in range(min(curr_mtx.shape)):
            min_val = np.min(curr_mtx)
            if min_val == np.inf: break
            flat_idx = np.argmin(curr_mtx)
            r, p = divmod(flat_idx, curr_mtx.shape[1])
            matches.append((r, p))
            curr_mtx[r, :] = np.inf
            curr_mtx[:, p] = np.inf
        
        # --- PER-LIGAND METRICS ---
        row_dockqc = []
        row_global_lrms = []
        row_internal_lrms = []
        row_ligand_lddt = []
        
        for (r_idx, p_idx) in matches:
            ref_lig = ref_ligands[r_idx]
            pred_lig = pred_ligands[p_idx]
            
            # ==============================================================================
            # SCHEME 2: INTERNAL ALIGNMENT (Metrics 2 & 5)
            # ==============================================================================
            # We align the ligand to itself (SVD) to check internal conformation quality.
            
            # Metric 5: Internal Ligand RMSD
            # Chem.rdMolAlign.GetBestRMS performs an optimal alignment (SVD) internally.
            # This isolates the ligand shape from its protein pocket.
            internal_lrms = 100.0
            try:
                # Same logic as before, GetBestRMS handles the alignment automatically
                ref_mol = ref_lig.rdkit_mol
                pred_mol = pred_lig.rdkit_mol
                mcs = rdFMCS.FindMCS([ref_mol, pred_mol], timeout=2, 
                                     matchValences=False, ringMatchesRingOnly=True, 
                                     completeRingsOnly=True)
                if mcs.smartsString:
                    pattern = Chem.MolFromSmarts(mcs.smartsString)
                    ref_match = ref_mol.GetSubstructMatch(pattern)
                    pred_match = pred_mol.GetSubstructMatch(pattern)
                    if ref_match and pred_match:
                        atom_map = list(zip(pred_match, ref_match))
                        # GetBestRMS = "Align then calculate RMSD"
                        internal_lrms = Chem.rdMolAlign.GetBestRMS(pred_mol, ref_mol, map=[atom_map])
            except: pass

            # Metric 2: Ligand-aligned LDDT
            # LDDT is based on internal distance matrices, so it is invariant to rigid body rotation.
            # It effectively assesses the "Internal Aligned" quality naturally.
            lig_lddt = calculate_lddt(ref_lig.coords, pred_lig.coords)

            # ==============================================================================
            # SCHEME 3: POCKET ALIGNMENT (Metrics 3 - DockQC)
            # ==============================================================================
            # We align the PREDICTED protein POCKET onto the REFERENCE protein POCKET.
            # We then apply this transform to the predicted ligand to see if it sits correctly in the pocket.
            
            # 1. Define the Pocket: Ref CA atoms within 10A of Ref Ligand
            ref_lig_all_coords = ref_lig.coords
            ref_ca_coords_arr = np.array([a.coord for a in ref_atoms_align]) # Uses aligned set
            
            # Distance from every Ref CA to every Ref Ligand Atom
            d_mat = distance_matrix(ref_ca_coords_arr, ref_lig_all_coords)
            min_d_per_res = np.min(d_mat, axis=1)
            pocket_mask = min_d_per_res < 10.0
            
            # 2. Extract Pocket Atoms for Alignment
            ref_pocket_atoms = []
            pred_pocket_atoms = []
            
            for k, is_in_pocket in enumerate(pocket_mask):
                if is_in_pocket:
                    ref_pocket_atoms.append(ref_atoms_align[k])
                    pred_pocket_atoms.append(pred_atoms_align[k])
            
            # 3. Create Pocket-Aligned Ligand Coordinates
            # Note: pred_lig is currently "Globally Aligned". We calculate the delta transform
            # to move it to "Pocket Aligned".
            
            if len(ref_pocket_atoms) > 3:
                sup_pocket = Superimposer()
                sup_pocket.set_atoms(ref_pocket_atoms, pred_pocket_atoms)
                rot, tran = sup_pocket.rotran
                
                # Apply transform: x_new = dot(x, rot) + tran
                pred_pocket_coords = np.dot(pred_lig.coords, rot) + tran
                
                # Create a temporary object for DockQC calculation
                pred_lig_pocket = copy.copy(pred_lig)
                pred_lig_pocket.coords = pred_pocket_coords
                # Re-generate helper objects with new coords
                pred_lig_pocket.glycan_obj = Glycan(pred_lig_pocket.coords, pred_lig.atom_names, BOND_CUTOFF=RING_BOND_CUTOFF)
                pred_lig_pocket.rdkit_mol = pred_lig_pocket._to_rdkit()
                
                # Calculate DockQC metrics using the Pocket-Aligned Pose
                lrms = calculate_lrms(ref_lig, pred_lig_pocket)
                rirms = calculate_rirms(ref_lig, pred_lig_pocket)
                
                # Fnat is contact-based (internal distance), so it is robust to alignment method.
                # However, it relies on protein atoms. Since we aren't moving the protein atoms 
                # (we only moved the ligand virtually), we must pass the "offset" correctly.
                # The user's fnat functions take (ref_lig, pred_lig, ref_prot, pred_prot).
                # To be chemically strictly correct for Fnat, we should use the GLOBAL alignment
                # because Fnat measures "Are these atoms touching?". 
                # Since we aligned the protein globally, the contacts are preserved.
                fnat_full = calculate_fnat_full(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, residue_offset)
                fnat_res = calculate_fnat_res(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, residue_offset)
                
            else:
                # Fallback to global if pocket is undefined
                lrms = calculate_lrms(ref_lig, pred_lig)
                rirms = calculate_rirms(ref_lig, pred_lig)
                fnat_full = calculate_fnat_full(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, residue_offset)
                fnat_res = calculate_fnat_res(ref_lig, pred_lig, ref_prot_atoms, pred_prot_atoms, residue_offset)

            # 4. Final DockQC Score
            dqc = calculate_dockqc(fnat_res, fnat_full, lrms, rirms)

            row_dockqc.append(dqc)
            row_global_lrms.append(lrms) # This is now technically "Pocket LRMS", which is what DockQC wants
            row_internal_lrms.append(internal_lrms)
            row_ligand_lddt.append(lig_lddt)

        if not row_dockqc: continue

        results.append({
            'Target': target_name,
            'Model': model_id,
            'LDDT_Global': global_lddt,
            'LDDT_Ligand': row_ligand_lddt,
            'DockQC': row_dockqc,
            'Global_Ligand_RMSD': row_global_lrms,
            'Internal_Ligand_RMSD': row_internal_lrms,
            'Global_RMSD': global_rmsd
        })
            
    return results
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("pdb", type=str)
    parser.add_argument("out", type=str)
    parser.add_argument("--format", type=str, default="af3", choices=["af3", "boltz", "chai"])
    parser.add_argument("--workers", type=int, default=cpu_count())
    args = parser.parse_args()
    
    targets = [f for f in Path(args.data).iterdir() if f.is_dir()]
    print(f"Processing {len(targets)} targets with {args.workers} workers...")
    
    func = partial(process_target, pdb_dir=args.pdb, args=args)
    
    all_rows = []
    with Pool(args.workers) as p:
        with tqdm(total=len(targets)) as pbar:
            for res in p.imap_unordered(func, targets):
                all_rows.extend(res)
                pbar.update()
                
    if not all_rows:
        print("No results found.")
        sys.exit(0)
    
    # --- Write to CSV ---
    print(f"Writing detailed results to {args.out}...")
    with open(args.out, 'w', newline='') as csvfile:
        fieldnames = ['Target', 'Model', 'LDDT_Global', 'LDDT_Ligand', 'DockQC', 'Global_Ligand_RMSD', 'Internal_Ligand_RMSD']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in all_rows:
            num_ligands = len(row['DockQC'])
            for i in range(num_ligands):
                writer.writerow({
                    'Target': row['Target'],
                    'Model': row['Model'],
                    'LDDT_Global': row['LDDT_Global'], 
                    'LDDT_Ligand': row['LDDT_Ligand'][i],
                    'DockQC': row['DockQC'][i],
                    'Global_Ligand_RMSD': row['Global_Ligand_RMSD'][i],
                    'Internal_Ligand_RMSD': row['Internal_Ligand_RMSD'][i]
                })

    # --- Summary Report ---
    all_global_lddt = [r['LDDT_Global'] for r in all_rows]
    all_ligand_lddt = []
    all_dockqc = []
    all_global_ligand_rmsd = []
    all_internal_rmsd = []

    # NEW: collect true global structure RMSD per model
    all_global_rmsd = []

    for row in all_rows:
        all_ligand_lddt.extend(row['LDDT_Ligand'])
        all_dockqc.extend(row['DockQC'])
        all_global_ligand_rmsd.extend(row['Global_Ligand_RMSD'])
        all_internal_rmsd.extend(row['Internal_Ligand_RMSD'])

        # Global_RMSD is one value per model row; skip NaNs safely
        gr = row.get('Global_RMSD', np.nan)
        if gr is not None and not (isinstance(gr, float) and np.isnan(gr)):
            all_global_rmsd.append(gr)

    # Convert to arrays
    all_global_lddt = np.array(all_global_lddt)
    all_ligand_lddt = np.array(all_ligand_lddt)
    all_dockqc = np.array(all_dockqc)
    all_global_ligand_rmsd = np.array(all_global_ligand_rmsd)
    all_internal_rmsd = np.array(all_internal_rmsd)
    all_global_rmsd = np.array(all_global_rmsd) if len(all_global_rmsd) > 0 else np.array([])

    def print_metric_stat(name, mean, ci):
        print(f"{name:<45} {mean:.3f} +/- {ci:.3f}")

    print("\n--- METRICS REPORT ---")
    
    # 1. Global LDDT (Mean)
    m, h = get_ci(all_global_lddt)
    print_metric_stat("1. Mean Global LDDT:", m, h)

    # 2. Ligand-aligned LDDT (Mean)
    m, h = get_ci(all_ligand_lddt)
    print_metric_stat("2. Mean Ligand-aligned LDDT:", m, h)

    # 3. DockQC
    binary_dqc_low = (all_dockqc > 0.25)
    m, h = get_ci(binary_dqc_low)
    print_metric_stat("3a. Ratio DockQC > 0.25:", m, h)

    binary_dqc_low = (all_dockqc > 0.5)
    m, h = get_ci(binary_dqc_low)
    print_metric_stat("3b. Ratio DockQC > 0.5:", m, h)

    binary_dqc_low = (all_dockqc > 0.8)
    m, h = get_ci(binary_dqc_low)
    print_metric_stat("3c. Ratio DockQC > 0.8:", m, h)

    # 4. TRUE Global RMSD of the structure (protein heavy atoms), threshold > 2.0 Å
    binary_global_rmsd = (all_global_rmsd < 2.0)
    m, h = get_ci(binary_global_rmsd)
    print_metric_stat("4. Ratio Global RMSD < 2.0 Å:", m, h)

    # 5. Ratio Internal Ligand RMSD < 1.0 (Note threshold 1.0)
    binary_internal = (all_internal_rmsd < 1.0)
    m, h = get_ci(binary_internal)
    print_metric_stat("5. Ratio Internal Ligand RMSD < 1.0 Å:", m, h)

if __name__ == "__main__":
    main()
