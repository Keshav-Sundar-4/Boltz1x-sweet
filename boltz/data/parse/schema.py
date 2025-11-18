from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List, Mapping, Dict
import sys

import click
import re
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondStereo, Conformer, Mol
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from boltz.data import const
from boltz.data.types import (
    Atom,
    Bond,
    Chain,
    ChainInfo,
    ChiralAtomConstraint,
    Connection,
    InferenceOptions,
    Interface,
    PlanarBondConstraint,
    PlanarRing5Constraint,
    PlanarRing6Constraint,
    RDKitBoundsConstraint,
    Record,
    Residue,
    ResidueConstraints,
    StereoBondConstraint,
    Structure,
    GlycosylationSite,
    StructureInfo,
    Target,
)

####################################################################################################
# DATACLASSES
####################################################################################################

@dataclass
class MonosaccharideFeatures:
    """Stores detailed features for a specific monosaccharide instance (chain)."""
    asym_id: int
    ccd_code: str
    source_glycan_idx: int
    anomeric_config: Optional[str] = None  # 'a' or 'b' from the donating linkage spec

# Add a type hint for the global monosaccharide map for clarity
MonosaccharideFeatureMapType = Dict[Tuple[int, int], MonosaccharideFeatures]

@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True)
class ParsedRDKitBoundsConstraint:
    """A parsed RDKit bounds constraint object."""

    atom_idxs: tuple[int, int]
    is_bond: bool
    is_angle: bool
    upper_bound: float
    lower_bound: float


@dataclass(frozen=True)
class ParsedChiralAtomConstraint:
    """A parsed chiral atom constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_reference: bool
    is_r: bool


@dataclass(frozen=True)
class ParsedStereoBondConstraint:
    """A parsed stereo bond constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_check: bool
    is_e: bool


@dataclass(frozen=True)
class ParsedPlanarBondConstraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing5Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing6Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
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
    rdkit_bounds_constraints: Optional[list[ParsedRDKitBoundsConstraint]] = None
    chiral_atom_constraints: Optional[list[ParsedChiralAtomConstraint]] = None
    stereo_bond_constraints: Optional[list[ParsedStereoBondConstraint]] = None
    planar_bond_constraints: Optional[list[ParsedPlanarBondConstraint]] = None
    planar_ring_5_constraints: Optional[list[ParsedPlanarRing5Constraint]] = None
    planar_ring_6_constraints: Optional[list[ParsedPlanarRing6Constraint]] = None


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""

    entity: str
    type: str
    residues: list[ParsedResidue]
    cyclic_period: int


####################################################################################################
# HELPERS
####################################################################################################


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    Tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    # Try using the computed conformer
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to the ideal coordinates
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    msg = "Conformer does not exist."
    raise ValueError(msg)


def compute_geometry_constraints(mol: Mol, idx_map):
    if mol.GetNumAtoms() <= 1:
        return []

    bounds = GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bonds = set(
        tuple(sorted(b)) for b in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    )
    angles = set(
        tuple(sorted([a[0], a[2]]))
        for a in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    )

    constraints = []
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1)):
        if i in idx_map and j in idx_map:
            constraint = ParsedRDKitBoundsConstraint(
                atom_idxs=(idx_map[i], idx_map[j]),
                is_bond=tuple(sorted([i, j])) in bonds,
                is_angle=tuple(sorted([i, j])) in angles,
                upper_bound=bounds[i, j],
                lower_bound=bounds[j, i],
            )
            constraints.append(constraint)
    return constraints


def compute_chiral_atom_constraints(mol, idx_map):
    constraints = []
    if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
        for center_idx, orientation in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False
        ):
            center = mol.GetAtomWithIdx(center_idx)
            neighbors = [
                (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                for neighbor in center.GetNeighbors()
            ]
            neighbors = sorted(
                neighbors, key=lambda neighbor: neighbor[1], reverse=True
            )
            neighbors = tuple(neighbor[0] for neighbor in neighbors)
            is_r = orientation == "R"

            if len(neighbors) > 4:
                continue

            atom_idxs = (*neighbors[:3], center_idx)
            if all(i in idx_map for i in atom_idxs):
                constraints.append(
                    ParsedChiralAtomConstraint(
                        atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                        is_reference=True,
                        is_r=is_r,
                    )
                )

            if len(neighbors) == 4:
                for skip_idx in range(3):
                    chiral_set = neighbors[:skip_idx] + neighbors[skip_idx + 1 :]
                    if skip_idx % 2 == 0:
                        atom_idxs = chiral_set[::-1] + (center_idx,)
                    else:
                        atom_idxs = chiral_set + (center_idx,)
                    if all(i in idx_map for i in atom_idxs):
                        constraints.append(
                            ParsedChiralAtomConstraint(
                                atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                                is_reference=False,
                                is_r=is_r,
                            )
                        )
    return constraints


def compute_stereo_bond_constraints(mol, idx_map):
    constraints = []
    if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
        for bond in mol.GetBonds():
            stereo = bond.GetStereo()
            if stereo in {BondStereo.STEREOE, BondStereo.STEREOZ}:
                start_atom_idx, end_atom_idx = (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                )
                start_neighbors = [
                    (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                    for neighbor in mol.GetAtomWithIdx(start_atom_idx).GetNeighbors()
                    if neighbor.GetIdx() != end_atom_idx
                ]
                start_neighbors = sorted(
                    start_neighbors, key=lambda neighbor: neighbor[1], reverse=True
                )
                start_neighbors = [neighbor[0] for neighbor in start_neighbors]
                end_neighbors = [
                    (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                    for neighbor in mol.GetAtomWithIdx(end_atom_idx).GetNeighbors()
                    if neighbor.GetIdx() != start_atom_idx
                ]
                end_neighbors = sorted(
                    end_neighbors, key=lambda neighbor: neighbor[1], reverse=True
                )
                end_neighbors = [neighbor[0] for neighbor in end_neighbors]
                is_e = stereo == BondStereo.STEREOE

                atom_idxs = (
                    start_neighbors[0],
                    start_atom_idx,
                    end_atom_idx,
                    end_neighbors[0],
                )
                if all(i in idx_map for i in atom_idxs):
                    constraints.append(
                        ParsedStereoBondConstraint(
                            atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                            is_check=True,
                            is_e=is_e,
                        )
                    )

                if len(start_neighbors) == 2 and len(end_neighbors) == 2:
                    atom_idxs = (
                        start_neighbors[1],
                        start_atom_idx,
                        end_atom_idx,
                        end_neighbors[1],
                    )
                    if all(i in idx_map for i in atom_idxs):
                        constraints.append(
                            ParsedStereoBondConstraint(
                                atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                                is_check=False,
                                is_e=is_e,
                            )
                        )
    return constraints


def compute_flatness_constraints(mol, idx_map):
    planar_double_bond_smarts = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    aromatic_ring_5_smarts = Chem.MolFromSmarts("[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1")
    aromatic_ring_6_smarts = Chem.MolFromSmarts(
        "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1"
    )

    planar_double_bond_constraints = []
    aromatic_ring_5_constraints = []
    aromatic_ring_6_constraints = []
    for match in mol.GetSubstructMatches(planar_double_bond_smarts):
        if all(i in idx_map for i in match):
            planar_double_bond_constraints.append(
                ParsedPlanarBondConstraint(atom_idxs=tuple(idx_map[i] for i in match))
            )
    for match in mol.GetSubstructMatches(aromatic_ring_5_smarts):
        if all(i in idx_map for i in match):
            aromatic_ring_5_constraints.append(
                ParsedPlanarRing5Constraint(atom_idxs=tuple(idx_map[i] for i in match))
            )
    for match in mol.GetSubstructMatches(aromatic_ring_6_smarts):
        if all(i in idx_map for i in match):
            aromatic_ring_6_constraints.append(
                ParsedPlanarRing6Constraint(atom_idxs=tuple(idx_map[i] for i in match))
            )

    return (
        planar_double_bond_constraints,
        aromatic_ring_5_constraints,
        aromatic_ring_6_constraints,
    )


####################################################################################################
# PARSING
####################################################################################################


def parse_ccd_residue(
    name: str,
    ref_mol: Mol,
    res_idx: int,
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Remove hydrogens
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
    Chem.AssignStereochemistry(ref_mol, cleanIt=True, force=True)

    # Check if this is a single atom CCD residue
    if ref_mol.GetNumAtoms() == 1:
        pos = (0, 0, 0)
        ref_atom = ref_mol.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            str(ref_atom.GetChiralTag()), unk_chirality
        )
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            element=ref_atom.GetAtomicNum(),
            charge=ref_atom.GetFormalCharge(),
            coords=pos,
            conformer=(0, 0, 0),
            is_present=True,
            chirality=chirality_type,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            orig_idx=None,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
        )
        return residue

    # Get reference conformer coordinates
    conformer = get_conformer(ref_mol)

    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")
        charge = atom.GetFormalCharge()
        element = atom.GetAtomicNum()
        ref_coords = conformer.GetAtomPosition(atom.GetIdx())
        ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
        chirality_type = const.chirality_type_ids.get(
            str(atom.GetChiralTag()), unk_chirality
        )

        # Get PDB coordinates, if any
        coords = (0, 0, 0)
        atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=atom_is_present,
                chirality=chirality_type,
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1  # noqa: SIM113

    # Load bonds
    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        # Skip bonds with atoms ignored
        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        idx_1 = idx_map[idx_1]
        idx_2 = idx_map[idx_2]
        start = min(idx_1, idx_2)
        end = max(idx_1, idx_2)
        bond_type = bond.GetBondType().name
        bond_type = const.bond_type_ids.get(bond_type, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    rdkit_bounds_constraints = compute_geometry_constraints(ref_mol, idx_map)
    chiral_atom_constraints = compute_chiral_atom_constraints(ref_mol, idx_map)
    stereo_bond_constraints = compute_stereo_bond_constraints(ref_mol, idx_map)
    planar_bond_constraints, planar_ring_5_constraints, planar_ring_6_constraints = (
        compute_flatness_constraints(ref_mol, idx_map)
    )

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        orig_idx=None,
        is_standard=False,
        is_present=True,
        rdkit_bounds_constraints=rdkit_bounds_constraints,
        chiral_atom_constraints=chiral_atom_constraints,
        stereo_bond_constraints=stereo_bond_constraints,
        planar_bond_constraints=planar_bond_constraints,
        planar_ring_5_constraints=planar_ring_5_constraints,
        planar_ring_6_constraints=planar_ring_6_constraints,
    )


def parse_polymer(
    sequence: list[str],
    entity: str,
    chain_type: str,
    components: dict[str, Mol],
    cyclic: bool,
    glycosylated_residue_indices: set = frozenset(),
) -> Optional[ParsedChain]:
    """(FINAL CORRECTED VERSION) Process a sequence into a chain object."""
    ref_res = set(const.tokens)
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    parsed = []
    for res_idx, res_name in enumerate(sequence):
        res_corrected = res_name if res_name != "MSE" else "MET"

        # If it's a standard AA that's glycosylated, use our new special parser
        if res_corrected in ref_res and res_idx in glycosylated_residue_indices:
            if res_corrected not in components:
                raise ValueError(f"Component definition for '{res_corrected}' not found.")
            
            # Call the new function that generates bonds but respects the curated atom list
            residue = parse_standard_residue_with_bonds(
                name=res_corrected,
                ref_mol=components[res_corrected],
                res_idx=res_idx,
            )
            if residue is not None:
                # Flip the flag for the tokenizer
                residue = replace(residue, is_standard=False)
            parsed.append(residue)
            continue
        
        # If it's a true non-standard residue (ligand in sequence), use the generic CCD parser
        elif res_corrected not in ref_res:
            if res_corrected not in components:
                raise ValueError(f"Component definition for '{res_corrected}' not found.")
            
            residue = parse_ccd_residue(
                name=res_corrected,
                ref_mol=components[res_corrected],
                res_idx=res_idx,
            )
            parsed.append(residue)
            continue

        # This is the original path for standard, non-glycosylated residues (no bonds needed).
        ref_mol = components[res_corrected]
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
        ref_conformer = get_conformer(ref_mol)
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_corrected]]

        atoms: list[ParsedAtom] = []
        for ref_atom in ref_atoms:
            atom_name = ref_atom.GetProp("name")
            idx = ref_atom.GetIdx()
            ref_coords = ref_conformer.GetAtomPosition(idx)
            ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    element=ref_atom.GetAtomicNum(),
                    charge=ref_atom.GetFormalCharge(),
                    coords=(0, 0, 0),
                    conformer=ref_coords,
                    is_present=True,
                    chirality=const.chirality_type_ids.get(str(ref_atom.GetChiralTag()), unk_chirality),
                )
            )

        parsed.append(
            ParsedResidue(
                name=res_corrected,
                type=const.token_ids[res_corrected],
                atoms=atoms,
                bonds=[],
                idx=res_idx,
                atom_center=const.res_to_center_atom_id[res_corrected],
                atom_disto=const.res_to_disto_atom_id[res_corrected],
                is_standard=True,
                is_present=True,
                orig_idx=None,
            )
        )

    cyclic_period = len(sequence) if cyclic else 0

    return ParsedChain(
        entity=entity,
        residues=parsed,
        type=chain_type,
        cyclic_period=cyclic_period,
    )


def parse_boltz_schema(  # noqa: C901, PLR915, PLR912
    name: str,
    schema: dict,
    ccd: Mapping[str, Mol],
) -> Target:
    """
    (Corrected Version) Parses a Boltz input yaml / json.
    This version is modified to correctly handle glycosylated standard amino acids
    by pre-parsing glycosylation sites and passing this information to the
    polymer parser, ensuring correct bond generation. It also correctly separates
    inter-chain bonds (like glycosylation sites) into the 'connections' array.
    """
    version = schema.get("version", 1)
    if version != 1:
        msg = f"Invalid version {version} in input!"
        raise ValueError(msg)

    blocker = rdBase.BlockLogs()

    # --- FIX: Step 1 - Pre-parse glycosylation sites for easy lookup ---
    glycosylation_sites_lookup = set()
    if "glycosylation" in schema:
        for site in schema.get("glycosylation", []):
            try:
                prot_spec = site["site"]["protein"]
                p_chain, p_res_1based = prot_spec[0], prot_spec[1]
                glycosylation_sites_lookup.add((p_chain, p_res_1based - 1))  # Store as 0-based index
            except (IndexError, KeyError) as e:
                raise ValueError(f"Malformed protein site in glycosylation block: {site}. Details: {e}") from e

    # --- GLYCAN-SPECIFIC DATA STRUCTURES (Unchanged) ---
    local_glycan_feature_map: MonosaccharideFeatureMapType = {}
    local_atom_to_mono_idx_maps: Dict[int, np.ndarray] = {}
    all_glycan_prelim_features: Dict[str, dict] = {}
    all_glycan_atom_maps: Dict[str, np.ndarray] = {}
    glycan_source_indices: Dict[str, int] = {}

    # --- Grouping and initial setup (Unchanged) ---
    items_to_group: Dict[Tuple[str, str], List[Tuple[int, dict]]] = {}
    for item_idx, item in enumerate(schema["sequences"]):
        entity_type = next(iter(item.keys())).lower()
        if entity_type not in {"protein", "dna", "rna", "ligand", "glycan"}:
            msg = f"Invalid entity type: {entity_type} in item {item_idx}"
            raise ValueError(msg)
        data = item[entity_type]
        definition = ""
        if entity_type in {"protein", "dna", "rna"}:
            definition = str(data["sequence"])
        elif entity_type == "ligand":
            definition = f"smiles:{data['smiles']}" if "smiles" in data else f"ccd:{data['ccd']}"
        elif entity_type == "glycan":
            definition = f"iupac:{data['iupac']}:idx:{item_idx}"
        items_to_group.setdefault((entity_type, definition), []).append((item_idx, item))

    # --- Main Parsing Loop ---
    chains: dict[str, ParsedChain] = {}
    chain_to_msa: dict[str, str] = {}
    entity_to_seq: dict[str, str] = {}
    definition_to_entity_id: Dict[str, int] = {}
    entity_counter = 0
    is_msa_custom = False
    is_msa_auto = False

    for (entity_type, definition), item_tuples in items_to_group.items():
        items = [item_dict for _, item_dict in item_tuples]
        chem_definition = definition
        if entity_type == 'glycan':
            chem_definition = definition.split(':idx:')[0]
        elif entity_type == 'ligand' and chem_definition.startswith('ccd:'):
            chem_definition = f"ccd:{str(items[0][entity_type]['ccd'])}"

        current_entity_id = definition_to_entity_id.setdefault(chem_definition, entity_counter)
        if current_entity_id == entity_counter:
            entity_to_seq[str(entity_counter)] = chem_definition
            entity_counter += 1

        # --- FIX: Modified Protein Parsing Logic ---
        if entity_type == "protein":
            seq = items[0][entity_type]["sequence"]
            msa_str = items[0][entity_type].get("msa", 0)
            msa_value = -1 if msa_str == "empty" else (msa_str if msa_str not in (0, None, "") else 0)
            if msa_value != 0 and msa_value != -1: is_msa_custom = True
            elif msa_value == 0: is_msa_auto = True

            token_map = const.prot_letter_to_token
            unk_token = const.unk_token["PROTEIN"]
            seq_tokens = [token_map.get(c, unk_token) for c in list(seq)]
            for mod in items[0][entity_type].get("modifications", []):
                idx = mod["position"] - 1
                if 0 <= idx < len(seq_tokens): seq_tokens[idx] = mod["ccd"]
                else: raise ValueError(f"Modification position {mod['position']} out of bounds.")

            chain_type_id = const.chain_type_ids["PROTEIN"]
            cyclic = items[0][entity_type].get("cyclic", False)

            # Loop through each instance (chain) of this protein entity
            for _, item in item_tuples:
                chain_ids = item[entity_type]["id"]
                if isinstance(chain_ids, str): chain_ids = [chain_ids]
                for chain_name in chain_ids:
                    if chain_name in chains:
                        raise ValueError(f"Duplicate chain name '{chain_name}' in input.")

                    # For THIS specific chain, find which of its residues are glycosylated
                    glycosylated_indices_for_this_chain = {
                        res_idx for c_name, res_idx in glycosylation_sites_lookup if c_name == chain_name
                    }

                    # Parse the chain with its specific glycosylation info
                    parsed_chain = parse_polymer(
                        sequence=seq_tokens,
                        entity=str(current_entity_id),
                        chain_type=chain_type_id,
                        components=ccd,
                        cyclic=cyclic,
                        glycosylated_residue_indices=glycosylated_indices_for_this_chain
                    )
                    if parsed_chain is None:
                        raise ValueError(f"Failed to parse polymer for chain '{chain_name}'")

                    chains[chain_name] = parsed_chain
                    chain_to_msa[chain_name] = msa_value
            continue # Skip to the next entity group

        # --- Unchanged logic for other entity types ---
        parsed_chain_template: Optional[ParsedChain] = None
        if entity_type in {"dna", "rna"}:
            seq = items[0][entity_type]["sequence"]
            token_map = const.rna_letter_to_token if entity_type == "rna" else const.dna_letter_to_token
            unk_token = const.unk_token[entity_type.upper()]
            seq_tokens = [token_map.get(c, unk_token) for c in list(seq)]
            chain_type_id = const.chain_type_ids[entity_type.upper()]
            cyclic = items[0][entity_type].get("cyclic", False)
            parsed_chain_template = parse_polymer(
                sequence=seq_tokens, entity=str(current_entity_id),
                chain_type=chain_type_id, components=ccd, cyclic=cyclic
            )
        elif entity_type == "ligand":
            ligand_residues = []
            if chem_definition.startswith("ccd:"):
                ccd_codes = items[0][entity_type]["ccd"]
                if isinstance(ccd_codes, str): ccd_codes = [ccd_codes]
                for res_idx, code in enumerate(ccd_codes):
                    residue = parse_ccd_residue(name=code, ref_mol=ccd[code], res_idx=res_idx)
                    ligand_residues.append(residue)
            elif chem_definition.startswith("smiles:"):
                smiles = chem_definition.split(":", 1)[1]
                mol = AllChem.MolFromSmiles(smiles)
                mol = AllChem.AddHs(mol)
                canonical_order = list(AllChem.CanonicalRankAtoms(mol))
                for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
                    atom.SetProp("name", atom.GetSymbol().upper() + str(can_idx + 1))
                compute_3d_conformer(mol)
                mol_no_h = AllChem.RemoveHs(mol, sanitize=False)
                Chem.AssignStereochemistry(mol_no_h, cleanIt=True, force=True)
                residue = parse_ccd_residue(name="LIG", ref_mol=mol_no_h, res_idx=0)
                ligand_residues.append(residue)
            parsed_chain_template = ParsedChain(entity=str(current_entity_id), residues=ligand_residues, type=const.chain_type_ids["NONPOLYMER"], cyclic_period=0)
        elif entity_type == "glycan":
            source_glycan_idx, item = item_tuples[0]
            glycan_data = item[entity_type]
            iupac = glycan_data["iupac"]
            chain_name = str(glycan_data.get("id", f"G{source_glycan_idx + 1}"))
            if chain_name in chains: raise ValueError(f"Duplicate chain ID '{chain_name}'.")
            parsed_residue, prelim_features, atom_map = parse_glycan(iupac, ccd)
            if parsed_residue:
                chains[chain_name] = ParsedChain(entity=str(current_entity_id), type=const.chain_type_ids["NONPOLYMER"], residues=[parsed_residue], cyclic_period=0)
                chain_to_msa[chain_name] = 0
                all_glycan_prelim_features[chain_name] = prelim_features
                all_glycan_atom_maps[chain_name] = atom_map
                glycan_source_indices[chain_name] = source_glycan_idx
            continue

        if parsed_chain_template is not None:
            for _, item in item_tuples:
                ids = item[entity_type]["id"]
                if isinstance(ids, str): ids = [ids]
                for chain_name in ids:
                    if chain_name in chains: raise ValueError(f"Duplicate chain name '{chain_name}'.")
                    chains[chain_name] = parsed_chain_template
                    chain_to_msa[chain_name] = 0
    
    # --- Step 3: Build final flat data arrays from ParsedChain objects ---
    atom_data, res_data, chain_data, bond_data = [], [], [], []
    rdkit_bounds_constraint_data, chiral_atom_constraint_data, stereo_bond_constraint_data = [], [], []
    planar_bond_constraint_data, planar_ring_5_constraint_data, planar_ring_6_constraint_data = [], [], []

    atom_idx, res_idx = 0, 0
    connections_list = [] # <<< FIX: INITIALIZE CONNECTIONS LIST
    sym_count = {}
    chain_to_idx: Dict[str, int] = {}
    atom_idx_map: Dict[Tuple[str, int, str], int] = {}
    res_map: Dict[Tuple[str, int], int] = {} # Map (chain_name, local_res_idx) -> global_res_idx

    for asym_id, (chain_name, chain) in enumerate(sorted(chains.items())):
        res_num = len(chain.residues)
        atom_num = sum(len(res.atoms) for res in chain.residues)
        entity_id = int(chain.entity)
        sym_id = sym_count.get(entity_id, 0)
        
        chain_data.append(
            (chain_name, chain.type, entity_id, sym_id, asym_id, atom_idx,
             atom_num, res_idx, res_num, chain.cyclic_period)
        )
        chain_to_idx[chain_name] = asym_id
        sym_count[entity_id] = sym_id + 1

        for res in chain.residues:
            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append(
                (res.name, res.type, res.idx, atom_idx, len(res.atoms),
                 atom_center, atom_disto, res.is_standard, res.is_present)
            )
            res_map[(chain_name, res.idx)] = res_idx

            if res.rdkit_bounds_constraints is not None:
                for c in res.rdkit_bounds_constraints:
                    rdkit_bounds_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs), c.is_bond, c.is_angle, c.upper_bound, c.lower_bound))
            if res.chiral_atom_constraints is not None:
                for c in res.chiral_atom_constraints:
                    chiral_atom_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs), c.is_reference, c.is_r))
            if res.stereo_bond_constraints is not None:
                for c in res.stereo_bond_constraints:
                    stereo_bond_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs), c.is_check, c.is_e))
            if res.planar_bond_constraints is not None:
                for c in res.planar_bond_constraints:
                    planar_bond_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs),))
            if res.planar_ring_5_constraints is not None:
                for c in res.planar_ring_5_constraints:
                    planar_ring_5_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs),))
            if res.planar_ring_6_constraints is not None:
                for c in res.planar_ring_6_constraints:
                    planar_ring_6_constraint_data.append((tuple(a + atom_idx for a in c.atom_idxs),))

            for bond in res.bonds:
                bond_data.append((atom_idx + bond.atom_1, atom_idx + bond.atom_2, bond.type))

            for local_atom_idx, atom in enumerate(res.atoms):
                atom_idx_map[(chain_name, res.idx, atom.name)] = atom_idx
                atom_data.append(
                    (convert_atom_name(atom.name), atom.element, atom.charge,
                     atom.coords, atom.conformer, atom.is_present, atom.chirality)
                )
                atom_idx += 1
            res_idx += 1

    # --- Step 4: Populate the final glycan feature maps ---
    for chain_name, prelim_features in all_glycan_prelim_features.items():
        asym_id = chain_to_idx[chain_name]
        source_idx = glycan_source_indices[chain_name]
        local_atom_to_mono_idx_maps[asym_id] = all_glycan_atom_maps[chain_name]
        for mono_idx, features in prelim_features.items():
            final_key = (asym_id, mono_idx)
            local_glycan_feature_map[final_key] = MonosaccharideFeatures(
                asym_id=asym_id, ccd_code=features["ccd_code"],
                source_glycan_idx=source_idx, anomeric_config=features["anomeric_config"]
            )

    # --- Step 5: Process schema-level constraints ---
    pocket_binders, pocket_residues, glycosylation_sites_data = [], [], []
    
    # Process legacy bond/pocket constraints
    for constraint in schema.get("constraints", []):
        if "bond" in constraint:
            try:
                c1_name, r1_idx, a1_name = tuple(constraint["bond"]["atom1"])
                c2_name, r2_idx, a2_name = tuple(constraint["bond"]["atom2"])
                a1_global = atom_idx_map[(c1_name, r1_idx - 1, a1_name)]
                a2_global = atom_idx_map[(c2_name, r2_idx - 1, a2_name)]
                # This is an inter-residue bond, so it's a connection
                connections_list.append((chain_to_idx[c1_name], chain_to_idx[c2_name], res_map[(c1_name, r1_idx - 1)], res_map[(c2_name, r2_idx - 1)], a1_global, a2_global))
            except Exception as e:
                raise ValueError(f"Error processing bond constraint: {e}") from e
        elif "pocket" in constraint:
            binder_chain_name = constraint["pocket"]["binder"]
            if len(pocket_binders) > 0 and pocket_binders[-1] != chain_to_idx[binder_chain_name]:
                raise ValueError("Only one pocket binder is supported!")
            if not pocket_binders:
                pocket_binders.append(chain_to_idx[binder_chain_name])
            
            pocket_residues.extend(
                [(chain_to_idx[c_name], r_idx - 1) for c_name, r_idx in constraint["pocket"]["contacts"]]
            )
            
    # Process new glycosylation sites
    if "glycosylation" in schema:
        for site in schema.get("glycosylation", []):
            try:
                prot_spec = site["site"]["protein"]
                p_chain, p_res_1based, p_atom = prot_spec if len(prot_spec) == 3 else (*prot_spec, None)
                p_res_0based = p_res_1based - 1

                g_chain, g_mono_0based, g_atom = site["site"]["glycan"]
                
                if p_atom is None:
                    prot_global_res_idx = res_map.get((p_chain, p_res_0based))
                    if prot_global_res_idx is not None:
                        prot_res_name = res_data[prot_global_res_idx][0]
                        p_atom = {"ASN": "ND2", "SER": "OG", "THR": "OG1"}.get(prot_res_name)
                if p_atom is None:
                    raise ValueError(f"Could not infer attachment atom for protein residue {p_chain}:{p_res_1based}")

                prot_atom_global_idx = atom_idx_map.get((p_chain, p_res_0based, p_atom))
                glycan_atom_global_idx = atom_idx_map.get((g_chain, 0, g_atom))

                if prot_atom_global_idx is None or glycan_atom_global_idx is None:
                    raise ValueError(f"Could not find atoms for glycosylation site: {site}")

                # <<< FIX: REMOVE this line >>>
                # bond_data.append((prot_atom_global_idx, glycan_atom_global_idx, const.bond_type_ids["SINGLE"]))
                
                # <<< FIX: ADD this line >>>
                connections_list.append((chain_to_idx[p_chain], chain_to_idx[g_chain], res_map[(p_chain, p_res_0based)], res_map[(g_chain, 0)], prot_atom_global_idx, glycan_atom_global_idx))

                prot_chain_id_int = chain_to_idx.get(p_chain)
                glycan_chain_id_int = chain_to_idx.get(g_chain)
                if prot_chain_id_int is None or glycan_chain_id_int is None:
                    raise ValueError(f"Could not find chain ID for protein '{p_chain}' or glycan '{g_chain}' in site.")

                site_tuple = (prot_chain_id_int, p_res_0based, p_atom, glycan_chain_id_int, g_mono_0based, g_atom)
                glycosylation_sites_data.append(site_tuple)
            except Exception as e:
                raise ValueError(f"Error processing glycosylation site: {site}. Details: {e}") from e

    # --- Step 6: Final Assembly ---
    structure_data = Structure(
        atoms=np.array(atom_data, dtype=Atom),
        bonds=np.array(sorted(list(set(bond_data))), dtype=Bond),
        residues=np.array(res_data, dtype=Residue),
        chains=np.array(chain_data, dtype=Chain),
        connections=np.array(connections_list, dtype=Connection), # <<< FIX: USE the populated list
        interfaces=np.array([], dtype=Interface),
        mask=np.ones(len(chain_data), dtype=bool),
        glycosylation_sites=np.array(glycosylation_sites_data, dtype=GlycosylationSite),
        glycan_feature_map=local_glycan_feature_map,
        atom_to_mono_idx_map=local_atom_to_mono_idx_maps,
    )

    residue_constraints = ResidueConstraints(
        rdkit_bounds_constraints=np.array(rdkit_bounds_constraint_data, dtype=RDKitBoundsConstraint),
        chiral_atom_constraints=np.array(chiral_atom_constraint_data, dtype=ChiralAtomConstraint),
        stereo_bond_constraints=np.array(stereo_bond_constraint_data, dtype=StereoBondConstraint),
        planar_bond_constraints=np.array(planar_bond_constraint_data, dtype=PlanarBondConstraint),
        planar_ring_5_constraints=np.array(planar_ring_5_constraint_data, dtype=PlanarRing5Constraint),
        planar_ring_6_constraints=np.array(planar_ring_6_constraint_data, dtype=PlanarRing6Constraint),
    )

    struct_info = StructureInfo(num_chains=len(chain_data))
    chain_infos = [
        ChainInfo(
            chain_id=int(c[4]), chain_name=c[0], mol_type=int(c[1]),
            cluster_id=-1, msa_id=chain_to_msa.get(c[0], 0), num_residues=int(c[8]),
            valid=True, entity_id=int(c[2])
        ) for c in chain_data
    ]

    options = InferenceOptions(binders=pocket_binders, pocket=pocket_residues) if pocket_binders else None
    record = Record(
        id=name, structure=struct_info, chains=chain_infos,
        interfaces=[], inference_options=options,
    )

    del blocker

    return Target(
        record=record,
        structure=structure_data,
        sequences=entity_to_seq,
        residue_constraints=residue_constraints,
    )

def parse_standard_residue_with_bonds(
    name: str,
    ref_mol: Mol,
    res_idx: int,
) -> Optional[ParsedResidue]:
    """
    Parses a standard amino acid using a curated atom list (from const.py)
    but generates its internal bond graph from the full RDKit component.
    This avoids including unwanted atoms like OXT for internal residues.
    """
    ref_mol_no_h = AllChem.RemoveHs(ref_mol, sanitize=False)
    ref_conformer = get_conformer(ref_mol_no_h)
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Get the curated list of atom names for an internal residue
    ref_atom_names = const.ref_atoms.get(name)
    if ref_atom_names is None:
        raise ValueError(f"Residue '{name}' not found in const.ref_atoms.")

    # Create a map of all atoms in the full RDKit molecule
    full_atom_map = {a.GetProp("name"): a for a in ref_mol_no_h.GetAtoms()}

    # Parse only the atoms present in our curated list
    atoms: list[ParsedAtom] = []
    # Map from the RDKit atom's original index to its new local index in our list
    rdkit_idx_to_local_idx: Dict[int, int] = {}
    for local_idx, atom_name in enumerate(ref_atom_names):
        if atom_name not in full_atom_map:
            continue # Should not happen if CCD and const.py are consistent
        
        ref_atom = full_atom_map[atom_name]
        rdkit_idx = ref_atom.GetIdx()
        ref_coords = ref_conformer.GetAtomPosition(rdkit_idx)

        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=ref_atom.GetAtomicNum(),
                charge=ref_atom.GetFormalCharge(),
                coords=(0, 0, 0),
                conformer=(ref_coords.x, ref_coords.y, ref_coords.z),
                is_present=True,
                chirality=const.chirality_type_ids.get(str(ref_atom.GetChiralTag()), unk_chirality),
            )
        )
        rdkit_idx_to_local_idx[rdkit_idx] = local_idx

    # Generate bonds ONLY between the atoms we have kept
    bonds: list[ParsedBond] = []
    for bond in ref_mol_no_h.GetBonds():
        start_idx_rdkit, end_idx_rdkit = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start_idx_rdkit in rdkit_idx_to_local_idx and end_idx_rdkit in rdkit_idx_to_local_idx:
            start_idx_local = rdkit_idx_to_local_idx[start_idx_rdkit]
            end_idx_local = rdkit_idx_to_local_idx[end_idx_rdkit]
            bond_type = const.bond_type_ids.get(bond.GetBondType().name, const.bond_type_ids[const.unk_bond_type])
            bonds.append(ParsedBond(min(start_idx_local, end_idx_local), max(start_idx_local, end_idx_local), bond_type))
            
    return ParsedResidue(
        name=name,
        type=const.token_ids[name],
        idx=res_idx,
        atoms=atoms,
        bonds=bonds,
        orig_idx=None,
        atom_center=const.res_to_center_atom_id[name],
        atom_disto=const.res_to_disto_atom_id[name],
        is_standard=True, # Start with True, will be flipped later
        is_present=True,
    )


# --- New helper classes and functions for branching ---
class GlycanToken:
    def __init__(self, token_type: str, value: str, bond_spec: Optional[Tuple[str, int, int]] = None):
        """
        token_type: 'residue', 'open', 'close', 'open_curly', 'close_curly'
        value: For residues, the monosaccharide code (e.g. "FRU")
        bond_spec: If the token is a residue and has a bond spec (e.g. from "(a1-4)"),
                   then (alpha_beta_char, donor_num, acceptor_num) e.g. ('a', 1, 4)
        """
        self.type = token_type
        self.value = value
        self.bond_spec = bond_spec
        self.residue_index: Optional[int] = None # will be set after residues are created
        self.is_cyclic_acceptor: bool = False # The special flag

def tokenize_cyclodextrin(iupac: str) -> List[GlycanToken]:
    pattern = re.compile(r'([A-Z0-9]+)(?:\(([abAB])(\d+)-(\d+)\))?|([\[\]{}])')
    tokens = []
    pos = 0
    while pos < len(iupac):
        match = pattern.match(iupac, pos)
        if not match:
            raise ValueError(f"Could not parse IUPAC string starting at: {iupac[pos:]}")

        if match.group(1):
            code = match.group(1)
            bond_spec = None
            if match.group(2):
                alpha_beta = match.group(2).lower()
                donor_num = int(match.group(3))
                acceptor_num = int(match.group(4))
                bond_spec = (alpha_beta, donor_num, acceptor_num)
            tokens.append(GlycanToken('residue', code, bond_spec))
            pos = match.end()
        elif match.group(5):
            symbol = match.group(5)
            type_map = {'[': 'open', ']': 'close', '{': 'open_curly', '}': 'close_curly'}
            tokens.append(GlycanToken(type_map[symbol], symbol))
            pos = match.end()
        else:
             raise ValueError(f"Unexpected parsing state at: {iupac[pos:]}")

    for i, token in enumerate(tokens):
        if token.type == 'open_curly':
            if i == 0:
                raise ValueError("Cyclic notation '{' cannot be at the start of the string.")
            acceptor_token = tokens[i-1]
            if acceptor_token.type != 'residue':
                raise ValueError(f"The character before '{{' must be a residue, but found '{acceptor_token.value}'.")
            acceptor_token.is_cyclic_acceptor = True

    return tokens

def compute_cyclodextrin_bonds(iupac: str, ccd: Mapping[str, Mol]) -> Tuple[List[ParsedResidue], List[Tuple[int, int, Tuple[str, int, int]]], Optional[int]]:
    """
    Takes the original IUPAC string, tokenizes it to find the acceptor,
    then builds the linear/branched connections and returns the reordered residues
    along with the stable original index of the acceptor.
    """
    # /-------------------------- THE ONLY CHANGE IS THIS ONE LINE --------------------------/
    tokens = tokenize_cyclodextrin(iupac)
    # /------------------------------------ END OF FIX ------------------------------------/
    
    residues: List[ParsedResidue] = []
    
    cyclic_acceptor_original_idx: Optional[int] = None

    for token in tokens:
        if token.type == 'residue':
            idx = len(residues)
            token.residue_index = idx
            if token.is_cyclic_acceptor:
                cyclic_acceptor_original_idx = idx

            if token.value not in ccd:
                raise ValueError(f"CCD structure for glycan residue '{token.value}' not found!")
            res = parse_ccd_residue(token.value, ccd[token.value], res_idx=idx)
            if res is None:
                 raise ValueError(f"Failed to parse CCD residue '{token.value}'")
            residues.append(res)

    stack: List[GlycanToken] = []
    connections: List[Tuple[int, int, Tuple[str, int, int]]] = []
    connect_tokens = [t for t in tokens if t.type in ('residue', 'open', 'close')]

    for token in reversed(connect_tokens):
        if token.type == 'residue':
            if not stack:
                stack.append(token)
            else:
                if token.bond_spec is not None:
                    if stack and stack[-1].type == 'close':
                        stack.pop()
                        if not stack or stack[-1].type != 'residue':
                            raise ValueError("Malformed glycan string: expected residue below close bracket.")
                        target = stack[-1]
                        connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.append(target)
                        stack.append(token)
                    elif stack and stack[-1].type == 'residue':
                        target = stack[-1]
                        connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.pop()
                        stack.append(token)
                    else:
                        raise ValueError(f"Malformed glycan string: unexpected token {stack[-1].type if stack else 'empty'} after residue {token.value}")
                else:
                     if stack and stack[-1].type == 'residue':
                         raise ValueError(f"Residue '{token.value}' is missing bond specification but is followed by '{stack[-1].value}'.")
                     stack.append(token)
        elif token.type == 'close':
            stack.append(token)
        elif token.type == 'open':
            while stack and stack[-1].type != 'close':
                stack.pop()
            if stack and stack[-1].type == 'close':
                stack.pop()

    if not residues:
        return [], [], None
    
    num_residues = len(residues)
    if not connections:
        return residues, [], cyclic_acceptor_original_idx

    child_indices = {c[1] for c in connections}
    root_candidates = [i for i in range(num_residues) if i not in child_indices]
    if len(root_candidates) != 1:
        raise ValueError(f"Glycan parsing error: Found {len(root_candidates)} possible roots. Expected 1.")
    root_idx = root_candidates[0]

    adj = {i: [] for i in range(num_residues)}
    for p, c, _ in connections:
        adj[p].append(c)

    new_order_indices = []
    queue = [root_idx]
    visited = {root_idx}
    
    while queue:
        parent_idx = queue.pop(0)
        new_order_indices.append(parent_idx)
        for child_idx in sorted(adj.get(parent_idx, [])):
            if child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)

    reordered_residues = [residues[i] for i in new_order_indices]
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order_indices)}
    reordered_connections = [(old_to_new_idx_map[p], old_to_new_idx_map[c], spec) for p, c, spec in connections]

    return reordered_residues, reordered_connections, cyclic_acceptor_original_idx


def tokenize_glycan(iupac: str) -> List[GlycanToken]:
    """
    Tokenize the glycan IUPAC string into monosaccharide tokens (with optional bond spec including alpha/beta)
    and branch markers (i.e. '[' and ']').
    """
    # Modified pattern to capture alpha/beta char in group 2, donor in 3, acceptor in 4
    pattern = re.compile(r'([A-Z0-9]+)(?:\(([abAB])(\d+)-(\d+)\))?|([\[\]])')
    tokens = []
    pos = 0
    while pos < len(iupac):
        match = pattern.match(iupac, pos)
        if not match:
            raise ValueError(f"Could not parse IUPAC string starting at: {iupac[pos:]}")

        if match.group(1): # Matched a residue code (potentially with bond spec)
            code = match.group(1)
            bond_spec = None
            if match.group(2): # Bond spec exists
                alpha_beta = match.group(2).lower() # Capture 'a' or 'b'
                donor_num = int(match.group(3))
                acceptor_num = int(match.group(4))
                bond_spec = (alpha_beta, donor_num, acceptor_num)
            tokens.append(GlycanToken('residue', code, bond_spec))
            pos = match.end() # Advance past the whole match including bond spec
        elif match.group(5): # Matched a bracket
            symbol = match.group(5)
            if symbol == '[':
                tokens.append(GlycanToken('open', symbol))
            elif symbol == ']':
                tokens.append(GlycanToken('close', symbol))
            pos = match.end() # Advance past the bracket
        else:
             # This case should ideally not be reached if the regex covers all possibilities
             raise ValueError(f"Unexpected parsing state at: {iupac[pos:]}")

    return tokens


def compute_branching_bonds(iupac: str, ccd: Mapping[str, Mol]) -> Tuple[List[ParsedResidue], List[Tuple[int, int, Tuple[str, int, int]]]]:
    """
    Implements the branching heuristic. It now reorders the output residues
    to be in a root-first traversal order for more intuitive file outputs,
    without changing the underlying chemical topology.

    Returns:
      - a list of residues (in root-first traversal order)
      - a list of connections, where each connection is a tuple:
            (parent_residue_index, child_residue_index, bond_spec)
        bond_spec is (alpha_beta_char, donor_num, acceptor_num) e.g. ('a', 1, 6).
    """
    tokens = tokenize_glycan(iupac)
    residues: List[ParsedResidue] = []
    residue_tokens = [] # Keep track of just the residue tokens for indexing

    for token in tokens:
        if token.type == 'residue':
            idx = len(residues)
            token.residue_index = idx
            if token.value not in ccd:
                raise ValueError(f"CCD structure for glycan residue '{token.value}' not found!")
            # Ensure res_idx is set correctly during initial parsing (used for chain index later)
            res = parse_ccd_residue(token.value, ccd[token.value], res_idx=idx)
            if res is None:
                 raise ValueError(f"Failed to parse CCD residue '{token.value}'")
            residues.append(res)
            residue_tokens.append(token) # Add to the list maintaining order

    # Now process the tokens in reverse (right-to-left) with a stack
    stack: List[GlycanToken] = []
    connections: List[Tuple[int, int, Tuple[str, int, int]]] = [] # Store full bond spec

    for token in reversed(tokens):
        if token.type == 'residue':
            if not stack:
                stack.append(token)
            else:
                if token.bond_spec is not None:
                    if stack and stack[-1].type == 'close':
                        close_token = stack.pop()
                        if not stack or stack[-1].type != 'residue':
                            raise ValueError("Malformed glycan string: expected residue below close bracket.")
                        target = stack[-1]
                        connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.append(target)
                        stack.append(close_token)
                        stack.append(token)
                    elif stack and stack[-1].type == 'residue':
                        target = stack[-1]
                        connections.append((target.residue_index, token.residue_index, token.bond_spec))
                        stack.pop()
                        stack.append(token)
                    else:
                        raise ValueError(f"Malformed glycan string: unexpected token {stack[-1].type if stack else 'empty'} after residue {token.value}")
                else:
                     if stack:
                          if stack[-1].type == 'residue':
                              raise ValueError(f"Residue '{token.value}' (index {token.residue_index}) is missing bond specification but is followed by '{stack[-1].value}'.")
                     stack.append(token)

        elif token.type == 'close':
            stack.append(token)
        elif token.type == 'open':
            while stack and stack[-1].type != 'close':
                stack.pop()
            if stack and stack[-1].type == 'close':
                stack.pop()

    # --- NEW LOGIC: Reorder residues to be root-first (cosmetic change) ---
    if not residues:
        return [], []
    
    num_residues = len(residues)
    if not connections: # Handle single monosaccharide case
        return residues, []

    # 1. Find the root (the only node that is never a child)
    child_indices = {c[1] for c in connections}
    root_candidates = [i for i in range(num_residues) if i not in child_indices]
    if len(root_candidates) != 1:
        raise ValueError(f"Glycan parsing error: Found {len(root_candidates)} possible roots. Expected 1.")
    root_idx = root_candidates[0]

    # 2. Perform a Breadth-First Search (BFS) to get a root-first ordering
    adj = {i: [] for i in range(num_residues)}
    for p, c, _ in connections:
        adj[p].append(c)

    new_order_indices = []
    queue = [root_idx]
    visited = {root_idx}
    
    while queue:
        parent_idx = queue.pop(0)
        new_order_indices.append(parent_idx)
        for child_idx in sorted(adj.get(parent_idx, [])): # Sort for deterministic output
            if child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)

    # 3. Create the reordered list of residues
    reordered_residues = [residues[i] for i in new_order_indices]

    # 4. Create a map from old indices to new indices
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order_indices)}

    # 5. Update the connection indices to match the new residue order
    reordered_connections = [
        (old_to_new_idx_map[p], old_to_new_idx_map[c], spec)
        for p, c, spec in connections
    ]

    return reordered_residues, reordered_connections



def _parse_cyclic_bond_from_original_iupac(
    original_iupac: str,
    residues: List[ParsedResidue],
    iupac_res_names: List[str],
    acceptor_original_idx: int
) -> Tuple[int, int, Tuple[str, int, int]]:
    bond_spec_match = re.search(r'(\w+)\(([abAB])(\d+)-(\d+)\)\s*\}$', original_iupac)
    if not bond_spec_match:
        raise ValueError(f"Could not parse cyclic bond specification from: {original_iupac}")

    alpha_beta = bond_spec_match.group(2).lower()
    donor_num = int(bond_spec_match.group(3))
    acceptor_num = int(bond_spec_match.group(4))
    cyclic_bond_spec = (alpha_beta, donor_num, acceptor_num)

    donor_idx = 0
    donor_name = iupac_res_names[-1]
    if residues[donor_idx].name != donor_name:
         raise ValueError(f"Cyclodextrin root residue mapping failed. Expected {donor_name}, found {residues[donor_idx].name}.")

    try:
        acceptor_reordered_idx = next(i for i, r in enumerate(residues) if r.idx == acceptor_original_idx)
    except StopIteration:
        raise ValueError(f"Logic error: Could not find the flagged acceptor residue (original index {acceptor_original_idx}) in the reordered list.")
    
    acceptor_idx = acceptor_reordered_idx
    return (acceptor_idx, donor_idx, cyclic_bond_spec)

def relax_glycan_structure(
    chains: Dict[str, ParsedChain],
    chain_connections: List[Tuple[str, str, ParsedBond]]
) -> Dict[str, ParsedChain]:
    """
    Merge all provided chains into a temporary RDKit molecule, perform a 
    force-field relaxation, and update the coordinate information in the original
    ParsedChain objects.

    In the single-chain glycan model, this function is called with a single chain
    representing the entire glycan and an empty list of connections.

    Parameters:
        chains: A dictionary mapping chain names to ParsedChain objects.
        chain_connections: A list representing inter-chain bonds (obsolete for single-chain glycans).

    Returns:
        The updated chains dictionary with relaxed coordinates.
    """
    blocker = rdBase.BlockLogs()

    merged_mol = Chem.RWMol()
    # Maps (chain_name, local_atom_index_in_residue) to merged_mol atom index.
    mapping: Dict[Tuple[str, int], int] = {}

    # Add atoms from each chain's residue(s).
    for chain_name, chain in chains.items():
        if not chain.residues: continue
        # Note: This handles both single large glycan residues and multi-residue chains.
        for residue in chain.residues:
            for local_idx, atom in enumerate(residue.atoms):
                rd_atom = Chem.Atom(atom.element)
                rd_atom.SetFormalCharge(atom.charge)
                rd_idx = merged_mol.AddAtom(rd_atom)
                # The key must be unique per atom. We use chain_name and its global atom index
                # from the original structure as a unique key during mapping.
                # Let's assume the local_idx within the residue is sufficient if chain names are unique.
                mapping[(chain_name, local_idx)] = rd_idx

    # Add all internal bonds for each chain.
    for chain_name, chain in chains.items():
        if not chain.residues: continue
        residue = chain.residues[0] # Assuming one residue per chain for simplicity here
        for bond in residue.bonds:
            key1 = (chain_name, bond.atom_1)
            key2 = (chain_name, bond.atom_2)
            if key1 in mapping and key2 in mapping:
                idx1, idx2 = mapping[key1], mapping[key2]
                # FIX: All bonds in glycans are single bonds. This removes the dependency on the non-existent const attribute.
                rd_bond_type = Chem.BondType.SINGLE
                if merged_mol.GetBondBetweenAtoms(idx1, idx2) is None:
                    try:
                        merged_mol.AddBond(idx1, idx2, rd_bond_type)
                    except RuntimeError:
                        print(f"[relax] RDKit failed to add bond between atoms {idx1}-{idx2} for chain {chain_name}")


    # Create a conformer and set the initial coordinates.
    if merged_mol.GetNumAtoms() == 0:
        del blocker
        return chains

    conf = Chem.Conformer(merged_mol.GetNumAtoms())
    for (chain_name, local_idx), rd_idx in mapping.items():
        if chain_name in chains and chains[chain_name].residues:
            atom = chains[chain_name].residues[0].atoms[local_idx]
            # Use the idealized conformer coordinates to build the molecule for relaxation
            x, y, z = atom.conformer
            conf.SetAtomPosition(rd_idx, (x, y, z))

    merged_mol.AddConformer(conf, assignId=True)

    # --- Perform Relaxation ---
    try:
        Chem.SanitizeMol(merged_mol, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY, catchErrors=True)
        # Try UFF first
        uff_code = AllChem.UFFOptimizeMolecule(merged_mol, maxIters=500)
        if uff_code != 0: # 0 is success, 1 is not converged
            # Try MMFF as a fallback
            if AllChem.MMFFHasAllMoleculeParams(merged_mol):
                AllChem.MMFFOptimizeMolecule(merged_mol, maxIters=500)
            else:
                print(f"[relax] UFF failed/not converged and MMFF parameters not available for a molecule.")
    except Exception as e:
        print(f"[relax] Warning: RDKit sanitization or relaxation failed: {e}")

    # Extract the final coordinates and update the original ParsedAtom objects.
    if merged_mol.GetNumConformers() > 0:
        final_conf = merged_mol.GetConformer(0)
        for (chain_name, local_idx), rd_idx in mapping.items():
            if chain_name in chains and chains[chain_name].residues and 0 <= local_idx < len(chains[chain_name].residues[0].atoms):
                pos = final_conf.GetAtomPosition(rd_idx)
                atom = chains[chain_name].residues[0].atoms[local_idx]
                
                updated_atom = ParsedAtom(
                    name=atom.name, element=atom.element, charge=atom.charge,
                    # Preserve the original ground truth coordinates
                    coords=atom.coords,
                    # Update only the conformer with the new relaxed coordinates
                    conformer=(pos.x, pos.y, pos.z),
                    is_present=atom.is_present, chirality=atom.chirality,
                )
                chains[chain_name].residues[0].atoms[local_idx] = updated_atom

    del blocker
    return chains

def stitch_monosaccharides(parent: ParsedResidue, child: ParsedResidue, bond_spec: tuple[str, int, int]) -> tuple[ParsedResidue, ParsedBond]:
    """
    Rigidly translate child based on parent acceptor and child donor atoms derived from bond_spec.
    Removes donor oxygen from the child and returns the transformed child and the glycosidic bond definition.

    Args:
        parent: The parent ParsedResidue.
        child: The child ParsedResidue to be transformed.
        bond_spec: Tuple (alpha_beta_char, donor_num, acceptor_num), e.g., ('a', 1, 4). Alpha/beta not used for geometry here.

    Returns:
        - The transformed and processed child ParsedResidue.
        - A ParsedBond representing the glycosidic bond (using local indices before O removal for child donor C).
    """
    alpha_beta, donor_num, acceptor_num = bond_spec # Unpack the full spec
    donor_atom_name = f"C{donor_num}"
    acceptor_atom_name = f"O{acceptor_num}"
    oxygen_to_remove_name = f"O{donor_num}" # Assume O matches C number for removal

    # --- Find Donor Atom (Child - Carbon) using CONFORMER coordinates ---
    donor_idx = -1
    donor_conformer_coord = None
    for i, atom in enumerate(child.atoms):
        # Use upper() for case-insensitive matching, although CCD names should be consistent
        if atom.name.upper() == donor_atom_name.upper():
            donor_idx = i
            donor_conformer_coord = np.array(atom.conformer)
            break
    if donor_idx == -1:
        atom_names = [a.name for a in child.atoms]
        raise ValueError(f"[stitch] Donor atom {donor_atom_name} not found in child {child.name}. Atoms: {atom_names}")

    # --- Find Acceptor Atom (Parent) using CONFORMER coordinates ---
    acceptor_idx = -1
    acceptor_conformer_coord = None
    for i, atom in enumerate(parent.atoms):
        if atom.name.upper() == acceptor_atom_name.upper():
            acceptor_idx = i
            acceptor_conformer_coord = np.array(atom.conformer)
            break
    if acceptor_idx == -1:
        atom_names = [a.name for a in parent.atoms]
        raise ValueError(f"[stitch] Acceptor atom {acceptor_atom_name} not found in parent {parent.name}. Atoms: {atom_names}")

    # --- Calculate Target Position for Donor ---
    v_donor_to_acceptor = acceptor_conformer_coord - donor_conformer_coord
    dist = np.linalg.norm(v_donor_to_acceptor)
    unit_v = v_donor_to_acceptor / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
    bond_length = 1.4 # Target glycosidic bond length
    target_donor_position = acceptor_conformer_coord - unit_v * bond_length

    # --- Calculate Translation Vector ---
    translation = target_donor_position - donor_conformer_coord

    # --- Apply Translation RIGIDLY to all Child Atoms (Temporary list) ---
    translated_atoms_temp = []
    for original_atom in child.atoms:
        new_conformer_position = np.array(original_atom.conformer) + translation
        translated_atoms_temp.append(
            ParsedAtom(
                name=original_atom.name, element=original_atom.element, charge=original_atom.charge,
                coords=original_atom.coords,  # Preserve original ground truth
                conformer=tuple(new_conformer_position),  # Update idealized conformer
                is_present=original_atom.is_present, chirality=original_atom.chirality
            )
        )

    # --- Find and Remove the Donor Oxygen (O{donor_num}) ---
    oxygen_to_remove_idx = -1
    for i, atom in enumerate(translated_atoms_temp):
        if atom.name.upper() == oxygen_to_remove_name.upper():
            oxygen_to_remove_idx = i
            break

    final_atoms = []
    idx_map = {} # old index -> new index
    new_idx_counter = 0
    original_num_atoms = len(translated_atoms_temp)

    if oxygen_to_remove_idx != -1:
        # print(f"[stitch] Removing oxygen '{oxygen_to_remove_name}' (idx {oxygen_to_remove_idx}) from {child.name}")
        for i, atom in enumerate(translated_atoms_temp):
            if i == oxygen_to_remove_idx:
                continue
            final_atoms.append(atom)
            idx_map[i] = new_idx_counter
            new_idx_counter += 1
    else:
        # print(f"[stitch] Warning: Oxygen '{oxygen_to_remove_name}' not found in {child.name}. Proceeding without removal.")
        final_atoms = translated_atoms_temp # No atoms removed
        idx_map = {i: i for i in range(original_num_atoms)} # Identity map

    # --- Update Bond Indices ---
    final_bonds = []
    if oxygen_to_remove_idx != -1: # Only remap if oxygen was actually removed
        for bond in child.bonds:
            if bond.atom_1 == oxygen_to_remove_idx or bond.atom_2 == oxygen_to_remove_idx:
                continue # Skip bonds involving the removed oxygen
            try:
                new_a1 = idx_map[bond.atom_1]
                new_a2 = idx_map[bond.atom_2]
                final_bonds.append(ParsedBond(min(new_a1, new_a2), max(new_a1, new_a2), bond.type))
            except KeyError:
                 print(f"[stitch] Warning: Could not remap bond {bond} for {child.name} after O removal.")
    else:
        final_bonds = child.bonds # Bonds remain unchanged if no oxygen removed

    # --- Update atom_center and atom_disto indices ---
    new_atom_center = idx_map.get(child.atom_center, 0) # Default to 0 if original center removed or not found
    new_atom_disto = idx_map.get(child.atom_disto, 0)   # Default to 0 if original disto removed or not found
    if child.atom_center == oxygen_to_remove_idx: print(f"[stitch] Warning: Center atom was removed oxygen for {child.name}")
    if child.atom_disto == oxygen_to_remove_idx: print(f"[stitch] Warning: Disto atom was removed oxygen for {child.name}")

    # --- Construct Final Processed Child Residue ---
    processed_child = ParsedResidue(
        name=child.name, type=child.type, idx=child.idx, # Keep original residue index from compute_branching_bonds
        atoms=final_atoms, bonds=final_bonds,
        orig_idx=child.orig_idx, # This should be the index from compute_branching_bonds
        atom_center=new_atom_center, atom_disto=new_atom_disto,
        is_standard=child.is_standard, is_present=child.is_present
    )

    # --- Create Glycosidic Bond Definition (using original local indices) ---
    glyco_bond = ParsedBond(
        atom_1=acceptor_idx, # Local index in parent
        atom_2=donor_idx,    # Local index of donor CARBON in child (before O removal)
        type=const.bond_type_ids.get("glycosidic", const.bond_type_ids[const.unk_bond_type]) # Or SINGLE?
    )

    return processed_child, glyco_bond

def _delete_anomeric_oxygen(child_residue: ParsedResidue, donor_c_num: int) -> ParsedResidue:
    oxygen_to_remove_name = f"O{donor_c_num}"

    oxygen_to_remove_idx = -1
    for i, atom in enumerate(child_residue.atoms):
        if atom.name.upper() == oxygen_to_remove_name.upper():
            oxygen_to_remove_idx = i
            break

    if oxygen_to_remove_idx == -1:
        return child_residue

    final_atoms = []
    idx_map = {}
    new_idx_counter = 0
    for i, atom in enumerate(child_residue.atoms):
        if i == oxygen_to_remove_idx:
            continue
        final_atoms.append(atom)
        idx_map[i] = new_idx_counter
        new_idx_counter += 1

    final_bonds = []
    for bond in child_residue.bonds:
        if bond.atom_1 == oxygen_to_remove_idx or bond.atom_2 == oxygen_to_remove_idx:
            continue
        try:
            new_a1 = idx_map[bond.atom_1]
            new_a2 = idx_map[bond.atom_2]
            final_bonds.append(ParsedBond(min(new_a1, new_a2), max(new_a1, new_a2), bond.type))
        except KeyError:
            pass

    new_atom_center = idx_map.get(child_residue.atom_center, 0)
    new_atom_disto = idx_map.get(child_residue.atom_disto, 0)

    return replace(child_residue,
                   atoms=final_atoms,
                   bonds=final_bonds,
                   atom_center=new_atom_center,
                   atom_disto=new_atom_disto)

def parse_glycan(iupac: str, ccd: Mapping[str, Mol]) -> Tuple[Optional[ParsedResidue], Optional[Dict[int, dict]], Optional[np.ndarray]]:
    """
    Parses a glycan IUPAC string by dispatching to the correct specialized parser.
    - For regular/branched glycans, it uses the original, unmodified logic.
    - For cyclodextrins, it uses a new, dedicated pathway.
    This ensures complete logical separation and correctness for both cases.
    """
    original_iupac = iupac
    is_cyclodextrin = '{' in iupac

    if is_cyclodextrin:
        # --- CYCLODEXTRIN PATH ---
        try:
            # 1. Call the NEW, dedicated cyclodextrin parser
            residues, connections, cyclic_acceptor_original_idx = compute_cyclodextrin_bonds(original_iupac, ccd)
        except ValueError as e:
            raise ValueError(f"Error parsing cyclodextrin IUPAC '{iupac}' during branching: {e}") from e

        if not residues:
            return None, None, None
        
        if cyclic_acceptor_original_idx is None:
            raise ValueError("Detected cyclodextrin notation '{}' but could not identify the acceptor residue.")
        
        # 2. Add the cyclic bond
        # We need the original order of names for the donor check
        sanitized_iupac = original_iupac.replace('{', '').replace('}', '')
        iupac_res_names = [t.value for t in tokenize_glycan(sanitized_iupac) if t.type == 'residue']
        
        cyclic_connection = _parse_cyclic_bond_from_original_iupac(original_iupac, residues, iupac_res_names, cyclic_acceptor_original_idx)
        connections.append(cyclic_connection)

        # 3. Perform the robust two-pass modification and bond creation
        # PASS 1: Modify all donor residues
        residue_mods = {i: res for i, res in enumerate(residues)}
        for _, child_idx, bond_spec in connections:
            child_res_to_modify = residue_mods[child_idx]
            _, donor_c_num, _ = bond_spec
            modified_child = _delete_anomeric_oxygen(child_res_to_modify, donor_c_num)
            residue_mods[child_idx] = modified_child
        
        processed_residues = [residue_mods[i] for i in range(len(residues))]

        # PASS 2: Determine local indices for bonds from the final residue states
        glycosidic_bonds_local = []
        for parent_idx, child_idx, bond_spec in connections:
            parent_res = processed_residues[parent_idx]
            child_res = processed_residues[child_idx]
            _, donor_c_num, acceptor_o_num = bond_spec
            donor_atom_name = f"C{donor_c_num}"
            acceptor_atom_name = f"O{acceptor_o_num}"

            try:
                donor_c_idx_local = next(i for i, a in enumerate(child_res.atoms) if a.name.upper() == donor_atom_name.upper())
                acceptor_o_idx_local = next(i for i, a in enumerate(parent_res.atoms) if a.name.upper() == acceptor_atom_name.upper())
            except StopIteration:
                raise ValueError(f"Could not find atoms for bond between {parent_res.name} and {child_res.name}") from None
            
            glycosidic_bonds_local.append((parent_idx, child_idx, ParsedBond(acceptor_o_idx_local, donor_c_idx_local, const.bond_type_ids["SINGLE"])))
        
        # 4. Aggregate into a single residue (NO relaxation)
        master_atoms, master_bonds, atom_to_mono_idx_list, atom_offsets = [], [], [], {}
        current_offset = 0
        for mono_idx, res in enumerate(processed_residues):
            num_atoms_in_mono = len(res.atoms)
            atom_offsets[mono_idx] = current_offset
            master_atoms.extend(res.atoms)
            atom_to_mono_idx_list.extend([mono_idx] * num_atoms_in_mono)
            for bond in res.bonds:
                master_bonds.append(ParsedBond(bond.atom_1 + current_offset, bond.atom_2 + current_offset, bond.type))
            current_offset += num_atoms_in_mono

        single_bond_type = const.bond_type_ids["SINGLE"]
        for parent_idx, child_idx, local_bond in glycosidic_bonds_local:
            parent_offset = atom_offsets[parent_idx]
            child_offset = atom_offsets[child_idx]
            global_atom_1 = local_bond.atom_1 + parent_offset
            global_atom_2 = local_bond.atom_2 + child_offset
            master_bonds.append(ParsedBond(global_atom_1, global_atom_2, single_bond_type))

        final_residue = ParsedResidue("GLYCAN", const.unk_token_ids["PROTEIN"], 0, master_atoms, master_bonds, None, 0, 0, False, True)

    else:
        # --- REGULAR/BRANCHED GLYCAN PATH (Original, Unmodified Logic) ---
        try:
            # 1. Call the ORIGINAL, restored branching function
            residues, connections = compute_branching_bonds(original_iupac, ccd)
        except ValueError as e:
            raise ValueError(f"Error parsing glycan IUPAC '{iupac}' during branching: {e}") from e

        if not residues:
            return None, None, None

        # 2. Perform the original root oxygen removal
        child_idxs = {c[1] for c in connections}
        root_candidates = [i for i in range(len(residues)) if i not in child_idxs]
        if len(root_candidates) == 1:
            root_idx = root_candidates[0]
            root_res = residues[root_idx]
            oxy_idxs = [i for i, atom in enumerate(root_res.atoms) if atom.name.upper().startswith("O")]
            if oxy_idxs:
                idx_name_pairs = [(i, root_res.atoms[i].name) for i in oxy_idxs]
                remove_idx, _ = min(idx_name_pairs, key=lambda x: x[1])
                new_atoms, idx_map, new_bonds = [], {}, []
                new_i = 0
                for old_i, atom in enumerate(root_res.atoms):
                    if old_i == remove_idx: continue
                    new_atoms.append(atom)
                    idx_map[old_i] = new_i
                    new_i += 1
                for bond in root_res.bonds:
                    if bond.atom_1 == remove_idx or bond.atom_2 == remove_idx: continue
                    a1, a2 = idx_map[bond.atom_1], idx_map[bond.atom_2]
                    new_bonds.append(ParsedBond(min(a1, a2), max(a1, a2), bond.type))
                new_center = idx_map.get(root_res.atom_center, 0)
                new_disto = idx_map.get(root_res.atom_disto, 0)
                residues[root_idx] = replace(root_res, atoms=new_atoms, bonds=new_bonds, atom_center=new_center, atom_disto=new_disto)
        
        # 3. Perform the original stitching process
        glycosidic_bonds_local = []
        for parent_idx, child_idx, bond_spec in connections:
            parent_res, child_res = residues[parent_idx], residues[child_idx]
            try:
                transformed_child, glyco_bond_local = stitch_monosaccharides(parent_res, child_res, bond_spec)
                residues[child_idx] = transformed_child
                glycosidic_bonds_local.append((parent_idx, child_idx, glyco_bond_local))
            except ValueError as e:
                raise ValueError(f"Error stitching {parent_res.name}({parent_idx}) to {child_res.name}({child_idx}): {e}") from e

        # 4. Aggregate and Relax the structure
        master_atoms, master_bonds, atom_to_mono_idx_list, atom_offsets = [], [], [], {}
        current_offset = 0
        for mono_idx, res in enumerate(residues):
            num_atoms_in_mono = len(res.atoms)
            atom_offsets[mono_idx] = current_offset
            master_atoms.extend(res.atoms)
            atom_to_mono_idx_list.extend([mono_idx] * num_atoms_in_mono)
            for bond in res.bonds:
                master_bonds.append(ParsedBond(bond.atom_1 + current_offset, bond.atom_2 + current_offset, bond.type))
            current_offset += num_atoms_in_mono

        single_bond_type = const.bond_type_ids["SINGLE"]
        for parent_idx, child_idx, local_bond in glycosidic_bonds_local:
            parent_offset, child_offset = atom_offsets[parent_idx], atom_offsets[child_idx]
            global_atom_1, global_atom_2 = local_bond.atom_1 + parent_offset, local_bond.atom_2 + child_offset
            master_bonds.append(ParsedBond(global_atom_1, global_atom_2, single_bond_type))

        temp_residue = ParsedResidue("GLYCAN", const.unk_token_ids["PROTEIN"], 0, master_atoms, master_bonds, None, 0, 0, False, True)
        temp_chain = ParsedChain("temp_glycan_chain", const.chain_type_ids["NONPOLYMER"], [temp_residue], 0)
        try:
            relaxed_chains = relax_glycan_structure({"temp_glycan_chain": temp_chain}, [])
            final_residue = relaxed_chains["temp_glycan_chain"].residues[0]
        except Exception:
            print(f"[parse_glycan] Warning: Structure relaxation failed for IUPAC '{iupac}'. Using unrelaxed coordinates.")
            final_residue = temp_residue

    # --- COMMON LOGIC FOR BOTH PATHS ---
    prelim_features: Dict[int, dict] = {}
    connections_by_child = {c[1]: c for c in connections}
    for i, res_initial in enumerate(residues):
        ccd_code = res_initial.name
        conn_info = connections_by_child.get(i)
        anomeric_config = conn_info[2][0] if conn_info else None
        prelim_features[i] = {"ccd_code": ccd_code, "anomeric_config": anomeric_config}

    # --- MODIFICATION START ---
    # Zero out the conformer space for all atoms in the final glycan residue.
    # This ensures that for glycan chains, the idealized coordinates are all (0,0,0).
    if final_residue:
        zeroed_atoms = [
            replace(atom, conformer=(0.0, 0.0, 0.0)) for atom in final_residue.atoms
        ]
        final_residue = replace(final_residue, atoms=zeroed_atoms)
    # --- MODIFICATION END ---

    atom_to_mono_idx_array = np.array(atom_to_mono_idx_list, dtype=np.int32)

    return final_residue, prelim_features, atom_to_mono_idx_array
