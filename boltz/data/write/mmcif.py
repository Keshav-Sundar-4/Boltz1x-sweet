# boltz/data/write/mmcif.py (Final Corrected Version)

import io
from collections import defaultdict
from collections.abc import Iterator
from typing import Optional

import ihm
import modelcif
from modelcif import Assembly, AsymUnit, Entity, System, dumper
from modelcif.model import AbInitioModel, Atom, ModelGroup
from rdkit import Chem
from torch import Tensor

from boltz.data import const
from boltz.data.types import Structure


def to_mmcif(structure: Structure, plddts: Optional[Tensor] = None) -> str:  # noqa: C901, PLR0915, PLR0912
    """
    Write a structure into an MMCIF file, correctly handling proteins,
    nucleic acids, simple ligands, and complex glycans with correct pLDDTs.
    """
    system = System()
    periodic_table = Chem.GetPeriodicTable()

    # --- Step 1: Define Entities and Sequences (Glycan-Aware) ---
    entity_to_chains = defaultdict(list)
    entity_to_moltype = {}
    for chain in structure.chains:
        entity_id = chain["entity_id"]
        entity_to_chains[entity_id].append(chain)
        entity_to_moltype[entity_id] = chain["mol_type"]

    sequences = {}
    for entity_id, chains in entity_to_chains.items():
        chain = chains[0]
        chain_idx = chain["asym_id"]

        is_glycan_entity = (
            structure.glycan_feature_map is not None
            and structure.atom_to_mono_idx_map is not None
            and chain_idx in structure.atom_to_mono_idx_map
        )

        if is_glycan_entity:
            all_monos_for_chain = {
                mono_idx: features['ccd_code'] if isinstance(features, dict) else features.ccd_code
                for (cid, mono_idx), features in structure.glycan_feature_map.items()
                if cid == chain_idx
            }
            sequence = [all_monos_for_chain[i] for i in sorted(all_monos_for_chain.keys())]
        else:
            res_start, res_end = chain["res_idx"], chain["res_idx"] + chain["res_num"]
            residues = structure.residues[res_start:res_end]
            sequence = [str(res["name"]) for res in residues]
        sequences[entity_id] = sequence

    entities_map = {}
    for entity_id, sequence in sequences.items():
        mol_type = entity_to_moltype[entity_id]
        if mol_type == const.chain_type_ids["PROTEIN"]:
            alphabet, chem_comp = ihm.LPeptideAlphabet(), lambda x: ihm.LPeptideChemComp(id=x, code=x, code_canonical="X")
        elif mol_type == const.chain_type_ids["DNA"]:
            alphabet, chem_comp = ihm.DNAAlphabet(), lambda x: ihm.DNAChemComp(id=x, code=x, code_canonical="N")
        elif mol_type == const.chain_type_ids["RNA"]:
            alphabet, chem_comp = ihm.RNAAlphabet(), lambda x: ihm.RNAChemComp(id=x, code=x, code_canonical="N")
        elif len(sequence) > 1: # Glycans
            alphabet, chem_comp = {}, lambda x: ihm.SaccharideChemComp(id=x)
        else: # Ligands
            alphabet, chem_comp = {}, lambda x: ihm.NonPolymerChemComp(id=x)

        seq = [alphabet[item] if item in alphabet else chem_comp(item) for item in sequence]
        model_e = Entity(seq)
        for chain in entity_to_chains[entity_id]:
            entities_map[chain["asym_id"]] = model_e

    # --- Step 2: Create Asymmetric Units ---
    asym_unit_map = {}
    for chain in structure.chains:
        chain_idx, chain_tag = chain["asym_id"], str(chain["name"])
        asym = AsymUnit(entities_map[chain_idx], details=f"Model subunit {chain_tag}", id=chain_tag)
        asym_unit_map[chain_idx] = asym
    modeled_assembly = Assembly(list(asym_unit_map.values()), name="Modeled assembly")

    class _LocalPLDDT(modelcif.qa_metric.Local, modelcif.qa_metric.PLDDT):
        name, software, description = "pLDDT", None, "Predicted lddt"

    class _MyModel(AbInitioModel):
        def get_atoms(self) -> Iterator[Atom]:
            # Boltz-1x state machine variables
            res_num, prev_polymer_resnum, ligand_index_offset = 0, -1, 0

            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                is_glycan_chain = (
                    structure.glycan_feature_map is not None
                    and structure.atom_to_mono_idx_map is not None
                    and chain_idx in structure.atom_to_mono_idx_map
                )

                if is_glycan_chain:
                    # --- Glycan-Specific Atom and B-factor Logic ---
                    # This logic correctly handles the multiple "residues" (monosaccharides)
                    # within a single glycan chain.
                    chain_atom_start = chain["atom_idx"]
                    mono_idx_map = structure.atom_to_mono_idx_map[chain_idx]
                    
                    atoms_by_mono = defaultdict(list)
                    for i in range(chain["atom_num"]):
                        global_idx = chain_atom_start + i
                        if structure.atoms[global_idx]["is_present"]:
                            atoms_by_mono[mono_idx_map[i]].append((i, global_idx))

                    for mono_idx in sorted(atoms_by_mono.keys()):
                        residue_index = mono_idx + 1 # 1-based seq_id
                        for local_idx_in_chain, global_atom_idx in atoms_by_mono[mono_idx]:
                            atom = structure.atoms[global_atom_idx]
                            name = "".join([chr(c + 32) for c in atom["name"] if c != 0])
                            element = periodic_table.GetElementSymbol(atom["element"].item()).upper()
                            pos = atom["coords"]
                            
                            biso = 100.00
                            if plddts is not None:
                                # Glycans are HETATMs, so plddt is per-atom.
                                # The index is calculated relative to the last polymer residue.
                                plddt_idx = prev_polymer_resnum + ligand_index_offset + local_idx_in_chain + 1
                                biso = round(plddts[plddt_idx].item() * 100, 3)

                            yield Atom(
                                asym_unit=asym_unit_map[chain_idx], type_symbol=element,
                                seq_id=residue_index, atom_id=name,
                                x=f"{pos[0]:.5f}", y=f"{pos[1]:.5f}", z=f"{pos[2]:.5f}",
                                het=True, biso=biso, occupancy=1)
                    
                    # After processing all atoms in the glycan chain, update the offset
                    ligand_index_offset += chain["atom_num"]
                else:
                    # --- Original Boltz-1x Atom and B-factor Logic (for non-glycans) ---
                    het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                    record_type = "ATOM" if not het else "HETATM"
                    res_start, res_end = chain["res_idx"], chain["res_idx"] + chain["res_num"]

                    for residue in structure.residues[res_start:res_end]:
                        atom_start, atom_end = residue["atom_idx"], residue["atom_idx"] + residue["atom_num"]
                        for i, atom in enumerate(structure.atoms[atom_start:atom_end]):
                            if not atom["is_present"]: continue
                            name = "".join([chr(c + 32) for c in atom["name"] if c != 0])
                            element = periodic_table.GetElementSymbol(atom["element"].item()).upper()
                            pos = atom["coords"]
                            
                            biso = 100.00
                            if plddts is not None:
                                if record_type == 'ATOM':
                                    biso = round(plddts[res_num + ligand_index_offset].item() * 100, 3)
                                else:
                                    plddt_idx = prev_polymer_resnum + ligand_index_offset + i + 1
                                    biso = round(plddts[plddt_idx].item() * 100, 3)
                            
                            yield Atom(
                                asym_unit=asym_unit_map[chain_idx], type_symbol=element,
                                seq_id=residue["res_idx"] + 1, atom_id=name,
                                x=f"{pos[0]:.5f}", y=f"{pos[1]:.5f}", z=f"{pos[2]:.5f}",
                                het=het, biso=biso, occupancy=1)

                        if record_type == 'ATOM':
                            res_num += 1
                            prev_polymer_resnum = res_num - 1
                        else:
                            ligand_index_offset += residue["atom_num"]

        def add_plddt(self, plddts):
            # This method adds per-residue QA scores. Since per-atom pLDDTs are
            # already in the B-factor column, this logic remains mostly the same,
            # but we must correctly handle the glycan chain's single residue entry.
            res_num, prev_polymer_resnum, ligand_index_offset = 0, -1, 0

            for chain in structure.chains:
                chain_idx = chain["asym_id"]
                het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
                record_type = "ATOM" if not het else "HETATM"
                res_start, res_end = chain["res_idx"], chain["res_idx"] + chain["res_num"]

                for residue in structure.residues[res_start:res_end]:
                    # For both glycans and simple ligands, we calculate an average pLDDT.
                    # For polymers, we take the single residue score.
                    if record_type == 'ATOM':
                        plddt_val = round(plddts[res_num + ligand_index_offset].item() * 100, 3)
                        prev_polymer_resnum = res_num
                        res_num += 1
                    else:
                        start_idx = prev_polymer_resnum + ligand_index_offset + 1
                        end_idx = start_idx + residue["atom_num"]
                        plddt_val = round(plddts[start_idx:end_idx].mean().item() * 100, 2)
                        ligand_index_offset += residue["atom_num"]

                    # The glycan chain has only one residue in `structure.residues`, but its
                    # Entity sequence has multiple. We must link to the correct seq_id.
                    if het and structure.glycan_feature_map and chain_idx in structure.atom_to_mono_idx_map:
                         # For a glycan, associate the average pLDDT with each monosaccharide.
                         num_monos = len(sequences[chain['entity_id']])
                         for i in range(num_monos):
                            self.qa_metrics.append(_LocalPLDDT(asym_unit_map[chain_idx].residue(i + 1), plddt_val))
                    else:
                         # Standard logic for proteins and simple ligands
                         self.qa_metrics.append(_LocalPLDDT(asym_unit_map[chain_idx].residue(residue["res_idx"] + 1), plddt_val))

    model = _MyModel(assembly=modeled_assembly, name="Model")
    if plddts is not None:
        model.add_plddt(plddts)

    model_group = ModelGroup([model], name="All models")
    system.model_groups.append(model_group)

    fh = io.StringIO()
    dumper.write(fh, [system])
    return fh.getvalue()
