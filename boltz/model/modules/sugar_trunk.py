from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Any, Tuple
import sys
import pkg_resources
import json

from boltz.data import const
from boltz.data.feature.featurizer import MONO_TYPE_MAP
import torch.nn.functional as F

from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.transition import Transition
from boltz.model.layers.dropout import get_dropout_mask
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

#############################################################################################################
#############################################################################################################
#CONSTANTS
#############################################################################################################
#############################################################################################################


NUM_MONO_TYPES_PLACEHOLDER = 931 # As derived from the provided map example (+1 for OTHER)
NUM_ANOMERIC_TYPES = 3
NUM_MONO_TYPES = len(MONO_TYPE_MAP)
D_MONO_EMB = 64
NUM_AMINO_ACIDS = 22

#############################################################################################################
#############################################################################################################
#CONSTANTS
#############################################################################################################
#############################################################################################################



#############################################################################################################
#############################################################################################################
#HELPERS
#############################################################################################################
#############################################################################################################

def build_couplet_pair_mask(feats: Dict[str, Tensor]) -> Tensor:
    """
    Locate glycosidic couplets and return a boolean mask [B, L, L] where
    mask[b, i, j] = True iff token i is a nucleophile (O or N), token j is carbon,
    they are bonded, and they belong to different monosaccharides.
    """
    token_bonds    = feats["token_bonds"].squeeze(-1).bool()      # [B, L, L]
    token_to_mono = feats["token_to_mono_idx"]                  # [B, L]

    # derive atomic numbers
    ref_elem_oh = feats["ref_element"].float()                  # [B, A, E]
    selector    = feats["token_to_rep_atom"].float()            # [B, L, A]
    elem_oh     = torch.einsum("bla,bae->ble", selector, ref_elem_oh)
    atom_num    = elem_oh.argmax(dim=-1)                          # [B, L]
    is_O        = atom_num == 8
    is_N        = atom_num == 7
    is_C        = atom_num == 6

    # A glycosidic nucleophile can be Oxygen or Nitrogen
    is_nucleophile = is_O | is_N

    # inter‑monosaccharide bonds only
    mono_i      = token_to_mono.unsqueeze(2)                    # [B, L, 1]
    mono_j      = token_to_mono.unsqueeze(1)                    # [B, 1, L]
    inter_mono  = token_bonds & (mono_i != mono_j)

    # mask nucleophile→carbon bonds
    mask_nuc_to_C = inter_mono & is_nucleophile.unsqueeze(2) & is_C.unsqueeze(1) # True if row_idx is O/N, col_idx is C
    # also identify carbon->nucleophile bonds for symmetry
    mask_C_to_nuc = inter_mono & is_C.unsqueeze(2) & is_nucleophile.unsqueeze(1) # True if row_idx is C, col_idx is O/N
    
    return mask_nuc_to_C | mask_C_to_nuc

def _decode_atom_name(one_hot_encoded_name: torch.Tensor) -> str:
    integer_indices = torch.argmax(one_hot_encoded_name, dim=-1)
    chars = []
    for idx_tensor in integer_indices:
        num = idx_tensor.item()
        char_code = num + 32
        if char_code > 32:
            chars.append(chr(char_code))
    return "".join(chars).strip()

def _decode_int_to_str(encoded_name: torch.Tensor) -> str:
    """Decodes a tensor of 4 integers back into a string atom name."""
    # Add 32 to convert back to ASCII character codes
    char_codes = [c.item() + 32 for c in encoded_name]
    # Convert codes to characters and join, stripping trailing whitespace
    return "".join([chr(c) for c in char_codes]).strip()

def _decode_one_hot_to_str(one_hot_encoded_name: torch.Tensor) -> str:
    """Decodes a one-hot encoded name from ref_atom_name_chars."""
    # Find the integer index for each of the 4 character positions
    integer_indices = torch.argmax(one_hot_encoded_name, dim=-1)
    # Add 32 to convert back to ASCII character codes
    char_codes = [idx.item() + 32 for idx in integer_indices]
    # Filter out null characters (code 32) and join
    return "".join([chr(c) for c in char_codes if c > 32]).strip()

def _get_glycosylation_features(feats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    (Streamlined & Instrumented Version)
    Extracts the token indices of the specific protein-glycan covalent attachment points.
    This version is simplified to only compute what is used by the debug print function,
    removing extraneous feature generation.
    """
    device = feats["token_pad_mask"].device
    B, L = feats["token_pad_mask"].shape

    raw_sites_tensors = feats.get('raw_glycosylation_sites')
    if raw_sites_tensors is None:
        raw_sites_tensors = [torch.empty((0, 12), device=device, dtype=torch.long)] * B

    empty_result = {
        "t_glycosylation_indices": torch.empty((0, 2), dtype=torch.long, device=device),
        "t_batch_idx": torch.empty((0,), dtype=torch.long, device=device),
    }

    all_t_indices, all_t_batch_idx = [], []
    any_sites_found = False

    for b in range(B):
        sites_tensor_b = raw_sites_tensors[b]
        if sites_tensor_b is None or sites_tensor_b.numel() == 0:
            continue
        
        any_sites_found = True
        
        atom_to_token = feats["atom_to_token"][b].argmax(-1)
        token_asym_ids = feats["asym_id"][b]
        token_res_indices = feats["residue_index"][b]
        atom_asym_ids = torch.gather(token_asym_ids, 0, atom_to_token)
        atom_res_indices = torch.gather(token_res_indices, 0, atom_to_token)
        ref_name_chars = feats["ref_atom_name_chars"][b]
        
        for site_idx, site_data_tensor in enumerate(sites_tensor_b):
            p_chain_id, p_res_id = site_data_tensor[0].item(), site_data_tensor[1].item()
            g_chain_id = site_data_tensor[6].item()
            
            tgt_p_name = _decode_int_to_str(site_data_tensor[2:6]).upper()
            tgt_g_name = _decode_int_to_str(site_data_tensor[8:12]).upper()

            p_res_mask = (atom_asym_ids == p_chain_id) & (atom_res_indices == p_res_id)
            p_atoms_in_res = torch.where(p_res_mask)[0]
            
            glycan_chain_atoms = torch.where(atom_asym_ids == g_chain_id)[0]
            
            trg_p_atom = -1
            for p_atom_idx in p_atoms_in_res:
                decoded_name = _decode_one_hot_to_str(ref_name_chars[p_atom_idx]).upper()
                if decoded_name == tgt_p_name:
                    trg_p_atom = p_atom_idx.item()
                    break
            
            trg_g_atom = -1
            for g_atom_idx in glycan_chain_atoms:
                 decoded_name = _decode_one_hot_to_str(ref_name_chars[g_atom_idx]).upper()
                 if decoded_name == tgt_g_name:
                    trg_g_atom = g_atom_idx.item()
                    break
            
            if trg_p_atom != -1 and trg_g_atom != -1:
                p_tok_idx = atom_to_token[trg_p_atom].item()
                g_tok_idx = atom_to_token[trg_g_atom].item()
                
                all_t_indices.append([p_tok_idx, g_tok_idx])
                all_t_batch_idx.append(b)

    if not any_sites_found:
        return empty_result

    final_result = {
        "t_glycosylation_indices": torch.tensor(all_t_indices, dtype=torch.long, device=device) if all_t_indices else torch.empty((0, 2), dtype=torch.long, device=device),
        "t_batch_idx": torch.tensor(all_t_batch_idx, dtype=torch.long, device=device),
    }

    return final_result

def _get_glycosylation_linkage_mask(feats: Dict[str, Any], device: torch.device) -> Tensor:
    """
    Creates a boolean mask [B, L, L] that is True for token pairs
    forming a protein-glycan covalent bond, using the ground-truth feature extractor.
    """
    B, L = feats["token_pad_mask"].shape
    linkage_mask = torch.zeros((B, L, L), dtype=torch.bool, device=device)

    glyco_features = _get_glycosylation_features(feats)
    batch_indices = glyco_features["t_batch_idx"]
    token_pairs = glyco_features["t_glycosylation_indices"]

    if token_pairs.numel() > 0:
        p_tokens = token_pairs[:, 0]
        g_tokens = token_pairs[:, 1]
        linkage_mask[batch_indices, p_tokens, g_tokens] = True
        linkage_mask[batch_indices, g_tokens, p_tokens] = True

    return linkage_mask

class AnomericBondBias(nn.Module):
    """
    Apply a learned pairwise bias to glycosidic bonds based on the anomeric
    configuration (alpha/beta/other).

    UPDATED LOGIC:
    This module now strictly applies bias ONLY to Intra-Glycan bonds.
    It intentionally ignores Protein-Glycan bonds to avoid injecting noise
    into these highly conserved, deterministic linkages.
    """

    def __init__(self, token_z: int, num_anomeric_types: int = 3) -> None:
        super().__init__()
        if token_z <= 0:
            raise ValueError(f"token_z must be positive, got {token_z}")
        if num_anomeric_types <= 0:
            raise ValueError(
                f"num_anomeric_types must be positive, got {num_anomeric_types}"
            )

        # Linear map from anomeric one-hot -> bias vector in z space.
        self.pairwise_bias_embedding = nn.Linear(
            num_anomeric_types, token_z, bias=False
        )
        # Small init to act as a gentle bias.
        nn.init.normal_(self.pairwise_bias_embedding.weight, mean=0.0, std=0.01)

    # ------------------------- helpers -------------------------

    @staticmethod
    def _get_token_bonds(feats: Dict[str, Any]) -> torch.Tensor:
        token_bonds = feats["token_bonds"]
        if token_bonds.dim() == 4 and token_bonds.size(-1) == 1:
            token_bonds = token_bonds.squeeze(-1)
        if token_bonds.dim() != 3:
            raise ValueError(
                f"token_bonds must be [B, L, L] (or [B, L, L, 1]), got {token_bonds.shape}"
            )
        return token_bonds > 0.5

    # ------------------------- forward -------------------------

    def forward(self, z: torch.Tensor, feats: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            z: Pairwise embeddings [B, L, L, token_z]
            feats: feature dictionary
        Returns:
            z with anomeric biases added ONLY to Sugar-Sugar bonds.
        """
        if z.dim() != 4:
            raise ValueError(f"z must be [B, L, L, Cz], got {z.shape}")

        B, L, L2, Cz = z.shape
        device = z.device

        token_bonds = self._get_token_bonds(feats)  # [B, L, L]
        token_to_mono_idx = feats["token_to_mono_idx"]  # [B, L]

        # ----------------------------------------------------------------------
        # 1. Identify Entity Types (Sugar vs Protein/Other)
        # ----------------------------------------------------------------------
        # A token is part of a sugar if it has a valid mono_idx (!= -1).
        # Conversely, Protein/Ligand/Water tokens have mono_idx == -1.
        is_sugar_token = token_to_mono_idx != -1  # [B, L]
        
        # ----------------------------------------------------------------------
        # 2. Identify Connectivity to Non-Sugar (Protein)
        # ----------------------------------------------------------------------
        # We need to know if any token is covalently bonded to a Protein.
        # This is critical for O-linked glycans where the bond is C1 -> O1 -> Protein.
        # We must NOT bias C1->O1 if O1 is attached to a Protein.
        
        is_non_sugar_token = ~is_sugar_token # [B, L]
        
        # Shape: [B, L] -> True if token i is bonded to ANY non-sugar token
        # logic: (bond exists) AND (target is non-sugar)
        connected_to_protein = (token_bonds & is_non_sugar_token.unsqueeze(1)).any(dim=2)

        # ----------------------------------------------------------------------
        # 3. Topology & Element Masks
        # ----------------------------------------------------------------------
        mono_i = token_to_mono_idx.unsqueeze(2)  # [B, L, 1]
        mono_j = token_to_mono_idx.unsqueeze(1)  # [B, 1, L]
        
        same_entity = (mono_i == mono_j) & is_sugar_token.unsqueeze(2)
        different_entity = (mono_i != mono_j) & is_sugar_token.unsqueeze(2) & is_sugar_token.unsqueeze(1)

        # Get Element Types
        ref_elem_oh = feats["ref_element"].float()
        token_to_atom = feats["token_to_rep_atom"].float()
        token_elem_oh = torch.einsum("bta,bae->bte", token_to_atom, ref_elem_oh)
        token_elem_idx = token_elem_oh.argmax(dim=-1)
        
        OXYGEN_INDEX = 8
        is_oxygen = token_elem_idx == OXYGEN_INDEX  # [B, L]

        # ----------------------------------------------------------------------
        # 4. Identify Anomeric Carbon (C1 or C2) via Ring Topology
        # ----------------------------------------------------------------------
        # Convert ref_atom_name_chars [B, A, 4, 64] -> Token names [B, L, 4, 64]
        ref_names = feats["ref_atom_name_chars"].float()
        token_names = torch.einsum("bta,banv->btnv", token_to_atom, ref_names)
        token_name_indices = token_names.argmax(dim=-1) # [B, L, 4]

        # Targets (ASCII - 32): 'C'=35, '1'=17, '2'=18, Pad=0
        c1_target = torch.tensor([35, 17, 0, 0], device=device).view(1, 1, 4)
        c2_target = torch.tensor([35, 18, 0, 0], device=device).view(1, 1, 4)

        is_named_c1 = (token_name_indices == c1_target).all(dim=-1) # [B, L]
        is_named_c2 = (token_name_indices == c2_target).all(dim=-1) # [B, L]

        intra_bonds = token_bonds & same_entity
        intra_degree = intra_bonds.float().sum(dim=2)
        
        # Ring Oxygen: Oxygen with >= 2 bonds within residue
        is_ring_oxygen = is_oxygen & (intra_degree >= 2)

        # Find which Carbon is bonded to the Ring Oxygen
        connected_to_ring_o = (intra_bonds & is_ring_oxygen.unsqueeze(1)).any(dim=2)
        is_anomeric_carbon = (is_named_c1 | is_named_c2) & connected_to_ring_o

        # ----------------------------------------------------------------------
        # 5. Determine Target Bond Partners (Strictly Glycan-Only)
        # ----------------------------------------------------------------------
        
        # --- Type A: Inter-residue Glycosidic Bond (Condensed Representation) ---
        # C1 -> O4' (Next Sugar). 
        # CRITICAL FIX: We strictly enforce that the target (different_entity) is a SUGAR.
        # This implicitly excludes N-linked bonds (C1->Asn) because Asn is not a sugar.
        mask_inter = token_bonds & is_anomeric_carbon.unsqueeze(2) & different_entity

        # --- Type B: Intra-residue Exocyclic Bond (Bridging Atom) ---
        # C1 -> O1.
        # CRITICAL FIX: We enforce that the neighbor (O1) is NOT connected to a protein.
        # If O1 is connected to Ser/Thr, connected_to_protein[neighbor] is True.
        # We mask these out to prevent biasing the O-linked interface.

        # [B, L_anom, L_neighbor]
        anomeric_neighbors = intra_bonds & is_anomeric_carbon.unsqueeze(2)
        
        # Exocyclic Filter: Degree 1 within the residue
        is_exocyclic = intra_degree == 1
        
        # FILTER: Neighbor must NOT be bonded to protein
        is_pure_sugar_neighbor = is_exocyclic & (~connected_to_protein)

        exocyclic_candidates = anomeric_neighbors & is_pure_sugar_neighbor.unsqueeze(1) 

        # Weight by Atomic Number (O > C > H) to find heaviest
        atomic_weights = token_elem_idx.float() # [B, L]
        candidate_weights = exocyclic_candidates * atomic_weights.unsqueeze(1) 
        
        # Find the max weight neighbor for each anomeric carbon
        max_vals, max_indices = candidate_weights.max(dim=2) # [B, L_anom]
        
        # Reconstruct mask
        col_indices = torch.arange(L, device=device).view(1, 1, L).expand(B, L, L)
        selected_neighbor_indices = max_indices.unsqueeze(2) 
        
        mask_intra = (
            is_anomeric_carbon.unsqueeze(2) & 
            (max_vals.unsqueeze(2) > 0) & 
            (col_indices == selected_neighbor_indices)
        )

        # Combine Masks
        final_bias_mask = mask_inter | mask_intra

        # ----------------------------------------------------------------------
        # 6. Compute and Apply Bias
        # ----------------------------------------------------------------------
        anomeric_config = feats["mono_anomeric"]  # [B, L, num_anomeric_types]
        pairwise_bias = self.pairwise_bias_embedding(anomeric_config)  # [B, L, Cz]
        pairwise_bias = pairwise_bias.unsqueeze(2)  # [B, L, 1, Cz]

        mask_expanded = final_bias_mask.unsqueeze(-1).to(pairwise_bias.dtype)
        bias_to_add = pairwise_bias * mask_expanded  # [B, L, L, Cz]

        # Symmetrize
        z = z + bias_to_add + bias_to_add.transpose(1, 2)

        # DDP-safe no-op
        if not final_bias_mask.any():
            z = z + sum(p.sum() * 0.0 for p in self.parameters())

        return z
        
#############################################################################################################
#############################################################################################################
#HELPERS
#############################################################################################################
#############################################################################################################

class SugarPairformerLayer(nn.Module):
    """
    A single layer of the SugarPairformer refinery. This version is simplified
    and assumes its inputs (s, z) correspond to a single, isolated glycan.
    Therefore, no internal masking is required.
    """
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_heads: int,
        pairwise_num_heads: int,
        dropout: float = 0.25,
        pairwise_head_width: int = 32,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.attention = AttentionPairBias(token_s, token_z, num_heads)
        self.transition_s = Transition(token_s, token_s * 4)

        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.transition_z = Transition(token_z, token_z * 4)
        
    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor, 
        chunk_size_tri_attn: int | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for an isolated glycan.
        """
        padding_pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        
        # --- Z (Pairwise) Updates ---
        z = z + get_dropout_mask(self.dropout, z, self.training) * self.tri_mul_out(z, mask=padding_pair_mask)
        z = z + get_dropout_mask(self.dropout, z, self.training) * self.tri_mul_in(z, mask=padding_pair_mask)
        z = z + get_dropout_mask(self.dropout, z, self.training) * self.tri_att_start(z, mask=padding_pair_mask, chunk_size=chunk_size_tri_attn)
        z = z + get_dropout_mask(self.dropout, z, self.training, columnwise=True) * self.tri_att_end(z, mask=padding_pair_mask, chunk_size=chunk_size_tri_attn)
        z = z + self.transition_z(z)

        # --- S (Single) Updates ---
        s = s + self.attention(s, z, mask=mask)
        s = s + self.transition_s(s)
            
        return s, z


class SugarPairformerModule(nn.Module):
    """
    A stack of SugarPairformer layers. This module processes a single,
    isolated glycan tensor.
    """
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        num_heads: int,
        pairwise_num_heads: int,
        dropout: float,
        pairwise_head_width: int,
        activation_checkpointing: bool = True,
        offload_to_cpu: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = SugarPairformerLayer(
                token_s,
                token_z,
                num_heads,
                pairwise_num_heads,
                dropout,
                pairwise_head_width,
            )
            if activation_checkpointing:
                layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
            self.layers.append(layer)

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor, # Pass the standard padding mask down.
    ) -> Tuple[Tensor, Tensor]:

        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_size_tri_attn = 128
            else:
                chunk_size_tri_attn = 512
        else:
            chunk_size_tri_attn = None

        for layer in self.layers:
            s, z = layer(
                s, 
                z, 
                mask=mask, 
                chunk_size_tri_attn=chunk_size_tri_attn
            )

        return s, z

class SugarPairformer(nn.Module):
    """
    (Corrected for DDP)
    A specialist module that refines glycan representations.
    This version uses a gather-batch-process-scatter approach to guarantee
    mathematical isolation for each glycan chain while remaining compatible
    with DDP and activation checkpointing by making only a single call
    to its core processing stack.
    """
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int = 4,
        num_heads: int = 16,
        pairwise_num_heads: int = 8,
        pairwise_head_width: int = 16,
        dropout: float = 0.25,
        activation_checkpointing: bool = True,
        offload_to_cpu: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.sugar_pairformer_stack = SugarPairformerModule(
            token_s=token_s,
            token_z=token_z,
            num_blocks=num_blocks,
            num_heads=num_heads,
            pairwise_num_heads=pairwise_num_heads,
            pairwise_head_width=pairwise_head_width,
            dropout=dropout,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

    def forward(self, s_input: Tensor, z_input: Tensor, feats: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """
        Extracts glycan representations, processes them in a single batched call,
        and scatters the results back. This module is designed to operate only on
        tokens identified as part of a monosaccharide, leaving all other token
        representations mathematically unchanged. If no glycans are present in a batch,
        it uses a parameter-summing mechanism to ensure compatibility with Distributed
        Data Parallel (DDP) training without performing any data processing.
        """
        B, N, _ = s_input.shape
        
        is_glycan_token = feats['is_monosaccharide'].squeeze(-1).bool()
        
        # Initialize lists to gather glycan-specific data
        glycan_s_list, glycan_z_list, glycan_mask_list = [], [], []
        scatter_map = []

        # 1. GATHER (Only if there are any glycans in the batch)
        if torch.any(is_glycan_token):
            s_out = s_input.clone()
            z_out = z_input.clone()
            asym_id_all: Tensor = feats['asym_id']
            pad_mask_all: Tensor = feats['token_pad_mask']

            for b in range(B):
                is_glycan_token_b = is_glycan_token[b]
                if not torch.any(is_glycan_token_b): continue
                token_asym_id = asym_id_all[b]
                unique_chain_ids = torch.unique(token_asym_id[is_glycan_token_b])
                for chain_id in unique_chain_ids:
                    glycan_indices = torch.where((token_asym_id == chain_id) & is_glycan_token_b)[0]
                    if glycan_indices.numel() == 0: continue
                    
                    glycan_s_list.append(s_input[b, glycan_indices, :])
                    glycan_z_list.append(z_input[b, glycan_indices][:, glycan_indices])
                    glycan_mask_list.append(pad_mask_all[b, glycan_indices])
                    scatter_map.append({'batch_idx': b, 'indices': glycan_indices})

        # If no glycan chains were gathered across the entire batch, apply the DDP-safe exit.
        if not glycan_s_list:
            # DDP-safe exit: "touch" all parameters in the stack by summing them.
            # This adds them to the computation graph without running a forward pass.
            # The result is multiplied by 0.0, ensuring no mathematical impact.
            dummy_loss = 0.0
            for p in self.sugar_pairformer_stack.parameters():
                dummy_loss += p.sum()
            
            return s_input + (dummy_loss * 0.0), z_input + (dummy_loss * 0.0)

        # 2. PAD & BATCH
        max_glycan_len = max(s.shape[0] for s in glycan_s_list)
        s_padded_list, z_padded_list, mask_padded_list = [], [], []
        for s, z, m in zip(glycan_s_list, glycan_z_list, glycan_mask_list):
            pad_len = max_glycan_len - s.shape[0]
            s_padded = F.pad(s, (0, 0, 0, pad_len)) if pad_len > 0 else s
            z_padded = F.pad(z, (0, 0, 0, pad_len, 0, pad_len)) if pad_len > 0 else z
            m_padded = F.pad(m, (0, pad_len)) if pad_len > 0 else m
            s_padded_list.append(s_padded)
            z_padded_list.append(z_padded)
            mask_padded_list.append(m_padded)
        s_batch = torch.stack(s_padded_list, dim=0)
        z_batch = torch.stack(z_padded_list, dim=0)
        mask_batch = torch.stack(mask_padded_list, dim=0)

        # 3. SINGLE PROCESS CALL
        s_refined_batch, z_refined_batch = self.sugar_pairformer_stack(
            s=s_batch, z=z_batch, mask=mask_batch.float()
        )

        # 4. UNBATCH & SCATTER
        for i, meta in enumerate(scatter_map):
            b, original_indices = meta['batch_idx'], meta['indices']
            original_len = len(original_indices)
            s_refined = s_refined_batch[i, :original_len, :]
            z_refined = z_refined_batch[i, :original_len, :original_len, :]
            s_out[b, original_indices, :] = s_refined
            rows, cols = torch.meshgrid(original_indices, original_indices, indexing='ij')
            z_out[b, rows, cols, :] = z_refined
            
        return s_out, z_out
