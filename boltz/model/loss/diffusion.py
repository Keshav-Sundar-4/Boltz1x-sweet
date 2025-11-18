# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from einops import einsum
import torch
import torch.nn.functional as F
import sys
from typing import Dict, List, Any, Tuple, Optional

def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """

    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords.detach_()

    return aligned_coords


def smooth_lddt_loss(
    pred_coords,
    true_coords,
    is_nucleotide,
    coords_mask,
    nucleic_acid_cutoff: float = 30.0,
    other_cutoff: float = 15.0,
    multiplicity: int = 1,
):
    """Compute weighted alignment.

    Parameters
    ----------
    pred_coords: torch.Tensor
        The predicted atom coordinates
    true_coords: torch.Tensor
        The ground truth atom coordinates
    is_nucleotide: torch.Tensor
        The weights for alignment
    coords_mask: torch.Tensor
        The atoms mask
    nucleic_acid_cutoff: float
        The nucleic acid cutoff
    other_cutoff: float
        The non nucleic acid cutoff
    multiplicity: int
        The multiplicity
    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """
    B, N, _ = true_coords.shape
    true_dists = torch.cdist(true_coords, true_coords)
    is_nucleotide = is_nucleotide.repeat_interleave(multiplicity, 0)

    coords_mask = coords_mask.repeat_interleave(multiplicity, 0)
    is_nucleotide_pair = is_nucleotide.unsqueeze(-1).expand(
        -1, -1, is_nucleotide.shape[-1]
    )

    mask = (
        is_nucleotide_pair * (true_dists < nucleic_acid_cutoff).float()
        + (1 - is_nucleotide_pair) * (true_dists < other_cutoff).float()
    )
    mask = mask * (1 - torch.eye(pred_coords.shape[1], device=pred_coords.device))
    mask = mask * (coords_mask.unsqueeze(-1) * coords_mask.unsqueeze(-2))

    # Compute distances between all pairs of atoms
    pred_dists = torch.cdist(pred_coords, pred_coords)
    dist_diff = torch.abs(true_dists - pred_dists)

    # Compute epsilon values
    eps = (
        (
            (
                F.sigmoid(0.5 - dist_diff)
                + F.sigmoid(1.0 - dist_diff)
                + F.sigmoid(2.0 - dist_diff)
                + F.sigmoid(4.0 - dist_diff)
            )
            / 4.0
        )
        .view(multiplicity, B // multiplicity, N, N)
        .mean(dim=0)
    )

    # Calculate masked averaging
    eps = eps.repeat_interleave(multiplicity, 0)
    num = (eps * mask).sum(dim=(-1, -2))
    den = mask.sum(dim=(-1, -2)).clamp(min=1)
    lddt = num / den

    return 1.0 - lddt.mean()

# Ported helper function to avoid circular imports.
def _decode_one_hot_to_str(one_hot_encoded_name: torch.Tensor) -> str:
    """Decodes a one-hot encoded name from ref_atom_name_chars."""
    # Find the integer index for each of the 4 character positions
    integer_indices = torch.argmax(one_hot_encoded_name, dim=-1)
    # Add 32 to convert back to ASCII character codes
    char_codes = [idx.item() + 32 for idx in integer_indices]
    # Filter out null characters (code 32) and join
    return "".join([chr(c) for c in char_codes if c > 32]).strip()

def compute_monosaccharide_mse_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    feats: Dict[str, Any],
    multiplicity: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Per-Mono MSE with local rigid alignment in ATOM space.

    For each logical batch b (pre-multiplicity):
      1) Build N sets: atoms belonging to monosaccharide k, PLUS any glycosidic linker
         heteroatom(s) (O / N / S) bridging k to a neighboring mono (found via token bonds).
         * Linker heteroatom is double-counted in alignments (appears in both donor & acceptor sets).
      2) For each set, independently rigid-align (pred -> true) and accumulate the SSE
         ONLY over atoms in that set.
      3) For a (b, m) sample, sum SSE over sets and divide by the number of UNIQUE glycan atoms
         (linker heteroatom counted once here). If no glycan atoms, mark sample invalid.
    Returns per-sample values so the caller can apply per-sample sigma weighting.

    Returns
    -------
    mono_mse_per_sample : torch.Tensor [Bm]
        Per-sample (b,m) loss = (sum over set SSEs) / (# unique glycan atoms in b).
        Zero for invalid samples (no glycan atoms).
    mono_valid_mask     : torch.Tensor [Bm] (bool)
        True where a sample has at least one glycan atom (valid denominator).
    total_alignments    : int
        Number of local set alignments performed.
    """
    device = pred_coords.device
    dtype  = pred_coords.dtype

    # ----- 0) Pull features -----
    atom_mono_idx: torch.Tensor      = feats["atom_mono_idx"]              
    atom_to_token_1hot: torch.Tensor = feats["atom_to_token"]              
    ref_element_1hot: torch.Tensor   = feats["ref_element"]                
    atom_present_mask: torch.Tensor  = feats.get("atom_resolved_mask", None)  

    token_to_mono_idx: torch.Tensor  = feats["token_to_mono_idx"]          
    token_bonds: torch.Tensor        = feats["token_bonds"].squeeze(-1).bool()  

    Bm, A, _ = pred_coords.shape          
    B0 = atom_mono_idx.shape[0]
    assert Bm % multiplicity == 0 and (Bm // multiplicity) == B0, \
        "Batch/multiplicity mismatch: pred_coords batch must equal B0 * multiplicity."

    atom_to_token = atom_to_token_1hot.argmax(dim=-1)         
    atomic_number = ref_element_1hot.argmax(dim=-1)           
    is_oxygen_atom   = (atomic_number == 8)   # O-linker
    is_nitrogen_atom = (atomic_number == 7)   # N-linker (e.g., N-glycosidic)
    is_sulfur_atom   = (atomic_number == 16)  # S-linker (rare)
    is_linker_atom   = is_oxygen_atom | is_nitrogen_atom | is_sulfur_atom

    if atom_present_mask is None:
        atom_present_mask = torch.ones((B0, A), dtype=torch.bool, device=device)

    mono_mse_per_sample = torch.zeros((Bm,), device=device, dtype=dtype)
    mono_valid_mask     = torch.zeros((Bm,), device=device, dtype=torch.bool)
    total_alignments    = 0

    def _build_sets_for_batch(b: int):
        mono_idx_b  = atom_mono_idx[b]                       
        atom2tok_b  = atom_to_token[b]                       
        tok2mono_b  = token_to_mono_idx[b]                   
        bonds_b     = token_bonds[b]                         
        linker_b    = is_linker_atom[b] & atom_present_mask[b]
        present_b   = atom_present_mask[b]

        denom_mask = (mono_idx_b != -1) & present_b

        mono_ids = torch.unique(mono_idx_b)
        mono_ids = mono_ids[(mono_ids != -1)]

        if mono_ids.numel() == 0:
            return [], [], denom_mask

        T = tok2mono_b.shape[0]
        mono_i = tok2mono_b.unsqueeze(1).expand(T, T)
        mono_j = tok2mono_b.unsqueeze(0).expand(T, T)
        cross_mono_edges = bonds_b & (mono_i != -1) & (mono_j != -1) & (mono_i != mono_j)
        tok_has_cross_mono_neighbor = cross_mono_edges.any(dim=1)  

        set_masks, mono_id_list = [], []

        for m_id in mono_ids.tolist():
            core_atom_mask = (mono_idx_b == m_id) & present_b
            if not core_atom_mask.any():
                continue

            core_tokens = torch.unique(atom2tok_b[core_atom_mask])  

            if core_tokens.numel() > 0:
                neighbor_tok_mask = bonds_b[core_tokens].any(dim=0)                  
                neighbor_tok_mask = neighbor_tok_mask & cross_mono_edges[core_tokens].any(dim=0)
            else:
                neighbor_tok_mask = torch.zeros((T,), dtype=torch.bool, device=device)

            atom_tok_in_core = torch.isin(atom2tok_b, core_tokens) if core_tokens.numel() > 0 else torch.zeros_like(atom2tok_b, dtype=torch.bool)
            nbr_tok_indices = torch.nonzero(neighbor_tok_mask, as_tuple=False).flatten()
            atom_tok_in_nbrs = torch.isin(atom2tok_b, nbr_tok_indices) if nbr_tok_indices.numel() > 0 else torch.zeros_like(atom2tok_b, dtype=torch.bool)

            linker_on_core_cross = (
                linker_b & atom_tok_in_core & tok_has_cross_mono_neighbor[atom2tok_b]
            )
            linker_on_nbr_side   = linker_b & atom_tok_in_nbrs

            glyco_linker_mask = linker_on_core_cross | linker_on_nbr_side

            final_mask = core_atom_mask | glyco_linker_mask

            if final_mask.sum() >= 3:
                set_masks.append(final_mask)
                mono_id_list.append(m_id)

        return mono_id_list, set_masks, denom_mask

    for b in range(B0):
        mono_ids, masks, denom_mask = _build_sets_for_batch(b)
        denom_count = int(denom_mask.sum().item())
        valid = denom_count > 0

        for m in range(multiplicity):
            sample_idx = b * multiplicity + m

            if not valid:
                mono_mse_per_sample[sample_idx] = torch.tensor(0.0, device=device, dtype=dtype)
                mono_valid_mask[sample_idx]     = False
                continue

            true_b = true_coords[sample_idx]  
            pred_b = pred_coords[sample_idx]  

            sample_sse = torch.tensor(0.0, device=device, dtype=torch.float32)

            for mono_id, mask in zip(mono_ids, masks):
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                K = idx.shape[0]

                tc = true_b[idx].unsqueeze(0).float()  
                pc = pred_b[idx].unsqueeze(0).float()  

                aligned = weighted_rigid_align(
                    true_coords=tc,   
                    pred_coords=pc,
                    weights=torch.ones((1, K), device=device, dtype=torch.float32),
                    mask=torch.ones((1, K), device=device, dtype=torch.float32),
                ).squeeze(0)  

                sse = ((pc.squeeze(0) - aligned) ** 2).sum()  
                sample_sse = sample_sse + sse
                total_alignments += 1

            sample_loss = sample_sse / max(1, 3 * denom_count)

            mono_mse_per_sample[sample_idx] = sample_loss.to(dtype)
            mono_valid_mask[sample_idx]     = True

    return mono_mse_per_sample, mono_valid_mask, total_alignments

def Linkage_Loss(
    feats: Dict[str, torch.Tensor],
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor, # This should be the UN-ALIGNED true coordinates
    loss_weights: torch.Tensor,
    multiplicity: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Calculates a distance-based loss for glycosylation bonds.
    Returns the final scalar loss tensor if any glycosylation sites are present
    in the batch, otherwise returns None.
    """
    from boltz.model.modules.sugar_trunk import _get_glycosylation_features
    glyco_feats = _get_glycosylation_features(feats)

    # 1. Get the TOKEN bond pairs and their batch indices from the helper function.
    token_bond_pairs = glyco_feats.get("t_glycosylation_indices")
    batch_indices = glyco_feats.get("t_batch_idx")

    # If no glycosylation sites are found in the entire batch, return None.
    # The caller (compute_loss) is responsible for handling this.
    if token_bond_pairs is None or token_bond_pairs.numel() == 0:
        return None

    # --- Section A: Convert Token Indices to Atom Indices ---
    B_orig = feats['atom_pad_mask'].shape[0]
    token_to_rep_atom_idx = feats["token_to_rep_atom"].argmax(-1)
    p_token_indices = token_bond_pairs[:, 0]
    g_token_indices = token_bond_pairs[:, 1]
    p_atom_indices = token_to_rep_atom_idx[batch_indices, p_token_indices]
    g_atom_indices = token_to_rep_atom_idx[batch_indices, g_token_indices]

    # --- Section B: Index Preparation for Multiplicity ---
    num_bonds_in_batch = token_bond_pairs.shape[0]
    offset = torch.arange(0, multiplicity, device=device) * B_orig
    offset_expanded = offset.repeat_interleave(num_bonds_in_batch)
    final_batch_indices = batch_indices.repeat(multiplicity) + offset_expanded
    final_protein_atom_indices = p_atom_indices.repeat(multiplicity)
    final_glycan_atom_indices = g_atom_indices.repeat(multiplicity)

    # --- Section C: Dense Tensor Gathering ---
    p_true_coords = true_coords[final_batch_indices, final_protein_atom_indices]
    g_true_coords = true_coords[final_batch_indices, final_glycan_atom_indices]
    p_pred_coords = pred_coords[final_batch_indices, final_protein_atom_indices]
    g_pred_coords = pred_coords[final_batch_indices, final_glycan_atom_indices]

    # --- Section D: Loss Calculation on Dense Tensors ---
    true_dist = torch.linalg.norm(p_true_coords - g_true_coords, dim=-1)
    pred_dist = torch.linalg.norm(p_pred_coords - g_pred_coords, dim=-1)
    per_bond_loss = (pred_dist - true_dist) ** 2

    # --- Section E: Aggregation ---
    B_mult = pred_coords.shape[0]
    dist_loss_per_item = torch.zeros(B_mult, device=device).scatter_add_(0, final_batch_indices, per_bond_loss)
    ones = torch.ones_like(per_bond_loss)
    counts = torch.zeros(B_mult, device=device).scatter_add_(0, final_batch_indices, ones)
    avg_dist_loss_per_item = dist_loss_per_item / counts.clamp(min=1)

    # --- Section F: Final Sigma-Weighted Mean ---
    valid_items_mask = counts > 0
    # This check is now redundant due to the check at the top, but kept for safety.
    if not valid_items_mask.any():
        return None

    final_loss = (avg_dist_loss_per_item[valid_items_mask] * loss_weights[valid_items_mask]).mean()

    # Handle potential NaN from division by zero if all weights are zero for valid items
    return final_loss if not torch.isnan(final_loss) else torch.tensor(0.0, device=device)

