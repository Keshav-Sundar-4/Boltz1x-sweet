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

def Glyco_AA_MSE_Loss(
    feats: Dict[str, torch.Tensor],
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    loss_weights: torch.Tensor,
    multiplicity: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Calculates the Mean Squared Error (MSE) of the glycosylated amino acid and the 
    attached glycan atom after performing an independent rigid alignment.
    
    This enforces the correct internal geometry of the entire amino acid and the 
    linkage bond simultaneously, preventing backbone warping.
    """
    # Robustly determine original batch size (feats tensors are [B, N, ...])
    B_orig = feats['atom_to_token'].shape[0]
    
    raw_sites_tensors = feats.get('raw_glycosylation_sites')
    # Pre-fetch resolved mask: [B, N]
    atom_resolved_mask_orig = feats["atom_resolved_mask"]

    if raw_sites_tensors is None:
        return None

    total_mse_loss = torch.tensor(0.0, device=device)
    site_count = torch.tensor(0.0, device=device)

    # Iterate through each structure in the batch 'b'
    for b in range(B_orig):
        sites_tensor_b = raw_sites_tensors[b]
        if sites_tensor_b is None or sites_tensor_b.numel() == 0:
            continue

        # Get atom mapping data for this batch item
        # atom_to_token: [N, L] -> argmax -> [N]
        atom_to_token = feats["atom_to_token"][b].argmax(-1)
        token_asym_ids = feats["asym_id"][b]
        token_res_indices = feats["residue_index"][b]
        ref_name_chars = feats["ref_atom_name_chars"][b]

        # Map atoms to Chain/Residue
        atom_asym_ids = torch.gather(token_asym_ids, 0, atom_to_token)
        atom_res_indices = torch.gather(token_res_indices, 0, atom_to_token)

        for site_data_tensor in sites_tensor_b:
            p_chain_id = site_data_tensor[0].item()
            p_res_id = site_data_tensor[1].item()
            
            g_chain_id = site_data_tensor[6].item()
            g_res_id = site_data_tensor[7].item()
            g_name = _decode_int_to_str(site_data_tensor[8:12]).upper()

            # 1. Get all atoms of the Glycosylated Amino Acid
            p_res_mask = (atom_asym_ids == p_chain_id) & (atom_res_indices == p_res_id)
            p_atoms_indices = torch.where(p_res_mask)[0]

            if p_atoms_indices.numel() == 0:
                continue

            # 2. Get the singular covalently bonded Glycan Atom (e.g., C1)
            g_res_mask = (atom_asym_ids == g_chain_id) & (atom_res_indices == g_res_id)
            g_atoms_in_res = torch.where(g_res_mask)[0]
            
            g_atom_index = -1
            for g_idx in g_atoms_in_res:
                decoded_name = _decode_one_hot_to_str(ref_name_chars[g_idx]).upper()
                if decoded_name == g_name:
                    g_atom_index = g_idx.item()
                    break
            
            if g_atom_index == -1:
                continue

            # 3. Combine Indices: [All AA Atoms] + [Linkage Glycan Atom]
            g_idx_tensor = torch.tensor([g_atom_index], device=device, dtype=torch.long)
            subset_indices = torch.cat([p_atoms_indices, g_idx_tensor])

            # 4. Handle Multiplicity and Extraction
            # The coordinate tensors have shape [B*M, N, 3]
            # The rows corresponding to batch 'b' are [b*M, ..., (b+1)*M - 1]
            batch_start = b * multiplicity
            batch_end = (b + 1) * multiplicity

            # Extract Coords: Shape [M, N_subset, 3]
            # We slice the batch dim [batch_start:batch_end] and select specific atoms [subset_indices]
            curr_pred = pred_coords[batch_start:batch_end][:, subset_indices, :]
            curr_true = true_coords[batch_start:batch_end][:, subset_indices, :]
            
            # Extract Masks: Shape [N_subset] -> Expand to [M, N_subset]
            curr_mask = atom_resolved_mask_orig[b, subset_indices]
            curr_mask = curr_mask.unsqueeze(0).expand(multiplicity, -1)
            
            # Extract Sigma Weights: Shape [M]
            curr_weights = loss_weights[batch_start:batch_end]
            
            # 5. Independent Rigid Alignment
            # Align 'True' onto 'Pred' using the validity mask as weights
            # Shape: [M, N_subset, 3]
            aligned_true = weighted_rigid_align(
                curr_true, 
                curr_pred, 
                curr_mask, 
                curr_mask
            )
            
            # 6. Calculate MSE on Aligned Subsets
            diff = aligned_true - curr_pred
            mse_per_atom = (diff ** 2).sum(dim=-1) # [M, N_subset]
            
            # Weighted average over atoms (ignoring unresolved atoms)
            sum_sq_error = (mse_per_atom * curr_mask).sum(dim=-1) # [M]
            num_valid_atoms = curr_mask.sum(dim=-1).clamp(min=1e-6) # [M]
            
            masked_mse_per_sample = sum_sq_error / num_valid_atoms
            
            # 7. Apply Sigma Weighting and Accumulate
            # Average over the M samples for this site
            weighted_site_loss = (masked_mse_per_sample * curr_weights).mean()
            
            total_mse_loss += weighted_site_loss
            site_count += 1

    if site_count == 0:
        return None

    return total_mse_loss / site_count

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

