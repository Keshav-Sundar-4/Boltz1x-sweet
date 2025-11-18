from typing import Dict, Iterator, List, Optional

import numpy as np
from numpy.random import RandomState

from boltz.data import const
from boltz.data.types import ChainInfo, InterfaceInfo, Record
from boltz.data.sample.sampler import Sample, Sampler


def get_chain_cluster(chain: ChainInfo, record: Record) -> str:  # noqa: ARG001
    """Get the cluster id for a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain id to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the chain.

    """
    return chain.cluster_id


def get_interface_cluster(interface: InterfaceInfo, record: Record) -> str:
    """Get the cluster id for an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the cluster id for.
    record : Record
        The record the interface is part of.

    Returns
    -------
    str
        The cluster id of the interface.

    """
    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    cluster_1 = str(chain1.cluster_id)
    cluster_2 = str(chain2.cluster_id)

    cluster_id = (cluster_1, cluster_2)
    cluster_id = tuple(sorted(cluster_id))

    return cluster_id


def get_chain_weight(
    chain: ChainInfo,
    record: Record,  # noqa: ARG001
    clusters: Dict[str, int],
    beta_chain: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Get the weight of a chain.

    Parameters
    ----------
    chain : ChainInfo
        The chain to get the weight for.
    record : Record
        The record the chain is part of.
    clusters : Dict[str, int]
        The cluster sizes.
    beta_chain : float
        The beta value for chains.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the chain.

    """
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    weight = beta_chain / clusters[chain.cluster_id]
    if chain.mol_type == prot_id:
        weight *= alpha_prot
    elif chain.mol_type in [rna_id, dna_id]:
        weight *= alpha_nucl
    elif chain.mol_type == ligand_id:
        weight *= alpha_ligand

    return weight


def get_interface_weight(
    interface: InterfaceInfo,
    record: Record,
    clusters: Dict[str, int],
    beta_interface: float,
    alpha_prot: float,
    alpha_nucl: float,
    alpha_ligand: float,
) -> float:
    """Get the weight of an interface.

    Parameters
    ----------
    interface : InterfaceInfo
        The interface to get the weight for.
    record : Record
        The record the interface is part of.
    clusters : Dict[str, int]
        The cluster sizes.
    beta_interface : float
        The beta value for interfaces.
    alpha_prot : float
        The alpha value for proteins.
    alpha_nucl : float
        The alpha value for nucleic acids.
    alpha_ligand : float
        The alpha value for ligands.

    Returns
    -------
    float
        The weight of the interface.

    """
    prot_id = const.chain_type_ids["PROTEIN"]
    rna_id = const.chain_type_ids["RNA"]
    dna_id = const.chain_type_ids["DNA"]
    ligand_id = const.chain_type_ids["NONPOLYMER"]

    chain1 = record.chains[interface.chain_1]
    chain2 = record.chains[interface.chain_2]

    n_prot = (chain1.mol_type) == prot_id
    n_nuc = chain1.mol_type in [rna_id, dna_id]
    n_ligand = chain1.mol_type == ligand_id

    n_prot += chain2.mol_type == prot_id
    n_nuc += chain2.mol_type in [rna_id, dna_id]
    n_ligand += chain2.mol_type == ligand_id

    weight = beta_interface / clusters[get_interface_cluster(interface, record)]
    weight *= alpha_prot * n_prot + alpha_nucl * n_nuc + alpha_ligand * n_ligand
    return weight


class ClusterSampler(Sampler):
    """The weighted sampling approach, as described in AF3.

    Each chain / interface is given a weight according
    to the following formula, and sampled accordingly:

    w = b / n_clust *(a_prot * n_prot + a_nuc * n_nuc
        + a_ligand * n_ligand)

    This sampler includes a modification for glycan modeling. If the dataset
    contains both contextual (e.g., glycoprotein) and non-contextual (e.g.,
    free glycan) samples, it enforces a specific sampling ratio between them,
    controlled by `glycan_context_prob`. If the dataset does not contain both
    types of samples, it reverts to the standard weighted sampling across all
    available data.
    """

    def __init__(
        self,
        alpha_prot: float = 3.0,
        alpha_nucl: float = 3.0,
        alpha_ligand: float = 1.0,
        beta_chain: float = 0.5,
        beta_interface: float = 1.0,
        glycan_context_prob: Optional[float] = 0.70,
    ) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        alpha_prot : float, optional
            The alpha value for proteins.
        alpha_nucl : float, optional
            The alpha value for nucleic acids.
        alpha_ligand : float, optional
            The alpha value for ligands.
        beta_chain : float, optional
            The beta value for chains.
        beta_interface : float, optional
            The beta value for interfaces.
        glycan_context_prob : float, optional
            The desired probability of sampling a contextual (glycoprotein/lectin)
            file. The probability of sampling a free-floating glycan will be
            (1 - glycan_context_prob). This is only active if the dataset
            contains both types of files. Defaults to 0.70.
            If set to ``None``, the two-pool logic is disabled and the sampler
            always falls back to single-pool weighted sampling.


        """
        self.alpha_prot = alpha_prot
        self.alpha_nucl = alpha_nucl
        self.alpha_ligand = alpha_ligand
        self.beta_chain = beta_chain
        self.beta_interface = beta_interface
        self.glycan_context_prob = glycan_context_prob

    def sample(self, records: List[Record], random: RandomState) -> Iterator[Sample]:  # noqa: C901, PLR0912
        """Sample a structure from the dataset infinitely.

        If the dataset contains both contextual (glycoprotein/lectin) and
        non-contextual (free glycan) files, it uses a two-pool sampling
        strategy to enforce a specific ratio. Otherwise, it reverts to the
        original single-pool weighted sampling.

        Parameters
        ----------
        records : List[Record]
            The records to sample from.
        random : RandomState
            The random state for reproducibility.

        Yields
        ------
        Sample
            A data sample.

        """
        # Compute chain cluster sizes
        chain_clusters: Dict[str, int] = {}
        for record in records:
            for chain in record.chains:
                if not chain.valid:
                    continue
                cluster_id = get_chain_cluster(chain, record)
                if cluster_id not in chain_clusters:
                    chain_clusters[cluster_id] = 0
                chain_clusters[cluster_id] += 1

        # Compute interface clusters sizes
        interface_clusters: Dict[str, int] = {}
        for record in records:
            for interface in record.interfaces:
                if not interface.valid:
                    continue
                cluster_id = get_interface_cluster(interface, record)
                if cluster_id not in interface_clusters:
                    interface_clusters[cluster_id] = 0
                interface_clusters[cluster_id] += 1

        # Separate items and weights into three pools: contextual, free glycan, and other.
        contextual_items, contextual_weights = [], []
        free_glycan_items, free_glycan_weights = [], []
        other_items, other_weights = [], []

        prot_id = const.chain_type_ids["PROTEIN"]
        glycan_id = const.chain_type_ids["NONPOLYMER"]

        for record in records:
            has_protein = any(c.mol_type == prot_id for c in record.chains)
            has_glycan = any(c.mol_type == glycan_id for c in record.chains)

            is_contextual = has_protein and has_glycan
            is_free_glycan = not has_protein and has_glycan

            # Assign the record's sampling targets to the correct pool
            if is_contextual:
                target_items, target_weights = contextual_items, contextual_weights
            elif is_free_glycan:
                target_items, target_weights = free_glycan_items, free_glycan_weights
            else:
                target_items, target_weights = other_items, other_weights

            # Compute weights for chains and interfaces for this record
            for chain_id, chain in enumerate(record.chains):
                if not chain.valid:
                    continue
                weight = get_chain_weight(
                    chain,
                    record,
                    chain_clusters,
                    self.beta_chain,
                    self.alpha_prot,
                    self.alpha_nucl,
                    self.alpha_ligand,
                )
                target_items.append((record, 0, chain_id))
                target_weights.append(weight)

            for int_id, interface in enumerate(record.interfaces):
                if not interface.valid:
                    continue
                weight = get_interface_weight(
                    interface,
                    record,
                    interface_clusters,
                    self.beta_interface,
                    self.alpha_prot,
                    self.alpha_nucl,
                    self.alpha_ligand,
                )
                target_items.append((record, 1, int_id))
                target_weights.append(weight)

        # --- CONDITIONAL SAMPLING ---
        # If we have both contextual and free glycan samples, use the new two-pool logic.
        if (self.glycan_context_prob is not None) and contextual_items and free_glycan_items:
            # Normalize the weights within each glycan-related pool
            contextual_weights = np.array(contextual_weights) / np.sum(contextual_weights)
            free_glycan_weights = np.array(free_glycan_weights) / np.sum(
                free_glycan_weights
            )

            # Start the two-pool sampling loop
            while True:
                # Step 1: Choose which pool to sample from
                if random.rand() < self.glycan_context_prob:
                    # Sample from the contextual pool
                    item_idx = random.choice(len(contextual_items), p=contextual_weights)
                    record, kind, index = contextual_items[item_idx]
                else:
                    # Sample from the free-floating glycan pool
                    item_idx = random.choice(len(free_glycan_items), p=free_glycan_weights)
                    record, kind, index = free_glycan_items[item_idx]

                # Step 2: Yield the chosen sample
                if kind == 0:
                    yield Sample(record=record, chain_id=index)
                else:
                    yield Sample(record=record, interface_id=index)
        else:
            # Fallback to original logic: combine all pools and sample from one.
            all_items = contextual_items + free_glycan_items + other_items
            all_weights = contextual_weights + free_glycan_weights + other_weights

            if not all_items:
                return  # Empty dataset, nothing to sample.

            # Sample infinitely using the original single-pool method
            weights = np.array(all_weights) / np.sum(all_weights)
            while True:
                item_idx = random.choice(len(all_items), p=weights)
                record, kind, index = all_items[item_idx]
                if kind == 0:
                    yield Sample(record=record, chain_id=index)
                else:
                    yield Sample(record=record, interface_id=index)
