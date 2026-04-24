"""
disorder — Bond-Network Entropy Library
========================================

A Python library for computing Bond-Network Entropy (BNE), a topological
descriptor that quantifies structural disorder in amorphous (glassy) materials.

BNE is based on the H1 persistent homology barcode of each atom's local
environment.  The distribution of barcodes across all atoms in a structure is
summarised by a single Shannon entropy value — the BNE.

Reference
---------
K. Iwanowski, G. Csányi, and M. Simoncelli,
*Phys. Rev. X* 15, 041041 (2025)

Quick start
-----------
::

    from ase.io import read
    from smooth_disorder.structural import obtain_distances_ase
    from smooth_disorder.barcode import (
        obtain_local_number_environment_big_structures,
        obtain_H1_barcode,
        reduce_barcode,
        mu,
    )
    import numpy as np

    atoms = read("structure.vasp")
    distances, idx_distances = obtain_distances_ase(atoms, n_smallest=300)

    cutoff = 2.1  # Å — Si-O bond cutoff for silica
    adjacency_matrix = ((distances < cutoff) & (distances > 0.1)).astype(int)

    # compute barcode for the first atom with a local environment of 30 atoms
    adj, layers, local_idx, global_idx = obtain_local_number_environment_big_structures(
        adjacency_matrix, 0, distances, idx_distances, 30
    )
    G, F = obtain_H1_barcode(adj, layers, mu)
    G = reduce_barcode(G)
    print("Barcode:", G)

Submodules
----------
- ``disorder.barcode``             — H1 barcode computation and BNE calculation
- ``disorder.structural``          — Structure I/O, distances, periodic boundaries
- ``disorder.disorder_linewidth``  — Phonon mesh, VDOS, linewidth model, fitting
- ``disorder.vis``                 — Visualization (notebook plotting)

The disorder linewidth submodule (``disorder.disorder_linewidth``) requires
phonopy and PyTorch.  It is imported automatically when both are available;
on a BNE-only installation it is silently skipped and must be imported
directly if needed.
"""

from smooth_disorder.barcode import (
    obtain_local_number_environment_big_structures,
    obtain_H1_barcode,
    reduce_barcode,
    clear_mu_cache,
    mu,
)
from smooth_disorder.structural import (
    obtain_positions_and_lattice_vectors,
    obtain_distances_ase,
    obtain_distances_big_structures,
    obtain_density,
)
try:
    from smooth_disorder.disorder_linewidth import (
        run_band_structure_manual,
        run_phonon_mesh,
        save_mesh_data_to_files,
        lorentzian_numpy,
        flatten_arrays,
        calculate_vdos_and_average_speed_with_frequency,
        save_vdos_speed_data_to_files,
        flatten_arrays_freq_only,
        calculate_vdos_with_frequency,
        save_vdos_data_to_files,
        prepare_fitting_inputs,
        evaluate_linewidth_and_model_prediction,
        lorentzian_torch,
    )
except ImportError:
    pass


__version__ = "1.0.0"
__authors__ = ["Kamil Iwanowski", "Michele Simoncelli"]

__all__ = [
    # barcode
    "obtain_local_number_environment_big_structures",
    "obtain_H1_barcode",
    "reduce_barcode",
    "clear_mu_cache",
    "mu",
    # structural
    "obtain_positions_and_lattice_vectors",
    "obtain_distances_ase",
    "obtain_distances_big_structures",
    "obtain_density",
    # disorder_linewidth
    "run_band_structure_manual",
    "run_phonon_mesh",
    "save_mesh_data_to_files",
    "lorentzian_numpy",
    "flatten_arrays",
    "calculate_vdos_and_average_speed_with_frequency",
    "save_vdos_speed_data_to_files",
    "flatten_arrays_freq_only",
    "calculate_vdos_with_frequency",
    "save_vdos_data_to_files",
    "prepare_fitting_inputs",
    "evaluate_linewidth_and_model_prediction",
    "lorentzian_torch",
]
