"""
5_BNE_workflow.py — Compute Bond-Network Entropy for a range of LAE sizes
==========================================================================

This script calculates the Bond-Network Entropy (BNE) for a silica glass
structure and saves the results to HDF5 files.  It is the data source for
notebook 6 (BNE growth rate analysis).

What is BNE?
------------
BNE is a set of numbers dependent on the sizes of local environments that 
describe how variable local environments are within a structure. Its growth rate 
can be used to quantify how disordered an atomic structure is.
It is computed from the H1 persistent homology barcodes of each atom's local
atomic environment (LAE).  
A barcode is a fingerprint of the local environment that finds 
algebraically independent rings in it.  
Overall, BNE is the Shannon entropy of the
distribution of these fingerprints across all atoms:

    BNE = -sum_i  p_i * log(p_i)

where p_i is the fraction of atoms that share the same barcode fingerprint.
Higher BNE means more variety of local environments → more topological disorder.

How to run
----------
From the examples/ directory:

    python 5_BNE_workflow.py

Pre-computed data for LAE sizes 1-80 is already in
examples/data/bond_network_entropy/ and were calculated using this script.
To change the LAE size range, edit `local_environment_nat` in the ``__main__``
block at the bottom of this file.

Output
------
One HDF5 file per LAE size, saved as:
    data/bond_network_entropy/structure_0/entropy_number_<N>.hdf5

Each file stores:
    - "entropy"         : scalar BNE value
    - "probabilities"   : array of barcode class probabilities
    - "number_of_atoms" : LAE size N used for this calculation
"""

import os

import numpy as np
import h5py
from tqdm import tqdm

# obtain_distances_big_structures is the fast manual MIC implementation suited
# for large structures.  It operates directly on NumPy arrays (positions and
# lattice vectors) rather than an ASE Atoms object, which avoids per-atom
# Python overhead inside the ASE call.
#
# WARNING — known limitations of this implementation:
#   1. It is only *exact* for orthorhombic cells (a, b, c mutually perpendicular).
#   2. For monoclinic or triclinic cells the single-image MIC wrap should only be accurate
#.     for distances shorter than the half the smallest cell dimension (d < min(|a|, |b|, |c|) / 2).
#      Distances beyond that threshold may be assigned to the wrong periodic image.
#      For the Si–O bond cutoff of 2.1 Å and an orthorhombic silica unit cell this is fine,
#      but always verify before applying to a different structure.
#
# For non-orthorhombic cells and large cutoffs, use obtain_distances_ase instead.
from smooth_disorder.structural import obtain_distances_big_structures, obtain_positions_and_lattice_vectors


from smooth_disorder.barcode import (
    obtain_local_number_environment_big_structures,
    obtain_H1_barcode,
    reduce_barcode,
    mu,
)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_bne_data_to_files(filename, probabilities, entropy, LE_nat, filepath):
    """
    Save BNE results for one LAE size to an HDF5 file.

    Parameters
    ----------
    filename : str
        Output file path without the .hdf5 extension.
    probabilities : np.ndarray
        Empirical probability of each barcode equivalence class.
    entropy : float
        Shannon entropy (BNE) computed from the probabilities.
    LE_nat : int
        Number of atoms in the local atomic environment (LAE size).
    filepath : str
        Path to the input structure file (stored for reference).
    """
    with h5py.File(f"{filename}.hdf5", "w") as w:
        w.create_dataset("probabilities", data=probabilities, compression="gzip")
        w.create_dataset("entropy", data=np.array([entropy]), compression="gzip")
        w.create_dataset("number_of_atoms", data=np.array([LE_nat]), compression="gzip")


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calculate_barcode(distances, idx_distances, adjacency_matrix, n_atoms, save_folder, structure_idx, filepath, local_environment_nat):
    """
    Compute BNE for a range of LAE sizes and save results to disk.

    For each LAE size N:
      1. Extract the N-atom local environment around every atom using BFS.
      2. Compute the H1 barcode (ring fingerprint) for each local environment.
      3. Count how many atoms share the same barcode.
      4. Convert counts to probabilities and compute Shannon entropy = BNE.
      5. Save results to an HDF5 file.

    Parameters
    ----------
    distances : np.ndarray, shape (n_atoms, n_smallest)
        Pre-computed sorted nearest-neighbour distances (Ångström).
    idx_distances : np.ndarray, shape (n_atoms, n_smallest)
        Global atom indices corresponding to those distances.
    adjacency_matrix : np.ndarray, shape (n_atoms, n_smallest)
        Binary matrix: entry [i, j] = 1 if atoms i and idx_distances[i, j] are bonded.
    n_atoms : int
        Total number of atoms in the structure.
    save_folder : str
        Root directory for output HDF5 files.
    structure_idx : int
        Index label for this structure (used in the output subdirectory name).
    filepath : str
        Path to the original structure file.
    local_environment_nat : list of int
        LAE sizes to compute BNE for.  Defined in the ``__main__`` block.
    """
    # Create the output directory if it does not exist yet.
    os.makedirs(f"{save_folder}/structure_{structure_idx}", exist_ok=True)

    for LE_nat in local_environment_nat:
        print(f"Current LAE size = {LE_nat}")

        # Gs collects barcodes grouped by their shape (dimensions of the array).
        # Two reduced barcodes with different shapes can never be equal, so grouping by
        # shape first makes the equality checks much faster.
        Gs = {}

        for atom_index in tqdm(range(n_atoms), desc="Atoms"):

            # Step 1 — extract the local atomic environment (LAE).
            # Starting from atom_index, we grow outward by BFS until we have
            # included LE_nat atoms.  The result is a small sub-graph.
            local_adjacency_matrix, layers, local_atom_index, global_index = \
                obtain_local_number_environment_big_structures(
                    adjacency_matrix=adjacency_matrix,
                    atom_index=atom_index,
                    distances=distances,
                    idx_distances=idx_distances,
                    n_environment_atoms=LE_nat,
                )

            # Step 2 — compute the H1 barcode of the local sub-graph.
            G, F = obtain_H1_barcode(
                adjacency_matrix=local_adjacency_matrix,
                layers=layers,
                mu=mu,
            )

            # Step 3 — reduce the barcode to its canonical form so that two
            # topologically identical environments always give the same array.
            G = reduce_barcode(G)

            # Accumulate barcodes, grouped by shape for efficient comparison.
            key = G.shape
            Gs.setdefault(key, []).append(G)

        # Step 4 — count equivalence classes (atoms with identical barcodes).
        # np.unique(..., axis=0) finds the distinct rows across all barcodes
        # that share the same shape, and counts how many times each barcode appears.
        class_counts = []
        for key, value in Gs.items():
            _, class_count = np.unique(value, return_counts=True, axis=0)
            class_counts.extend(class_count)

        class_counts = np.array(class_counts)

        # Convert raw counts to probabilities (must sum to 1).
        probabilities = class_counts / class_counts.sum()

        # Step 5 — Shannon entropy: BNE = -sum_i p_i * log(p_i).
        entropy = -(probabilities * np.log(probabilities)).sum()

        # Save results for this LAE size.
        BNE_save_filename = f"{save_folder}/structure_{structure_idx}/entropy_number_{LE_nat}"
        save_bne_data_to_files(BNE_save_filename, probabilities, entropy, LE_nat, filepath)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Output directory (relative to examples/).
    save_directory = "./data"
    BNE_folder = "bond_network_entropy"
    save_folder = f"{save_directory}/{BNE_folder}"

    # Input structure: silica glass with 5184 atoms.
    # Downloaded from https://www.pnas.org/doi/abs/10.1073/pnas.2422763122
    structure_filename = "./data/structural/silica_glass_5184_atoms/POSCAR"
    atomic_positions, lattice_vectors = obtain_positions_and_lattice_vectors(structure_filename)
    n_atoms = len(atomic_positions)

    # Compute nearest-neighbour distances for every atom.
    # n_smallest=300 is well above the LAE sizes we compute (up to 80 atoms),
    # so the pre-computed neighbour list is always large enough.
    distances, idx_distances = obtain_distances_big_structures(atomic_positions, lattice_vectors, n_smallest=300)

    # Build the adjacency matrix: two atoms are bonded if their distance is
    # between 0.1 Å (avoids self-distance = 0) and the cutoff.
    # 2.1 Å is the Si-O bond cutoff appropriate for silica glass.
    cutoff = 2.1
    adjacency_matrix = ((distances < cutoff) & (distances > 0.1)).astype(int)

    # Label for this structure (used in the output directory name).
    structure_idx = 0

    # LAE sizes to compute — edit this list to change the sweep range.
    # Pre-computed data covers 1–80; extend or restrict as needed.
    local_environment_nat = np.arange(1, 81, 1).tolist()

    calculate_barcode(distances, idx_distances, adjacency_matrix, n_atoms, save_folder, structure_idx, structure_filename, local_environment_nat)