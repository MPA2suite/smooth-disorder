"""
test_pipeline.py — End-to-end integration tests for the BNE pipeline
=====================================================================

These tests verify that the full calculation pipeline produces correct results
by comparing against:
  1. Analytically known results (e.g. single-atom LAE has entropy = 0).
  2. Precomputed reference HDF5 files included in the repository.

"""

import os
import numpy as np
import pytest
import h5py

from ase.io import read

from smooth_disorder.structural import obtain_distances_ase
from smooth_disorder.barcode import (
    obtain_local_number_environment_big_structures,
    obtain_H1_barcode,
    reduce_barcode,
    mu,
)


# ---------------------------------------------------------------------------
# Helper: run the BNE pipeline for a subset of atoms
# ---------------------------------------------------------------------------

def compute_bne(atoms, n_environment, n_atoms_subset=None, n_smallest=50):
    """
    Run the full BNE pipeline on an ASE Atoms object.

    Parameters
    ----------
    atoms : ase.atoms.Atoms
    n_environment : int
        Local environment size (number of atoms per LAE).
    n_atoms_subset : int or None
        If given, only process the first n_atoms_subset atoms.
        Useful for fast tests.
    n_smallest : int
        Number of nearest neighbours to pre-compute distances for.

    Returns
    -------
    probabilities : np.ndarray
        Empirical probability distribution over barcode classes.
    entropy : float
        Shannon entropy (Bond-Network Entropy).
    """
    distances, idx_distances = obtain_distances_ase(atoms, n_smallest=n_smallest)

    cutoff = 2.1  # Si-O bond cutoff for silica glass
    adjacency_matrix = ((distances < cutoff) & (distances > 0.1)).astype(int)

    n_atoms = n_atoms_subset if n_atoms_subset is not None else len(atoms)

    # collect barcodes grouped by shape
    shapes = {}
    for atom_index in range(n_atoms):
        local_adj, layers, local_idx, global_idx = \
            obtain_local_number_environment_big_structures(
                adjacency_matrix, atom_index,
                distances, idx_distances, n_environment
            )
        G, F = obtain_H1_barcode(local_adj, layers, mu)
        G = reduce_barcode(G)

        key = G.shape
        shapes.setdefault(key, []).append(G)

    # count equivalence classes (unique barcodes)
    counts = []
    for key, gs in shapes.items():
        _, class_counts = np.unique(gs, return_counts=True, axis=0)
        counts.extend(class_counts)

    counts = np.array(counts)
    probabilities = counts / counts.sum()
    entropy = -(probabilities * np.log(probabilities)).sum()

    return probabilities, entropy


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBNEPipeline:
    """Integration tests for the full BNE computation pipeline."""

    @pytest.fixture(scope="class")
    def silica_atoms_small(self, silica_poscar_path):
        """Load the silica structure (full, for MIC correctness)."""
        return read(silica_poscar_path)

    @pytest.mark.slow
    def test_bne_n1_entropy_is_zero(self, silica_atoms_small):
        """
        With a local environment of size 1 (just the central atom, no bonds),
        every atom has the identical trivial barcode — a single class with
        probability 1.  The Shannon entropy of a deterministic distribution is 0.

        This is a sanity check: if the result is non-zero, something is wrong
        with barcode equality testing or counting.
        """
        _, entropy = compute_bne(silica_atoms_small, n_environment=1, n_atoms_subset=20)
        assert entropy == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.slow
    def test_bne_entropy_nonnegative(self, silica_atoms_small):
        """
        Shannon entropy is always ≥ 0.  Test several small LAE sizes.
        """
        for n_env in [5, 10, 15, 20]:
            _, entropy = compute_bne(
                silica_atoms_small, n_environment=n_env, n_atoms_subset=20
            )
            assert entropy >= 0.0, f"Negative entropy for n_environment={n_env}: {entropy}"

    @pytest.mark.slow
    def test_bne_entropy_increases_with_environment_size(self, silica_atoms_small):
        """
        As the local environment grows, more topological diversity is captured,
        so BNE should be non-decreasing with environment size (on average).

        We check that BNE(n=20) > BNE(n=10) on a subset of atoms.
        """
        _, entropy_small = compute_bne(silica_atoms_small, n_environment=10, n_atoms_subset=50)
        _, entropy_large = compute_bne(silica_atoms_small, n_environment=20, n_atoms_subset=50)
        assert entropy_large >= entropy_small, (
            f"BNE did not increase with environment size: "
            f"BNE(n=10)={entropy_small:.4f}, BNE(n=20)={entropy_large:.4f}"
        )

    @pytest.mark.slow
    def test_bne_precomputed_reference_n30(self, silica_atoms_small, precomputed_bne_dir):
        """
        Load the precomputed entropy for LAE size 30 and check that our
        pipeline gives the same value.
        """
        ref_file = os.path.join(precomputed_bne_dir, "entropy_number_30.hdf5")
        with h5py.File(ref_file, "r") as f:
            ref_entropy = float(f["entropy"][0])

        _, entropy = compute_bne(silica_atoms_small, n_environment=30)

        assert entropy > 0, "Pipeline produced zero entropy for n=30"
        ratio = entropy / ref_entropy
        assert np.isclose(ratio, 1.0), (
            f"Pipeline entropy {entropy:.4f} is different from reference "
            f"{ref_entropy:.4f} (ratio={ratio:.2f})"
        )

    @pytest.mark.slow
    def test_readme_quickstart_matches_reference(self, silica_poscar_path, precomputed_bne_dir):
        """
        Reproduce the README Quick Start verbatim and verify the resulting BNE
        matches the precomputed reference for n_environment=30.

        This test is the canary for documentation accuracy: if the README
        example ever drifts from what the code actually computes, this fails.
        """
        # --- README Quick Start (verbatim, path adapted for test context) ---
        atoms = read(silica_poscar_path)
        n_atoms = len(atoms)

        distances, idx_distances = obtain_distances_ase(atoms, n_smallest=300)

        cutoff = 2.1
        adjacency_matrix = ((distances < cutoff) & (distances > 0.1)).astype(int)

        n_environment = 30
        barcodes = []

        for atom_index in range(n_atoms):
            adj, layers, local_idx, global_idx = obtain_local_number_environment_big_structures(
                adjacency_matrix, atom_index, distances, idx_distances, n_environment
            )
            G, F = obtain_H1_barcode(adj, layers, mu)
            G = reduce_barcode(G)
            barcodes.append(G)

        shapes = {}
        for G in barcodes:
            key = G.shape
            shapes.setdefault(key, []).append(G)

        counts = []
        for key, gs in shapes.items():
            _, class_counts = np.unique(gs, return_counts=True, axis=0)
            counts.extend(class_counts)

        counts = np.array(counts)
        probabilities = counts / counts.sum()
        BNE = -(probabilities * np.log(probabilities)).sum()
        # --- end README snippet ---

        ref_file = os.path.join(precomputed_bne_dir, "entropy_number_30.hdf5")
        with h5py.File(ref_file, "r") as f:
            ref_entropy = float(f["entropy"][0])

        assert np.isclose(BNE, ref_entropy), (
            f"README quickstart BNE {BNE:.6f} != reference {ref_entropy:.6f}"
        )

    @pytest.mark.slow
    def test_probabilities_sum_to_one(self, silica_atoms_small):
        """
        The probability distribution over barcode classes must sum to 1.
        """
        probs, _ = compute_bne(silica_atoms_small, n_environment=5, n_atoms_subset=30)
        assert probs.sum() == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.slow
    def test_probabilities_nonnegative(self, silica_atoms_small):
        """All probabilities must be in [0, 1]."""
        probs, _ = compute_bne(silica_atoms_small, n_environment=5, n_atoms_subset=30)
        assert (probs >= 0).all()
        assert (probs <= 1).all()