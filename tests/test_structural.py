"""
test_structural.py — Unit tests for disorder.structural
========================================================

Tests cover:
  - obtain_positions_and_lattice_vectors: correct shapes for silica
  - obtain_distances_ase: shape, non-negativity, sort order, index range
  - ASE vs manual distance agreement (values and neighbour indices) up to 10 Å
  - Coordination number calculation from the adjacency matrix:
      silica should have 4-coordinated Si and 2-coordinated O
  - obtain_density: silica density should be ~2.2 g/cm³

Note on scope
-------------
TestObtainDistancesAse uses the first 50 atoms with n_smallest=20 for speed.
TestDistanceMethodAgreement and TestCoordinationNumber run on the full
5184-atom structure — these are slower but required for meaningful statistics
and full coverage of all atoms.
"""

import numpy as np
import pytest

from ase.io import read

from smooth_disorder.structural import (
    obtain_positions_and_lattice_vectors,
    obtain_distances_ase,
    obtain_distances_big_structures,
    obtain_density,
)


# ---------------------------------------------------------------------------
# Structure I/O
# ---------------------------------------------------------------------------

class TestObtainPositionsAndLatticeVectors:
    """Tests for reading structure files."""

    def test_positions_shape(self, silica_poscar_path):
        """
        The silica POSCAR has 5184 atoms.  Positions should be a (5184, 3) array.
        """
        positions, lattice_vectors = obtain_positions_and_lattice_vectors(silica_poscar_path)
        assert positions.shape == (5184, 3)

    def test_lattice_vectors_shape(self, silica_poscar_path):
        """Lattice vectors should be a 3×3 matrix."""
        positions, lattice_vectors = obtain_positions_and_lattice_vectors(silica_poscar_path)
        assert lattice_vectors.shape == (3, 3)

    def test_positions_are_finite(self, silica_poscar_path):
        """All atomic coordinates should be finite numbers (no NaN or Inf)."""
        positions, _ = obtain_positions_and_lattice_vectors(silica_poscar_path)
        assert np.all(np.isfinite(positions))


# ---------------------------------------------------------------------------
# Distance calculation
# ---------------------------------------------------------------------------

class TestObtainDistancesAse:
    """Tests for obtain_distances_ase on a small slice of the silica structure."""

    @pytest.fixture(scope="class")
    def distances_small(self, silica_poscar_path):
        """
        Pre-compute distances for the first 50 atoms to the 20 nearest
        neighbours.  Scoped to the class so it is computed only once.
        """
        atoms_full = read(silica_poscar_path)
        # slice to the first 50 atoms for speed
        atoms = atoms_full[:50]
        distances, idx_distances = obtain_distances_ase(atoms, n_smallest=20)
        return distances, idx_distances

    def test_distances_shape(self, distances_small):
        """distances should be shape (n_atoms, n_smallest)."""
        distances, idx_distances = distances_small
        assert distances.shape == (50, 20)
        assert idx_distances.shape == (50, 20)

    def test_distances_nonnegative(self, distances_small):
        """All distances must be non-negative."""
        distances, _ = distances_small
        assert (distances >= 0).all()

    def test_distances_sorted(self, distances_small):
        """
        Distances for each atom should be sorted in ascending order —
        the first entry is the smallest distance.
        """
        distances, _ = distances_small
        for row in distances:
            assert np.all(np.diff(row) >= 0), "Distances are not sorted"

    def test_idx_distances_in_valid_range(self, distances_small):
        """All neighbour indices must be valid atom indices (0 … 49)."""
        _, idx_distances = distances_small
        assert (idx_distances >= 0).all()
        assert (idx_distances < 50).all()


# ---------------------------------------------------------------------------
# Coordination number from adjacency matrix
# ---------------------------------------------------------------------------

class TestCoordinationNumber:
    """
    Tests for the coordination number distribution in silica glass.

    In crystalline SiO₂ (and its glass), silicon is 4-coordinated (bonded to
    4 oxygen atoms) and oxygen is 2-coordinated (bonded to 2 silicon atoms).
    A bond cutoff of 2.1 Å selects only Si-O nearest-neighbour bonds.
    """

    @pytest.fixture(scope="class")
    def silica_adjacency(self, silica_poscar_path):
        """
        Build the adjacency matrix for the full 5184-atom silica structure.
        Returns (atoms, adjacency_matrix) so we can check species.
        """
        atoms = read(silica_poscar_path)
        distances, idx_distances = obtain_distances_ase(atoms, n_smallest=20)
        # bonds exist for distances in (0.1, 2.1] Å — the Si-O bond cutoff
        cutoff = 2.1
        adjacency_matrix = ((distances < cutoff) & (distances > 0.1)).astype(int)
        return atoms, adjacency_matrix

    def test_silicon_coordination_is_4(self, silica_adjacency):
        """
        ≥ 95% of Si atoms should have exactly 4 bonds.
        (A small fraction may be slightly distorted in the glass.)
        """
        atoms, adjacency_matrix = silica_adjacency
        symbols = np.array(atoms.get_chemical_symbols())
        si_indices = np.where(symbols == 'Si')[0]

        # coordination number = number of bonds per atom (row sum)
        coordination = adjacency_matrix.sum(axis=1)
        si_coordination = coordination[si_indices]

        fraction_4 = (si_coordination == 4).sum() / len(si_indices)
        assert fraction_4 >= 0.95, (
            f"Only {fraction_4:.1%} of Si atoms have coordination 4 "
            f"(expected ≥ 95%)"
        )

    def test_oxygen_coordination_is_2(self, silica_adjacency):
        """
        ≥ 95% of O atoms should have exactly 2 bonds.
        """
        atoms, adjacency_matrix = silica_adjacency
        symbols = np.array(atoms.get_chemical_symbols())
        o_indices = np.where(symbols == 'O')[0]

        coordination = adjacency_matrix.sum(axis=1)
        o_coordination = coordination[o_indices]

        fraction_2 = (o_coordination == 2).sum() / len(o_indices)
        assert fraction_2 >= 0.95, (
            f"Only {fraction_2:.1%} of O atoms have coordination 2 "
            f"(expected ≥ 95%)"
        )

    def test_total_atom_count(self, silica_adjacency):
        """
        The silica structure should contain 5184 atoms total:
        1728 Si + 3456 O.
        """
        atoms, _ = silica_adjacency
        symbols = np.array(atoms.get_chemical_symbols())
        assert (symbols == 'Si').sum() == 1728
        assert (symbols == 'O').sum() == 3456


# ---------------------------------------------------------------------------
# ASE vs manual distance agreement
# ---------------------------------------------------------------------------

class TestDistanceMethodAgreement:
    """
    Check that obtain_distances_ase and obtain_distances_big_structures
    agree for all atoms in the full silica structure, for distances up to 10 Å.

    Why this matters
    ----------------
    The two implementations use different MIC strategies:

    - obtain_distances_ase delegates to ASE's get_distances, which uses a
      rigorous MIC for any cell shape.
    - obtain_distances_big_structures uses a manual implementation that wraps
      fractional coordinates into [-0.5, 0.5].  This is exact for orthorhombic
      cells (all cell angles = 90°), but only approximate for triclinic cells.

    The silica glass structure used here has a cubic cell (~42.8 x 42.8 x 42.8 Å,
    all angles 90°).  Because the cell is orthorhombic, both methods are
    exact and must agree to numerical precision (rtol=1e-5).

    We run on the full 5184-atom structure so that every atom is verified.
    n_smallest = 400 neighbours is enough to cover all atoms within 10 Å
    in silica glass (~275 neighbours at that range on average).
    """

    N_SMALLEST = 400    # neighbours pre-computed by each method
    CUTOFF_ANG = 10.0   # distance threshold for comparison (Å)

    @pytest.fixture(scope="class")
    def both_distances(self, silica_poscar_path):
        """
        Compute distances with both methods for the full silica structure and
        return them as (distances_ase, idx_ase, distances_manual, idx_manual).
        """
        atoms = read(silica_poscar_path)
        positions = atoms.get_positions()
        lattice_vectors = np.array(atoms.cell)

        distances_ase, idx_ase = obtain_distances_ase(atoms, n_smallest=self.N_SMALLEST)
        distances_man, idx_man = obtain_distances_big_structures(
            positions, lattice_vectors, n_smallest=self.N_SMALLEST
        )
        return distances_ase, idx_ase, distances_man, idx_man

    def test_short_distance_values_agree(self, both_distances):
        """
        For every atom, collect all distances < 10 Å from both methods
        and check they are element-wise close (rtol=1e-5).  No sorting is
        needed because both functions return distances in ascending order.

        Because the cell is orthorhombic, the two MIC implementations are
        both exact at this range and must agree to floating-point precision.
        """
        distances_ase, _, distances_man, _ = both_distances
        n_atoms = distances_ase.shape[0]

        for i in range(n_atoms):
            # distances are already sorted ascending by both functions
            d_ase = distances_ase[i][distances_ase[i] < self.CUTOFF_ANG]
            d_man = distances_man[i][distances_man[i] < self.CUTOFF_ANG]

            assert len(d_ase) == len(d_man), (
                f"Atom {i}: ASE found {len(d_ase)} neighbours within {self.CUTOFF_ANG} Å "
                f"but manual found {len(d_man)}"
            )
            assert np.allclose(d_ase, d_man, rtol=1e-5), (
                f"Atom {i}: distance values disagree within {self.CUTOFF_ANG} Å\n"
                f"  max abs diff = {np.abs(d_ase - d_man).max():.2e} Å"
            )

    def test_short_distance_neighbours_agree(self, both_distances):
        """
        For every atom, check that the neighbour index arrays within 10 Å
        are identical (element-wise) between the two methods.

        Because both functions sort idx_distances by distance, a direct
        array comparison is valid.  For an orthorhombic cell both MIC
        implementations are exact, so the arrays must match exactly.
        """
        distances_ase, idx_ase, distances_man, idx_man = both_distances
        n_atoms = distances_ase.shape[0]

        for i in range(n_atoms):
            mask_ase = distances_ase[i] < self.CUTOFF_ANG
            mask_man = distances_man[i] < self.CUTOFF_ANG

            # idx_distances arrays are sorted by distance, so compare directly
            neighbours_ase = idx_ase[i][mask_ase]
            neighbours_man = idx_man[i][mask_man]

            assert np.array_equal(neighbours_ase, neighbours_man), (
                f"Atom {i}: neighbour index arrays differ within {self.CUTOFF_ANG} Å\n"
                f"  ASE:    {neighbours_ase}\n"
                f"  manual: {neighbours_man}"
            )


# ---------------------------------------------------------------------------
# Full-matrix agreement (ASE vs manual, all 5184 × 5184 pairs)
# ---------------------------------------------------------------------------

class TestFullDistanceMatrixAgreement:
    """
    Compare the full (5184, 5184) distance and neighbour-index matrices
    produced by obtain_distances_ase and obtain_distances_big_structures.

    Unlike TestDistanceMethodAgreement (which uses n_smallest=400 and a 10 Å
    cutoff), these tests set n_smallest=5184 so every atom-pair distance is
    included, and compare the entire array without any distance filter.

    Because the silica cell is orthorhombic, both MIC implementations are
    exact for all pair distances and must agree to floating-point precision.
    """

    @pytest.fixture(scope="class")
    def full_distance_matrices(self, silica_poscar_path):
        """
        Compute the full distance and index matrices (n_smallest = n_atoms)
        with both methods.  Returns (distances_ase, idx_ase, distances_man, idx_man).
        """
        atoms = read(silica_poscar_path)
        n_atoms = len(atoms)
        positions = atoms.get_positions()
        lattice_vectors = np.array(atoms.cell)

        distances_ase, idx_ase = obtain_distances_ase(atoms, n_smallest=n_atoms)
        distances_man, idx_man = obtain_distances_big_structures(
            positions, lattice_vectors, n_smallest=n_atoms
        )
        return distances_ase, idx_ase, distances_man, idx_man

    def test_full_distances_agree(self, full_distance_matrices):
        """
        All (5184, 5184) distance values must agree between the two methods
        to floating-point precision (rtol=1e-5).

        Both functions return distances sorted ascending, so element-wise
        comparison is valid without any reordering.
        """
        distances_ase, _, distances_man, _ = full_distance_matrices
        assert distances_ase.shape == distances_man.shape == (5184, 5184)
        assert np.allclose(distances_ase, distances_man, rtol=1e-5), (
            f"Full distance matrices disagree: "
            f"max abs diff = {np.abs(distances_ase - distances_man).max():.2e} Å"
        )

    def test_full_idx_distances_agree(self, full_distance_matrices):
        """
        All (5184, 5184) neighbour index values must be identical between the
        two methods.

        Both functions sort idx_distances by distance, so for an orthorhombic
        cell where both MIC implementations are exact, the index arrays must
        match element-wise exactly.
        """
        _, idx_ase, _, idx_man = full_distance_matrices
        assert idx_ase.shape == idx_man.shape == (5184, 5184)
        assert np.array_equal(idx_ase, idx_man), (
            f"Full neighbour index matrices disagree: "
            f"{(idx_ase != idx_man).sum()} out of {idx_ase.size} entries differ"
        )


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

class TestObtainDensity:
    """Tests for the density calculation."""

    def test_silica_density_reasonable(self, silica_poscar_path):
        """
        Silica glass has a density of approximately 2.2 g/cm³.
        We allow ± 0.5 g/cm³ to account for slightly different models.
        """
        atoms = read(silica_poscar_path)
        density = obtain_density(atoms)
        assert 1.7 < density < 2.7, (
            f"Silica density {density:.3f} g/cm³ is outside the expected range "
            f"(1.7 – 2.7 g/cm³)"
        )

    def test_density_is_positive(self, silica_poscar_path):
        """Density must always be a positive number."""
        atoms = read(silica_poscar_path)
        density = obtain_density(atoms)
        assert density > 0
