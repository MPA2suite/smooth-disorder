"""
test_barcode.py — Unit tests for disorder.barcode
==================================================

Tests cover:
  - recursive_find_mu: base case and a few known values
  - obtain_H1_barcode: triangle (1 ring) and line (0 rings)
  - reduce_barcode: trailing-zero trimming, no-reduction, 1×1 base case

All tests use small hand-crafted inputs whose correct outputs can be verified
by hand, so that any change to the algorithm logic will cause a test failure.
"""

import numpy as np
import pytest

from smooth_disorder.barcode import (
    recursive_find_mu,
    obtain_H1_barcode,
    reduce_barcode,
    mu,
)


# ---------------------------------------------------------------------------
# recursive_find_mu
# ---------------------------------------------------------------------------

class TestRecursiveFindMu:
    """Tests for the Möbius inversion function."""

    def test_base_case_returns_one(self):
        """
        μ((a,b),(a,b)) = 1 for any valid (a,b).
        This is the definition of the base case in the Möbius inversion.
        """
        assert recursive_find_mu(mu, 0, 0, 0, 0) == 1
        assert recursive_find_mu(mu, 1, 2, 1, 2) == 1
        assert recursive_find_mu(mu, 3, 5, 3, 5) == 1

    def test_invalid_subshell_raises(self):
        """
        Calling with arguments that violate the ordering (a≥c, b≤d, b≥a, d≥c)
        should raise an Exception.
        """
        with pytest.raises(Exception, match="invalid subshell"):
            recursive_find_mu(mu, 0, 0, 1, 1)  # a < c

    def test_result_is_integer(self):
        """The Möbius function always returns an integer value."""
        result = recursive_find_mu(mu, 0, 1, 0, 1)
        assert isinstance(result, (int, np.integer))

    def test_memoization_caches_result(self):
        """
        After the first call, the result should be stored in the mu dict
        so that subsequent calls do not recompute it.
        """
        key = "(1,1),(0,1)"  # will be computed by the call below
        recursive_find_mu(mu, 1, 1, 0, 1)
        assert key in mu

    def test_known_value_mu_1_1_0_1(self):
        """
        Hand-computed value: μ((1,1),(0,1)).

        By the recursive formula:
            μ((1,1),(0,1)) = -μ((1,1),(1,1))
                           = -1

        Verify the recursion gives -1.
        """
        result = recursive_find_mu(mu, 1, 1, 0, 1)
        assert result == -1


# ---------------------------------------------------------------------------
# obtain_H1_barcode
# ---------------------------------------------------------------------------

class TestObtainH1Barcode:
    """Tests for the H1 barcode computation on small known graphs."""

    def test_triangle_has_one_ring(self, tiny_adjacency_triangle):
        """
        A 3-atom triangle (3 nodes, 3 edges, 1 component) has exactly 1
        independent ring.  The Euler formula gives: β₁ = edges - atoms + components
        = 3 - 3 + 1 = 1.

        The barcode G should contain a total ring count of 1.
        """
        adj, layers = tiny_adjacency_triangle
        G, F = obtain_H1_barcode(adj, layers, mu)
        # total rings across all barcode entries must be 1
        assert G.sum() == pytest.approx(1.0)

    def test_line_has_zero_rings(self, tiny_adjacency_line):
        """
        A 3-atom line (path graph: 3 nodes, 2 edges, 1 component) has no rings.
        Euler formula: β₁ = 2 - 3 + 1 = 0.

        All entries of G should be zero.
        """
        adj, layers = tiny_adjacency_line
        G, F = obtain_H1_barcode(adj, layers, mu)
        assert np.allclose(G, 0.0)

    def test_returns_square_matrices(self, tiny_adjacency_triangle):
        """G and F should be square matrices of size (n_layers, n_layers)."""
        adj, layers = tiny_adjacency_triangle
        G, F = obtain_H1_barcode(adj, layers, mu)
        n = len(layers)
        assert G.shape == (n, n)
        assert F.shape == (n, n)

    def test_F_is_nonnegative(self, tiny_adjacency_triangle):
        """
        F[i,j] counts rings (a non-negative integer), so all entries must
        be ≥ 0.
        """
        adj, layers = tiny_adjacency_triangle
        G, F = obtain_H1_barcode(adj, layers, mu)
        assert (F >= 0).all()

    def test_single_atom_environment(self):
        """
        A local environment with only 1 atom has no bonds and no rings.
        G and F should both be 1×1 zero matrices.
        """
        adj = np.array([[0]], dtype=int)
        layers = [[0]]
        G, F = obtain_H1_barcode(adj, layers, mu)
        assert G.shape == (1, 1)
        assert np.allclose(G, 0.0)
        assert np.allclose(F, 0.0)


# ---------------------------------------------------------------------------
# reduce_barcode
# ---------------------------------------------------------------------------

class TestReduceBarcode:
    """Tests for barcode matrix reduction (canonical form)."""

    def test_1x1_matrix_unchanged(self):
        """A 1×1 matrix is already minimal and must not be changed."""
        G = np.array([[3.0]])
        result = reduce_barcode(G)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(3.0)

    def test_trailing_zero_column_removed(self):
        """
        If the last column (and its matching row) sums to zero, the matrix
        should be reduced by one dimension.
        """
        G = np.array([
            [1.0, 0.0],
            [0.0, 0.0],
        ])
        result = reduce_barcode(G)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0)

    def test_no_reduction_when_last_column_nonzero(self):
        """
        If the last column has any non-zero entry, the matrix must not
        be reduced.
        """
        G = np.array([
            [1.0, 1.0],
            [0.0, 0.0],
        ])
        result = reduce_barcode(G)
        assert result.shape == (2, 2)

    def test_recursive_reduction(self):
        """
        Multiple trailing zero rows/cols should all be stripped recursively.
        A 3×3 matrix with only G[0,0] non-zero should reduce to 1×1.
        """
        G = np.zeros((3, 3))
        G[0, 0] = 2.0
        result = reduce_barcode(G)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(2.0)

    def test_triangle_barcode_reduces_correctly(self, tiny_adjacency_triangle):
        """
        The triangle barcode (computed from obtain_H1_barcode) should reduce
        to a 2×2 matrix because ring information lives at G[0,1].
        """
        adj, layers = tiny_adjacency_triangle
        G, _ = obtain_H1_barcode(adj, layers, mu)
        reduced = reduce_barcode(G)
        # layers has 2 entries → G is 2×2 already, no trailing zeros expected
        assert reduced.shape == (2, 2)