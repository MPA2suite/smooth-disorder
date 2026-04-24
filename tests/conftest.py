"""
conftest.py — Shared pytest fixtures for the disorder test suite
================================================================

Fixtures defined here are automatically available to all test files without
needing to import them.

Fixture overview
----------------
silica_poscar_path
    Path to the example silica POSCAR file included in the repository.

precomputed_bne_dir
    Path to the directory of precomputed BNE HDF5 files for structure_0.

tiny_adjacency_triangle
    A hand-crafted 3-atom triangle adjacency matrix and its BFS layers.
    Useful for testing barcode functions on a known topology.

tiny_adjacency_line
    A hand-crafted 3-atom line adjacency matrix and its BFS layers.

reset_mu_cache (autouse)
    Clears the global Möbius inversion cache before every test so that
    memoized values from one test cannot affect another.
"""

import os
import numpy as np
import pytest

# Path to the repository root (one level up from tests/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def silica_poscar_path():
    """
    Absolute path to the silica glass POSCAR file (5184 atoms).

    The file is part of the repository under tutorials/bond_network_entropy/data/.
    """
    path = os.path.join(
        REPO_ROOT,
        "tutorials", "bond_network_entropy", "data", "structural",
        "silica_glass_5184_atoms", "POSCAR"
    )
    assert os.path.exists(path), f"POSCAR not found at {path}"
    return path


@pytest.fixture(scope="class")
def dl_test_dir():
    """
    Path to the dl_test directory containing reference HDF5 files for the
    disorder linewidth workflow.
    """
    path = os.path.join(REPO_ROOT, "tutorials", "disorder_linewidth", "dl_test")
    assert os.path.exists(path), f"dl_test reference directory not found at {path}"
    return path


@pytest.fixture(scope="class")
def precomputed_bne_dir():
    """
    Path to the directory containing precomputed BNE HDF5 files.

    Files are named entropy_number_1.hdf5 … entropy_number_50.hdf5.
    """
    path = os.path.join(
        REPO_ROOT,
        "tutorials", "bond_network_entropy", "data", "bond_network_entropy", "structure_0"
    )
    assert os.path.exists(path), f"Precomputed BNE directory not found at {path}"
    return path


@pytest.fixture(scope="class")
def workflows_dl_test_dir():
    """Path to workflows/dl_test, the committed DL workflow reference files."""
    path = os.path.join(REPO_ROOT, "workflows", "dl_test")
    assert os.path.exists(path), f"workflows/dl_test not found at {path}"
    return path


@pytest.fixture(scope="class")
def workflows_bne_test_dir():
    """Path to workflows/bne_test/bond_network_entropy/structure_0, the committed BNE reference files."""
    path = os.path.join(REPO_ROOT, "workflows", "bne_test", "bond_network_entropy", "structure_0")
    assert os.path.exists(path), f"workflows/bne_test/bond_network_entropy/structure_0 not found at {path}"
    return path


# ---------------------------------------------------------------------------
# Small synthetic graph fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_adjacency_triangle():
    """
    3-atom triangle: atoms 0-1-2-0 all bonded (one closed ring).

    Adjacency matrix:
        [[0, 1, 1],
         [1, 0, 1],
         [1, 1, 0]]

    BFS from atom 0:
        layer 0 = [0]          (the central atom)
        layer 1 = [1, 2]       (both neighbours reached in 1 hop)

    Expected H1 result: 1 independent ring (the triangle itself).
    The ring spans layers 0–1, so G[0, 1] = 1.
    """
    adj = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ], dtype=int)
    # layers as produced by obtain_local_number_environment_big_structures
    layers = [[0], [1, 2]]
    return adj, layers


@pytest.fixture
def tiny_adjacency_line():
    """
    3-atom line: atoms bonded as 0-1-2 (no closed ring).

    Adjacency matrix:
        [[0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]

    BFS from atom 0:
        layer 0 = [0]
        layer 1 = [1]
        layer 2 = [2]

    Expected H1 result: 0 rings (a path graph has no loops).
    """
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=int)
    layers = [[0], [1], [2]]
    return adj, layers


# ---------------------------------------------------------------------------
# Cache management (autouse — runs around every test)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_mu_cache():
    """
    Clear the Möbius inversion cache before and after every test.

    The global ``mu`` dict in disorder.barcode accumulates results across
    calls.  Without clearing it, values computed in one test could affect
    another test, making tests order-dependent.
    """
    from smooth_disorder.barcode import clear_mu_cache
    clear_mu_cache()   # clear before test
    yield              # run the test
    clear_mu_cache()   # clear after test
