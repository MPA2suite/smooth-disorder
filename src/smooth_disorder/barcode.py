"""
barcode.py — H1 Persistent Homology Barcodes and Bond-Network Entropy
======================================================================

This module implements the core algorithm for computing the H1 persistent
homology barcode of a local atomic environment and the Bond-Network Entropy
(BNE) derived from it.

**Background**

Amorphous solids do not have long-range order and instead we can look
at the local atomic environments to understand their structure.
In here we will treat a local atomic environment as a selection of atoms
around a given atom and construct a subgraph where atoms are nodes and bonds
between them are edges.

A "barcode" in persistent homology counts topological loops (rings) in that
subgraph at different distance scales. Each atom has a corresponding barcode; the
distribution of barcodes across all atoms is summarized with a single number —
the information entropy of barcode probability distribution 
— called the Bond-Network Entropy (BNE).

**References**

B. Schweinhart, D. Rodney, and J.K. Mason,
*Phys. Rev. E* 101, 052312 (2020)  — definition of H1 barcode used here.

K. Iwanowski, G. Csányi, and M. Simoncelli,
*Phys. Rev. X* 15, 041041 (2025)   — BNE applied to disordered energy materials.
"""

import string, re, struct, sys, math, os
import scipy
import numpy as np
from ase.io import read
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Local environment extraction
# ---------------------------------------------------------------------------

def obtain_local_number_environment_big_structures(
    adjacency_matrix: np.ndarray,
    atom_index: int,
    distances: np.ndarray,
    idx_distances: np.ndarray,
    n_environment_atoms: int,
):
    """
    Extract the local atomic environment (LAE) centred on a given atom.

    The LAE is the subgraph formed by the n_environment_atoms atoms that are
    geometrically closest to the central atom (including central atom).  
    We also record which "layer" (bond-distance shell) each atom belongs to, 
    which is needed for the barcode calculation.

    **How the layer discovery works**

    Layers are found using a breadth-first search (BFS) implemented via
    matrix multiplication:

        current_layer = A @ previous_layer_indicator

    where A is the local adjacency matrix and the indicator vector has 1 at
    atoms already found and 0 elsewhere.  Multiplying spreads the "signal"
    to all bonded neighbours in one step.  By taking the XOR of the new
    indicator with the old one we find only the *newly discovered* atoms,
    i.e. the next shell.

    Parameters
    ----------
    adjacency_matrix : np.ndarray, shape (N, K)
        Global adjacency matrix for the full structure.  Entry [i, j] = 1
        means atoms i and idx_distances[i, j] are bonded, 0 otherwise.
    atom_index : int
        Index of the central atom in the global structure.
    distances : np.ndarray, shape (N, K)
        Pre-computed distances from each atom to itself and its K-1 nearest
        neighbours (K in total; output of obtain_distances_ase or
        obtain_distances_big_structures).
    idx_distances : np.ndarray, shape (N, K)
        Global atom indices corresponding to the distances array.
    n_environment_atoms : int
        Number of atoms to include in the local environment (n in "LAE of n").

    Returns
    -------
    local_adjacency_matrix : np.ndarray, shape (n_environment_atoms, n_environment_atoms)
        Adjacency matrix restricted to the local environment.
    layers : list of lists
        layers[k] contains the local indices of atoms first reached in hop k.
        layers[0] = [local_atom_index] (the central atom itself).
    local_atom_index : int
        Index of the central atom within the local environment (0 … n-1).
    global_index : np.ndarray, shape (n_environment_atoms,)
        Global atom indices of the atoms in the local environment, sorted.
    """

    # -----------------------------------------------------------------------
    # Step 1: identify which n atoms form the local environment
    # -----------------------------------------------------------------------

    # argpartition gives the n_environment_atoms smallest distances in O(n)
    # (faster than a full sort which would be O(n log n))
    local_index = np.argpartition(distances[atom_index], n_environment_atoms)[:n_environment_atoms]

    # convert local position indices back to global atom indices
    global_index = idx_distances[atom_index][local_index]

    # sort global_index so the local adjacency matrix rows/cols have a
    # consistent ordering
    sorted_global_index_idx = np.argsort(global_index)
    global_index = global_index[sorted_global_index_idx]

    # -----------------------------------------------------------------------
    # Step 2: build the local adjacency matrix
    # -----------------------------------------------------------------------
    # For each atom in the local environment, take the row of the global
    # adjacency matrix and restrict it to the other atoms in the environment.

    local_adjacency_matrix = []
    for idx in global_index:
        # which of this atom's neighbours are also in the local environment?
        mask = np.isin(idx_distances[idx], global_index)

        # adjacency entries for the neighbours that are in the environment
        row_adjacency_matrix = adjacency_matrix[idx][mask]
        
        row_connection_order = list(idx_distances[idx][mask])

        # pad to length n_environment_atoms in case fewer neighbours exist
        pad_amount = n_environment_atoms - len(row_connection_order)
        row_adjacency_matrix = np.pad(row_adjacency_matrix, (0, pad_amount), constant_values=(0, 0))
        row_connection_order = np.array(
            row_connection_order + list(global_index[np.logical_not(np.isin(global_index, row_connection_order))]))

        # reorder columns to match the sorted global_index ordering
        local_adjacency_matrix.append(row_adjacency_matrix[np.argsort(row_connection_order)])

    local_adjacency_matrix = np.array(local_adjacency_matrix)

    # -----------------------------------------------------------------------
    # Step 3: BFS to find which bond-distance layer each atom belongs to
    # -----------------------------------------------------------------------

    # find the row index of the central atom within the local environment
    local_atom_index = np.arange(0, n_environment_atoms, 1)[global_index == atom_index][0]

    # indicator vector: 1 at the central atom, 0 everywhere else
    atom_int_array = np.zeros(n_environment_atoms)
    atom_int_array[local_atom_index] = 1

    # boolean mask of all atoms found so far (starts with just the centre)
    local_environment = np.zeros(n_environment_atoms).astype(bool)
    local_environment[local_atom_index] = True

    # layer 0 is the central atom itself
    layers = [[local_atom_index]]

    search_bool = True
    while search_bool:
        # matrix–vector multiply propagates the "signal" to all bonded atoms
        atom_int_array = local_adjacency_matrix @ atom_int_array

        # XOR: atoms that are newly reached (in current shell but not before)
        current_environment = local_environment.copy()
        local_environment = local_environment | atom_int_array.astype(bool)
        current_layer = local_environment ^ current_environment

        if np.isclose(current_layer.sum(), 0.0):  # no new atoms → BFS done
            search_bool = False
            break

        layers.append(list(np.arange(0, n_environment_atoms, 1)[current_layer]))

    return local_adjacency_matrix, layers, local_atom_index, global_index


# ---------------------------------------------------------------------------
# Möbius inversion cache
# ---------------------------------------------------------------------------

# mu stores previously computed values of the Möbius inversion function.
# This avoids recomputing the same values repeatedly (memoization).
# The key is a string encoding the four integer arguments: "(a,b),(c,d)".
mu = {}


def clear_mu_cache():
    """
    Clear the Möbius inversion memoization cache.

    The mu dictionary accumulates values across calls to recursive_find_mu.
    Call this function in test teardown to prevent state leaking between tests.
    In normal usage (single script) it never needs to be called.
    """
    mu.clear()


# ---------------------------------------------------------------------------
# Möbius inversion function
# ---------------------------------------------------------------------------

def recursive_find_mu(mu: dict, a: int, b: int, c: int, d: int):
    """
    Recursively compute the Möbius inversion function μ((a,b),(c,d)).

    **Background**

    The Möbius inversion is a mathematical tool from combinatorics that
    "inverts" a cumulative sum over a partially ordered set.  Here the
    partial order is defined on pairs of non-negative integers (i, j) with
    i ≤ j, ordered by (a,b) ≤ (c,d)  iff  c ≤ a  and  d ≥ b.

    In the barcode calculation, F[a,b] counts rings in a *union* of shells
    a through b.  The Möbius inversion is used to extract G[c,d] — rings
    that first appear in the interval [c,d] specifically — from the
    cumulative F values.

    Reference: B. Schweinhart et al., Phys. Rev. E 101, 052312 (2020).

    **Memoization**

    Results are stored in the `mu` dictionary so each (a,b,c,d) combination
    is computed only once, then reused on later calls.

    Parameters
    ----------
    mu : dict
        Lookup table passed by reference; modified in-place.
    a, b, c, d : int
        Integer indices satisfying c ≤ a, a ≤ b, b ≤ d.

    Returns
    -------
    int
        The value of the Möbius function μ((a,b),(c,d)).
    """
    # sanity check: the arguments must satisfy the ordering constraints
    if a < c or b > d or b < a or d < c:
        raise Exception("invalid subshell")

    element = f"({a},{b}),({c},{d})"

    if element in mu:
        # result already known — return immediately
        return mu[element]

    else:
        if a == c and b == d:
            # base case: μ((a,b),(a,b)) = 1 by definition
            mu[element] = 1
            return mu[element]

        else:
            # recursive case: sum over all strictly smaller (e,f) intervals
            result = 0

            for e in range(c, a + 1):
                for f in range(b, d + 1):
                    if e == c and f == d:
                        continue  # skip (c,d) itself to avoid circular reference

                    new_element = f"({a},{b}),({e},{f})"
                    if new_element in mu:
                        result += -mu[new_element]
                    else:
                        result += -recursive_find_mu(mu, a, b, e, f)

            
            mu[element] = result
            return mu[element]


# ---------------------------------------------------------------------------
# H1 barcode computation
# ---------------------------------------------------------------------------

def obtain_H1_barcode(adjacency_matrix: np.ndarray, layers: List[List[int]], mu: dict):
    """
    Compute the H1 persistent homology barcode for a local atomic environment.

    **What is an H1 barcode?**

    H1 (first homology) counts closed loops (rings) in a graph.  A "barcode"
    in persistent homology is a record of which loops exist at different
    distance scales.  Here we count loops present in each pair of shells
    (i, j), where shell i contains all atoms reachable in exactly i bond hops
    from the centre.

    **Algorithm overview**

    1. Compute F[i,j] = number of *algebraically independent* rings in the union
       of shells i through j, using the Euler characteristic formula
       ``rank = (components) - (atoms) + (edges)``.
       The rank of the first homology equals the number of independent loops.

    2. Apply the Möbius inversion to go from cumulative F to local G:
       ``G[c,d] = Σ_{(a,b) ≤ (c, d)} F[a,b] * μ((a,b),(c,d))``.
       G[c,d] counts rings that *start* in shell c and *end* in shell d
       (i.e. rings that first close when shell d is reached).

    Parameters
    ----------
    adjacency_matrix : np.ndarray, shape (n, n)
        Local adjacency matrix of the environment (from
        obtain_local_number_environment_big_structures).
    layers : list of lists
        Layer assignment from the BFS (output of same function).
    mu : dict
        Möbius function cache (pass the module-level `mu` dict).

    Returns
    -------
    G : np.ndarray, shape (n_layers, n_layers)
        H1 barcode matrix.  G[c,d] = number of independent rings that first
        appear between shells c and d.
    F : np.ndarray, shape (n_layers, n_layers)
        Cumulative ring count matrix.  F[i,j] = total independent rings in
        shells i through j.
    """
    n_steps = len(layers)  # number of shells (including the central atom as shell 0)

    # -----------------------------------------------------------------------
    # Step 1: compute F[i,j] for all shell pairs using the Euler characteristic
    # -----------------------------------------------------------------------
    F = np.zeros((n_steps, n_steps))

    for i in range(0, n_steps):
        for j in range(i, n_steps):

            # collect all atom indices in shells i through j (inclusive)
            shell_layer_atoms = []
            for layer in layers[i:j + 1]:
                shell_layer_atoms += layer

            n_shell_atoms = len(shell_layer_atoms)

            # restrict the adjacency matrix to only these atoms
            shell_adjacency_matrix = adjacency_matrix[shell_layer_atoms, :][:, shell_layer_atoms]

            # count bonds (edges) — each bond is counted twice in the matrix,
            # so divide by 2
            n_shell_edges = int(shell_adjacency_matrix.sum() / 2)

            # count connected components using SciPy's sparse graph routine
            n_shell_components = scipy.sparse.csgraph.connected_components(shell_adjacency_matrix)[0]

            # Euler characteristic formula for the first Betti number (H1 rank):
            #   β₁ = edges - atoms + connected_components
            rank_shell = n_shell_components - n_shell_atoms + n_shell_edges
            F[i, j] = rank_shell

    # -----------------------------------------------------------------------
    # Step 2: apply Möbius inversion to get the local barcode G from F
    # -----------------------------------------------------------------------
    G = np.zeros((n_steps, n_steps))

    for c in range(0, n_steps):
        for d in range(c, n_steps):
            result = 0
            for a in range(c, d + 1):
                for b in range(a, d + 1):
                    element = f"({a},{b}),({c},{d})"
                    if element in mu:
                        result += F[a, b] * mu[element]
                    else:
                        result += F[a, b] * recursive_find_mu(mu, a, b, c, d)
            G[c, d] = result

    # G is the H1 barcode; F is the cumulative ring-count matrix
    return G, F


# ---------------------------------------------------------------------------
# Barcode reduction (canonical form)
# ---------------------------------------------------------------------------

def reduce_barcode(G_matrix):
    """
    Remove trailing all-zero rows and columns from a barcode matrix.

    **Why this is needed**

    Two barcodes should be considered identical if they encode the same
    topological information.  A zero-padded matrix and its trimmed version
    describe the same rings, but would compare as different arrays.  Reducing
    to a canonical (minimal) form allows us to count how many atoms share the
    same barcode type.

    The function works recursively: if the last column sums to zero, it (and
    the matching last row) carry no ring information and are removed.  We keep
    trimming until either a 1x1 matrix remains or the last column is non-zero.

    Parameters
    ----------
    G_matrix : np.ndarray, shape (m, m)
        Barcode matrix to reduce.

    Returns
    -------
    np.ndarray
        Reduced barcode matrix with trailing zero rows/columns removed.
    """
    if G_matrix.shape == (1, 1):
        # base case: cannot reduce further
        reduced_G_matrix = G_matrix
    elif np.isclose(G_matrix[:, -1].sum(), 0.0):
        # last column is all zeros — remove it (and the matching last row)
        reduced_G_matrix = G_matrix[:-1, :-1]
        reduced_G_matrix = reduce_barcode(reduced_G_matrix)
    else:
        # last column has non-zero entries — stop
        reduced_G_matrix = G_matrix

    return reduced_G_matrix