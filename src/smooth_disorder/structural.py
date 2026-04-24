"""
structural.py — Structure I/O, Distance Calculations, and Periodic Boundaries
==============================================================================

This module handles everything related to reading atomic structures and
computing the distances between atoms, taking periodic boundary conditions
(PBC) into account.

**Background**

Crystal and glass structures are described by:
  - **Atomic positions**: the (x, y, z) coordinates of each atom in Å.
  - **Lattice vectors**: three vectors (a, b, c) that define the repeating
    unit cell.  The real material is infinite — it is the unit cell tiled in
    all directions.

Because the disordered materials are treated as large periodic unit cells,
the shortest distance between two atoms is
*not* simply the straight-line distance between their coordinates in the
primary cell.  We must also consider copies of each atom in neighbouring
cells.  Selecting the shortest such distance is called the
**Minimum Image Convention (MIC)**.

Two MIC implementations are provided:
  - ``obtain_distances_ase``: uses the ASE library (recommended, more accurate
    for non-orthorhombic cells).
  - ``obtain_distances_big_structures``: manual implementation, faster for
    very large structures but approximate for large off-diagonal cells.

**Reference**

Appendix C of https://pure.rug.nl/ws/portalfiles/portal/2839530/03_c3.pdf
explains the limitations of the single-image MIC approximation.
"""

import string, re, struct, sys, math, os
import scipy
import numpy as np
import ase
from ase.io import read
from typing import Tuple, List
from scipy.constants import physical_constants
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

rY_TO_CMM1 = 109737.315701113          # Rydberg to cm⁻¹ conversion
hARTREE_SI = 4.35974394E-18            # Hartree energy in Joules
rYDBERG_SI = hARTREE_SI / 2.0          # Rydberg energy in Joules
H_PLANCK_SI = 6.62606896E-34           # Planck's constant in J·s
ryau_sec = H_PLANCK_SI / rYDBERG_SI    # Rydberg atomic unit of time in seconds
k_BOLTZMANN_SI = 1.3806504E-23         # Boltzmann constant in J/K
aU_SEC = H_PLANCK_SI / (2. * np.pi) / hARTREE_SI   # atomic unit of time in seconds
aU_TERAHERTZ = aU_SEC * 1.0E+12        # atomic unit of frequency in THz
rY_TO_THZ = 1.0 / aU_TERAHERTZ / (4 * np.pi)       # Rydberg to THz
k_BOLTZMANN_RY = k_BOLTZMANN_SI / rYDBERG_SI        # Boltzmann in Rydberg/K
BOHR_TO_M = 0.52917720859E-10          # Bohr radius in metres
ryvel_si = BOHR_TO_M / ryau_sec        # Rydberg velocity unit in m/s
SpeedOfLight = 299792458               # Speed of light in m/s
THzToCm = 1.0e12 / (SpeedOfLight * 100)  # THz to cm⁻¹: ≈ 33.356410
Angstrom = 1.0e-10                     # 1 Ångström in metres
THz = 1.0e12                           # 1 THz in Hz
tpi = 2.0 * np.pi                      # 2π


# ---------------------------------------------------------------------------
# Structure I/O
# ---------------------------------------------------------------------------

def obtain_positions_and_lattice_vectors(primitive_filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an atomic structure file and return atomic positions and lattice vectors.

    Uses ASE (Atomic Simulation Environment), which supports many common
    formats: VASP POSCAR, CIF, XYZ, extended XYZ, and more.  ASE autodetects
    the format from the filename extension or content.

    Parameters
    ----------
    primitive_filename : str
        Path to the structure file (e.g. ``"POSCAR"``, ``"structure.cif"``).

    Returns
    -------
    positions : np.ndarray, shape (num_atoms, 3)
        Cartesian coordinates of each atom in Ångström.
    lattice_vectors : np.ndarray, shape (3, 3)
        Rows are the three lattice vectors a, b, c (in Ångström).
        ``lattice_vectors[0]`` = **a**, ``lattice_vectors[1]`` = **b**, etc.
    """
    cell = read(primitive_filename)
    positions = cell.get_positions()
    lattice_vectors = np.array(cell.cell)

    return positions, lattice_vectors


# ---------------------------------------------------------------------------
# Distance calculations
# ---------------------------------------------------------------------------

def obtain_distances_big_structures(
    atomic_positions: np.ndarray,
    lattice_vectors: np.ndarray,
    n_smallest: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the n_smallest nearest-neighbour distances for every atom using a
    manual Minimum Image Convention (MIC).

    This function is a manual implementation of the MIC.  It is provided as
    an alternative to ``obtain_distances_ase`` for very large structures where
    the ASE call may be slow.  Note: it is only an approximation for large
    distances in non-orthorhombic cells.

    **How it works**

    For each atom i:
      1. Compute the vector from i to every other atom j: Δr = r_j - r_i.
      2. Express Δr in fractional coordinates (units of the lattice vectors)
         by multiplying by the inverse lattice matrix.
      3. Apply the MIC: if any fractional coordinate is outside [-0.5, 0.5],
         shift it by ±1 so it moves inside that range.  This picks the
         nearest periodic image of atom j.
      4. Convert back to Cartesian coordinates and compute the distance.
      5. Keep only the n_smallest distances using argpartition (O(n)).

    Parameters
    ----------
    atomic_positions : np.ndarray, shape (N, 3)
        Cartesian positions in Ångström.
    lattice_vectors : np.ndarray, shape (3, 3)
        Lattice vectors as rows.
    n_smallest : int
        Number of nearest neighbours (including central atom) to keep per atom.

    Returns
    -------
    distances : np.ndarray, shape (N, n_smallest)
        Sorted nearest-neighbour distances for each atom (in Ångström).
    idx_distances : np.ndarray, shape (N, n_smallest)
        Global atom indices corresponding to those distances.
    """

    distances = []
    idx_distances = []
    nat = len(atomic_positions)

    k = 0
    for atom_pos in tqdm(atomic_positions, desc="Distances"):

        # displacement vectors from this atom to all others (shape: N × 3)
        difference = atomic_positions - atom_pos

        # convert to fractional coordinates: each component is now in units
        # of the corresponding lattice vector
        scaled_difference = np.matmul(difference, np.linalg.inv(lattice_vectors))

        # sanity checks: scaled differences should be in (-1.5, 1.5)
        assert (scaled_difference >= 1.5).sum() == 0
        assert (scaled_difference <= -1.5).sum() == 0
        assert np.isnan(scaled_difference).sum() == 0

        # minimum image: wrap fractional coordinates into [-0.5, 0.5]
        # If a component is > 0.5 we are closer via the −a image; subtract 1.
        # If a component is < −0.5 we are closer via the +a image; add 1.
        scaled_difference = np.where(scaled_difference > 0.5,
                                     scaled_difference - 1,
                                     np.where(scaled_difference < -0.5, scaled_difference + 1, scaled_difference))

        # convert back to Cartesian and compute Euclidean distance
        difference = np.matmul(scaled_difference, lattice_vectors)
        distance = np.sqrt(np.square(difference).sum(axis=1))

        # keep only the n_smallest nearest neighbours (O(n) via argpartition)
        if n_smallest < nat:
            idx_distance = np.argpartition(distance, n_smallest)[:n_smallest]
        else:
            idx_distance = np.arange(0, nat, 1)

        # sort within the selected n_smallest by distance
        idx_distance = idx_distance[np.argsort(distance[idx_distance])]

        idx_distances.append(idx_distance)
        distances.append(distance[idx_distance])

        k += 1

    distances = np.array(distances)
    idx_distances = np.array(idx_distances)
    return distances, idx_distances


def obtain_distances_ase(
    atoms: ase.atoms.Atoms,
    n_smallest: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the n_smallest nearest-neighbour distances for every atom using
    the ASE Minimum Image Convention (preferred method).

    ASE's ``get_distances`` method handles the MIC correctly for all cell
    shapes (orthorhombic, monoclinic, triclinic), making it more accurate
    than the manual implementation for non-cubic cells.

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        ASE Atoms object containing positions and cell (lattice vectors).
        Load from file with ``ase.io.read(filename)``.
    n_smallest : int
        Number of nearest neighbours (including central atom) to keep per atom.

    Returns
    -------
    distances : np.ndarray, shape (N, n_smallest)
        Sorted nearest-neighbour distances (in Ångström).
    idx_distances : np.ndarray, shape (N, n_smallest)
        Global atom indices corresponding to those distances.
    """

    distances = []
    idx_distances = []

    nat = len(atoms)
    atom_indices = np.arange(0, nat, 1)

    for k in tqdm(range(len(atoms)), desc="Distances"):

        # ASE computes MIC-corrected distances from atom k to all others
        distance = atoms.get_distances(k, atom_indices, mic=True)

        # keep only the n_smallest nearest neighbours using argpartition
        if n_smallest < nat:
            idx_distance = np.argpartition(distance, n_smallest)[:n_smallest]
        else:
            idx_distance = np.arange(0, nat, 1)

        # sort by distance within the selected neighbours
        idx_distance = idx_distance[np.argsort(distance[idx_distance])]

        idx_distances.append(idx_distance)
        distances.append(distance[idx_distance])

    distances = np.array(distances)
    idx_distances = np.array(idx_distances)
    return distances, idx_distances


# ---------------------------------------------------------------------------
# Derived structural quantities
# ---------------------------------------------------------------------------

def obtain_density(atoms: ase.atoms.Atoms) -> float:
    """
    Compute the mass density of a material from its unit cell.

    The density is calculated as:
        ρ = total_mass / volume

    where total_mass is the sum of all atomic masses in the cell (in atomic
    mass units, converted to grams) and volume is the cell volume in cm³.

    Parameters
    ----------
    atoms : ase.atoms.Atoms
        ASE Atoms object with positions, cell, and chemical species.

    Returns
    -------
    float
        Density in g/cm³.
    """
    volume_A3 = atoms.get_volume()          # cell volume in Ångström³
    total_mass = atoms.get_masses().sum()   # sum of atomic masses in amu

    # convert amu/Å³ → kg/m³, then to g/cm³
    amu = physical_constants['atomic mass constant'][0]  # kg
    conversion_factor = amu * 1e30  # converts amu/Å³ to kg/m³

    density_kg_m3 = total_mass / volume_A3 * conversion_factor

    return density_kg_m3 / 1e3  # g/cm³
