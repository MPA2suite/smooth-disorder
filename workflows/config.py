import numpy as np
from smooth_disorder.structural import THzToCm

# ── Paths ──────────────────────────────────────────────────────────────────────

# Root output directory for BNE results
BNE_WORK_DIR = "./bne_workflow"
# Subdirectory name for BNE output files
BNE_FOLDER = "bond_network_entropy"
# Root output directory for disorder-linewidth results
DL_WORK_DIR = "./dl_workflow"

# Reference crystal structure (POSCAR format)
CRYSTAL_POSCAR = "./ref_crystal/POSCAR"
# Second-order force constants for the reference crystal
CRYSTAL_FC2 = "./ref_crystal/fc2.hdf5"
# Disordered system structure (POSCAR format)
DISORDERED_POSCAR = "./disordered_system/irg_t9_14009.vasp"
# Pre-computed vibrational frequencies for the disordered system
DISORDERED_FREQUENCIES = "./disordered_system/irg_t9_frequencies.hdf5"
# Pre-computed diffusivity data for the disordered system
DISORDERED_DIFFUSIVITY = "./disordered_system/irg_t9_diffusivity.hdf5"

# ── Phonon mesh ────────────────────────────────────────────────────────────────

# Supercell expansion matrix used to build the second order force constants
SUPERCELL_MATRIX = np.diag([8, 8, 2])
# q-point mesh dimensions for phonon sampling
MESH = [128, 128, 32]
# Whether the q-point mesh is Gamma-centered
GAMMA_CENTER = True
# Lorentzian half-width η for VDOS broadening [cm⁻¹]
GAMMA_BROADENING = 0.6

# ── Band structure ─────────────────────────────────────────────────────────────

# High-symmetry q-point path through the Brillouin zone (fractional coordinates)
BAND_PATH = [[[0, 0, 0], [0., 0., 0.5], [0.5, 0., 0.5],
              [0.5, 0., 0.], [0., 0., 0.],
              [0.33333333, 0.33333333, 0.], [0.33333333, 0.33333333, 0.5]]]
# Labels for the high-symmetry points along the band path
BAND_LABELS = [r"$\Gamma$", "A", "L", "M", r"$\Gamma$", "K", "H"]

# ── Fitting ────────────────────────────────────────────────────────────────────

# Initial grain boundary mean free path [nm]
L0 = 3.3
# Initial defect scattering amplitude [1e-6 THz cm nm³]
R0 = 5.54
# Learning rate for the L-BFGS optimizer
LR = 1.0
# Maximum number of L-BFGS iterations
MAX_ITER = 50
# Line search strategy for L-BFGS
LINE_SEARCH_FN = "strong_wolfe"
# Fallback phonon lifetime for frequencies above the maximum computed value [s]
EXTRAPOLATION_VALUE = 1 / 24 * THzToCm * 1e-12

# ── BNE ───────────────────────────────────────────────────────────────────────

# Bond distance cutoff for building the adjacency matrix [Å]
CUTOFF = 1.8
# Number of nearest-neighbor distances pre-computed per atom (must exceed max LAE size)
N_SMALLEST = 300
# Label index for this structure (used in output subdirectory names)
STRUCTURE_IDX = 0
# LAE sizes over which BNE is computed
LOCAL_ENVIRONMENT_NAT = list(range(10, 31))
# Start of LAE window used for growth-rate normalization in plots
N_START = 14
# End of LAE window used for growth-rate normalization in plots
N_STOP = 30
