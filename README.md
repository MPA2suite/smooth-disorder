# Smooth Disorder — Bond-Network Entropy and Disorder Linewidth Library

A Python library that introduces three complementary tools for connecting structural disorder to thermal transport:

- computation of **Allen-Feldman (AF) conductivity** and diffusivity starting from a structure's unit cell and second-order force constants,
- computation of **Bond-Network Entropy (BNE)**, a descriptor of disorder for network solids,
- fitting of **Disorder Linewidth (DL)**, a measure of scattering due to disorder that allows decomposition of diffusivity in disordered materials.

See the detailed documentation and tutorials at https://smooth-disorder.readthedocs.io

**Paper:**
K. Iwanowski, G. Csányi, and M. Simoncelli,
*Physical Review X* **15**, 041041 (2025)
— [DOI: 10.1103/w4p6-b9mp](https://doi.org/10.1103/w4p6-b9mp)

**Library Authors:** Kamil Iwanowski, Michele Simoncelli (Columbia University)

---

## Installation

Two variants are available depending on which functionality you need.

### Requirements

- Python 3.12 or later

### BNE only (lightweight)

Installs the bond-network entropy module with Jupyter and the test suite.
No phonopy or PyTorch required.

```bash
bash setup_bne.sh
```

Or manually, from the repository root:

```bash
python3.12 -m venv .venv_bne
source .venv_bne/bin/activate
pip install -e ".[jupyter,dev]"
```

### BNE + Disorder Linewidth

Installs everything above plus PyTorch and phonopy.

Manually, from the repository root:

```bash
python3.12 -m venv .venv_dl
source ./.venv_dl/bin/activate

pip install --upgrade pip
pip install numpy
pip install phonopy
pip install -e ".[dl,jupyter,dev]"
```

See also the provided setup script with optional optimization flags: `setup_dl.sh`.

> **Note:** For platform-specific phonopy installation instructions, see the
> [phonopy documentation](https://phonopy.github.io/phonopy/install.html).

### Phono3py (for AF diffusivity calculation)

See the provided setup script with optional optimization flags in `setup_phono3py.sh`.

> **Note:** For platform-specific phonopy and phono3py installation instructions, see the
> [phonopy](https://phonopy.github.io/phonopy/install.html) and
> [phono3py](https://phonopy.github.io/phono3py/install.html) documentation.

---

## Quick start

### Bond-Network Entropy

```python
from ase.io import read
from smooth_disorder.structural import obtain_distances_ase
from smooth_disorder.barcode import (
    obtain_local_number_environment_big_structures,
    obtain_H1_barcode,
    reduce_barcode,
    mu,
)
import numpy as np

# 1. Load a structure (VASP POSCAR, CIF, XYZ, …)
atoms = read("tutorials/bond_network_entropy/data/structural/silica_glass_5184_atoms/POSCAR")
n_atoms = len(atoms)

# 2. Compute nearest-neighbour distances (MIC-corrected)
distances, idx_distances = obtain_distances_ase(atoms, n_smallest=300)

# 3. Build the adjacency matrix (bonds = distances below the cutoff)
cutoff = 2.1  # Å — Si-O bond cutoff for silica glass
adjacency_matrix = ((distances < cutoff) & (distances > 0.1)).astype(int)

# 4. Compute the H1 barcode for each atom and collect the distribution
n_environment = 30  # local environment size (number of atoms)
barcodes = []

for atom_index in range(n_atoms):
    adj, layers, local_idx, global_idx = obtain_local_number_environment_big_structures(
        adjacency_matrix, atom_index, distances, idx_distances, n_environment
    )
    G, F = obtain_H1_barcode(adj, layers, mu)
    G = reduce_barcode(G)
    barcodes.append(G)

# 5. Compute the Shannon entropy (Bond-Network Entropy)
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

print(f"Bond-Network Entropy (n={n_environment}): {BNE:.4f}")
```

### Disorder Linewidth

The DL module computes the phonon mesh and VDOS in the reference crystal from second-order force constants, applies a
frequency-shift correction to align the crystal and disordered VDOS, then fits the disorder
linewidth parameters *L* and *R* against the disordered VDOS.  Given the disorder linewidth it decomposes thermal diffusivity into propagation velocity and mean free path.

For the theoretical background see the [paper](https://doi.org/10.1103/w4p6-b9mp).
For a step-by-step walkthrough, work through the six notebooks in `tutorials/disorder_linewidth/`.
For direct code implementation, run the `workflows/` 2-series scripts (requires the full
`setup_dl.sh` installation).

---

## Tutorials

### Allen-Feldman Diffusivity

The `tutorials/diffusivity/` directory provides a terminal-level tutorial for computing the
Allen-Feldman (AF) diffusivity of irradiated graphite (IRG T9, 216-atom unit cell). The resulting
diffusivity and frequencies arrays feed directly into the disorder linewidth workflow.

#### Required inputs

- `POSCAR` — primitive cell of IRG T9 (216 atoms, included in the directory)
- `fc2.hdf5` — second-order force constants (file is too large for GitHub; please download it from
  [Google Drive](https://drive.google.com/drive/folders/16loux_gkvg3oDMCR8urRwfGbaMyPewpc?usp=sharing) and place in `tutorials/diffusivity/`)

#### Setup

Activate the phono3py environment before running any script:

```bash
source tutorials/diffusivity/activate_phono3py.sh
```

> **Note:** The mesh variable must be consistent across scripts `1a`, `2b`, `3b`, and `3c`.

#### Script sequence

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1a_calc_vel_ops.py` | Compute velocity-operator elements per irreducible q-point; saves `velocity_operators/save_{iq}.hdf5` |
| 2 | `2a_convergence_serial.py` | Run AF diffusivity over a range of Lorentzian smearing values η and temperatures to find the converged η |
| 3 | `2b_save_convergence.py` | Load raw convergence data, apply BZ-weight averaging over irreducible q-points, save convergence summary |
| 4 | `2c_plot_convergence.py` | Plot AF conductivity vs smearing η at each temperature; plateau identifies the converged η |
| 5 | `3c_launch_serial.py` | Serial launcher — computes AF conductivity tensors per q-point using the converged η |
| 6 | `3b_tensor_conductivity_save_process.py` | Assembles per-q-point tensors into a single dataset |
| 7 | `4a_prepare_inputs_for_dl_fitting.py` | Packages AF results as HDF5 inputs for the disorder linewidth workflow |

#### Workflow

```bash
cd tutorials/diffusivity/
source activate_phono3py.sh
python 1a_calc_vel_ops.py
python 2a_convergence_serial.py
python 2b_save_convergence.py
python 2c_plot_convergence.py
python 3c_launch_serial.py
python 3b_tensor_conductivity_save_process.py
python 4a_prepare_inputs_for_dl_fitting.py
```

---

### Bond-Network Entropy

The `tutorials/bond_network_entropy/` directory contains a step-by-step tutorial series that walks through computing BNE
for silica glass, from raw structure file to growth-rate analysis.

#### Setup

Install the package with Jupyter support, then launch the notebooks:

```bash
pip install -e ".[jupyter]"
jupyter notebook tutorials/bond_network_entropy/
```

#### Notebook sequence

| Step | File | Topic |
|------|------|-------|
| 1 | `1_structure_preparation_and_bonding.ipynb` | Load a structure (POSCAR/CIF/XYZ), compute pairwise MIC distances, identify the bond cutoff from the distance distribution |
| 2 | `2_plot_coordination_number_distribution.ipynb` | Build the adjacency matrix, plot coordination number distributions |
| 3 | `3_H1_barcode.ipynb` | Extract local atomic environments (LAE), compute H₁ persistent homology barcodes, interpret bar positions as algebraically independent rings |
| 4 | `4_Bond_Network_Entropy.ipynb` | Compute the full barcode distribution for all atoms, calculate BNE via Shannon entropy, visualise the most common environments |
| 5 | `5_BNE_workflow.py` | **Run from the terminal** — computes BNE for all LAE sizes 1–80 and saves HDF5 results (precomputed for the notebook 6) |
| 6 | `6_BNE_growth_rate.ipynb` | Load the HDF5 results, plot BNE as a function of LAE size, compute the growth rate and saturation behaviour |

#### Running the workflow script

Notebook 6 requires pre-computed data. In a separate terminal, run:

```bash
cd tutorials/bond_network_entropy/
python 5_BNE_workflow.py
```

This saves results to `tutorials/bond_network_entropy/data/bond_network_entropy/`. Pre-computed reference data for
silica glass (LAE sizes 1–80) is already included in that directory.

---

### Disorder Linewidth

The `tutorials/disorder_linewidth/` directory contains a six-notebook series covering the full DL
pipeline for irradiated graphite, from phonon band structure and VDOS to disorder linewidth fitting
and thermal transport decomposition.

#### Setup

```bash
bash setup_dl.sh                          # full installation
jupyter notebook tutorials/disorder_linewidth/
```

#### Notebook sequence

| Step | File | Topic |
|------|------|-------|
| 1 | `1_crystals_and_BTE_vs_disordered_systems.ipynb` | Phonon band structure of crystal graphite; contrast BTE in simple crystals vs WTE in disordered systems |
| 2 | `2_VDOS_comparison_between_crystal_and_irradiated_graphite.ipynb` | Compute VDOS for crystal and irradiated graphite; observe disorder-induced broadening |
| 3 | `3_Debye_model_and_Lorentzian_spectral_functions.ipynb` | Debye model, Lorentzian spectral functions, and the defect-scattering linewidth ansatz |
| 4 | `4_frequency_shift_correction.ipynb` | Frequency-shift correction to align crystal and disordered VDOS before fitting |
| 5 | `5_disorder_linewidth_and_mean_free_path.ipynb` | Disorder linewidth, phonon mean free path, and their relation to thermal transport |
| 6 | `6_fit_disorder_linewidth_parameters.ipynb` | Fit *L* and *R* parameters with L-BFGS via PyTorch autodiff |

The `workflows/` 2-series scripts implement the same pipeline end-to-end; the notebooks are the
recommended starting point to understand what each workflow step does.

---

## Workflows

The `workflows/` directory provides standalone terminal scripts for computing BNE and disorder
linewidth for IRG T9 irradiated graphite. Shared configuration (file paths, mesh settings,
fitting hyperparameters) lives in `workflows/config.py`.

### 1-series — Bond-Network Entropy for IRG T9

| Script | Purpose |
|--------|---------|
| `1a_BNE_workflow.py` | Compute BNE for IRG T9 across all configured LAE sizes; save results to HDF5 |
| `1b_BNE_plot.py` | Plot BNE vs LAE size and growth-rate analysis |

### 2-series — Disorder Linewidth for IRG T9

| Script | Purpose |
|--------|---------|
| `2a_DL_workflow_precompute.py` | Compute phonon mesh, VDOS, and frequency-shift correction for crystal and IRG T9; save to HDF5 |
| `2b_DL_fit_params.py` | Fit *L* and *R* disorder-linewidth parameters using L-BFGS (PyTorch) |
| `2c_DL_diffusivity_decomposition.py` | Decompose thermal diffusivity into propagation velocity and mean free path |

---

## Repository layout

```
src/smooth_disorder/       Core library
  barcode.py               H1 barcode computation and BNE
  structural.py            Structure I/O, distances, periodic boundaries
  disorder_linewidth.py    Phonon mesh, VDOS, linewidth model, fitting
  vis/
    interactive.py         Plotting for Jupyter notebooks

tests/                     Automated test suite (run with pytest)
tutorials/
  bond_network_entropy/    BNE tutorial notebooks, workflow script, and reference data
  diffusivity/             AF diffusivity tutorial scripts (phono3py environment)
  disorder_linewidth/      DL tutorial notebooks and workflow scripts
workflows/                 Standalone terminal scripts for BNE and DL (IRG T9)
```

---

## Running tests

```bash
pytest tests/ -m "not slow" -v   # fast tests only
pytest tests/ -v                 # full suite (~4 min)
```

---

## Citation

If you use this software, please cite:

```bibtex
@article{iwanowski_bond-network_2025,
	title = {Bond-{Network} {Entropy} {Governs} {Heat} {Transport} in {Coordination}-{Disordered} {Solids}},
	volume = {15},
	url = {https://link.aps.org/doi/10.1103/w4p6-b9mp},
	doi = {10.1103/w4p6-b9mp},
	abstract = {Understanding how the vibrational and thermal properties of solids are influenced by atomistic structural disorder is of fundamental scientific interest and paramount to designing materials for next-generation energy technologies. While several studies indicate that structural disorder strongly influences the thermal conductivity, the fundamental physics governing the disorder-conductivity relation remains elusive. Here we show that order-of-magnitude, disorder-induced variations of conductivity in network solids can be predicted from a “bond-network” entropy, an atomistic structural descriptor that quantifies heterogeneity in the topology of the atomic-bond network. We employ the Wigner formulation of thermal transport to demonstrate the existence of a relation between the bond-network entropy and observables such as smoothness of the vibrational density of states and macroscopic conductivity. We also show that the smoothing of the vibrational density of states encodes information about the thermal resistance induced by disorder and can be directly related to phenomenological models for phonon-disorder scattering based on the semiclassical Peierls-Boltzmann equation. Our findings rationalize the conductivity variations of disordered carbon polymorphs ranging from nanoporous electrodes to defective graphite used as a moderator in nuclear reactors.},
	number = {4},
	urldate = {2025-12-12},
	journal = {Physical Review X},
	publisher = {American Physical Society},
	author = {Iwanowski, Kamil and Csányi, Gábor and Simoncelli, Michele},
	month = dec,
	year = {2025},
	pages = {041041},
}
```

---

## Licence

This software is distributed under the Academic Software Licence.
See [LICENSE](LICENSE) for details.
If you are interested in a commercial use of this software, please contact Michele Simoncelli (michele.simoncelli@gmail.com).