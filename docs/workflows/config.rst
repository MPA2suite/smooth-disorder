config.py — Shared Workflow Configuration
==========================================

Central configuration file imported by all workflow scripts. Edit this file
to point to your own structures, force-constant files, and diffusivity data
before running any 1-series or 2-series script.

Parameter groups
----------------

**Paths**
   Output directories for BNE and DL results (``BNE_WORK_DIR``, ``DL_WORK_DIR``),
   the reference crystal POSCAR and FC2 (``CRYSTAL_POSCAR``, ``CRYSTAL_FC2``),
   the disordered system structure (``DISORDERED_POSCAR``), and pre-computed
   vibrational frequencies and diffusivity for the disordered system
   (``DISORDERED_FREQUENCIES``, ``DISORDERED_DIFFUSIVITY``).

**Phonon mesh**
   Supercell expansion matrix (``SUPERCELL_MATRIX``), BZ mesh dimensions
   (``MESH = [128, 128, 32]``), Γ-centering flag, and Lorentzian half-width
   η for VDOS broadening (``GAMMA_BROADENING = 0.6 cm⁻¹``).

**Band structure**
   High-symmetry q-point path (``BAND_PATH``) and labels (``BAND_LABELS``)
   for the phonon dispersion plot produced by ``2a_DL_workflow_precompute.py``.

**Fitting**
   Initial grain-boundary mean free path ``L0`` [nm], defect scattering
   amplitude ``R0`` [1e-6 THz cm nm³], L-BFGS learning rate ``LR``,
   maximum iterations ``MAX_ITER``, line-search strategy
   ``LINE_SEARCH_FN = "strong_wolfe"``, and fallback phonon lifetime for
   frequencies above the computed range (``EXTRAPOLATION_VALUE``).

**BNE**
   Bond distance cutoff (``CUTOFF = 1.8 Å``), number of pre-computed
   nearest-neighbour distances per atom (``N_SMALLEST = 300``), structure
   label index (``STRUCTURE_IDX``), LAE sizes (``LOCAL_ENVIRONMENT_NAT``),
   and growth-rate normalisation window (``N_START``, ``N_STOP``).

.. literalinclude:: config.py
   :language: python
   :linenos: