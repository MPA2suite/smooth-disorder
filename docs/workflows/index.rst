Workflows
=========

Standalone terminal scripts for computing Bond-Network Entropy and Disorder
Linewidth for IRG T9 irradiated graphite. All shared configuration — file
paths, phonon mesh, fitting hyperparameters, and BNE settings — lives in
:doc:`config`; every workflow script imports from it.

The structure of irradiated graphite used in the workflow can be found at the following link: https://doi.org/10.24435/materialscloud:jm-cg. 
If you use this structure in your research, you should cite both :cite:t:`wf-iwanowski_bond-network_2025` and :cite:t:`wf-farbos_time-dependent_2017`.

**Configuration**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - :doc:`config`
     - All paths, phonon mesh, band-structure path, L-BFGS fitting parameters, and BNE settings

**1-series — Bond-Network Entropy**

.. list-table::
   :header-rows: 1
   :widths: 5 45 50

   * - Step
     - Script
     - Topic
   * - 1a
     - :doc:`1a_BNE_workflow`
     - Compute BNE for IRG T9 across all configured LAE sizes; save one HDF5 per size
   * - 1b
     - :doc:`1b_BNE_plot`
     - Read HDF5 results from 1a; plot BNE vs LAE size and growth-rate analysis

**2-series — Disorder Linewidth**

.. list-table::
   :header-rows: 1
   :widths: 5 45 50

   * - Step
     - Script
     - Topic
   * - 2a
     - :doc:`2a_DL_workflow_precompute`
     - Phonon mesh, crystal VDOS + group speed, disordered VDOS, density-shifted VDOS (mirrors Notebooks 1-4)
   * - 2b
     - :doc:`2b_DL_fit_params`
     - Fit *L* and *R* with L-BFGS via PyTorch autodiff; saves ``model_parameters.hdf5`` (mirrors Notebook 6)
   * - 2c
     - :doc:`2c_DL_diffusivity_decomposition`
     - Disorder linewidth, propagation velocity, mean free path (mirrors Notebook 5)

.. note::

   Run scripts in order: **1a → 1b** (BNE series) and **2a → 2b → 2c** (DL series).
   All scripts must be executed from the ``workflows/`` directory.
   The 2-series requires the BNE + DL installation (see `setup_dl.sh`).

.. bibliography::
   :filter: docname in docnames
   :labelprefix: W
   :keyprefix: wf-

.. toctree::
   :maxdepth: 1
   :hidden:

   config
   1a_BNE_workflow
   1b_BNE_plot
   2a_DL_workflow_precompute
   2b_DL_fit_params
   2c_DL_diffusivity_decomposition