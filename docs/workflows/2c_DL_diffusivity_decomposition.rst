2c\_DL\_diffusivity\_decomposition.py — Diffusivity Decomposition
===================================================================

Post-processing step that decomposes thermal diffusivity into propagation
velocity and mean free path using the fitted *L* and *R* parameters (mirrors
Notebook 5). Reads the model parameters from ``dl_workflow/model_parameters.hdf5``
and precomputed vdos/velocity/diffusivity data, then:

1. Computes the disorder linewidth Γ(ω) from the fitted model.
2. Evaluates the PDC model for the disordered VDOS.
3. Decomposes the Allen-Feldman diffusivity into propagation velocity
   *v*\ :sub:`prop` and mean free path ℓ as a function of frequency.
4. Saves results and generates decomposition plots.

**Run after** ``2b_DL_fit_params.py``.

.. code-block:: bash

   cd workflows
   python 2c_DL_diffusivity_decomposition.py

.. literalinclude:: 2c_DL_diffusivity_decomposition.py
   :language: python
   :linenos: