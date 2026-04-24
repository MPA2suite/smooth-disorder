2b\_DL\_fit\_params.py — Fit L and R Parameters
=================================================

Fits the two disorder-linewidth model parameters (mirrors Notebook 6):

- **L** [nm] — grain-boundary size (Casimir scattering length)
- **R** [1e-6 THz cm nm³] — defect scattering amplitude (Rayleigh-type)

using L-BFGS with strong-Wolfe line search via PyTorch autodiff. Starting
values are taken from ``config.L0`` and ``config.R0``; the optimizer runs for
up to ``config.MAX_ITER`` iterations.

The fitted parameters are saved to ``dl_workflow/model_parameters.hdf5``
(keys: ``final_loss``, ``final_model_params``).

**Run after** ``2a_DL_workflow_precompute.py``.

.. code-block:: bash

   cd workflows
   python 2b_DL_fit_params.py

.. literalinclude:: 2b_DL_fit_params.py
   :language: python
   :linenos: