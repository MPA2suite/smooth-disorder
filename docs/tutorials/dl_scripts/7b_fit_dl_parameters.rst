7b\_fit\_dl\_parameters.py — Fit L and R Parameters
======================================================

This script fits the two disorder-linewidth parameters (mirrors Notebook 6):

- **L** [nm] — grain-boundary mean free path (Casimir scattering length)
- **R** [cm THz nm³] — defect scattering amplitude (Rayleigh-type)

using L-BFGS with strong-Wolfe line search via PyTorch autodiff.

**Run after** ``7a_workflow_precompute_quantities.py``.

The fitted parameters are saved to ``dl_workflow/model_parameters.hdf5``
(keys: ``final_loss``, ``final_model_params``).

.. code-block:: bash

   cd tutorials/disorder_linewidth
   python 7b_fit_dl_parameters.py

.. literalinclude:: 7b_fit_dl_parameters.py
   :language: python
   :linenos:
