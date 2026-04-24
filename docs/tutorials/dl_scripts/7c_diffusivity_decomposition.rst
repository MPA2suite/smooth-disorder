7c\_diffusivity\_decomposition.py — Transport Decomposition
=============================================================

This script uses the fitted L and R parameters from ``7b_fit_dl_parameters.py``
to compute and plot the full transport decomposition (mirrors Notebook 5):

- **Disorder linewidth** Γ(ω) = Γ_defect + Γ_Casimir and its two contributions
- **PDC model VDOS** — crystal VDOS Lorentzian-broadened by Γ(ω)/2, compared to measured disordered VDOS
- **Propagation velocity** :math:`v_{\rm prop}(\omega) = \sqrt{D(\omega) / \tau (\omega)}` [km/s]
- **Mean free path** :math:`\lambda(\omega) = \sqrt{D(\omega) \cdot \tau (\omega)}` [Å]

where D(ω) is loaded from ``2_irg_t2/diffusivity.hdf5`` (pre-computed externally)
and τ(ω) = 1/Γ(ω) is derived from the fitted linewidth.

**Run after** ``7b_fit_dl_parameters.py``.

.. code-block:: bash

   cd tutorials/disorder_linewidth
   python 7c_diffusivity_decomposition.py

Plots are saved to ``dl_workflow/``:
``disorder_linewidth.png``, ``disordered_vs_pdc_vdos.png``,
``propagation_velocity.png``, ``mean_free_path.png``.

.. literalinclude:: 7c_diffusivity_decomposition.py
   :language: python
   :linenos:
