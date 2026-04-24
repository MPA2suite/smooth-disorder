2a\_DL\_workflow\_precompute.py — Precompute Phonon Quantities
===============================================================

Precomputation phase of the Disorder Linewidth pipeline (mirrors Notebooks 1-4).
Must be run first; its HDF5 outputs are used by ``2b`` and ``2c``.

What it computes
----------------

1. **Band structure** — phonon dispersion of the reference crystal along the
   path defined in ``config.BAND_LABELS``; saved as
   ``dl_workflow/crystal_band_structure.png``.
2. **Phonon mesh** — frequencies and group velocities on ``config.MESH``
   (default ``[128, 128, 32]``) Γ-centred BZ mesh; saved to
   ``dl_workflow/mesh_data.hdf5``.
3. **Crystal VDOS and mean group speed** — extracted using Lorentzians 
   broadened with η = ``config.GAMMA_BROADENING``; saved to
   ``dl_workflow/crystal_vdos_group_vel.hdf5``.
4. **Disordered VDOS** — obtained from pre-computed frequencies in
   ``config.DISORDERED_FREQUENCIES``; saved to
   ``dl_workflow/disordered_vdos.hdf5``.
5. **Density-shifted crystal VDOS and speed** — obtained from frequencies scaled by
   ``(ρ_dis/ρ_crys)^(1/3)`` to align with the disordered system; saved to
   ``dl_workflow/reduced_density_crystal_vdos_group_vel.hdf5``.

**Prerequisites:** BNE + DL installation (see `setup_dl.sh`).

.. code-block:: bash

   cd workflows
   python 2a_DL_workflow_precompute.py

.. literalinclude:: 2a_DL_workflow_precompute.py
   :language: python
   :linenos: