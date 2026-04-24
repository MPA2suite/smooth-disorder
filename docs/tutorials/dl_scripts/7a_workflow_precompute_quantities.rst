7a\_workflow\_precompute\_quantities.py — Precompute Phonon Quantities
=======================================================================

This script covers the precomputation phase of the Disorder Linewidth pipeline
(mirrors Notebooks 1-4).  It must be run first; its HDF5 outputs are consumed
by scripts 7b and 7c.

What it computes
----------------

1. **Band structure** — phonon dispersion of reference crystal graphite along a
   Γ-A-L-M-Γ-K-H path; saved as ``dl_workflow/crystal_band_structure.png``.
2. **Phonon mesh** — frequencies and group velocities on a ``[128, 128, 32]``
   Γ-centred BZ mesh; saved to ``dl_workflow/mesh_data.hdf5``.
3. **Crystal VDOS and mean group speed** — Lorentzian-broadened (half-width η = 0.6 cm⁻¹);
   saved to ``dl_workflow/crystal_vdos_group_vel.hdf5``.
4. **Disordered VDOS** — Lorentzian-broadened from pre-computed frequencies in
   ``2_irg_t2/irg_t2_frequencies.hdf5``; saved to ``dl_workflow/disordered_vdos.hdf5``.
5. **Density-shifted crystal VDOS and speed** — frequencies scaled by
   ``(ρ_dis/ρ_crys)^(1/3)`` to align with the disordered system; saved to
   ``dl_workflow/reduced_density_crystal_vdos_group_vel.hdf5``.

**Prerequisites:** BNE + DL installation (see `setup_dl.sh`).

.. code-block:: bash

   cd tutorials/disorder_linewidth
   python 7a_workflow_precompute_quantities.py

.. literalinclude:: 7a_workflow_precompute_quantities.py
   :language: python
   :linenos:
