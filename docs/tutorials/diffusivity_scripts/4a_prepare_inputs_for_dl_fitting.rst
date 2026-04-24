4a\_prepare\_inputs\_for\_dl\_fitting.py — Prepare DL Fitting Inputs
======================================================================

Extracts phonon frequencies and diffusivities from
``data_save/IC_dataset_tensor.npz`` (produced by step 3b) and saves them
to ``irg_t9_216_frequencies.hdf5`` and ``irg_t9_216_diffusivity.hdf5``. 
This file is the bridge between the Allen-Feldman diffusivity workflow 
and the Disorder Linewidth fitting workflow.

**Run after** ``3b_tensor_conductivity_save_process.py``.

.. code-block:: bash

   cd tutorials/diffusivity
   python 4a_prepare_inputs_for_dl_fitting.py

.. literalinclude:: 4a_prepare_inputs_for_dl_fitting.py
   :language: python
   :linenos: