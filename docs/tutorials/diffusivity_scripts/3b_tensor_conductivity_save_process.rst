3b\_tensor\_conductivity\_save\_process.py — Assemble Conductivity Tensors
===========================================================================

Collects all per-q-point, per-mode-batch HDF5 files written by
``3a_tensor_conductivity_save.py``, applies BZ weighting, and assembles the
full Allen-Feldman conductivity tensor dataset into
``data_save/IC_dataset_tensor.npz``.

**Run after** ``3c_launch_serial.py`` (which calls ``3a`` for every q-point).

.. code-block:: bash

   cd tutorials/diffusivity
   python 3b_tensor_conductivity_save_process.py

.. literalinclude:: 3b_tensor_conductivity_save_process.py
   :language: python
   :linenos: