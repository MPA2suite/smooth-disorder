3a\_tensor\_conductivity\_save.py — Per-q Conductivity Tensor
===============================================================

Computes the Allen-Feldman conductivity tensor contribution for a single
irreducible q-point ``iq`` and a block of mode indices
``[num_start, num_stop)``. Saves the result as a partial HDF5 file postprocessed
by ``3b_tensor_conductivity_save_process.py``.

This script is not run directly; it is dispatched in a loop by
``3c_launch_serial.py``, which iterates over all q-points and mode batches.

.. code-block:: bash

   # Called internally by 3c_launch_serial.py:
   python 3a_tensor_conductivity_save.py <iq> <num_start> <num_stop> <gamma_min_plateau_cmm1>

.. literalinclude:: 3a_tensor_conductivity_save.py
   :language: python
   :linenos: