2b\_save\_convergence.py — Save Convergence Summary
=====================================================

Loads the raw convergence data from ``AFC_convergence_results.npz``,
applies BZ-weight averaging over all irreducible q-points, and saves
summary arrays (conductivity vs smearing for each temperature).

**Run after** ``2a_convergence_serial.py``.

.. code-block:: bash

   cd tutorials/diffusivity
   python 2b_save_convergence.py

.. literalinclude:: 2b_save_convergence.py
   :language: python
   :linenos: