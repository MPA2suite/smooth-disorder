2a\_convergence\_serial.py — AF Smearing Convergence Study
===========================================================

Runs the Allen-Feldman diffusivity calculation over a range of Lorentzian
smearing values η and temperatures. For each (η, T) pair it evaluates the
AF thermal conductivity using the velocity operators produced by step 1a.

Results are saved as ``AFC_convergence_results.npz`` and consumed by
``2b_save_convergence.py``.

**Run after** ``1a_calc_vel_ops.py``.

.. code-block:: bash

   cd tutorials/diffusivity
   python 2a_convergence_serial.py

.. literalinclude:: 2a_convergence_serial.py
   :language: python
   :linenos: