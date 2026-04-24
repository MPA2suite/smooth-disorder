3c\_launch\_serial.py — Serial Conductivity Launcher
======================================================

Iterates over all irreducible q-points and mode batches and calls
``3a_tensor_conductivity_save.py`` for each combination via
``os.system``. This is the entry point for the full conductivity tensor
computation.

The ``gamma_min_plateau_cmm1`` threshold (default 8 cm⁻¹) should be
determined from the convergence study (steps 2a-2c).

**Run after** ``1a_calc_vel_ops.py``.

.. code-block:: bash

   cd tutorials/diffusivity
   python 3c_launch_serial.py

.. literalinclude:: 3c_launch_serial.py
   :language: python
   :linenos: