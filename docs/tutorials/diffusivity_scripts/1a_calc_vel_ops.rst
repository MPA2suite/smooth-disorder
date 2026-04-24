1a\_calc\_vel\_ops.py — Compute Velocity Operators
====================================================

Computes the velocity-operator matrix elements at each irreducible q-point
using phono3py/WTE and saves them to ``velocity_operators/save_{iq}.hdf5``.
These files are required by all subsequent steps.

The velocity operator is built from the dynamical matrix and its
momentum-space derivatives evaluated on a Γ-centred BZ mesh. One HDF5 file
is written per irreducible q-point.

**Prerequisites:** phono3py environment (``source activate_phono3py.sh``).

.. code-block:: bash

   cd tutorials/diffusivity
   python 1a_calc_vel_ops.py

.. literalinclude:: 1a_calc_vel_ops.py
   :language: python
   :linenos: