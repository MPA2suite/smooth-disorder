5\_BNE\_workflow.py — BNE Batch Script
======================================

This script computes the Bond-Network Entropy for a configurable range of LAE sizes
and saves one HDF5 file per size.  It is the data source for Notebook 6
(BNE growth-rate analysis).

**When to run:** before opening Notebook 6, unless you are using the pre-computed
reference data already included in ``tutorials/bond_network_entropy/data/``.

.. code-block:: bash

   cd tutorials/bond_network_entropy
   python 5_BNE_workflow.py

Output files are written to ``data/bond_network_entropy/structure_0/entropy_number_<N>.hdf5``.
To change the LAE size range, edit the ``local_environment_nat`` list in the
``if __name__ == "__main__":`` block at the bottom of the script.

.. literalinclude:: 5_BNE_workflow.py
   :language: python
   :linenos:
