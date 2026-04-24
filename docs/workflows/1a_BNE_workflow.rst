1a\_BNE\_workflow.py — Compute Bond-Network Entropy
=====================================================

Computes Bond-Network Entropy (BNE) for the disordered system across all LAE
sizes defined in ``config.LOCAL_ENVIRONMENT_NAT``. For each size *N* it:

1. Reads the disordered structure from ``config.DISORDERED_POSCAR``.
2. Computes pairwise MIC distances up to ``config.N_SMALLEST`` neighbours.
3. Extracts the H₁ barcode for every atom's local environment of size *N*.
4. Collects the barcode distribution and computes the Shannon entropy (BNE).
5. Saves results to ``<BNE_WORK_DIR>/<BNE_FOLDER>/structure_0/entropy_number_<N>.hdf5``
   (keys: ``entropy``, ``probabilities``, ``number_of_atoms``).

**Prerequisites:** BNE installation (``bash setup_bne.sh``).

.. code-block:: bash

   cd workflows
   python 1a_BNE_workflow.py

.. literalinclude:: 1a_BNE_workflow.py
   :language: python
   :linenos: