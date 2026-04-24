1b\_BNE\_plot.py — Plot BNE Growth Rate
=========================================

Reads the HDF5 entropy files written by ``1a_BNE_workflow.py``, obtains
BNE as a function of LAE size, and computes the growth-rate normalisation
within the window ``[config.N_START, config.N_STOP]``. Produces plots of BNE
vs LAE size and the normalised growth rate.

**Run after** ``1a_BNE_workflow.py``.

.. code-block:: bash

   cd workflows
   python 1b_BNE_plot.py

.. literalinclude:: 1b_BNE_plot.py
   :language: python
   :linenos: