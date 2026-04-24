Installation
============

Two variants are available depending on which functionality you need.

Requirements
------------

- Python 3.12 or later

BNE only (lightweight)
----------------------

Installs the Bond-Network Entropy module with Jupyter and the test suite.
No phonopy or PyTorch required.

Using the provided setup script:

.. code-block:: bash

   bash setup_bne.sh

Or manually, from the repository root:

.. code-block:: bash

   python3.12 -m venv .venv_bne
   source .venv_bne/bin/activate
   pip install -e ".[jupyter,dev]"

BNE + Disorder Linewidth
------------------------

Installs everything above plus PyTorch and phonopy.

Manually, from the repository root:

.. code-block:: bash

   python3.12 -m venv .venv_dl
   source ./.venv_dl/bin/activate

   pip install --upgrade pip
   pip install numpy
   pip install phonopy
   pip install -e ".[dl,jupyter,dev]"

See also the provided setup script for optimization flags: `setup_dl.sh`.

.. note::

   For platform-specific phonopy installation instructions, see the
   `phonopy documentation <https://phonopy.github.io/phonopy/install.html>`_.


Phono3py installation for AF diffusivity calculation
----------------------------------------------------

See the provided setup script with optional optimization flags in `setup_phono3py.sh`.

.. note::

   For platform-specific phonopy and phono3py installation instructions, see the
   `phonopy <https://phonopy.github.io/phonopy/install.html>`_ and `phono3py <https://phonopy.github.io/phono3py/install.html>`_ documentations.

Running tests
-------------

.. code-block:: bash

   pytest tests/ -m "not slow" -v   # fast tests only
   pytest tests/ -v                 # full suite (~4 min)