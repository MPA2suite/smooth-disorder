import sys
import os
import shutil
import glob

sys.path.insert(0, os.path.abspath("../src"))

# Copy notebooks into the docs source tree at build time so nbsphinx can
# resolve image paths correctly. The copies are gitignored.
_docs = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_docs)
for _src, _dst in [
    (os.path.join(_root, "tutorials", "bond_network_entropy"), os.path.join(_docs, "tutorials", "bne_notebooks")),
    (os.path.join(_root, "tutorials", "disorder_linewidth"),   os.path.join(_docs, "tutorials", "dl_notebooks")),
]:
    os.makedirs(_dst, exist_ok=True)
    for _nb in glob.glob(os.path.join(_src, "*.ipynb")):
        shutil.copy2(_nb, _dst)

# Copy tutorial Python scripts so literalinclude can resolve them from their .rst pages.
for _src, _dst, _pat in [
    (os.path.join(_root, "tutorials", "bond_network_entropy"), os.path.join(_docs, "tutorials", "bne_scripts"),          "5_BNE_workflow.py"),
    (os.path.join(_root, "tutorials", "diffusivity"),          os.path.join(_docs, "tutorials", "diffusivity_scripts"),  "[0-9]*.py"),
    (os.path.join(_root, "tutorials", "disorder_linewidth"),   os.path.join(_docs, "tutorials", "dl_scripts"),           "7[abc]*.py"),
    (os.path.join(_root, "workflows"),                         os.path.join(_docs, "workflows"),                         "*.py"),
]:
    os.makedirs(_dst, exist_ok=True)
    for _f in glob.glob(os.path.join(_src, _pat)):
        shutil.copy2(_f, _dst)

# -- Project information -------------------------------------------------------

project = "Smooth Disorder"
copyright = "2026, Kamil Iwanowski and Michele Simoncelli"
author = "Kamil Iwanowski, Michele Simoncelli"
release = "1.0.0"

# -- General configuration -----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc -------------------------------------------------------------------

# Mock heavy optional dependencies so autodoc works without phonopy/torch installed.
# seaborn and matplotlib are also mocked to avoid a seaborn→scipy.stats crash at
# import time when building docs without a full runtime environment.
autodoc_mock_imports = ["phonopy", "torch", "seaborn", "matplotlib"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Napoleon (NumPy docstring parser) -----------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_returns = True
napoleon_use_rtype = False

# -- Autosection labels --------------------------------------------------------

# Prefix labels with the document name to avoid cross-file conflicts
autosectionlabel_prefix_document = True
# Only auto-label top-level sections; avoids conflicts from same-named sections in docstrings
autosectionlabel_maxdepth = 1

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable", None),
    "scipy":      ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "ase":        ("https://wiki.fysik.dtu.dk/ase", None),
}

# -- nbsphinx ------------------------------------------------------------------

# Never re-execute notebooks at build time; use pre-computed cell outputs
nbsphinx_execute = "never"

# -- Bibliography --------------------------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"

# -- HTML output ---------------------------------------------------------------

html_theme = "furo"
html_logo = "_static/SD_logo.001.jpeg"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/MPA2suite/smooth-disorder",
    "source_branch": "main",
    "source_directory": "docs/",
}
