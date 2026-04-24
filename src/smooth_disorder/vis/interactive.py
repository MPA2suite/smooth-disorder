"""
vis/interactive.py — Visualization for Jupyter Notebooks
=========================================================

This module configures matplotlib and seaborn for interactive use in
Jupyter notebooks.  Import this module at the top of a notebook to get
consistent, readable plot styling.

Usage
-----
.. code-block:: python

    from smooth_disorder.vis.interactive import Colors
    import matplotlib.pyplot as plt

    plt.plot(x, y, color=Colors.blue)
    # or use integer indexing to cycle through colors:
    plt.plot(x, y, color=Colors[0])  # red
    plt.plot(x, y, color=Colors[1])  # orange

"""

import os, sys

import numpy as np
import pandas as pd
import scipy
from scipy.constants import physical_constants

# Set the plot style to 'fivethirtyeight' (a clean, modern style)
matplotlib_style = 'fivethirtyeight'
import matplotlib.pyplot as plt
plt.style.use(matplotlib_style)

# Set seaborn context to 'notebook': increases font sizes and line widths
# relative to the default, making figures easier to read in notebooks
import seaborn as sns
sns.set_context('notebook')


class _Colors(object):
    """
    A collection of named hex colors for consistent plot styling.

    Named attributes (e.g. ``Colors.blue``) and integer indexing
    (e.g. ``Colors[0]``) are both supported.  Integer indexing cycles
    through a fixed sequence of 13 colors, which is useful when plotting
    many series in a loop.
    """
    # Named colors (hex codes)
    red = '#F15854'
    blue = '#5DA5DA'
    orange = '#FAA43A'
    green = '#60BD68'
    pink = '#F17CB0'
    brown = '#B2912F'
    purple = '#B276B2'
    yellow = '#DECF3F'
    gray = '#4D4D4D'
    cyan = '#00FFFF'
    rebecca_purple = '#663399'
    chartreuse = '#7FFF00'
    dark_red = '#8B0000'

    def __getitem__(self, i):
        """
        Return the i-th color from the cycle (wraps around after 13 colors).

        Parameters
        ----------
        i : int
            Color index.  Cycling is automatic via ``i % 13``.

        Returns
        -------
        str
            Hex color string.
        """
        color_list = [
            self.red,
            self.orange,
            self.green,
            self.blue,
            self.pink,
            self.brown,
            self.purple,
            self.yellow,
            self.gray,
            self.cyan,
            self.rebecca_purple,
            self.chartreuse,
            self.dark_red
        ]
        return color_list[i % len(color_list)]


# Module-level instance — import and use directly:
#   from smooth_disorder.vis.interactive import Colors
Colors = _Colors()
