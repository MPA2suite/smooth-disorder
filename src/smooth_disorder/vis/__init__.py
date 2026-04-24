"""
disorder.vis — Visualization utilities
=======================================

``disorder.vis.interactive``: notebook plotting (seaborn/fivethirtyeight style).

The top-level ``disorder.vis`` namespace exposes ``Colors`` from the
interactive module, so ``from smooth_disorder.vis import Colors`` works without
an explicit submodule import.
"""

# Default: expose interactive Colors at the disorder.vis level
from smooth_disorder.vis.interactive import Colors, _Colors

__all__ = ["Colors", "_Colors"]
