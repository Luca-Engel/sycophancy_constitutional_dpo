"""Shared matplotlib palette/style constants, so every chart in this repo
(``plot_comparison.py``, ``notebooks/data_exploration.ipynb``) reads as one
visual system instead of each picking its own defaults.

Import the color constants directly, and call ``apply_axes_style(ax, fig)``
after creating a figure to get the shared surface color, spine, and
gridline treatment.
"""

from __future__ import annotations

# Categorical palette, assigned in a fixed order rather than left to
# matplotlib's arbitrary default cycle.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#3fa66f",  # green
    "#b1459c",  # magenta
    "#c9a227",  # gold
    "#5a5dcc",  # indigo
    "#4fa6b0",  # teal
)

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"


def apply_axes_style(ax, fig=None) -> None:
    """Apply the shared surface/spine/gridline/tick treatment to ``ax``
    (and ``fig``'s facecolor, if given). Caller still sets title/labels."""
    if fig is not None:
        fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_LINE)
    ax.tick_params(colors=TEXT_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
