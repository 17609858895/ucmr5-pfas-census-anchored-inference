"""Shared imports and the series-to-colour map for this paper's figures (nature_new palette via the copied figkit)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "figkit"))
from publication_layout import *          # noqa: F401,F403  (save_modules, simple, style_axis, basemap, map_points, PAL, INK, MUTED ...)
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FD = OUTPUT / "figure_data"

# one series -> one exact palette colour, shared by every figure (PAL = nature_new)
C = {
    # monitoring status
    "census": PAL[1], "sampled": PAL[0], "unmonitored": PAL[6],
    # estimands / composition
    "systems": PAL[4], "people": PAL[3],
    "cls0": PAL[6], "cls1": PAL[0], "cls2": PAL[4], "cls3": PAL[1],
    # observed vs model
    "observed": PAL[0], "plugin": PAL[1], "recal": PAL[3], "additive": PAL[7], "ratio": PAL[4],
    # prior feature sets
    "m_sdwis": PAL[6], "m_geo": PAL[1], "m_nb": PAL[3],
    # designs + estimators
    "SRS + Hájek (classical)": PAL[6], "SRS + PPI++": PAL[4], "PPS-pop + Hájek (EPA-like)": PAL[0],
    "Neyman-H10 + HT (classical)": PAL[1], "Neyman-H10 + PPI++ (hybrid)": PAL[7], "SRS + GREG-ML": PAL[5],
    "ANCHOR (Poisson + PPI++)": PAL[3],
    # external validation sets
    "UCMR5 small sample (national)": PAL[3], "Michigan": PAL[1], "Minnesota": PAL[7], "New Hampshire": PAL[6],
}
CLASS_ORDER = ["≤100", "101–500", "501–1,000", "1,001–3,300"]


def pct(ax, axis="y", digits=0):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.{digits}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def legend_strip(fig, ss, handles, ncol, **kw):
    la = fig.add_subplot(ss); la.axis("off")
    la.legend(handles=handles, loc="center", ncol=ncol, **{"handletextpad": .35, "columnspacing": 1.0, **kw})
    return la
