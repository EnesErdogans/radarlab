"""Matplotlib style for the SIU 2027 paper figures (IEEE single column, 8 pt).

Figures are written as vector PDF (TrueType fonts embedded) to
docs/paper/siu2027/figures/ and the numbers behind them as JSON next to them,
so every value quoted in the paper can be traced to a file.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "docs" / "paper" / "siu2027" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

COL_W = 3.45  # inch; IEEE column is 3.5 in
# categorical order (validated: CVD ΔE ≥ 9 for adjacent pairs); slot 3 is below 3:1
# contrast on white, so every series also has its own marker / line style.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#7a4fc9"]
INK, MUTED, GRID = "#0b0b0b", "#5f5e58", "#e1e0d9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": "#8a8a84", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.formatter.use_locale": False,
})


def save(fig, name: str) -> Path:
    p = FIG / name
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {p}")
    return p


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def dump(obj, name: str) -> Path:
    p = FIG / name
    p.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False))
    print(f"wrote {p}")
    return p


def tr_num(ax):
    """Turkish decimal comma on linear axes."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: (f"{v:g}").replace(".", ","))
    return f
