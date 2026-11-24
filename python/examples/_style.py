"""Shared matplotlib styling for RadarLab figures (single-hue sequential ramp,
fixed categorical order, recessive grid)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#5f5e58", "#e1e0d9", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "radarlab_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "lines.markersize": 7, "font.size": 10, "figure.dpi": 130,
    "axes.prop_cycle": matplotlib.cycler(color=SERIES),
})


def save(fig, name: str) -> Path:
    p = OUT / name
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    print(f"wrote {p}")
    return p
