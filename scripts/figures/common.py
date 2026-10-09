"""Shared style for paper figures (print, 300 dpi, Elsevier widths). Palette matches the graphical abstract."""
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# categorical: clinical / dermoscopic (validated pair), error, neutral
CLIN, DERM = "#2a78d6", "#eb6834"
CLIN_L, DERM_L = "#d6e6f9", "#fbdccd"
ERR, OK = "#d03b3b", "#2e9e5b"
INK, INK2, MUTED, GRID, AXIS = "#1d1d1f", "#52514e", "#9a988f", "#ecebe6", "#bdbbb3"
NEUTRAL = "#8a8882"
SINGLE, DOUBLE = 3.54, 7.48  # inches: 90 mm / 190 mm
MS = 6.5  # inches: text width of the manuscript PDF (left/right margins 2.5 cm); main-text figures are drawn at this width

_family = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
mpl.rcParams.update({
    "font.family": _family, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print("saved", OUT / name)


def panel(ax, letter, title=None, x=-0.02):
    ax.text(x, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, color=INK, ha="right", va="bottom")
    if title:
        ax.text(x + 0.03, 1.06, title, transform=ax.transAxes, fontweight="bold", fontsize=8, color=INK, ha="left",
                va="bottom")
