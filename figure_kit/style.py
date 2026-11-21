"""Shared figure style for the paper: vector PDF, Helvetica 7-8 pt, thin marks, recessive axes.
Categorical colours: the first three slots of the validated reference palette (all-pairs CVD-safe, light mode);
each series also gets its own marker and a direct label, so identity never rests on colour alone."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
SLOTS = [C["blue"], C["orange"], C["aqua"]]
MARKERS = ["o", "s", "^"]
TEXT_W = 6.75  # inches, two-column conference text width (single-column arXiv: same width)


def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=INK, ha="right", va="bottom")


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print("wrote", path + ".pdf/.png")
