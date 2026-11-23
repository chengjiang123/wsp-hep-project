"""Story 3, SRBench check: reads summary.csv in this folder, writes fig_srbench.pdf/.png (one panel).
All 130 ground-truth laws: symbolic solution rate vs target noise. Grey: the 14 published methods (SRBench 2021 table);
our reruns of Operon (published settings) and PySR (SRBench config), Operon's published curve dashed. It validates the
SR baselines used in fig_laws against the published benchmark.
(The monomial-subset reader was dropped 2026-11-21: a fixed rounding rule read the fitted unit as well or better.)
Solution = SRBench's rule; band = 1 SE over seeds. Edit the settings block, then:  python plot.py [--preview]"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style as S  # noqa: E402

# ---------------------------------------------------------------- settings
NOISE = [0.0, 0.001, 0.01, 0.1]
NOISE_LAB = ["0", "10⁻³", "10⁻²", "10⁻¹"]
OURS = [  # (method, label, colour, marker, line style); colours follow fig_laws (PySR orange, Operon aqua)
    ("pysr", "PySR, our rerun", S.SLOTS[1], "s", "-"),
    ("operon2021", "Operon, our rerun", S.SLOTS[2], "^", "-"),
]
PUB_HIGHLIGHT = ("Operon", "Operon, published", S.SLOTS[2], "^")
PUB_LABEL = ["AIFeynman"]                         # published methods named next to their first point in panel a
GREY = "#c9c8c3"
SIZE = (3.25, 2.6)              # one column
# --------------------------------------------------------------------------


def curve(summ, src, method, subset, col="solved_fraction"):
    g = summ[(summ.source == src) & (summ.method == method) & (summ.subset == subset)].set_index("noise")
    if g.empty:
        return None
    x = [i for i, n in enumerate(NOISE) if n in g.index]
    m = np.array([g.loc[NOISE[i], col] for i in x], float)
    se = np.nan_to_num(np.array([g.loc[NOISE[i], "se_over_seeds"] for i in x], float)) if col == "solved_fraction" else 0 * m
    return np.array(x), m, se


def draw(ax, xs, m, se, color, marker, ls, hollow=False, z=3):
    ax.fill_between(xs, np.clip(m - se, 0, 1), np.clip(m + se, 0, 1), color=color, alpha=0.13, lw=0)
    ax.plot(xs, m, color=color, ls=ls, marker=marker, ms=4.2, mec=color if hollow else "white",
            mfc="white" if hollow else color, mew=0.9, zorder=z)


def axes_common(ax, title):
    ax.set_xticks(range(len(NOISE)))
    ax.set_xticklabels(NOISE_LAB)
    ax.set_xlim(-0.25, len(NOISE) - 0.75)
    ax.set_xlabel("target noise")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=7.5, color=S.INK, loc="left", pad=4)


def main(preview=False):
    S.setup()
    summ = pd.read_csv(os.path.join(HERE, "summary.csv"))
    fig, ax = plt.subplots(figsize=SIZE)
    pub = summ[(summ.source == "published") & (summ.subset == "all")]
    for m in sorted(pub.method.unique()):
        if m == PUB_HIGHLIGHT[0]:
            continue
        c = curve(summ, "published", m, "all")
        ax.plot(c[0], c[1], color=GREY, lw=0.8, zorder=1)
        if m in PUB_LABEL:
            ax.text(0.12, c[1][0] + 0.005, m, fontsize=5.5, color=S.MUTED, ha="left", va="bottom")
    c = curve(summ, "published", PUB_HIGHLIGHT[0], "all")
    if c is not None:
        draw(ax, c[0], c[1], 0 * c[1], PUB_HIGHLIGHT[2], PUB_HIGHLIGHT[3], (0, (3, 2)), hollow=True, z=2)
    for m, lab, col, mk, ls in OURS:
        c = curve(summ, "ours", m, "all")
        if c is not None:
            draw(ax, *c, col, mk, ls)
    ax.set_ylabel("symbolic solution rate")
    axes_common(ax, "SRBench ground truth, 130 laws\nsymbolic regression only, no LLM")
    ax.set_ylim(0, 0.82)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls=ls, mec="white", mfc=c, ms=4.2, lw=1.4) for _, _, c, mk, ls in OURS]
    labels = [lab for _, lab, *_ in OURS]
    handles += [plt.Line2D([], [], color=PUB_HIGHLIGHT[2], marker=PUB_HIGHLIGHT[3], ls=(0, (3, 2)), mfc="white", ms=4.2, lw=1.4),
                plt.Line2D([], [], color=GREY, lw=0.8)]
    labels += [PUB_HIGHLIGHT[1], "published (SRBench 2021)"]
    ax.legend(handles, labels, loc="upper right", fontsize=5.8, handlelength=2.6, ncol=2, columnspacing=1.0)
    if preview:
        n = summ[(summ.source == "ours") & (summ.subset == "all")].groupby("method").runs.sum()
        fig.text(0.995, -0.04, "preview: " + ", ".join(f"{m} {int(v)}" for m, v in n.items()) + " verified runs",
                 ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, os.path.join(HERE, "fig_srbench"))


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
