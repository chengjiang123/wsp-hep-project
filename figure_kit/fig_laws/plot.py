"""Story 3 figure (laws): reads data.csv in this folder, writes fig_laws.pdf/.png here.
  (a) laws recovered vs N      (b) out-of-range error / threshold vs N      (c) recovered by law family at N_FIXED
Edit the settings block, then:  python plot.py   (add --preview to stamp the run count)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style as S  # noqa: E402

# ---------------------------------------------------------------- settings
METHODS = [  # (method column value, legend label, short label at the line end); order = colour slot order
    ("reader_gptoss", "LLM reads weights", "LLM"),
    ("pysr", "PySR", "PySR"),
    ("operon", "Operon", "Operon"),
]
SHOW_EXACT = True        # dashed line: the reader's exact symbolic reading (the paper's stricter rule)
N_FIXED = 1000           # panel (c)
FAMILIES = [("kin", "kinematic\ninvariant"), ("cons", "conserved\nquantity"), ("decay", "decay\nin flight")]
FLOOR = 1e-3             # panel (b): errors below FLOOR x threshold are drawn at the floor (exact formulas give 0)
SIZE = (S.TEXT_W, 2.05)  # inches
OUT = os.path.join(HERE, "fig_laws")
# --------------------------------------------------------------------------


def main(preview=False):
    S.setup()
    df = pd.read_csv(os.path.join(HERE, "data.csv"))
    Ns = sorted(df.N.unique())
    fig, axs = plt.subplots(1, 3, figsize=SIZE, gridspec_kw={"width_ratios": [1.05, 0.95, 1.2], "wspace": 0.45})

    # (a) recovered fraction vs N: per seed, fraction of laws recovered; then mean and SE over seeds
    ax = axs[0]
    ends = []
    for i, (m, lab, short) in enumerate(METHODS):
        d = df[df.method == m]
        if d.empty:
            continue
        per_seed = d.groupby(["N", "seed"]).recovered.mean().groupby("N")
        mu, se = per_seed.mean(), per_seed.sem().fillna(0)
        ax.fill_between(mu.index, mu - se, mu + se, color=S.SLOTS[i], alpha=0.15, lw=0)
        ax.plot(mu.index, mu, color=S.SLOTS[i], marker=S.MARKERS[i], mec="white", mew=0.8, ms=4.5, zorder=3)
        ends.append((mu.iloc[-1], short, mu.index[-1]))
        if SHOW_EXACT and m.startswith("reader") and d.exact.notna().any():
            ex = d.groupby(["N", "seed"]).exact.mean().groupby("N").mean()
            ax.plot(ex.index, ex, color=S.SLOTS[i], ls=(0, (3, 2)), lw=1.0, zorder=2)
            ends.append((ex.iloc[-1], "exact", ex.index[-1]))
    ax.set_xscale("log")
    ax.set_xlim(min(Ns) * 0.7, max(Ns) * 3.2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("labelled examples $N$")
    ax.set_ylabel("laws recovered")
    placed = []
    for y, lab, x in sorted(ends):
        yy = y
        for p in placed:
            if abs(yy - p) < 0.075:
                yy = p + 0.075
        placed.append(yy)
        ax.text(x * 1.25, yy, lab, fontsize=6, color=S.INK2, va="center")
    S.panel_label(ax, "a")

    # (b) out-of-range error relative to the family threshold: median and interquartile range over laws and seeds
    ax = axs[1]
    df["rel"] = (df.err_out_of_range / df.threshold).clip(FLOOR, 1e4).fillna(1e4)
    for i, (m, lab, short) in enumerate(METHODS):
        g = df[df.method == m].groupby("N").rel
        if not len(g):
            continue
        q25, q50, q75 = g.quantile(0.25), g.quantile(0.5), g.quantile(0.75)
        ax.fill_between(q50.index, q25, q75, color=S.SLOTS[i], alpha=0.09, lw=0)
        ax.plot(q50.index, q50, color=S.SLOTS[i], marker=S.MARKERS[i], mec="white", mew=0.8, ms=4.5, zorder=3)
    ax.axhline(1, color=S.INK2, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(min(Ns) * 1.15, 0.62, "recovery threshold", fontsize=6, color=S.INK2, va="top", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.7, 2e4)
    ax.set_yticks([1e-3, 1e-1, 1e1, 1e3])
    ax.set_yticklabels(["≤10⁻³", "10⁻¹", "10¹", "10³"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("labelled examples $N$")
    ax.set_ylabel("out-of-range error / threshold")
    S.panel_label(ax, "b")

    # (c) by family at N_FIXED
    ax = axs[2]
    w = 0.26
    for i, (m, lab, short) in enumerate(METHODS):
        d = df[(df.method == m) & (df.N == N_FIXED)]
        vals = [d[d.family == f].recovered.mean() if (d.family == f).any() else np.nan for f, _ in FAMILIES]
        x = np.arange(len(FAMILIES)) + (i - (len(METHODS) - 1) / 2) * (w + 0.02)
        ax.bar(x, vals, width=w, color=S.SLOTS[i], edgecolor="white", linewidth=0.6, zorder=3)
        for xi, v in zip(x, vals):
            if v == 0:
                ax.text(xi, 0.012, "0", ha="center", va="bottom", fontsize=5.5, color=S.INK2)
    ax.set_xticks(np.arange(len(FAMILIES)))
    ax.set_xticklabels([n for _, n in FAMILIES], fontsize=6.3, linespacing=0.95)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50%", "100%"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(f"recovered at $N$ = {N_FIXED:,}")
    S.panel_label(ax, "c")

    handles = [plt.Line2D([], [], color=S.SLOTS[i], marker=S.MARKERS[i], mec=S.SLOTS[i], ms=4.5, lw=1.4) for i in range(len(METHODS))]
    labels = [lab for _, lab, _ in METHODS]
    if SHOW_EXACT:
        handles.append(plt.Line2D([], [], color=S.SLOTS[0], ls=(0, (3, 2)), lw=1.0))
        labels.append("LLM, exact symbolic reading")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.98), handlelength=2.2, columnspacing=1.6)
    if preview:
        n = df.groupby("method").size()
        fig.text(0.995, -0.04, "preview: " + ", ".join(f"{lab} {n.get(m, 0)}" for m, lab, _ in METHODS) + " runs",
                 ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, OUT)


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
