"""Where reading the weights works: reads ../fig_laws/data.csv, writes fig_coverage.pdf/.png.
One panel per law family; x = labelled examples N; y = fraction of laws recovered (shared out-of-range rule), on the
instances every method was run on (seed 0, N = 100 / 300 / 1000 / 3000 / 10000, all 65 laws). Methods are offset a
little along x so that equal values stay visible; n under each point = laws in that cell. Grey dashed: how often the trained
network itself passes the same rule (the most a weight reader can be expected to read).
Bottom row: the same, only on instances where the network itself passes. A cell where no seed-0 network passes uses
the passing networks of seeds 1-2 instead, for every method (marked n=k*).
Edit the settings block, then:  python plot.py [--preview]"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style as S  # noqa: E402

# ---------------------------------------------------------------- settings
NS = [100, 300, 1000, 3000, 10000]
DX = 0.035                      # x offset between methods, in decades (so overlapping lines stay visible)
SEEDS = [0]
FAMILIES = [("cons", "conserved quantity"), ("kin", "kinematic invariant"), ("decay", "decay in flight")]
BLUES = ["#1c4f8f", "#2a78d6", "#8fb8ea"]           # one hue for the weight readers, darker = larger model
METHODS = [  # (method, label, colour, marker, line style); Opus = hollow rings over Sonnet's dots, so both show at 100 %
    ("reader_opus55sub", "Opus 5.5 reads weights", BLUES[0], "o", "-"),
    ("reader_sonnet55sub", "Sonnet 5.5 reads weights", BLUES[1], "o", (0, (3, 1.5))),
    ("reader_gptoss", "gpt-oss-120b reads weights", BLUES[2], "o", "-"),
    ("pysr", "PySR", S.SLOTS[1], "s", "-"),
    ("operon", "Operon", S.SLOTS[2], "^", "-"),
]
SIZE = (S.TEXT_W, 4.1)
# --------------------------------------------------------------------------


def main(preview=False):
    S.setup()
    full = pd.read_csv(os.path.join(HERE, "..", "fig_laws", "data.csv"))
    d = full[full.N.isin(NS) & full.seed.isin(SEEDS)]
    net = d[d.method == "network"]
    fig, axs = plt.subplots(2, 3, figsize=SIZE, sharex=True, gridspec_kw={"hspace": 0.32, "wspace": 0.22})
    for row, only_pass in enumerate([False, True]):
        for c, (fam, flab) in enumerate(FAMILIES):
            ax = axs[row, c]
            sub = d[d.family == fam]
            fallback = set()
            if only_pass:
                sub = sub[sub.network_pass == 1]
                for N in NS:   # no seed-0 network passes here: use the passing networks of the other seeds, for every method
                    if sub[(sub.N == N) & (sub.method == METHODS[0][0])].empty:
                        extra = full[(full.family == fam) & (full.N == N) & (~full.seed.isin(SEEDS)) & (full.network_pass == 1)]
                        if not extra.empty:
                            sub = pd.concat([sub, extra])
                            fallback.add(N)
            else:
                g = net[net.family == fam].groupby("N").recovered.mean()
                ax.plot(g.index, g.values, color=S.MUTED, ls=(0, (3, 2)), lw=1.1, marker="D", ms=3, zorder=1)
            for i, (m, lab, col, mk, ls) in enumerate(METHODS):
                g = sub[sub.method == m].groupby("N").recovered.agg(["mean", "count"])
                if g.empty:
                    continue
                x = g.index.values * 10 ** (DX * (i - (len(METHODS) - 1) / 2))
                hollow = m == "reader_opus55sub"
                ax.plot(x, g["mean"], color=col, marker=mk, ms=5.2 if hollow else 3.4, mfc="none" if hollow else col,
                        mec=col if hollow else "white", mew=1.0 if hollow else 0.5, ls=ls, lw=1.2,
                        zorder=10 if hollow else 3 + (len(METHODS) - i))
            ax.axhspan(0.95, 1.02, color="#eef3fb", zorder=0, lw=0)
            ax.set_xscale("log")
            ax.set_xticks(NS)
            ax.set_xticklabels(["10²", "300", "10³", "3000", "10⁴"], fontsize=6)
            ax.set_xlim(70, 14000)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            ax.set_ylim(-0.15, 1.04)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.axhline(-0.055, color=S.GRID, lw=0.6, zorder=0)
            if c == 0:
                ax.set_ylabel("laws recovered" + ("\n(network passes)" if only_pass else "\n(all)"))
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(flab, fontsize=7.5, color=S.INK, loc="left", pad=4)
            else:
                ax.set_xlabel("labelled examples $N$")
            n = sub[sub.method == METHODS[0][0]].groupby("N").size()
            for N in NS:
                k = int(n.get(N, 0))
                ax.text(N, -0.11, f"n={k}" + ("*" if N in fallback else ""), fontsize=5.0, color=S.MUTED, ha="center", va="center")
                if N in fallback:
                    ax.text(0.40, 0.20, "* N = 10²: seeds 1–2\n(no seed-0 network passes)", transform=ax.transAxes, fontsize=4.8, color=S.MUTED)
                if k == 0:
                    ax.text(N * 1.15, 0.12, "no network\npasses", fontsize=5.0, color=S.MUTED, ha="left", va="center")
            S.panel_label(ax, "abcdef"[3 * row + c])
    handles = [plt.Line2D([], [], color=col, marker=mk, ls=ls, ms=5.2 if m == "reader_opus55sub" else 4,
                          mfc="none" if m == "reader_opus55sub" else col, mec=col if m == "reader_opus55sub" else "white",
                          mew=1.0 if m == "reader_opus55sub" else 0.5, lw=1.4) for m, _, col, mk, ls in METHODS]
    handles += [plt.Line2D([], [], color=S.MUTED, ls=(0, (3, 2)), marker="D", ms=3, lw=1.1), plt.Rectangle((0, 0), 1, 1, color="#eef3fb")]
    labels = [lab for _, lab, *_ in METHODS] + ["trained network itself", "95–100 %"]
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.97), handlelength=2.0, columnspacing=1.2)
    if preview:
        fig.text(0.995, 0.0, "preview", ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, os.path.join(HERE, "fig_coverage"))


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
