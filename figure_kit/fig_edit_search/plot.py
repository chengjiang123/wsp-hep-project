"""Paper Figure 4: direct LLM weight edits vs black-box search on the K0S selection. Reads data.csv here (written by
../export.py from the raw runs), writes fig_edit_search.pdf/.png.
Panels: N = 1e4 and 5e4 labels. y = test background rejection R95 of the round chosen on held-back labels (higher is
better); x = evaluations of the edit-set objective. Shown: the LLM edits (interactive; automated Opus, 3 seeds) and two
ablations without an LLM on the same 24 weights of the eight encoded cuts: CMA-ES (the strongest search) and random
search, plus TPE. Trained models vs label budget (refs_vs_N.csv) are connected curves over the x range of the LLM edits,
one point per N in REF_NS (10^2 ... the panel's N, left to right; their x position is not an evaluation count): dashed =
boosted trees / neural network trained from scratch, solid = the same started from LLM-written weights (WSP + BDT: Opus
5.5 writes and edits the trees, then LightGBM boosts on top; WSP + NN: the LLM-encoded network, then gradient). The search over the LLM's own 41-weight structure (space r7) is in data.csv; add it to SERIES.
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
NS = [(10000, r"$N = 10^4$ labels"), (50000, r"$N = 5\times10^4$ labels")]
SERIES = [  # (method, space, label, colour, marker, z) ; space "llm" for the LLM edits
    ("llm_interactive", "llm", "LLM edits, interactive", S.SLOTS[0], "*", 6),
    ("llm_automated", "llm", "LLM edits, automated (Opus, 3 seeds)", "#8fb8ea", "o", 5),
    ("cmaes", "r0", "CMA-ES on the same 24 weights", S.SLOTS[1], None, 3),
    ("tpe", "r0", "TPE on the same 24 weights", S.SLOTS[2], None, 3),
    ("random", "r0", "random search on the same 24 weights", S.MUTED, None, 2),
]
VSN = [  # trained models vs label budget (refs_vs_N.csv): connected curves over the x range of the LLM edits, one point per N
    # (reference in refs_vs_N.csv, legend label, colour, marker, line style, filled marker)
    ("boosted trees", "boosted trees (BDT)", "#8c564b", "s", (0, (3.5, 2)), False),
    ("LLM-written trees + boosting", "WSP + BDT: Opus 5.5 writes the trees, then boosting", "#8c564b", "s", "-", True),
    ("neural network, gradient from random weights", "neural network (NN) from random weights", S.INK2, "^", (0, (3.5, 2)), False),
    ("encoded network + gradient", "WSP + NN: LLM-encoded network, then gradient", S.INK2, "^", "-", True),
]
REF_NS = [100, 300, 1000, 3000, 10000, 30000, 50000]   # label budgets for BDT / NN (only those up to the panel's N)
METRIC, YLABEL = "test_r95", r"test rejection $R_{95}$ (higher is better)"
XMAX = 2000
SIZE = (S.TEXT_W, 2.6)
# --------------------------------------------------------------------------


def main(preview=False):
    S.setup()
    d = pd.read_csv(os.path.join(HERE, "data.csv"))
    vsn = pd.read_csv(os.path.join(HERE, "refs_vs_N.csv"))
    fig, axs = plt.subplots(1, 2, figsize=SIZE, sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (n, title) in zip(axs, NS):
        for method, space, lab, col, mk, z in SERIES:
            sub = d[(d.method == method) & (d.space == space) & (d.N == n)]
            if sub.empty:
                continue
            M = sub.pivot_table(index="evaluation", columns="seed", values=METRIC).ffill()
            ev, med = M.index.values, M.median(axis=1).values
            if M.shape[1] > 1:
                ax.fill_between(ev, M.min(axis=1), M.max(axis=1), color=col, alpha=0.15, lw=0, zorder=z - 1)
            ax.plot(ev, med, color=col, lw=1.5 if mk else 1.3, zorder=z, label=lab,
                    marker=mk, ms=7 if mk == "*" else 3.5, mec="white", mew=0.5, markevery=1 if mk else None)
            if ev[-1] < XMAX:   # the LLM stops after a few dozen evaluations: carry its final value as a dotted line
                ax.plot([ev[-1], XMAX], [med[-1]] * 2, color=col, lw=1.0, ls=(0, (1.5, 1.5)), zorder=z)
            if method == "llm_interactive":
                xllm = ev[-1]
        for ref, lab, col, mk, ls, filled in VSN:   # dashed: trained from scratch; solid: started from LLM-written weights
            g = vsn[vsn.reference == ref].groupby("N")[METRIC].median()
            ns = [x for x in REF_NS if x <= n and x in g.index]
            if not ns:
                continue
            xs = xllm ** (np.arange(len(ns)) / max(len(ns) - 1, 1))      # N = 10^2 ... panel N, log-spaced from 1 to the last LLM edit
            ax.plot(xs, [g[x] for x in ns], color=col, ls=ls, lw=1.2, marker=mk, ms=3.8, zorder=5, label=lab,
                    mfc=col if filled else "white", mec="white" if filled else col, mew=0.5 if filled else 0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, XMAX * 1.15)
        ax.set_ylim(1.5, 3000)
        ax.set_xlabel("evaluations of the edit-set objective")
        ax.set_title(title, fontsize=7.5, color=S.INK, loc="left", pad=3)
        S.panel_label(ax, "ab"[list(dict(NS)).index(n)])
    axs[0].set_ylabel(YLABEL)
    axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.97), handlelength=2.2, columnspacing=1.2)
    if preview:
        fig.text(0.995, -0.04, "preview", ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, os.path.join(HERE, "fig_edit_search"))


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
