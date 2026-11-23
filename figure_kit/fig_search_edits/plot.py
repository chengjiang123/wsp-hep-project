"""E1 figure: rejection gain over plain FT vs evaluations of the validation objective, for black-box search on the
LLM's own edit space (CMA-ES, TPE, random) and for the LLM editors (4 rounds). Reads data.csv here.
Rows: N = 10^3 (median and range over 3 seeds) and N = 10^5 (seed 0). Columns: task x pretrained model.
Edit the settings block, then: python plot.py"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style as S  # noqa: E402

COLS = [("top", "part", "top · ParT"), ("top", "omni", "top · OmniLearned"),
        ("qg", "part", "quark/gluon · ParT"), ("qg", "omni", "quark/gluon · OmniLearned")]
ROWS = [("1000", r"$N=10^3$"), ("100000", r"$N=10^5$")]
SEARCH = [("cmaes", "CMA-ES", S.SLOTS[1]), ("tpe", "TPE", S.SLOTS[2]), ("random", "random search", S.MUTED)]
LLM = [("llm_sonnet", "Sonnet 5.5 edits", "#2a78d6", "o"), ("llm_opus", "Opus 5.5 edits", "#1c4f8f", "*")]
YLIM = (-60, 30)
SIZE = (S.TEXT_W, 3.3)


def main():
    S.setup()
    d = pd.read_csv(os.path.join(HERE, "data.csv"), dtype={"N": str})
    fig, axs = plt.subplots(2, 4, figsize=SIZE, sharex=True, sharey=True, gridspec_kw={"hspace": 0.18, "wspace": 0.1})
    for r, (N, rlab) in enumerate(ROWS):
        for c, (ds, model, clab) in enumerate(COLS):
            ax = axs[r, c]
            sub = d[(d.ds == ds) & (d.model == model) & (d.N == N)]
            ax.axhline(0, color=S.INK2, lw=0.7)
            for m, lab, col in SEARCH:
                q = sub[sub.method == m].groupby("evaluations").gain
                if q.ngroups == 0:
                    continue
                ev = q.median().index.values
                ax.plot(ev, np.clip(q.median().values, *YLIM), color=col, lw=1.3, marker="o", ms=2.5, label=lab)
                if sub[sub.method == m].seed.nunique() > 1:
                    ax.fill_between(ev, np.clip(q.min().values, *YLIM), np.clip(q.max().values, *YLIM), color=col, alpha=0.15, lw=0)
            for m, lab, col, mk in LLM:
                g = sub[sub.method == m].gain
                if len(g):
                    ax.errorbar(4 * (1.12 if m == "llm_opus" else 0.89), np.clip(g.median(), *YLIM),
                                yerr=None if len(g) < 2 else [[g.median() - g.min()], [g.max() - g.median()]],
                                fmt=mk, color=col, ms=7 if mk == "*" else 4.5, mec="white", mew=0.5, elinewidth=0.8, zorder=5, label=lab)
            ax.set_xscale("log")
            ax.set_xlim(0.8, 330)
            ax.set_ylim(*YLIM)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
            if r == 0:
                ax.set_title(clab, fontsize=6.8, color=S.INK, loc="left", pad=3)
            else:
                ax.set_xlabel("evaluations on validation labels", fontsize=6.6)
            if c == 0:
                ax.set_ylabel(rlab + "\nrejection gain vs FT", fontsize=6.6)
            S.panel_label(ax, "abcdefgh"[4 * r + c])
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.97), columnspacing=1.2)
    S.save(fig, os.path.join(HERE, "fig_search_edits"))


if __name__ == "__main__":
    main()
