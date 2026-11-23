"""Story 1 figure (N-scan): reads data.csv in this folder, writes fig_nscan.pdf/.png (and fig_nscan_controls with
--controls). Rows = task, columns = pretrained model. Each panel: relative improvement in background rejection over
plain fine-tuning (arm "ft"), R_arm / R_FT - 1 per seed, at the standard working point of each task (top: R30,
quark-gluon: R50), vs the label budget N; band = 1 SE over seeds. Points beyond the y range sit on the edge as a
triangle with their value. Grey band: N where the ordering WSP->FT > FT > FT->WSP holds in this metric (both paired gains
beyond 2 SE). The strip under each panel is the plain-FT rejection.
Edit the settings block, then:  python plot.py [--controls] [--preview]"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style as S  # noqa: E402

# ---------------------------------------------------------------- settings
ROWS = [("top", "top vs QCD (TopLandscape)"), ("qg", "quark vs gluon")]
COLS = [("part", "ParT"), ("omni", "OmniLearned")]
MAIN = [("wsp_ft", "WSP → FT"), ("ft_wsp", "FT → WSP"), ("script_ft", "fixed-rule edit → FT")]   # gains over "ft"
CONTROLS = [("wsp_ft", "WSP → FT"), ("lpft", "linear probe → FT"), ("random_ft", "random patch → FT")]
N_FULL_X = 2.4e6         # where "all" (N = full) is drawn on the log axis
SIGMA = 2.0              # ordering band rule: both gains > SIGMA x SE
WP = {"top": "test_r30", "qg": "test_r50"}   # working point per task
YLIM = (-40, 40)         # percent; points outside are drawn on the edge
SIZE = (S.TEXT_W, 4.5)
# --------------------------------------------------------------------------
AR = lambda t: t.replace("→", r"$\rightarrow$").replace(" > ", r" $>$ ")
NX = lambda n: N_FULL_X if n == "full" else float(n)


def paired(cell, x, y="ft"):
    """cell: DataFrame indexed by seed with one column per arm (rejection). Returns (mean, se, n) of the relative gain
    100 (R_x / R_y - 1) over seeds, or None."""
    if x not in cell or y not in cell:
        return None
    d = (100 * (cell[x] / cell[y] - 1)).dropna()
    if d.empty:
        return None
    return d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan), len(d)


def main(controls=False, preview=False):
    S.setup()
    df = pd.read_csv(os.path.join(HERE, "data.csv"), dtype={"N": str})
    df["R"] = np.where(df.task == "top", df.test_r30, df.test_r50)
    wide = df.pivot_table(index=["task", "model", "N", "seed"], columns="arm", values="R")
    series = CONTROLS if controls else MAIN
    fig = plt.figure(figsize=SIZE)
    gs = fig.add_gridspec(5, 2, height_ratios=[3, 0.85, 0.75, 3, 0.85], hspace=0.1, wspace=0.28)
    for r, (ds, dlab) in enumerate(ROWS):
        for c, (mk, mlab) in enumerate(COLS):
            ax = fig.add_subplot(gs[3 * r, c])
            axb = fig.add_subplot(gs[3 * r + 1, c], sharex=ax)
            sub = wide.loc[(ds, mk)] if (ds, mk) in wide.index.droplevel([2, 3]) else None
            Ns = sorted(sub.index.get_level_values(0).unique(), key=NX) if sub is not None else []
            for n in Ns:
                cell = sub.loc[n]
                a, b = paired(cell, "wsp_ft"), paired(cell, "ft", "ft_wsp")
                if a and b and a[2] > 1 and b[2] > 1 and a[0] > SIGMA * a[1] and b[0] > SIGMA * b[1]:
                    ax.axvspan(NX(n) / 1.6, NX(n) * 1.6, color="#efeeea", zorder=0, lw=0)
            ax.axhline(0, color=S.INK2, lw=0.7, zorder=1)
            for i, (arm, lab) in enumerate(series):
                pts = [(NX(n), *paired(sub.loc[n], arm)) for n in Ns if paired(sub.loc[n], arm)]
                if not pts:
                    continue
                x, m, se, k = map(np.array, zip(*pts))
                se = np.nan_to_num(se)
                lo, hi = YLIM
                mc = np.clip(m, lo + 0.04 * (hi - lo), hi - 0.04 * (hi - lo))
                ax.fill_between(x, np.clip(m - se, lo, hi), np.clip(m + se, lo, hi), color=S.SLOTS[i], alpha=0.15, lw=0)
                ax.plot(x, mc, color=S.SLOTS[i], lw=1.2, zorder=3)
                inside = (m >= lo) & (m <= hi)
                ax.plot(x[inside], m[inside], color=S.SLOTS[i], marker=S.MARKERS[i], ls="", mec="white", mew=0.8, ms=4.5, zorder=4)
                for xx, mm in zip(x[~inside], m[~inside]):
                    yy = hi - 0.04 * (hi - lo) if mm > hi else lo + 0.04 * (hi - lo)
                    ax.plot(xx, yy, marker="^" if mm > hi else "v", color=S.SLOTS[i], ms=5, mec="white", mew=0.5, ls="", zorder=5)
                    pass   # values beyond the range are listed in summary.csv
            ax.set_xscale("log")
            ax.set_ylim(*YLIM)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
            ax.tick_params(labelbottom=False)
            ax.text(0.98, 0.95, "R$_{30}$" if ds == "top" else "R$_{50}$", transform=ax.transAxes, fontsize=5.8, color=S.MUTED, ha="right", va="top")
            if c == 0:
                ax.set_ylabel("rejection gain vs FT")
            ax.set_title(r"$\bf{" + "abcd"[2 * r + c] + "}$   " + f"{dlab} · {mlab}", fontsize=7.5, color=S.INK, loc="left", pad=4)
            if sub is not None and "ft" in sub:
                ftm = sub["ft"].groupby(level=0).mean()
                xs = sorted(ftm.index, key=NX)
                axb.plot([NX(n) for n in xs], [ftm[n] for n in xs], color=S.INK2, lw=1.0, marker="o", ms=2.5)
            axb.set_xscale("log")
            axb.set_yscale("log")
            axb.yaxis.set_minor_locator(plt.NullLocator())
            axb.grid(axis="x", visible=False)
            axb.tick_params(axis="y", labelsize=6)
            axb.set_xlim(60, N_FULL_X * 1.7)
            axb.set_xticks([1e2, 1e3, 1e4, 1e5, N_FULL_X])
            axb.set_xticklabels(["10²", "10³", "10⁴", "10⁵", "all"])
            axb.xaxis.set_minor_locator(plt.NullLocator())
            axb.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
            axb.yaxis.set_major_locator(plt.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=4))
            axb.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            if c == 0:
                axb.set_ylabel("FT R", fontsize=6.5)
            if r == len(ROWS) - 1:
                axb.set_xlabel("labelled jets $N$")
            else:
                axb.tick_params(labelbottom=False)
    handles = [plt.Line2D([], [], color=S.SLOTS[i], marker=S.MARKERS[i], mec=S.SLOTS[i], ms=4.5, lw=1.4) for i in range(len(series))]
    handles.append(plt.Line2D([], [], color=S.INK2, marker="v", ls="", ms=5))
    handles.append(plt.Rectangle((0, 0), 1, 1, color="#efeeea"))
    labels = [AR(l) + (" (gpt-oss-120b)" if "WSP" in l else "") for _, l in series] + [f"beyond {YLIM[0]:.0f}% / +{YLIM[1]:.0f}%",
              AR("WSP→FT > FT > FT→WSP") + f" (> {SIGMA:g} SE)"]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.955), handlelength=2.0, columnspacing=1.4)
    if preview:
        fig.text(0.995, 0.0, f"preview: {len(wide)} (N, seed) cells", ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, os.path.join(HERE, "fig_nscan_controls" if controls else "fig_nscan"))


if __name__ == "__main__":
    main(controls="--controls" in sys.argv, preview="--preview" in sys.argv)
