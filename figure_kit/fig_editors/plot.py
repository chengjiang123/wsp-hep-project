"""Story 1 by editing LLM: reads data.csv here, writes fig_editors.pdf/.png.
Columns: task x pretrained model. Rows: FT->WSP (the edit is final) and WSP->FT (edit, then fine-tune).
y = relative improvement in background rejection over the same cell's plain fine-tune, R_arm / R_FT - 1, at the
standard working point of each task (top: R30 = 1/eps_B at 30% signal efficiency; quark-gluon: R50).
x = label budget N (10^3: mean of seeds 0-2 with SE bar; 10^5 and all: seed 0).
Editors: gpt-oss-120b (light), Sonnet 5.5, Opus 5.5 (dark); grey diamond = fixed-rule edit (script_ft, the
"signal vs background class rows" head; WSP->FT row only). Points beyond the y range sit on the edge as a triangle
with their value written next to them.
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
COLS = [("top", "part", "top · ParT"), ("top", "omni", "top · OmniLearned"),
        ("qg", "part", "quark/gluon · ParT"), ("qg", "omni", "quark/gluon · OmniLearned")]
ROWS = [("ft_wsp", r"FT$\rightarrow$WSP" + "\n(edit is final)"), ("wsp_ft", r"WSP$\rightarrow$FT" + "\n(edit, then FT)")]
NS = [("1000", "10³"), ("100000", "10⁵"), ("full", "all")]
EDITORS = [("gpt-oss-120b", "#8fb8ea", "o"), ("Sonnet 5.5", "#2a78d6", "o"), ("Opus 5.5", "#1c4f8f", "o")]
WP = {"top": "r30", "qg": "r50"}          # working point per task
YLIM = (-30, 30)                           # percent; points outside are drawn on the edge
DX = 0.16
SIZE = (S.TEXT_W, 3.4)
# --------------------------------------------------------------------------


def gains(d):
    w = d.task.map(WP)
    pick = lambda pre: np.where(w == "r30", d[pre + "_r30"], d[pre + "_r50"]).astype(float)
    d = d.assign(R=pick("test"), Rft=pick("ft"), Rsc=pick("script"))
    return d.assign(g=100 * (d.R / d.Rft - 1), g_script=100 * (d.Rsc / d.Rft - 1))


def point(ax, x, m, se, col, mk, stack, ncol=1):
    """stack: dict counting clipped labels per (axis, edge, N column), so labels of the same column do not overlap"""
    lo, hi = YLIM
    if m > hi or m < lo:
        up = m > hi
        y = hi - 0.04 * (hi - lo) if up else lo + 0.04 * (hi - lo)
        ax.plot(x, y, marker="^" if up else "v", color=col, ms=5, mec="white", mew=0.5, ls="", zorder=4, clip_on=False)
        key = (id(ax), up, round(x))
        k = stack.get(key, 0)
        stack[key] = k + 1
        step = 0.085 * (hi - lo)
        ty = y - k * step if up else y + k * step          # stack within one N column only
        last = round(x) == len(NS) - 1                      # rightmost column: label to the left, inside the panel
        if last:                                             # rightmost column: centred just inside the triangle
            tx, ha = x, "center"
            ty = y - (k + 1) * step if up else y + (k + 1) * step
        elif ncol == 1:                                      # alone in its column: right next to its triangle
            tx, ha = x + 0.08, "left"
        else:                                                # several in the column: right of the whole group
            tx, ha = round(x) + 0.32, "left"
        ax.text(tx, ty, f"{m:+.0f}%", fontsize=5.0, color=col if col != S.MUTED else S.INK2, va="center", ha=ha, zorder=5)
    else:
        ax.errorbar(x, m, yerr=se, fmt=mk, color=col, ms=4 if mk == "o" else 3.5, mec="white", mew=0.5,
                    elinewidth=0.8, capsize=0, zorder=3)


def main(preview=False):
    S.setup()
    d = gains(pd.read_csv(os.path.join(HERE, "data.csv"), dtype={"N": str}))
    fig, axs = plt.subplots(2, 4, figsize=SIZE, sharex=True, sharey=True, gridspec_kw={"hspace": 0.2, "wspace": 0.12})
    stack = {}
    for r, (arm, rlab) in enumerate(ROWS):
        for c, (task, model, clab) in enumerate(COLS):
            ax = axs[r, c]
            sub = d[(d.task == task) & (d.model == model) & (d.arm == arm)]
            ax.axhline(0, color=S.INK2, lw=0.7, zorder=1)
            series = list(EDITORS) + ([("fixed rule", S.MUTED, "D")] if arm == "wsp_ft" else [])
            clipped = {}                                     # (edge, N column) -> number of clipped series
            for ed, _, _ in series:
                for j, (N, _) in enumerate(NS):
                    g = (sub[(sub.N == N) & (sub.editor == "Opus 5.5")].g_script if ed == "fixed rule"
                         else sub[(sub.N == N) & (sub.editor == ed)].g).dropna()
                    if len(g) and not (YLIM[0] <= g.mean() <= YLIM[1]):
                        clipped[(g.mean() > YLIM[1], j)] = clipped.get((g.mean() > YLIM[1], j), 0) + 1
            for i, (ed, col, mk) in enumerate(series):
                for j, (N, _) in enumerate(NS):
                    if ed == "fixed rule":
                        g = sub[(sub.N == N) & (sub.editor == "Opus 5.5")].g_script   # one value per cell
                    else:
                        g = sub[(sub.N == N) & (sub.editor == ed)].g
                    g = g.dropna()
                    if g.empty:
                        continue
                    se = g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0
                    point(ax, j + (i - (len(series) - 1) / 2) * DX, g.mean(), se, col, mk, stack,
                          clipped.get((g.mean() > YLIM[1], j), 1))
            ax.set_xticks(range(len(NS)))
            ax.set_xticklabels([lab for _, lab in NS])
            ax.set_xlim(-0.5, len(NS) - 0.5)
            ax.set_ylim(*YLIM)
            ax.set_yticks([-30, -15, 0, 15, 30])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0"))
            ax.tick_params(axis="y", labelsize=6)
            ax.text(0.97, 0.95, "R$_{30}$" if task == "top" else "R$_{50}$", transform=ax.transAxes, fontsize=5.8,
                    color=S.MUTED, ha="right", va="top")
            if r == 0:
                ax.set_title(clab, fontsize=6.8, color=S.INK, loc="left", pad=3)
            else:
                ax.set_xlabel("labelled jets $N$", fontsize=6.8)
            if c == 0:
                ax.set_ylabel(rlab + "\nrejection gain vs FT", fontsize=6.6)
            S.panel_label(ax, "abcdefgh"[4 * r + c])
    handles = [plt.Line2D([], [], color=col, marker=mk, ls="", mec="white", ms=4.5) for _, col, mk in EDITORS]
    handles.append(plt.Line2D([], [], color=S.MUTED, marker="D", ls="", mec="white", ms=4))
    labels = [f"{ed} edits weights" for ed, _, _ in EDITORS] + ["fixed-rule edit (no LLM)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.97), handletextpad=0.3, columnspacing=1.4)
    if preview:
        fig.text(0.995, -0.02, f"preview: {len(d)} rows", ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, os.path.join(HERE, "fig_editors"))


if __name__ == "__main__":
    main(preview="--preview" in sys.argv)
