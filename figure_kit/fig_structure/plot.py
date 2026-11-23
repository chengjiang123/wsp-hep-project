"""Story 2 figure (structure): reads pair_factors.csv, displacement.csv, edit_alone.csv and ../fig_nscan/data.csv;
writes fig_structure_<task>_<model>.pdf/.png here.
  (a) the LLM's pair-interaction head factors in its WSP->FT patches, heads x N (log2; red weakened, blue strengthened)
  (b) relative weight change per layer group during fine-tuning at N_FIXED: FT vs WSP->FT vs LP-FT
  (c) test AUC of each edit alone (no training) vs N, with plain FT for scale
Edit the settings block, then:  python plot.py [task model] [--preview]     e.g. python plot.py qg omni"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import style as S  # noqa: E402

# ---------------------------------------------------------------- settings
TASK, MODEL = "top", "part"
N_FIXED = "1000"                                   # panel (b)
DISP_ARMS = [("ft", "FT"), ("wsp_ft", "WSP → FT"), ("lpft", "LP-FT")]
ALONE = [("LLM edit", "LLM edit, no FT"), ("script edit", "script edit, no FT"), ("LR head", "LR head, no FT")]
LAYERS = ["particle emb.", "pair emb."] + [f"block {i}" for i in range(1, 9)] + ["class attn.", "head"]
N_FULL_X = 2.4e6
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]     # weakened / unchanged / strengthened
SIZE = (S.TEXT_W, 2.15)
# --------------------------------------------------------------------------
AR = lambda t: t.replace("→", r"$\rightarrow$")
NX = lambda n: N_FULL_X if n == "full" else float(n)


def main(task, model, preview=False):
    S.setup()
    rd = lambda f: pd.read_csv(os.path.join(HERE, f), dtype={"N": str})
    pf, disp, alone = rd("pair_factors.csv"), rd("displacement.csv"), rd("edit_alone.csv")
    ft = pd.read_csv(os.path.join(HERE, "..", "fig_nscan", "data.csv"), dtype={"N": str})
    sel = lambda d: d[(d.task == task) & (d.model == model)]
    pf, disp, alone, ft = sel(pf), sel(disp), sel(alone), sel(ft)
    fig, axs = plt.subplots(1, 3, figsize=SIZE, gridspec_kw={"width_ratios": [0.85, 1.35, 1.0], "wspace": 0.62})

    # (a)
    ax = axs[0]
    Ns = sorted(pf.N.unique(), key=NX)
    if Ns:
        pf = pf.assign(l2=np.log2(pf.factor.clip(lower=1 / 16)))
        M = pf.groupby(["head", "N"]).l2.mean().unstack()[Ns]
        im = ax.imshow(M.values, cmap=LinearSegmentedColormap.from_list("div", DIVERGING), norm=TwoSlopeNorm(0, -4, 4),
                       aspect="auto", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, ticks=[-4, 0, 4])
        cb.ax.set_yticklabels(["off", "×1", "×16"], fontsize=5.5)
        cb.outline.set_visible(False)
        ax.set_yticks(range(len(M.index)))
        ax.set_yticklabels([f"h{h}" for h in M.index], fontsize=6)
        lab = {"100": "10²", "1000": "10³", "10000": "10⁴", "full": "all"}
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels([lab.get(n, "") for n in Ns], fontsize=6)
    ax.set_xlabel("labelled jets $N$")
    ax.set_ylabel("pair-interaction head")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    S.panel_label(ax, "a")

    # (b) per seed: median over the tensors of a layer group; then mean over seeds
    ax = axs[1]
    d = disp[disp.N == N_FIXED]
    x = np.arange(len(LAYERS))
    for i, (arm, lab) in enumerate(DISP_ARMS):
        g = d[d.arm == arm].groupby(["seed", "layer_group"]).rel_change.median().groupby("layer_group").mean()
        ax.plot(x, [g.get(l, np.nan) for l in LAYERS], color=S.SLOTS[i], marker=S.MARKERS[i], mec=S.SLOTS[i], ms=3.5, label=AR(lab))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS, rotation=55, ha="right", fontsize=5.8, rotation_mode="anchor")
    ax.set_ylabel(r"weight change $\|\Delta\theta\| / \|\theta\|$")
    ax.text(0.02, 0.97, f"$N$ = {int(NX(N_FIXED)):,}", transform=ax.transAxes, fontsize=6.5, color=S.INK2, va="top")
    ax.legend(loc="upper center", fontsize=6, handlelength=1.8, ncol=3, bbox_to_anchor=(0.55, 1.2))
    S.panel_label(ax, "b")

    # (c)
    ax = axs[2]
    for i, (what, lab) in enumerate(ALONE):
        g = alone[alone.what == what].groupby("N").test_auc.mean()
        if len(g):
            xs = sorted(g.index, key=NX)
            ax.plot([NX(n) for n in xs], [g[n] for n in xs], color=S.SLOTS[i], marker=S.MARKERS[i], mec=S.SLOTS[i], ms=3.5, label=lab)
    g = ft[ft.arm == "ft"].groupby("N").test_auc.mean()
    if len(g):
        xs = sorted(g.index, key=NX)
        ax.plot([NX(n) for n in xs], [g[n] for n in xs], color=S.INK2, lw=1.0, ls=(0, (3, 2)), marker="D", ms=2.5, label="plain FT")
    ax.set_xscale("log")
    ax.set_xlim(60, N_FULL_X * 1.7)
    ax.set_xticks([1e2, 1e3, 1e4, 1e5, N_FULL_X])
    ax.set_xticklabels(["10²", "10³", "10⁴", "10⁵", "all"], fontsize=6)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("labelled jets $N$")
    ax.set_ylabel("test AUC")
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.8)
    S.panel_label(ax, "c")
    if preview:
        fig.text(0.995, -0.08, f"preview: {task}/{model}", ha="right", va="top", fontsize=5.5, color=S.MUTED)
    S.save(fig, os.path.join(HERE, f"fig_structure_{task}_{model}"))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    t, m = (args + [TASK, MODEL][len(args):])[:2] if args else (TASK, MODEL)
    main(t, m, preview="--preview" in sys.argv)
