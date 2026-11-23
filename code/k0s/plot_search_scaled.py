"""Black-box search at labelled sets of 1e4 and 5e4 (30% held back), against gradient training at the same N.
usage: python plot_search_scaled.py DIR   (DIR holds search/*_n10000.json, *_n50000.json, improve_nets.json, improve_trees.json)
-> DIR/search_scaled.png, DIR/search_scaled_summary.json
"""
import json, glob, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "."
COL = {"random": "#7f7f7f", "tpe": "#1f77b4", "cmaes": "#2ca02c"}
LAB = {"random": "random search", "tpe": "TPE", "cmaes": "CMA-ES"}
SPACE = {"r0": ("-", "8 written cuts (24 numbers)"), "r7": ("--", "LLM's round-7 structure (41 numbers)")}
nets = json.load(open(os.path.join(D, "improve_nets.json"))); trees = json.load(open(os.path.join(D, "improve_trees.json")))


def ref(rows, model, n, key):
    v = [r[key] for r in rows if r["model"] == model and r["n"] == n]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def curves(r):
    """best-so-far chosen by the held-back stop-set AUC; edit AUC = running max (what is optimised)."""
    cur, t, e = -1, None, -1
    out = dict(edit=[], stop=[], tauc=[], trej=[], trej95=[])
    for h in r["hist"]:
        if h["stop_auc"] > cur and "test_auc" in h:
            cur, t = h["stop_auc"], h
        e = max(e, h["edit_auc"])
        out["edit"].append(e); out["stop"].append(cur)
        out["tauc"].append(t["test_auc"]); out["trej"].append(t["test_rej"]); out["trej95"].append(t["test_rej95"])
    return {k: np.array(v) for k, v in out.items()}


fig, ax = plt.subplots(2, 3, figsize=(19, 10)); summary = {}
for row, n in enumerate((10000, 50000)):
    for sp, (ls, splab) in SPACE.items():
        for opt in ("random", "tpe", "cmaes"):
            runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(D, f"search/{sp}_{opt}_*_n{n}.json")))]
            if not runs:
                continue
            cs = [curves(r) for r in runs]; ev = np.arange(1, len(cs[0]["edit"]) + 1)
            for k, (key, tr) in enumerate((("edit", lambda v: 1 - v), ("tauc", lambda v: 1 - v), ("trej95", lambda v: v))):
                M = np.array([tr(c[key]) for c in cs])
                ax[row, k].plot(ev, np.median(M, 0), color=COL[opt], ls=ls, lw=1.6, label=f"{LAB[opt]}, {splab}")
                ax[row, k].fill_between(ev, M.min(0), M.max(0), color=COL[opt], alpha=0.08)
            summary[f"n{n}_{sp}_{opt}"] = dict(
                test_auc=[float(c["tauc"][-1]) for c in cs], test_rej90=[float(c["trej"][-1]) for c in cs],
                test_rej95=[float(c["trej95"][-1]) for c in cs], stop_auc=[float(c["stop"][-1]) for c in cs],
                edit_auc=[float(c["edit"][-1]) for c in cs])
    hf = os.path.join(D, f"hand_n{n}/test_all.json")
    if os.path.exists(hf):
        H = json.load(open(hf)); cur, t, e = -1, None, -1; E, TA, TR = [], [], []
        for h in H:                      # same rule as the search curves: final round = best held-back AUC so far
            if h["stop_auc"] > cur:
                cur, t = h["stop_auc"], h
            e = max(e, h["edit_auc"]); E.append(1 - e); TA.append(1 - t["test_auc"]); TR.append(t["test_rej95"])
        ev = np.arange(1, len(H) + 1); xmax = max(len(json.load(open(f))["hist"]) for f in glob.glob(os.path.join(D, f"search/*_n{n}.json")))
        raw = ([1 - h["edit_auc"] for h in H], [1 - h["test_auc"] for h in H], [h["test_rej95"] for h in H])
        for k, y in enumerate((E, TA, TR)):
            ax[row, k].scatter(ev, raw[k], s=14, color="#ff7f0e", alpha=0.45, zorder=4,
                               label="every LLM round (own value, not chosen)" if k == 1 else None)
            ax[row, k].plot([ev[-1], xmax], [y[-1], y[-1]], color="#ff7f0e", lw=1.6, ls="--", zorder=4,
                            label="LLM final, held flat (no more rounds)" if k == 1 else None)
            ax[row, k].plot(ev, y, color="#ff7f0e", lw=1.8, marker="*", ms=11, mec="k", mew=0.4, zorder=5,
                            label=f"LLM hand edits ({len(H)} evaluations" + (", starts from the 1e4 result)" if n == 50000 else ")"))
        summary[f"n{n}_llm_hand"] = dict(evaluations=len(H), final=os.path.basename(t["spec"]), test_auc=t["test_auc"],
                                       test_rej90=t["test_rej"], test_rej95=t["test_rej95"], stop_auc=t["stop_auc"],
                                       edit_auc=1 - E[-1])
    for tag, col, lab in (("opus55", "#e377c2", "Opus 5.5 automated loop via CBorg"),):
        runs = sorted(glob.glob(os.path.join(D, f"cborg/{tag}_n{n}_s*")))
        cs = []
        for rd in runs:                  # test.json is in glob order (r0, r1, r10, ...): match rounds by name
            h = json.load(open(os.path.join(rd, "history.json")))
            T = {int(os.path.basename(x["spec"])[1:-5]): x for x in json.load(open(os.path.join(rd, "test.json")))}
            cur, t, e, E, TA, TR = -1, None, -1, [], [], []
            for (r, ea, _), st in zip(h["history"], h["stop_auc"]):
                if st > cur:
                    cur, t = st, T[r]
                e = max(e, ea); E.append(1 - e); TA.append(1 - t["auc"]); TR.append(t["rej95"])
            cs.append((E, TA, TR, t))
        if not cs:
            continue
        L = max(len(c[0]) for c in cs); xmax = 2000
        for k in range(3):
            M = np.array([c[k] + [c[k][-1]] * (L - len(c[k])) for c in cs]); ev = np.arange(1, L + 1)
            ax[row, k].plot(ev, np.median(M, 0), color=col, lw=1.8, marker="P", ms=7, mec="k", mew=0.4, zorder=5,
                            label=f"{lab} (3 seeds, median)")
            ax[row, k].fill_between(ev, M.min(0), M.max(0), color=col, alpha=0.15, zorder=3)
            ax[row, k].plot([L, xmax], [np.median(M[:, -1])] * 2, color=col, lw=1.4, ls="--", zorder=4)
        summary[f"n{n}_{tag}_cborg"] = dict(rounds=[len(c[0]) - 1 for c in cs], test_auc=[c[3]["auc"] for c in cs],
                                           test_rej90=[c[3]["rej"] for c in cs], test_rej95=[c[3]["rej95"] for c in cs])
    refs = [("[log] written, prior max(N,N0) + syst*", nets, "#d62728", "written* + gradient"),
            ("[log] random init, 64 units", nets, "#9467bd", "random-start net + gradient"),
            ("boosted trees, from scratch", trees, "#8c564b", "boosted trees"),
            ("boosted trees on top of tuned written net*", trees, "k", "boosted trees on written net*")]
    for model, rows, c, lab in refs:
        a, _ = ref(rows, model, n, "auc"); r95, _ = ref(rows, model, n, "rej95")
        summary[f"n{n}_ref_{lab}"] = dict(test_auc=a, test_rej95=r95, test_rej90=ref(rows, model, n, "rej")[0])
        ax[row, 1].axhline(1 - a, color=c, ls=":", lw=1.5, label=f"{lab}, N={n:.0e}")
        ax[row, 2].axhline(r95, color=c, ls=":", lw=1.5, label=f"{lab}, N={n:.0e}")
    for k in range(3):
        ax[row, k].set_xscale("log"); ax[row, k].set_xlabel(f"evaluations on {int(0.7 * n)} edit candidates (N = {n:.0e}, 30% held back)")
    ax[row, 0].set_yscale("log"); ax[row, 1].set_yscale("log"); ax[row, 2].set_yscale("log")
    ax[row, 0].set_ylabel("1 - AUC on the edit part (optimised)"); ax[row, 1].set_ylabel("1 - test AUC (chosen by held-back set)")
    ax[row, 2].set_ylabel(r"test rejection at $\epsilon_S$ = 95%")
    ax[row, 0].set_title(f"N = {n:.0e}: objective"); ax[row, 1].set_title(f"N = {n:.0e}: test AUC"); ax[row, 2].set_title(f"N = {n:.0e}: test rejection")
    ax[row, 0].legend(fontsize=7, loc="upper right"); ax[row, 1].legend(fontsize=6.5, loc="upper right"); ax[row, 2].legend(fontsize=6.5, loc="lower right")
fig.suptitle(r"K$^0_S$, weight space: black-box search on 1e4 and 5e4 labels vs gradient training at the same N "
             "; orange stars = LLM hand edits in this session, pink = automated LLM loop via CBorg\n(test: 30k K$^0_S$ + 100k background; median and range of 3 seeds; * = LLM-written)", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(D, "search_scaled.png"), dpi=140)
json.dump(summary, open(os.path.join(D, "search_scaled_summary.json"), "w"), indent=1)
for k, v in summary.items():
    print(k.ljust(52), {a: (np.round(np.mean(b), 4) if isinstance(b, list) else (round(b, 4) if not isinstance(b, str) else b)) for a, b in v.items()})
