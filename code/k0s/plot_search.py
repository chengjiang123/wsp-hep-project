"""Study 1 figure: evaluations needed by black-box search vs LLM hand edits, same weight space (search.png)."""
import json, glob, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
# LLM rounds: edit-set AUC per round {"r0": ..} and test of every round (list from handedit.py test). Override for scaled runs.
LLM_EDIT = json.load(open(os.environ.get("LLM_EDIT", "hand/edit_auc.json")))
H = json.load(open(os.environ.get("LLM_TEST", "handedit_test_all.json")))
NL = len(LLM_EDIT)
# written* + gradient at the same N (mean of seeds), from improve_nets.json if present
GRAD = dict(auc=0.9975, rej95=207)
_N = int(os.environ.get("NEDIT", 1000))
if os.path.exists("improve_nets.json"):
    _g = [r for r in json.load(open("improve_nets.json")) if r["model"] == "[log] written, prior max(N,N0) + syst*" and r["n"] == _N]
    if _g:
        GRAD = dict(auc=float(np.mean([r["auc"] for r in _g])), rej95=float(np.mean([r["rej95"] for r in _g])))
TARGET = float(os.environ.get("TARGET", 0.9980))      # edit-set AUC the LLM reached (round 7 at N = 1000)
runs = [json.load(open(f)) for f in sorted(glob.glob(f"search/*_[0-9]{__import__('os').environ.get('TAG', '')}.json"))]
COL = {"random": "#7f7f7f", "tpe": "#1f77b4", "cmaes": "#2ca02c"}
LAB = {"random": "random search", "tpe": "TPE (Optuna)", "cmaes": "CMA-ES"}


def curves(r):
    """Best-so-far: selected by the held-back stop-set AUC when it exists, otherwise by the edit-set AUC."""
    key = "stop_auc" if "stop_auc" in r["hist"][0] else "edit_auc"
    cur, t, e = -1, None, -1
    out = dict(edit=[], tauc=[], trej=[], trej95=[])
    for h in r["hist"]:
        if h[key] > cur and "test_auc" in h:
            cur = h[key]; t = h
        e = max(e, h["edit_auc"])
        out["edit"].append(e); out["tauc"].append(t["test_auc"]); out["trej"].append(t["test_rej"]); out["trej95"].append(t["test_rej95"])
    return {k: np.array(v) for k, v in out.items()}


fig, ax = plt.subplots(2, 3, figsize=(19, 10))
summary = []
for row, space in enumerate(("r0", "r7")):
    for opt in ("random", "tpe", "cmaes"):
        cs = [curves(r) for r in runs if r["space"] == space and r["opt"] == opt]
        if not cs:
            continue
        ev = np.arange(1, len(cs[0]["edit"]) + 1)
        for k, (key, tr) in enumerate((("edit", lambda v: 1 - v), ("tauc", lambda v: 1 - v), ("trej95", lambda v: v))):
            M = np.array([tr(c[key]) for c in cs])
            ax[row, k].plot(ev, np.median(M, 0), color=COL[opt], lw=1.8, label=LAB[opt])
            ax[row, k].fill_between(ev, M.min(0), M.max(0), color=COL[opt], alpha=0.15)
        hit = [int(np.argmax(c["edit"] >= TARGET)) + 1 if (c["edit"] >= TARGET).any() else None for c in cs]
        at = [(c["tauc"][h - 1], c["trej"][h - 1], c["trej95"][h - 1]) if h else None for c, h in zip(cs, hit)]
        fin = [(c["tauc"][-1], c["trej"][-1], c["trej95"][-1]) for c in cs]
        summary.append(dict(space=space, opt=opt, nparams=[r["nparams"] for r in runs if r["space"] == space][0],
                            evals_to_target=hit, test_at_target=at, test_at_1000=fin))
    # LLM: one evaluation per round
    le = np.arange(1, NL + 1)
    ed = np.array([LLM_EDIT[f"r{i}"] for i in range(NL)])
    acc = [int(np.argmax(ed[:i + 1])) for i in range(NL)]          # the round kept so far (best edit AUC, earliest on ties)
    ta = np.array([h["auc"] for h in H]); tr95 = np.array([h["rej95"] for h in H])
    ax[row, 0].plot(le, 1 - ed, "k*", ms=6, alpha=0.3)
    ax[row, 0].plot(le, 1 - ed[acc], "k*-", ms=9, label=f"LLM hand edits ({NL} evaluations; faint = rejected tries)")
    ax[row, 1].plot(le, 1 - ta, "k*", ms=6, alpha=0.3); ax[row, 1].plot(le, 1 - ta[acc], "k*-", ms=9, label="LLM hand edits (kept round)")
    ax[row, 2].plot(le, tr95, "k*", ms=6, alpha=0.3); ax[row, 2].plot(le, tr95[acc], "k*-", ms=9, label="LLM hand edits (kept round)")
    ax[row, 1].axhline(1 - GRAD["auc"], color="#d62728", ls="--", lw=1.2, label=f"written* + gradient, same {_N} labels")
    ax[row, 2].axhline(GRAD["rej95"], color="#d62728", ls="--", lw=1.2, label=f"written* + gradient, same {_N} labels")
    ax[row, 1].legend(fontsize=7.5, loc="upper right"); ax[row, 2].legend(fontsize=7.5, loc="lower right")
    ax[row, 0].axhline(1 - TARGET, color="k", ls=":", lw=1); ax[row, 0].text(1.1, (1 - TARGET) * 0.88, "LLM round 7 (old target)", fontsize=8)
    for k in range(3):
        ax[row, k].set_xscale("log"); ax[row, k].set_xlabel(f"evaluations on the {_N} labelled candidates")
    for k in (0, 1):
        ax[row, k].set_yscale("log")
    ax[row, 2].set_yscale("log")
    nm = "8 written cuts (24 numbers), start = written net" if space == "r0" else \
        "LLM's final structure (41 numbers), start = neutral values"
    ax[row, 0].set_ylabel(f"{nm}\n\n1 - AUC on edit set (what is optimised)", fontsize=9)
    ax[row, 1].set_ylabel("1 - AUC on test set"); ax[row, 2].set_ylabel(r"test rejection at $\epsilon_S$ = 95%")
    ax[row, 0].set_title("Objective on the edit set"); ax[row, 1].set_title("Test AUC of the best-so-far")
    ax[row, 2].set_title("Test rejection of the best-so-far")
    ax[row, 0].legend(fontsize=8, loc="upper right")
for k, (key, lab) in enumerate((("rej95", None),)):
    pass
fig.suptitle(f"K$^0_S$, weight space only: LLM hand edits vs black-box search on the same numbers and the same {_N} labels "
             "(median and min-max of 3 seeds)", fontsize=12)
plt.tight_layout(); plt.savefig(f"search{os.environ.get('TAG', '')}.png", dpi=150)
json.dump(summary, open(f"search_summary{os.environ.get('TAG', '')}.json", "w"), indent=1)
for s in summary:
    print(s["space"], s["opt"], "evals to edit AUC >= 0.998:", s["evals_to_target"],
          "| test at target:", [tuple(round(v, 4) if v < 1 else round(v) for v in t) if t else None for t in s["test_at_target"]],
          "| test at 1000:", [tuple(round(v, 4) if v < 1 else round(v) for v in t) for t in s["test_at_1000"]])
