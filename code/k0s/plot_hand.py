"""Hand editing in weight space vs gradient training, K0S (handedit.png)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = json.load(open("handedit_test.json"))
rows = json.load(open("improve_nets.json")) + json.load(open("improve_trees.json"))
EDITS = ["start:\nwritten net*", "tighten cos PA,\nDCA, mL/p", "Λ veto as real\nveto (w=3)", "radius < 40 cm,\nd_V0 unit",
         "graded second\ncos PA, DCA cuts", "2nd Armenteros\nstep", "smooth ramps\nDCA, radius", "gentler ramps\n(final)"]
REF = [("[log] written, prior max(N,N0) + syst*", "#d62728", "written* + gradient"),
       ("boosted trees, from scratch", "#2ca02c", "boosted trees, from scratch"),
       ("[log] random init, 16 units", "#1f77b4", "random start + gradient"),
       ("[log] written, layer 1 frozen*", "#fdae6b", "written*, gradient on output layer only")]


def stat(m, n, key):
    v = [r[key] for r in rows if r["model"] == m and r["n"] == n]
    return np.mean(v), np.std(v)


fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
x = np.arange(len(H))
for i, (key, lab) in enumerate((("rej", r"rejection at $\epsilon_S$=90%"), ("rej95", r"rejection at $\epsilon_S$=95%"))):
    a = ax[i]
    a.plot(x, [h[key] for h in H], "k*-", ms=12, lw=1.5, label="hand-edited* (no gradient, sees the same 1000 labels)", zorder=5)
    for m, c, l in REF:
        mu, sd = stat(m, 1000, key)
        a.axhspan(mu - sd, mu + sd, color=c, alpha=0.12); a.axhline(mu, color=c, lw=1.5, label=f"{l}, N=1000")
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([f"r{j}" for j in x])
    a.set_xlabel("hand-edit round"); a.set_ylabel(lab); a.set_title(f"Test {lab} per round")
    for j, t in enumerate(EDITS):
        a.annotate(t, (j, H[j][key]), textcoords="offset points", xytext=(0, -34 if j % 2 else 10), ha="center", fontsize=6.5, color="#444")
ax[0].legend(fontsize=7.5, loc="upper left")
ax[0].text(0.98, 0.03, "final round chosen on the edit set (best AUC) before testing", transform=ax[0].transAxes,
           ha="right", fontsize=7, color="gray")

# where hand editing sits on the learning curves
a = ax[2]
for m, c, l in REF:
    ns = sorted({r["n"] for r in rows if r["model"] == m})
    mu = [stat(m, n, "rej95")[0] for n in ns]; sd = [stat(m, n, "rej95")[1] for n in ns]
    a.errorbar(ns, mu, sd, color=c, marker="o", ms=3.5, capsize=2, label=l)
a.plot([1000], [H[-1]["rej95"]], "k*", ms=18, label="hand-edited* (final), 1000 labels", zorder=5)
a.plot([1000], [H[0]["rej95"]], "kx", ms=10, label="written*, no edits")
a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("labelled training candidates N")
a.set_ylabel(r"rejection at $\epsilon_S$=95%"); a.set_title("Hand editing against the learning curves")
a.legend(fontsize=7.5, loc="upper left")
fig.suptitle(r"K$^0_S$: LLM editing the weights by hand from diagnostics, no gradient steps (* = LLM-written).  Test: 10k K$^0_S$ + 25k background", fontsize=12)
plt.tight_layout(); plt.savefig("handedit.png", dpi=150)
