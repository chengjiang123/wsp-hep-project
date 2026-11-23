"""Figures for the three directions: improve.png, fingerprint.png, anomaly.png"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 1. tagger improvement
rows = json.load(open("improve_nets.json")) + json.load(open("improve_trees.json"))
SHOW = [("[lin] random init, 16 units", "#9ecae1", "--", "random start, linear layer (old)"),
        ("[lin] written, prior max(N,N0) + syst*", "#f4a3a3", "--", "written*, linear layer (old best)"),
        ("[log] random init, 16 units", "#1f77b4", "-", "random start, log layer"),
        ("[log] random init, 64 units", "#08306b", "-", "random start, log layer, 64 units"),
        ("[log] written, layer 1 frozen*", "#fdae6b", "-", "written*, layer 1 frozen"),
        ("[log] written, prior 1/N*", "#e6550d", "-", "written*, prior 1/N"),
        ("[log] written, prior max(N,N0) + syst*", "#d62728", "-", "written*, prior max(N,N0) + syst. widths"),
        ("[log] written + 48 free units, prior max(N,N0) + syst*", "#7b3fa0", "-", "written* + 48 free units"),
        ("boosted trees, from scratch", "#2ca02c", "-", "boosted trees, from scratch"),
        ("boosted trees on top of tuned written net*", "#000000", "-", "boosted trees on top of written net*")]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
for key, c, ls, lab in SHOW:
    rr = [r for r in rows if r["model"] == key]; ns = sorted({r["n"] for r in rr})
    for i, m in enumerate(("auc", "rej", "rej95")):
        a = np.array([[r[m] for r in rr if r["n"] == n] for n in ns])
        y = 1 - a if m == "auc" else a
        ax[i].errorbar(ns, y.mean(1), y.std(1), color=c, ls=ls, marker="o", ms=3.5, capsize=2, label=lab)
for i, m in enumerate(("auc", "rej", "rej95")):
    z = [r for r in rows if r["model"] == "[log] written soft + smooth, no training*"][0]
    ax[i].axhline(1 - z[m] if m == "auc" else z[m], color="k", ls=":", lw=1.2, label="written*, no training (log layer)")
    ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].set_xlabel("labelled training candidates N")
ax[0].set_ylabel("1 - AUC (lower is better)"); ax[0].set_title("1 - AUC")
ax[1].set_ylabel(r"background rejection $1/\epsilon_B$"); ax[1].set_title(r"Rejection at $\epsilon_S$ = 90%")
ax[2].set_ylabel(r"background rejection $1/\epsilon_B$"); ax[2].set_title(r"Rejection at $\epsilon_S$ = 95%")
ax[1].axhline(25000, color="gray", lw=0.8); ax[1].text(110, 18000, "1 test bkg candidate left", fontsize=7, color="gray")
ax[0].legend(fontsize=7.5, loc="lower left")
fig.suptitle(r"1. Tagger improvement: K$^0_S$ selection, 10k K$^0_S$ + 25k background test candidates, 5 seeds (* = LLM-written)")
plt.tight_layout(); plt.savefig("improve.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 2. sensitivity fingerprints
fp = json.load(open("fingerprint.json"))
lab = fp["labels"]; nom = np.array(fp["nominal"]); keys = list(fp["effects"])
noise = nom.std(0, ddof=1) * np.sqrt(1 / 3 + 1 / len(nom)) + 1e-3
D = np.array([np.mean(fp["effects"][k]["runs"], 0) - nom.mean(0) for k in keys])
Z = D / noise
my = np.mean(fp["mystery"]["runs"], 0) - nom.mean(0)
fig = plt.figure(figsize=(19, 7.5))
a = fig.add_axes([0.17, 0.30, 0.55, 0.6])
im = a.imshow(np.clip(Z, -10, 10), cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if abs(Z[i, j]) > 3:
            a.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                   color="w" if abs(Z[i, j]) > 7 else "k")
a.set_yticks(range(len(keys))); a.set_yticklabels([fp["effects"][k]["label"] for k in keys], fontsize=9)
a.set_xticks(range(len(lab))); a.set_xticklabels(lab, rotation=60, ha="right", fontsize=7.5)
for x in (7.5, 15.5):
    a.axvline(x, color="k", lw=0.8)
a.set_title("Fingerprint of each effect: change of the tuned written net* vs nominal\n"
            "colour = change / seed noise (clipped at 10), number = change (cut shift in syst. units, or weight)", fontsize=10)
cb = fig.colorbar(im, cax=fig.add_axes([0.73, 0.30, 0.008, 0.6])); cb.set_label("change / noise")
b = fig.add_axes([0.85, 0.30, 0.13, 0.6])
dc = fp["decode"]; truth = fp["mystery"]["truth"]
yy = np.arange(len(dc["keys"]))
b.barh(yy, dc["coef"], xerr=dc["err"], color="#7b3fa0", alpha=0.8, label="decoded")
b.scatter([truth.get(k, 0) for k in dc["keys"]], yy, color="k", marker="|", s=300, zorder=3, label="injected")
SHORT = dict(ip="IP res.", mom="mom. res.", disp="displaced res.", mult="multiplicity", lam="Lambda yield", pt="K0S pT")
b.set_yticks(yy); b.set_yticklabels([SHORT[k] for k in dc["keys"]], fontsize=8)
b.invert_yaxis(); b.set_xlabel("coefficient (1 = full single-effect size)")
b.set_title(f"Mystery sample decoded\n({fp['mystery']['label']})", fontsize=9); b.legend(fontsize=8)
plt.savefig("fingerprint.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 3. anomaly detection
MOD = [("no selection", "#bbbbbb", ":"), ("written, no training*", "k", "--"), ("random init", "#1f77b4", "-"),
       ("boosted trees", "#2ca02c", "-"), ("written + prior*", "#d62728", "-"), ("written + fixed prior*", "#e377c2", "-"),
       ("written, layer 1 frozen*", "#fdae6b", "-"),
       ("supervised (truth labels, upper bound)", "#7b3fa0", ":")]
fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
for col, anom, title in ((0, "displaced", r"anomaly c$\tau$ = 3 cm (prior matches)"),
                         (1, "short", r"anomaly c$\tau$ = 0.3 cm (prior mismatch)")):
    d = json.load(open(f"cwola_{anom}.json")); rr = d["rows"]
    for m, c, ls in MOD:
        x = sorted({r["n_s"] for r in rr})
        a = np.array([[r["sig_0.01"] for r in rr if r["model"] == m and r["n_s"] == n] for n in x])
        ax[col].errorbar(np.array(x) + 0.0, a.mean(1), a.std(1), color=c, ls=ls, marker="o", ms=3.5, capsize=2, label=m)
    ax[col].axhline(5, color="gray", lw=0.8); ax[col].text(11, 5.5, r"5$\sigma$", color="gray", fontsize=8)
    ax[col].set_xscale("symlog", linthresh=10); ax[col].set_yscale("symlog", linthresh=1); ax[col].set_xlim(0, 500); ax[col].set_ylim(0, 60)
    ax[col].set_xlabel(f"injected signal in SR (background in SR: {d['B_SR']})")
    ax[col].set_ylabel(r"$Z = \epsilon_S n_S / \sqrt{\epsilon_B B_{SR}}$ at SR background eff. 1%")
    ax[col].set_title(title)
ax[0].legend(fontsize=7.5, loc="lower right")
# sculpting check: SR vs SB separation on background only, n_s = 0
d = json.load(open("cwola_displaced.json"))["rows"]
names = [m for m, *_ in MOD if m != "no selection"]
v = [[r["bkg_sr_auc"] for r in d if r["model"] == m and r["n_s"] == 0] for m in names]
ax[2].bar(range(len(names)), [np.mean(x) for x in v], yerr=[np.std(x) for x in v], color=[c for m, c, _ in MOD if m != "no selection"])
ax[2].axhline(0.5, color="k", lw=0.8); ax[2].set_ylim(0.45, 0.6)
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[2].set_ylabel("AUC of score for background SR vs SB"); ax[2].set_title("Mass sculpting check (no signal injected)")
fig.suptitle(r"3. Anomaly detection: CWoLa bump hunt for S$\to\pi^+\pi^-$ at 0.70 GeV in combinatorial V0 background (* = LLM-written)")
plt.tight_layout(); plt.savefig("anomaly.png", dpi=150); plt.close()
