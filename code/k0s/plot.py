import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import models as M

COL = {"MLP on raw helix parameters": "#888888",
       "features, random init": "#1f77b4",
       "programmed + physics prior (sigma=1.0)": "#f4a3a3",
       "programmed + physics prior (sigma=0.3)": "#d62728",
       "programmed + physics prior (sigma=syst)": "#7b3fa0"}
LAB = {k: k.replace("sigma=syst", r"$\sigma$ from syst. variations").replace("sigma=", r"$\sigma$=") for k in COL}
FEAT_SHOW = sys.argv[1] if len(sys.argv) > 1 else "cos_pa"

sf = json.load(open("scan_feat.json")); sr = json.load(open("scan_raw.json"))
scan = sf["rows"] + sr["rows"]; eff = sf["eff"]
ro = json.load(open("readout.json"))
fig, ax = plt.subplots(2, 3, figsize=(17, 10))

# row 1: performance vs N
for k, c in COL.items():
    rows = [r for r in scan if r["model"] == k]; ns = sorted({r["n"] for r in rows})
    for i, key in enumerate(("auc", "rej")):
        a = np.array([[r[key] for r in rows if r["n"] == n] for n in ns])
        ax[0, i].errorbar(ns, a.mean(1), a.std(1), color=c, marker="o", ms=4, capsize=2, label=LAB[k])
zs = [r for r in scan if r["model"] == "programmed, no training"][0]
bdt = [r for r in scan if r["model"].startswith("BDT")][0]
cut = [r for r in scan if r["model"].startswith("plain cuts")][0]
for i, key in enumerate(("auc", "rej")):
    ax[0, i].axhline(zs[key], color="k", ls="--", lw=1.2, label="written, no training")
    ax[0, i].axhline(bdt[key], color="g", ls=":", lw=1.2, label="BDT on 8 features, N=30k")
    ax[0, i].set_xscale("log"); ax[0, i].set_xlabel("labelled training candidates N")
ax[0, 0].set_ylabel("test AUC"); ax[0, 0].set_title("AUC")
ax[0, 0].legend(fontsize=8, loc="lower right")
ax[0, 1].set_yscale("log"); ax[0, 1].set_ylabel(rf"background rejection $1/\epsilon_B$ at $\epsilon_S={eff}$")
ax[0, 1].set_title(f"Rejection at {int(eff*100)}% K$^0_S$ efficiency")
ax[0, 1].text(0.03, 0.95, f"plain cuts: $\\epsilon_S$={cut['eff_s']:.2f}, $1/\\epsilon_B$={cut['rej']:.0f}",
              transform=ax[0, 1].transAxes, fontsize=9, va="top")
ax[0, 1].text(0.03, 0.89, f"BDT line: {round(1e4 / bdt['rej'])} of 10k test bkg pass, so only a rough scale",
              transform=ax[0, 1].transAxes, fontsize=8, va="top", color="g")

# mass spectra, background reweighted to the natural mix
ms = ro["mass"]; m = np.array(ms["all"]["m"]); w = np.array(ms["all"]["w"])
bins = np.linspace(0.42, 0.58, 81); a = ax[0, 2]
for key, c, lab in (("pre", "#bbbbbb", "preselection"), ("plain_cuts", "k", "plain cuts"),
                    ("written", "#444444", f"written net, $\\epsilon_S$={eff}"),
                    ("tuned", "#d62728", f"tuned ($\\sigma$=0.3, N=30k), $\\epsilon_S$={eff}")):
    sel = np.ones(len(m), bool) if key == "pre" else np.array(ms[key], bool)
    ls = ":" if key == "plain_cuts" else "-"
    a.hist(m[sel], bins=bins, weights=w[sel], histtype="step" if key != "pre" else "stepfilled",
           color=c, ls=ls, lw=1.3, label=lab, alpha=0.6 if key == "pre" else 1)
a.set_yscale("log"); a.set_xlabel(r"$m_{\pi\pi}$ [GeV]"); a.set_ylabel("candidates (natural S/B)")
a.set_title(r"$m_{\pi\pi}$ after selection (not used as input)"); a.legend(fontsize=8)

# row 2: readout
names = [u[0] for u in M.programmed_units()]; x = np.arange(len(names))
keys = (("programmed_sigma0.3_N1000", "#f4a3a3", r"$\sigma$=0.3, N=1k"),
        ("programmed_sigma0.3_N30000", "#d62728", r"$\sigma$=0.3, N=30k"),
        ("programmed_sigmasyst_N1000", "#c9a6e0", "syst. $\\sigma$, N=1k"),
        ("programmed_sigmasyst_N30000", "#7b3fa0", "syst. $\\sigma$, N=30k"))
for i, (k, c, lab) in enumerate(keys):
    ax[1, 0].bar(x + (i - 1.5) * 0.2, [r["shift_in_dt"] for r in ro[k][:8]], 0.2, color=c, label=lab)
    ax[1, 1].bar(x + (i - 1.5) * 0.2, [r["out_weight"] for r in ro[k][:8]], 0.2, color=c, label=lab)
ax[1, 0].axhspan(-1, 1, color="g", alpha=0.08, label="within syst. variation")
ax[1, 0].set_ylabel(r"threshold shift / syst. variation $\Delta t$"); ax[1, 0].set_title("How far each written cut moved")
ax[1, 1].axhline(1.0, color="k", ls="--", lw=1, label="written")
ax[1, 1].set_ylabel("output weight"); ax[1, 1].set_title("How much each cut still counts")
for a in ax[1, :2]:
    a.set_xticks(x); a.set_xticklabels(names, rotation=35, ha="right", fontsize=8); a.axhline(0, color="k", lw=0.5)
ax[1, 0].legend(fontsize=7); ax[1, 1].legend(fontsize=7)

# one feature with its cut, written and tuned
fi = M.FEATS.index(FEAT_SHOW); h = ro["hist"][FEAT_SHOW]; a = ax[1, 2]
lo, hi = np.percentile(h["bkg"] + h["sig"], [0.5, 99.5])
a.hist(h["bkg"], bins=60, range=(lo, hi), histtype="step", density=True, color="#1f77b4", label="background")
a.hist(h["sig"], bins=60, range=(lo, hi), histtype="step", density=True, color="#d62728", label=r"K$^0_S$")
for k, ls, c, lab in (("programmed_N0", "--", "k", "written"), ("programmed_sigma0.3_N30000", "-", "#d62728", r"tuned $\sigma$=0.3, N=30k"),
                      ("programmed_sigmasyst_N30000", "-", "#7b3fa0", "tuned syst. $\\sigma$, N=30k")):
    a.axvline(ro[k][fi]["threshold"], color=c, ls=ls, lw=1.2, label=lab)
u = M.programmed_units()[fi]
a.axvspan(u[2] - u[5], u[2] + u[5], color="g", alpha=0.1, label="syst. variation")
a.set_yscale("log"); a.set_xlabel(FEAT_SHOW + (f" [{M.UNITS_OF[fi]}]" if M.UNITS_OF[fi] else ""))
a.set_title(f"{FEAT_SHOW}: written vs tuned cut"); a.legend(fontsize=7)

fig.suptitle(r"Hand-written K$^0_S\to\pi^+\pi^-$ topological selection: Pythia8 pp 13.6 TeV (5 overlaid MB events), ALICE-like tracking toy", fontsize=12)
plt.tight_layout(); plt.savefig("results.png", dpi=150)
