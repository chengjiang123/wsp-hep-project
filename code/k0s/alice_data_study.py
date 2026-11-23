"""Item 3 on real ALICE data only (Run 1 open data, pp 7 TeV, no Monte Carlo): the K0S labels come from the mass peak.
  * a fit of m(pi pi) gives the peak position mu and width sigma
  * signal region SR = |m - mu| < 3 sigma, sidebands SB = 6 to 12 sigma on both sides (pure background)
  * any selection is scored by sideband subtraction on a held-out half of the data:
        K0S kept  S(c) = N_SR(c) - alpha N_SB(c),  background under the peak  B(c) = alpha N_SB(c),  alpha = 0.5
    which gives the K0S efficiency, the background rejection at 90% / 95% efficiency, and an AUC, all from data
  * models that need labels learn SR against SB (mixed samples, as in CWoLa), with N labelled candidates
Checks: the background shape in the left and right sidebands (if it depends on the mass, sideband subtraction is biased).
usage: python alice_data_study.py DATA.npz OUTDIR [--specs a.json ...] [--ns 100,1000,10000] [--seeds 3]
"""
import argparse, json, os, numpy as np, torch
from scipy.optimize import curve_fit
import models_v2 as V
import handedit as H
from improve import gbm

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
MK = 0.497611
NAMES = ["dca_dau", "cos_pa", "dca_pos", "dca_neg", "radius", "mL/p", "qT/|alpha|", "dm_Lambda"]


def fit_peak(m):
    h, e = np.histogram(m, bins=160, range=(0.42, 0.58)); c = 0.5 * (e[1:] + e[:-1])
    f = lambda x, a, mu, s, b0, b1, b2: a * np.exp(-0.5 * ((x - mu) / s) ** 2) + b0 + b1 * (x - MK) + b2 * (x - MK) ** 2
    p, _ = curve_fit(f, c, h, p0=[h.max(), MK, 0.004, np.median(h), 0, 0], sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000)
    return p[1], abs(p[2]), dict(hist=h.tolist(), edges=e.tolist(), par=p.tolist())


class Regions:
    def __init__(self, m, mu, s):
        d = np.abs(m - mu) / s
        self.sr = d < 3; self.sb = (d > 6) & (d < 12); self.alpha = 6.0 / 12.0

    def curve(self, score, idx):
        """sideband-subtracted K0S efficiency and background, for thresholds over the score range"""
        sr, sb = self.sr[idx], self.sb[idx]; s = score
        qs = np.unique(np.quantile(s[sr | sb], np.linspace(0, 1, 801)))
        o_sr = np.sort(s[sr]); o_sb = np.sort(s[sb])
        nsr = len(o_sr) - np.searchsorted(o_sr, qs, side="left"); nsb = len(o_sb) - np.searchsorted(o_sb, qs, side="left")
        S = nsr - self.alpha * nsb; B = self.alpha * nsb
        return S / S[0], B / B[0], qs

    def metrics(self, score, idx):
        eff, fb, qs = self.curve(score, idx)
        eff = np.minimum.accumulate(np.clip(eff, 0, None)); fb = np.minimum.accumulate(fb)
        out = {}
        for e0 in (0.9, 0.95):
            k = np.where(eff >= e0)[0]; k = k[-1] if len(k) else 0
            out[f"rej{int(e0 * 100)}"] = float(1 / max(fb[k], 1e-6)); out[f"cut{int(e0 * 100)}"] = float(qs[k])
        x, y = fb[::-1], eff[::-1]
        out["auc"] = float(np.sum(np.diff(x) * 0.5 * (y[1:] + y[:-1])) + (1 - x[-1]) * 1.0)
        return out


def fit_sb(m, w=None, bins=112, rng=(0.43, 0.57)):
    """Gaussian + quadratic fit -> K0S yield S, background B under +-3 sigma, sigma, mu; and B from a fit of the sidebands only"""
    h, e = np.histogram(m, bins=bins, range=rng); c = 0.5 * (e[1:] + e[:-1]); bw = e[1] - e[0]
    q = lambda x, b0, b1, b2: b0 + b1 * (x - MK) + b2 * (x - MK) ** 2
    f = lambda x, a, mu, s, b0, b1, b2: a * np.exp(-0.5 * ((x - mu) / s) ** 2) + q(x, b0, b1, b2)
    try:
        p, _ = curve_fit(f, c, h, p0=[max(h.max() - np.median(h), 1), MK, 0.0039, np.median(h), 0, 0],
                         sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000)
    except Exception:
        return None
    a, mu, sg = p[0], p[1], abs(p[2]); xs = np.linspace(mu - 3 * sg, mu + 3 * sg, 200); trap = getattr(np, "trapezoid", None) or np.trapz
    S = a * sg * np.sqrt(2 * np.pi) / bw; B = trap(q(xs, *p[3:]), xs) / bw
    side = (np.abs(c - mu) > 6 * sg)
    try:
        pb, _ = curve_fit(q, c[side], h[side], p0=p[3:], sigma=np.sqrt(np.maximum(h[side], 1)))
        Bsb = trap(q(xs, *pb), xs) / bw
    except Exception:
        Bsb = float("nan")
    return dict(S=float(S), B=float(max(B, 1e-9)), sigma=float(sg), mu=float(mu), B_sideband=float(Bsb))


def fit_metrics(score, m, base):
    """K0S efficiency and background rejection from mass fits after cuts on the score (no sideband-shape assumption);
    plus the peak width and B(full fit) / B(sideband-only fit) at the 95% working point, as sculpting checks"""
    qs = np.quantile(score, np.linspace(0, 0.97, 50)); pts = []
    for c in qs:
        r = fit_sb(m[score >= c])
        if r:
            pts.append((r["S"] / base["S"], base["B"] / r["B"], r))
    out = {}
    for e0 in (0.9, 0.95):
        ok = [t for t in pts if t[0] >= e0]
        best = ok[-1] if ok else pts[0]
        out[f"rej{int(e0 * 100)}"] = float(best[1]); out[f"eff_at{int(e0 * 100)}"] = float(best[0])
        if e0 == 0.95:
            out["sigma95_MeV"] = 1000 * best[2]["sigma"]; out["B_full_over_sideband95"] = best[2]["B"] / max(best[2]["B_sideband"], 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--specs", nargs="*", default=[]); ap.add_argument("--ns", default="100,1000,10000"); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = dict(np.load(a.data, allow_pickle=True)); m = D["m"]; F = D["F"]
    mu, s, fit = fit_peak(m); R = Regions(m, mu, s)
    rng = np.random.default_rng(0); perm = rng.permutation(len(m)); half = len(m) // 2
    te, trp = perm[:half], perm[half:]
    Xall = V.phys(F, "log")
    print(f"{len(m)} candidates, peak at {1000 * mu:.2f} MeV, sigma {1000 * s:.2f} MeV; SR {R.sr.sum()}, SB {R.sb.sum()}, "
          f"K0S in SR (sideband-subtracted) {R.sr.sum() - R.alpha * R.sb.sum():.0f}", flush=True)

    # check: background in the left and right sidebands should look alike (else sideband subtraction is biased)
    left = R.sb & (m < mu); right = R.sb & (m > mu); chk = {}
    for i, n in enumerate(NAMES):
        ql, qr = np.quantile(F[left, i], [0.25, 0.5, 0.75]), np.quantile(F[right, i], [0.25, 0.5, 0.75])
        chk[n] = dict(left=ql.round(4).tolist(), right=qr.round(4).tolist())
    print("background quartiles, left vs right sideband:"); [print(f"  {n:11s} {v['left']}  {v['right']}") for n, v in chk.items()]

    res = []
    pool = trp[(R.sr | R.sb)[trp]]

    base = fit_sb(m[te])

    def rec(name, n, seed, score):
        sbm = R.metrics(score[te], te); fm = fit_metrics(score[te], m[te], base)
        r = dict(model=name, n=n, seed=seed, auc_sb=sbm["auc"], rej90_sb=sbm["rej90"], rej95_sb=sbm["rej95"], **fm); res.append(r)
        print(f"{name:40s} N={n:6d} seed {seed}: fit R90 {r['rej90']:6.1f}  R95 {r['rej95']:6.1f}  (sideband R95 {r['rej95_sb']:6.1f}, AUC {r['auc_sb']:.4f})"
              f"  peak sigma {r['sigma95_MeV']:.2f} MeV, B fit/sideband {r['B_full_over_sideband95']:.2f}", flush=True)
        return r

    # the standard ALICE selection as a hard rectangular cut: one working point
    alice = (F[:, 0] < 1.0) & (F[:, 1] > 0.97) & (F[:, 2] > 0.06) & (F[:, 3] > 0.06) & (F[:, 4] > 0.5) & (F[:, 5] < 20) & (F[:, 7] > 0.005)
    srt, sbt = R.sr[te], R.sb[te]
    S0 = srt.sum() - R.alpha * sbt.sum(); B0 = R.alpha * sbt.sum()
    Sc = (srt & alice[te]).sum() - R.alpha * (sbt & alice[te]).sum(); Bc = R.alpha * (sbt & alice[te]).sum()
    fa = fit_sb(m[te][alice[te]])
    alice_pt = dict(eff=float(Sc / S0), rej=float(B0 / max(Bc, 1)), eff_fit=fa["S"] / base["S"], rej_fit=base["B"] / fa["B"], sigma_MeV=1000 * fa["sigma"])
    print(f"ALICE rectangular cuts (mass fit): K0S efficiency {alice_pt['eff_fit']:.3f}, rejection {alice_pt['rej_fit']:.1f}")
    print(f"ALICE rectangular cuts: K0S efficiency {alice_pt['eff']:.3f}, background rejection {alice_pt['rej']:.1f}")

    specs = [("ALICE cuts, written soft", "hand/r0.json")] + [(os.path.splitext(os.path.basename(p))[0], p) for p in a.specs]
    for name, path in specs:
        spec = json.load(open(path))
        rec(f"{name}, as written", 0, 0, V.predict(H.build(spec), Xall))
    NS = [int(x) for x in a.ns.split(",")]
    for n in NS:
        for seed in range(a.seeds):
            r = np.random.default_rng(1000 * seed + n); idx = r.choice(pool, size=min(n, len(pool)), replace=False)
            y = R.sr[idx].astype(np.float32); X = Xall[idx]
            for name, path in specs[:1]:
                net = H.build(json.load(open(path))); mu0 = [p.detach().clone() for p in net.parameters()]
                rec(f"{name} + prior", n, seed, V.predict(V.train(net, X, y, mu=mu0, sig=0.3, mode="max"), Xall))
            rec("random start, 16 units", n, seed, V.predict(V.train(V.Net("random", space="log", H=16, seed=seed), X, y), Xall))
            rec("boosted trees", n, seed, gbm(F[idx], y, seed, Fte=F))
    json.dump(dict(peak=dict(mu=mu, sigma=s, **fit), alice_rect=alice_pt, sideband_check=chk, results=res),
              open(os.path.join(a.out, "alice_data_study.json"), "w"), indent=1)
    plot(res, alice_pt, fit, a.out)


def plot(res, alice_pt, fit, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(dict.fromkeys(r["model"] for r in res if r["n"] > 0))
    for nm, c in zip(names, ["#ff7f0e", "#7f7f7f", "#bcbd22", "#9467bd"]):
        ns = sorted({r["n"] for r in res if r["model"] == nm})
        v = [[r["rej95"] for r in res if r["model"] == nm and r["n"] == n] for n in ns]   # from mass fits
        ax[0].errorbar(ns, [np.mean(x) for x in v], yerr=[np.std(x) for x in v], fmt="o-", color=c, capsize=3, label=nm)
    for r, c in zip([r for r in res if r["n"] == 0], ["#d62728", "#e377c2", "#8c564b", "#17becf"]):
        ax[0].axhline(r["rej95"], ls=":", color=c, label=r["model"] + " (0 labels)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("labelled candidates (signal region vs sidebands)")
    ax[0].set_ylabel("background rejection at 95% K0S eff. (data)"); ax[0].set_title("ALICE Run 1 pp 7 TeV, real data only", fontsize=10)
    ax[0].legend(fontsize=7)
    e = np.array(fit["edges"]); ax[1].step(0.5 * (e[1:] + e[:-1]), fit["hist"], where="mid", color="k")
    ax[1].set_xlabel(r"$m(\pi^+\pi^-)$ [GeV]"); ax[1].set_ylabel("candidates"); ax[1].set_title(
        f"all candidates; ALICE rectangular cuts keep {alice_pt['eff_fit']:.2f} of the K0S, reject {alice_pt['rej_fit']:.0f}x", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(out, "alice_data_study.png"), dpi=150); print("plot:", os.path.join(out, "alice_data_study.png"))


if __name__ == "__main__":
    main()
