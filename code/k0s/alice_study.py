"""Item 3: the written K0S network on real ALICE data (Run 1 open data, pp 7 TeV), against models without written weights.
MC (alice_mc.npz) has labels: train on N labelled candidates, score AUC / R90 / R95 on an MC test set.
Data (alice_data.npz) has none: for each model the score cut is set at 95% K0S efficiency in MC, then the m(pi pi)
spectrum of the real candidates is fitted (Gaussian peak + quadratic background) before and after the cut. This gives
  * the K0S yield kept in data / yield before the cut = the efficiency in data, to compare with 95% in MC
  * the background under the peak (+-3 sigma) and the signal purity in data
usage:  python alice_study.py MC.npz DATA.npz OUTDIR [--specs a.json b.json ...] [--ns 100,1000,10000] [--seeds 3]
"""
import argparse, json, os, numpy as np, torch
from scipy.optimize import curve_fit
from sklearn.metrics import roc_auc_score
import models_v2 as V
import handedit as H
from improve import sample, rej, gbm

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
MK = 0.497611


def split(D, ntest_s, ntest_b):
    r = np.random.default_rng(0)
    S = r.permutation(np.where(D["y"])[0]); B = r.permutation(np.where(~D["y"])[0])
    ntest_s = min(ntest_s, len(S) // 2); ntest_b = min(ntest_b, len(B) // 2)
    return (S[ntest_s:], B[ntest_b:]), np.r_[S[:ntest_s], B[:ntest_b]]


def peak(m, fit_range=(0.43, 0.57)):
    """Gaussian + quadratic fit of m(pi pi). Returns signal yield, background under +-3 sigma, sigma, and the fit."""
    h, e = np.histogram(m, bins=112, range=fit_range); c = 0.5 * (e[1:] + e[:-1]); w = e[1] - e[0]
    f = lambda x, a, mu, s, b0, b1, b2: a * np.exp(-0.5 * ((x - mu) / s) ** 2) + b0 + b1 * (x - MK) + b2 * (x - MK) ** 2
    p0 = [max(h.max() - np.median(h), 1), MK, 0.004, np.median(h), 0, 0]
    try:
        p, _ = curve_fit(f, c, h, p0=p0, sigma=np.sqrt(np.maximum(h, 1)), maxfev=20000)
    except Exception:
        return dict(S=float("nan"), B=float("nan"), sigma=float("nan"), mu=float("nan"))
    a, mu, s = p[0], p[1], abs(p[2])
    S = a * s * np.sqrt(2 * np.pi) / w
    lo, hi = mu - 3 * s, mu + 3 * s; xs = np.linspace(lo, hi, 200)
    trap = getattr(np, "trapezoid", None) or np.trapz
    B = trap(p[3] + p[4] * (xs - MK) + p[5] * (xs - MK) ** 2, xs) / w
    return dict(S=float(S), B=float(B), sigma=float(s), mu=float(mu), hist=h.tolist(), edges=e.tolist(), par=p.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mc"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--specs", nargs="*", default=["hand/r0.json"]); ap.add_argument("--ns", default="100,1000,10000")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--ntest_s", type=int, default=30000); ap.add_argument("--ntest_b", type=int, default=100000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = dict(np.load(a.mc)); D["y"] = D["y"].astype(bool)
    G = dict(np.load(a.data)) if os.path.exists(a.data) else None
    tr, te = split(D, a.ntest_s, a.ntest_b)
    Xte = V.phys(D["F"][te], "log"); yte = D["y"][te].astype(float)
    Xd = V.phys(G["F"], "log") if G is not None else None
    print(f"MC: {len(D['y'])} candidates, {D['y'].sum()} K0S, test {int(yte.sum())} K0S + {int((1 - yte).sum())} background"
          + (f";  data: {len(G['m'])} candidates" if G is not None else ""), flush=True)
    res, data_res = [], {}
    if G is not None:
        data_res["no selection"] = peak(G["m"])

    def record(name, n, seed, p_te, p_data=None):
        r = dict(model=name, n=n, seed=seed, auc=float(roc_auc_score(yte, p_te)), rej90=float(rej(yte, p_te, 0.9)), rej95=float(rej(yte, p_te, 0.95)))
        res.append(r)
        msg = f"{name:38s} N={n:6d} seed {seed}: MC AUC {r['auc']:.5f}  R90 {r['rej90']:8.1f}  R95 {r['rej95']:7.1f}"
        if p_data is not None and seed == 0 and n in (0, 1000):
            cut = np.quantile(p_te[yte == 1], 0.05)                 # 95% K0S efficiency in MC
            pk = peak(G["m"][p_data >= cut]); base = data_res["no selection"]
            pk["eff_data"] = pk["S"] / base["S"]; pk["purity"] = pk["S"] / (pk["S"] + pk["B"]); data_res[f"{name}, N={n}"] = pk
            msg += f"  | data: K0S eff {pk['eff_data']:.3f} (MC 0.95), purity {pk['purity']:.3f}, B under peak {pk['B']:.0f}"
        print(msg, flush=True)

    NS = [int(x) for x in a.ns.split(",")]
    for sp in a.specs:
        spec = json.load(open(sp)); tag = os.path.splitext(os.path.basename(sp))[0] if "hand/r0" not in sp else "ALICE cuts (written)"
        net = H.build(spec)
        record(f"{tag}, as written", 0, 0, V.predict(net, Xte), V.predict(net, Xd) if Xd is not None else None)
        for n in NS:
            for seed in range(a.seeds):
                idx, y = sample(tr, n, 1000 * seed + n); X = V.phys(D["F"][idx], "log")
                net = H.build(spec); mu = [p.detach().clone() for p in net.parameters()]
                net = V.train(net, X, y, mu=mu, sig=0.3, mode="max")
                record(f"{tag} + prior", n, seed, V.predict(net, Xte), V.predict(net, Xd) if Xd is not None else None)
    for n in NS:
        for seed in range(a.seeds):
            idx, y = sample(tr, n, 1000 * seed + n); X = V.phys(D["F"][idx], "log")
            net = V.train(V.Net("random", space="log", H=16, seed=seed), X, y)
            record("random start, 16 units", n, seed, V.predict(net, Xte), V.predict(net, Xd) if Xd is not None else None)
            Fte = np.r_[D["F"][te], G["F"]] if G is not None else D["F"][te]
            s = gbm(D["F"][idx], y, seed, Fte=Fte)
            record("boosted trees", n, seed, s[:len(te)], s[len(te):] if G is not None else None)
    json.dump(dict(mc=res, data=data_res), open(os.path.join(a.out, "alice_study.json"), "w"), indent=1)
    plot(res, data_res, a.out)


def plot(res, data_res, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2 if data_res else 1, figsize=(13 if data_res else 6.5, 4.6)); ax = np.atleast_1d(ax)
    names = list(dict.fromkeys(r["model"] for r in res if r["n"] > 0))
    for nm, c in zip(names, ["#d62728", "#ff7f0e", "#9467bd", "#7f7f7f", "#bcbd22", "#1f77b4"]):
        ns = sorted({r["n"] for r in res if r["model"] == nm})
        v = [[r["rej95"] for r in res if r["model"] == nm and r["n"] == n] for n in ns]
        ax[0].errorbar(ns, [np.mean(x) for x in v], yerr=[np.std(x) for x in v], fmt="o-", color=c, capsize=3, label=nm)
    for r in res:
        if r["n"] == 0:
            ax[0].axhline(r["rej95"], ls=":", color="#d62728", label=r["model"])
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("labelled MC candidates"); ax[0].set_ylabel("MC background rejection at 95% K0S eff.")
    ax[0].set_title("ALICE Run 1 MC, pp 7 TeV", fontsize=10); ax[0].legend(fontsize=7)
    if data_res:
        for (nm, pk), c in zip(data_res.items(), ["k", "#d62728", "#ff7f0e", "#7f7f7f", "#bcbd22", "#9467bd", "#1f77b4"]):
            if "hist" not in pk:
                continue
            e = np.array(pk["edges"]); ax[1].step(0.5 * (e[1:] + e[:-1]), pk["hist"], where="mid", color=c, lw=1.2,
                                                  label=f"{nm}" + (f": eff {pk['eff_data']:.2f}, purity {pk['purity']:.2f}" if "eff_data" in pk else ""))
        ax[1].set_yscale("log"); ax[1].set_xlabel(r"$m(\pi^+\pi^-)$ [GeV]"); ax[1].set_ylabel("candidates")
        ax[1].set_title("ALICE Run 1 data, cut at 95% MC efficiency", fontsize=10); ax[1].legend(fontsize=6.5)
    plt.tight_layout(); plt.savefig(os.path.join(out, "alice_study.png"), dpi=150); print("plot:", os.path.join(out, "alice_study.png"))


if __name__ == "__main__":
    main()
