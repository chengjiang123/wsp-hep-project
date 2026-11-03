"""Direction 3: weakly supervised anomaly detection (CWoLa bump hunt) with the written net as prior.

Data: combinatorial V0 candidates with 0.62 < m_pipi < 0.78 GeV (no K0S, no Lambda peak there),
plus n_S injected S -> pi+ pi- (m = 0.70 GeV) candidates. Two anomalies:
  displaced  c*tau = 3 cm    (matches the displaced-vertex prior)
  short      c*tau = 0.3 cm  (prior mismatch)
CWoLa: signal region SR |m - 0.70| < 0.02 vs sidebands SB 0.62-0.68, 0.72-0.78, no signal labels.
2-fold: train on one half, score the other. Inputs are the 6 mass-independent topological features
(log layer), so the background should not separate SR from SB.

Working point: score threshold where the true SR background efficiency is 1% (or 0.1%).
Significance Z = eps_S * n_S / sqrt(eps_B * B_SR), so it never divides by a zero count.

usage: python cwola.py {displaced,short} [threads]
"""
import json, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import models_v2 as V

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
FEATS = [0, 1, 2, 3, 4, 5]
MS, HW, LO, HI = 0.70, 0.02, 0.62, 0.78
NINJ = [0, 10, 20, 50, 100, 200, 400]
EFFB = (0.01, 0.001)          # true background efficiency in the SR
SEEDS = 3


def get(f, cat=None):
    D = dict(np.load(f)); m = D["m"]
    k = (m > LO) & (m < HI) & (np.ones_like(m, bool) if cat is None else D["cat"] == cat)
    return D["F"][k], m[k]


def fit_net(X, y, init, seed, **kw):
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    if init == "random":
        net = V.Net("random", space="log", feats=FEATS, H=16, seed=seed)
        return V.train(net, X, y, w=w, steps=1000)
    net = V.Net("written", space="log", feats=FEATS, seed=seed); mu, sig = net.prior(syst=True)
    return V.train(net, X, y, w=w, mu=mu, sig=sig, steps=1000, **kw)


def fit_gbm(X, y, seed):
    r = np.random.default_rng(seed); p = r.permutation(len(y)); nv = len(y) // 5
    w = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    kw = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=50, verbose=-1, seed=seed)
    dtr = lgb.Dataset(X[p[nv:]], y[p[nv:]], weight=w[p[nv:]])
    dva = lgb.Dataset(X[p[:nv]], y[p[:nv]], weight=w[p[:nv]], reference=dtr)
    return lgb.train(kw, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])


def main(anom):
    Fb, mb = get("adbkg.npz")
    Fs, msig = get({"displaced": "adsig3.npz", "short": "adsig03.npz"}[anom], cat=4)
    Fs, msig = Fs[np.abs(msig - MS) < HW], msig[np.abs(msig - MS) < HW]      # injected counts quoted in the SR
    Xb, Xs = V.phys(Fb, "log")[:, FEATS], V.phys(Fs, "log")[:, FEATS]
    srb = np.abs(mb - MS) < HW
    print("bkg in SR", srb.sum(), "in SB", (~srb).sum(), "signal pool", len(Fs), flush=True)
    # upper bound: boosted trees trained with truth labels on separate signal MC vs background
    out = dict(B_SR=int(srb.sum()), B_SB=int((~srb).sum()), rows=[])
    for n_s in NINJ:
        for seed in range(SEEDS):
            r = np.random.default_rng(100 * seed + n_s)
            si = r.choice(len(Fs), n_s, replace=False)
            X = np.r_[Xb, Xs[si]]; sr = np.r_[srb, np.ones(n_s, bool)]; truth = np.r_[np.zeros(len(Xb)), np.ones(n_s)]
            fold = r.integers(2, size=len(X))
            scores = {"random init": np.zeros(len(X)), "boosted trees": np.zeros(len(X)),
                      "written + prior*": np.zeros(len(X)), "written + fixed prior*": np.zeros(len(X)),
                      "written, layer 1 frozen*": np.zeros(len(X)),
                      "supervised (truth labels, upper bound)": np.zeros(len(X))}
            rest = np.setdiff1d(np.arange(len(Fs)), si)
            mc = Xs[r.choice(rest, min(3000, len(rest)), replace=False)]
            for k in (0, 1):
                a, b = fold != k, fold == k
                ya = sr[a].astype(float)
                scores["random init"][b] = V.predict(fit_net(X[a], ya, "random", seed), X[b])
                scores["written + prior*"][b] = V.predict(fit_net(X[a], ya, "written", seed, mode="max"), X[b])
                scores["written + fixed prior*"][b] = V.predict(fit_net(X[a], ya, "written", seed, mode="fixed"), X[b])
                scores["written, layer 1 frozen*"][b] = V.predict(
                    fit_net(X[a], ya, "written", seed, mode="1/N", freeze_l1=True), X[b])
                scores["boosted trees"][b] = fit_gbm(X[a], ya, seed).predict(X[b], raw_score=True)
                bg = a & (truth == 0)
                scores["supervised (truth labels, upper bound)"][b] = fit_gbm(np.r_[X[bg], mc], np.r_[np.zeros(bg.sum()), np.ones(len(mc))], seed).predict(X[b], raw_score=True)
            scores["written, no training*"] = V.predict(V.Net("written", space="log", feats=FEATS), X)
            scores["no selection"] = np.zeros(len(X))
            for name, s in scores.items():
                row = dict(model=name, n_s=n_s, seed=seed)
                # sculpting check: SR vs SB separation on background only
                row["bkg_sr_auc"] = float(roc_auc_score(sr[truth == 0], s[truth == 0])) if name != "no selection" else 0.5
                for f in EFFB:
                    if name == "no selection":
                        es, eb = 1.0, 1.0
                    elif n_s == 0:
                        es, eb = 0.0, f
                    else:
                        # tie-safe: signal efficiency at background efficiency f, interpolated along the SR ROC
                        fpr, tpr, _ = roc_curve(truth[sr], s[sr])
                        es, eb = float(np.interp(f, fpr, tpr)), f
                    row[f"effS_{f}"] = es
                    row[f"sig_{f}"] = es * n_s / np.sqrt(eb * srb.sum())
                out["rows"].append(row)
                print(anom, n_s, seed, name, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()
                                               if k.startswith(("sig", "effS", "bkg"))}, flush=True)
    json.dump(out, open(f"cwola_{anom}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
