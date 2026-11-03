"""K0S vs background: data-efficiency scan.  usage: python run_scan.py {feat,raw} [threads]"""
import json, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import HistGradientBoostingClassifier
import models as M

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
NS = [100, 300, 1000, 3000, 10000, 30000]
NTEST = 10000
EFF = 0.9                      # signal efficiency for the rejection working point


def load():
    d = [np.load(f"k0s_{s}.npz") for s in "ab"]
    D = {k: np.concatenate([x[k] for x in d]) for k in ("F", "raw", "m", "y", "cat")}
    rng = np.random.default_rng(0)
    S = rng.permutation(np.where(D["y"])[0]); Bk = rng.permutation(np.where(~D["y"])[0])
    te = np.r_[S[:NTEST], Bk[:NTEST]]
    tr = (S[NTEST:], Bk[NTEST:])
    natural_ratio = len(Bk) / len(S)
    return D, tr, te, natural_ratio


def rej(y, p, eff=EFF):
    fpr, tpr, _ = roc_curve(y, p)
    return 1 / max(np.interp(eff, tpr, fpr), 1e-6)


def sample(tr, n, seed):
    r = np.random.default_rng(seed)
    idx = np.r_[r.choice(tr[0], n // 2, replace=False), r.choice(tr[1], n // 2, replace=False)]
    return idx, np.r_[np.ones(n // 2), np.zeros(n // 2)]


def raw_std(D, tr):
    pool = np.r_[tr[0], tr[1]]
    mu, sd = D["raw"][pool].mean(0), D["raw"][pool].std(0)
    return lambda X: ((X - mu) / sd).astype(np.float32)


def plain_cuts(F):
    ok = np.ones(len(F), bool)
    for _, f, t, k, sg, _ in M.programmed_units():
        ok &= (F[:, f] > t) if sg > 0 else (F[:, f] < t)
    return ok


def train_raw(X, y, seed):
    """Generic MLP with 20% validation split and early stopping (generous to the baseline)."""
    r = np.random.default_rng(seed); p = r.permutation(len(y)); nv = max(len(y) // 5, 20)
    va, tr = p[:nv], p[nv:]
    net = M.RawMLP(seed=seed); best, best_sd = -1, None
    for block in range(25):
        M.train_map(net, X[tr], y[tr], steps=200, lr=1e-3, batch=256, seed=seed * 100 + block)
        a = roc_auc_score(y[va], M.predict(net, X[va])) if len(set(y[va])) > 1 else 0
        if a > best:
            best, best_sd = a, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd)
    return net


def main(which):
    D, tr, te, ratio = load()
    yte = D["y"][te].astype(float)
    Fte = M.std_feats(D["F"][te])
    rs = raw_std(D, tr); Rte = rs(D["raw"][te])
    out = []
    def rec(model, n, seed, p):
        out.append(dict(model=model, n=n, seed=seed, auc=roc_auc_score(yte, p), rej=rej(yte, p)))
        print(model, n, seed, round(out[-1]["auc"], 4), round(out[-1]["rej"], 1), flush=True)
    if which == "feat":
        rec("programmed, no training", 0, 0, M.predict(M.FeatNet("prog"), Fte))
        ok = plain_cuts(D["F"][te])
        es, eb = ok[yte == 1].mean(), ok[yte == 0].mean()
        out.append(dict(model="plain cuts, same thresholds", n=0, seed=0, eff_s=es, eff_b=eb, rej=1 / eb))
        print("plain cuts eff_s", es, "eff_b", eb)
    for n in NS:
        for seed in range(5 if which == "feat" else 3):
            idx, y = sample(tr, n, 1000 * seed + n)
            if which == "feat":
                F = M.std_feats(D["F"][idx])
                rec("features, random init", n, seed, M.predict(M.train_map(M.FeatNet("random", seed), F, y), Fte))
                for s in (1.0, 0.3, "syst"):
                    net = M.FeatNet("prog", seed); mu = M.flat(net)
                    sig = M.syst_sigma(net) if s == "syst" else s
                    rec(f"programmed + physics prior (sigma={s})", n, seed,
                        M.predict(M.train_map(net, F, y, prior_mean=mu, prior_sigma=sig), Fte))
                if n == NS[-1] and seed == 0:
                    bdt = HistGradientBoostingClassifier(max_iter=500).fit(D["F"][idx], y)
                    rec("BDT on 8 topological features", n, 0, bdt.predict_proba(D["F"][te])[:, 1])
            else:
                rec("MLP on raw helix parameters", n, seed, M.predict(train_raw(rs(D["raw"][idx]), y, seed), Rte))
    json.dump(dict(rows=out, natural_bkg_per_sig=ratio, eff=EFF), open(f"scan_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
