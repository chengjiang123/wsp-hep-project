"""Direction 1: tagger improvement on K0S.  usage: python improve.py {nets,trees}
Stars mark LLM-written models (written weights, or written start)."""
import json, os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import models as M, models_v2 as V

torch.set_num_threads(1)
NS = [int(x) for x in os.environ.get("NS", "100,300,1000,3000,10000,30000").split(",")]
EFF = 0.9


def load(f=None, ntest_s=None, ntest_b=None):
    f = f or os.environ.get("DATA", "nom.npz")
    ntest_s = ntest_s or int(os.environ.get("NTEST_S", 10000)); ntest_b = ntest_b or int(os.environ.get("NTEST_B", 25000))
    D = dict(np.load(f))
    r = np.random.default_rng(0)
    S = r.permutation(np.where(D["y"])[0]); B = r.permutation(np.where(~D["y"])[0])
    te = np.r_[S[:ntest_s], B[:ntest_b]]
    return D, (S[ntest_s:], B[ntest_b:]), te


def sample(tr, n, seed, wS=None, wB=None):
    r = np.random.default_rng(seed)
    pS = None if wS is None else wS / wS.sum(); pB = None if wB is None else wB / wB.sum()
    idx = np.r_[r.choice(tr[0], n // 2, replace=False, p=pS), r.choice(tr[1], n // 2, replace=False, p=pB)]
    return idx, np.r_[np.ones(n // 2), np.zeros(n // 2)]


def rej(y, p, eff=EFF):
    fpr, tpr, _ = roc_curve(y, p)
    return 1 / max(np.interp(eff, tpr, fpr), 1e-6)


def fit_written(F, y, mode="max", syst=True, extra=0, freeze=False, seed=0, space="log"):
    net = V.Net("written", space=space, extra=extra, seed=seed)
    mu, sig = net.prior(syst=syst)
    return V.train(net, F, y, mu=mu, sig=sig, mode=mode, freeze_l1=freeze)


def gbm(F, y, seed, init=None, init_te=None, Fte=None):
    """LightGBM, 20% validation for early stopping. init: starting log-odds per candidate."""
    r = np.random.default_rng(seed); p = r.permutation(len(y)); nv = max(len(y) // 5, 20)
    va, tr = p[:nv], p[nv:]
    kw = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=max(5, min(20, len(tr) // 50)),
              feature_fraction=1.0, verbose=-1, seed=seed)
    dtr = lgb.Dataset(F[tr], y[tr], init_score=None if init is None else init[tr])
    dva = lgb.Dataset(F[va], y[va], init_score=None if init is None else init[va], reference=dtr)
    b = lgb.train(kw, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    s = b.predict(Fte, raw_score=True)
    return s if init_te is None else s + init_te


def main(which):
    D, tr, te = load()
    yte = D["y"][te].astype(float)
    Xte = {sp: V.phys(D["F"][te], sp) for sp in ("lin", "log")}
    out = []
    def rec(name, n, seed, p, net=None):
        r = dict(model=name, n=n, seed=seed, auc=roc_auc_score(yte, p), rej=rej(yte, p), rej95=rej(yte, p, 0.95))
        if net is not None and getattr(net, "U", None):
            rows, _ = V.read_units(net); r["purity"] = float(np.mean([u["purity"] for u in rows]))
        out.append(r); print(name, n, seed, round(r["auc"], 5), round(r["rej"], 1), round(r["rej95"], 1), r.get("purity"), flush=True)
    if which == "nets":
        for sp in ("lin", "log"):
            rec(f"[{sp}] written soft + smooth, no training*", 0, 0, V.predict(V.Net("written", space=sp), Xte[sp]))
    for n in NS:
        for seed in range(5):
            idx, y = sample(tr, n, 1000 * seed + n)
            X = {sp: V.phys(D["F"][idx], sp) for sp in ("lin", "log")}
            if which == "nets":
                for sp, H in (("lin", 16), ("log", 16), ("log", 64)):
                    net = V.train(V.Net("random", space=sp, H=H, seed=seed), X[sp], y)
                    rec(f"[{sp}] random init, {H} units", n, seed, V.predict(net, Xte[sp]))
                for sp, name, kw in (("lin", "written, prior max(N,N0) + syst*", dict(mode="max", syst=True)),
                                     ("log", "written, prior 1/N*", dict(mode="1/N", syst=False)),
                                     ("log", "written, layer 1 frozen*", dict(mode="1/N", syst=False, freeze=True)),
                                     ("log", "written, prior max(N,N0)*", dict(mode="max", syst=False)),
                                     ("log", "written, prior max(N,N0) + syst*", dict(mode="max", syst=True)),
                                     ("log", "written + 48 free units, prior max(N,N0) + syst*", dict(mode="max", syst=True, extra=48))):
                    net = fit_written(X[sp], y, seed=seed, space=sp, **kw)
                    rec(f"[{sp}] {name}", n, seed, V.predict(net, Xte[sp]), net)
            else:
                Fr, Fter = D["F"][idx], D["F"][te]
                rec("boosted trees, from scratch", n, seed, gbm(Fr, y, seed, Fte=Fter))
                net = fit_written(X["log"], y, seed=seed)
                rec("boosted trees on top of tuned written net*", n, seed,
                    gbm(Fr, y, seed, init=V.predict(net, X["log"]), init_te=V.predict(net, Xte["log"]), Fte=Fter))
    json.dump(out, open(f"improve_{which}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
