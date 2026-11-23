"""Gradient-free hand editing of the K0S net in weight space.

A spec (JSON) lists every hidden unit with hand-chosen weights:
  cut units:    {"cut": name, "f": feature index, "t": threshold (physical units), "dir": ">" or "<",
                 "k": sharpness per layer unit, "w": output weight}
  free units:   {"lin": {feature index: weight per layer unit}, "b": bias, "w": output weight, "name": ...}
  "bias": output bias
Weights are written into the log-layer net of models_v2 (same physics layer as the gradient runs).
No gradient step is ever taken.

usage:
  python handedit.py diag spec.json      diagnostics on the labelled edit candidates only (NEDIT, default 1000)
  python handedit.py diag2 spec.json     calibration view and worst-ranked candidates
  env: NEDIT (labelled set size), STOPFRAC (held-back share for stopping), DATA, NTEST_S, NTEST_B
  python handedit.py test spec1.json ... test-set numbers (run once at the end); HE_OUT=out.json to choose the output file
"""
import json, os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
import models as M, models_v2 as V
from improve import load, sample

N = int(os.environ.get("NEDIT", 1000))          # labelled candidates the editor may use (half K0S, half background)
STOPFRAC = float(os.environ.get("STOPFRAC", 0))  # share of them held back to decide when to stop (never shown to the editor)
NAMES = M.FEATS
_CACHE = {}


def labels():
    """(D, te, edit idx, edit y, stop idx, stop y). Seed 1000*0+N as before, so NEDIT=1000 STOPFRAC=0 reproduces the old set."""
    if "x" not in _CACHE:
        D, tr, te = load()
        idx, y = sample(tr, N, 1000 * 0 + N)
        h = N // 2; k = int(round(h * (1 - STOPFRAC)))
        e = np.r_[np.arange(k), h + np.arange(k)]; s_ = np.r_[np.arange(k, h), h + np.arange(k, h)]
        _CACHE["x"] = (D, te, idx[e], y[e], idx[s_], y[s_])
    return _CACHE["x"]


def stop_auc(spec):
    D, te, ie, ye, is_, ys = labels()
    return float(roc_auc_score(ys, V.predict(build(spec), V.phys(D["F"][is_], "log")))) if len(ys) else float("nan")


def layer_x(f, t):
    return float((V.tf(f, t) - V.LOG_C[f]) / V.LOG_S[f])


def build(spec):
    units = spec["units"]
    net = V.Net("random", space="log", H=len(units))
    W = np.zeros((len(units), V.NF)); b = np.zeros(len(units)); w2 = np.zeros(len(units))
    for j, u in enumerate(units):
        if "cut" in u:
            f = u["f"]; tl = layer_x(f, u["t"])
            up = (u["dir"] == ">") == V.INC[f]           # unit active for large layer values?
            s = 1.0 if up else -1.0
            W[j, f] = s * u["k"]; b[j] = -s * u["k"] * tl
        else:
            for f, c in u["lin"].items():
                W[j, int(f)] = c
            b[j] = u["b"]
        w2[j] = u["w"]
    with torch.no_grad():
        net.l1.weight.copy_(torch.tensor(W, dtype=torch.float32)); net.l1.bias.copy_(torch.tensor(b, dtype=torch.float32))
        net.l2.weight.copy_(torch.tensor(w2, dtype=torch.float32)[None]); net.l2.bias.fill_(float(spec["bias"]))
    return net


def rej(y, p, eff):
    fpr, tpr, _ = roc_curve(y, p)
    return 1 / max(np.interp(eff, tpr, fpr), 1e-6)


def unit_acts(net, X):
    with torch.no_grad():
        return torch.tanh(net.l1(torch.tensor(X))).numpy()


def diag(spec):
    D, te, idx, y, _, _ = labels()                       # NEDIT=1000: the same candidates the gradient runs used (seed 0)
    F = D["F"][idx]; X = V.phys(F, "log"); cat = D["cat"][idx]
    net = build(spec); p = V.predict(net, X); A = unit_acts(net, X)
    s, bkg = y == 1, y == 0
    print(f"edit set: {s.sum()} K0S, {bkg.sum()} bkg   AUC {roc_auc_score(y, p):.4f}   "
          f"rej90 {rej(y, p, .9):.0f}  rej95 {rej(y, p, .95):.0f}  rej98 {rej(y, p, .98):.0f}")
    thr90 = np.quantile(p[s], 0.10); thr95 = np.quantile(p[s], 0.05)
    print(f"bkg passing at 90% / 95% eff: {(p[bkg] >= thr90).sum()} / {(p[bkg] >= thr95).sum()} of {bkg.sum()}")
    print("\nunit                         pass(K0S)  pass(bkg)  w_out  | K0S lost by this unit alone")
    for j, u in enumerate(spec["units"]):
        nm = u.get("cut", u.get("name", f"free{j}"))
        print(f"{nm:28s} {np.mean(A[s, j] > 0):8.3f}  {np.mean(A[bkg, j] > 0):8.3f}  {u['w']:5.2f}  |"
              f" {np.sum((A[s, j] < 0) & (A.shape[1] > 0)):4d}")
    # background survivors and signal losers at the 95% working point
    surv = bkg & (p >= thr95); lose = s & (p < thr95)
    print(f"\nfeature quantiles (10/50/90%):  K0S | bkg survivors at 95% ({surv.sum()}) | K0S losers ({lose.sum()}) | all bkg")
    for f, nm in enumerate(NAMES):
        q = lambda m: " ".join(f"{v:8.4g}" for v in np.quantile(F[m, f], [.1, .5, .9])) if m.sum() else "   -"
        print(f"{nm:20s} {q(s)} | {q(surv)} | {q(lose)} | {q(bkg)}")
    print("bkg survivor categories (1 Lambda, 2 other true pair, 3 combinatorial):", np.bincount(cat[surv], minlength=4)[1:])
    # which unit is the most negative for each survivor / loser
    for lab, m in (("survivors: unit voting most for them", surv), ("losers: unit voting most against them", lose)):
        if m.sum():
            contrib = A[m] * np.array([u["w"] for u in spec["units"]])
            k = np.argmax(contrib, 1) if "surv" in lab else np.argmin(contrib, 1)
            names = [spec["units"][i].get("cut", spec["units"][i].get("name")) for i in k]
            vals, cnt = np.unique(names, return_counts=True)
            print(lab, dict(zip(vals, cnt.tolist())))


def test(specs):
    D, te = labels()[:2]; y = D["y"][te].astype(float); X = V.phys(D["F"][te], "log")
    out = []
    for f in specs:
        p = V.predict(build(json.load(open(f))), X)
        sp = json.load(open(f))
        r = dict(spec=f, auc=roc_auc_score(y, p), rej=rej(y, p, .9), rej95=rej(y, p, .95), stop_auc=stop_auc(sp))
        out.append(r); print(r)
    return out


def diag2(spec, out_json="hand/edit_auc.json", tag=None):
    """Calibration view: per feature bin, logit of the observed K0S fraction minus logit of the mean predicted probability
    (after one global affine map of the score to log-odds, display only; it does not change the ranking).
    Positive 'miss' = the data is more K0S-like than the net thinks there."""
    from sklearn.linear_model import LogisticRegression
    D, te, idx, y, _, _ = labels()
    F = D["F"][idx]; X = V.phys(F, "log")
    p = V.predict(build(spec), X)
    a = roc_auc_score(y, p)
    lr = LogisticRegression(C=1e6).fit(p[:, None], y); lo = lr.decision_function(p[:, None])
    print(f"edit AUC {a:.5f}   rej90 {rej(y, p, .9):.0f}  rej95 {rej(y, p, .95):.0f}  rej98 {rej(y, p, .98):.0f}   "
          f"pairs mis-ordered: {int((1 - a) * (y == 1).sum() * (y == 0).sum())} of {int((y == 1).sum() * (y == 0).sum())}")
    edges = np.array([-6, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 8])
    for f, nm in enumerate(NAMES):
        b = np.clip(np.digitize(X[:, f], edges) - 1, 0, len(edges) - 2)
        cells = []
        for i in range(len(edges) - 1):
            m = b == i
            ns, nb = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
            if ns + nb < 8:
                continue
            emp = (ns + 0.5) / (ns + nb + 1); mod = np.clip(np.mean(1 / (1 + np.exp(-lo[m]))), 1e-4, 1 - 1e-4)
            miss = np.log(emp / (1 - emp)) - np.log(mod / (1 - mod))     # logit(observed K0S fraction) - logit(predicted)
            cells.append(f"[{edges[i]:+.1f},{edges[i+1]:+.1f}] S{ns:3d}/B{nb:3d} miss {miss:+5.1f}")
        print(f"{nm:20s} " + " | ".join(cells))
    # the worst-ranked pairs: background with the highest score, K0S with the lowest
    order = np.argsort(-p); top_b = [i for i in order if y[i] == 0][:6]; low_s = [i for i in order[::-1] if y[i] == 1][:6]
    for lab, ii in (("highest background", top_b), ("lowest K0S", low_s)):
        print(lab + ":")
        for i in ii:
            print("   " + "  ".join(f"{nm[:10]}={F[i, f]:.4g}" for f, nm in enumerate(NAMES)) + f"   score {p[i]:+.2f}")
    if tag is not None:
        import os
        h = json.load(open(out_json)) if os.path.exists(out_json) else {}
        h[tag] = a; json.dump(h, open(out_json, "w"), indent=1)
    return a


if __name__ == "__main__":
    if sys.argv[1] == "diag":
        diag(json.load(open(sys.argv[2])))
    elif sys.argv[1] == "diag2":            # python handedit.py diag2 hand/rN.json  (records the edit AUC under rN)
        diag2(json.load(open(sys.argv[2])), tag=sys.argv[2].split("/")[-1].replace(".json", ""))
    else:
        import os                              # HE_OUT=file.json to keep separate results (e.g. for LLM runs)
        json.dump(test(sys.argv[2:]), open(os.environ.get("HE_OUT", "handedit_test.json"), "w"), indent=1)
