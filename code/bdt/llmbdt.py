"""LLM-searched boosted trees on K0S ("WSP + BDT"): the LLM writes and edits the branches of a few decision trees
directly (feature, threshold, leaf log-odds), then LightGBM boosts on top of them with the same recipe as the plain BDT.

Same data, test set, label samples, validation split and boosting recipe as code/k0s/improve.py (plain BDT baseline):
  DATA = $WSP_DATA/v0_k0s/big/nom.npz (made by code/k0s/gen.py), test = 30k K0S + 100k background, labels sample(tr, N, 1000*seed + N),
  20 % of the N labels held back (early stopping of the boosting, and choice of the LLM round).
The LLM sees only the training part (80 %) diagnostics and the held-back metrics; test metrics are never shown.

usage
  python llmbdt.py prep R        write batches/round<R>_<part>.md (R = 1, 2, 3)
  python llmbdt.py ingest R      read answers/round<R>_<part>.json, score every ensemble, store state/
  python llmbdt.py final         pick the round per cell (held-back log-loss of trees + boosting) -> results.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BDT_OUT", os.path.join(HERE, "..", "..", "results", "k0s_bdt"))   # batches/, answers/, state/, results.json
sys.path[:0] = [HERE, os.path.join(HERE, "..", "k0s")]
os.environ.setdefault("DATA", os.path.join(os.environ.get("WSP_DATA", os.path.join(HERE, "..", "..", "data")), "v0_k0s", "big", "nom.npz"))
os.environ.setdefault("NTEST_S", "30000")
os.environ.setdefault("NTEST_B", "100000")
import compat  # noqa: F401,E402  (hide the broken dask before lightgbm)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

import improve as I  # noqa: E402

NS = [100, 300, 1000, 3000, 10000, 30000, 50000]
SEEDS = range(5)
PARTS = {"A": [100, 300, 1000, 3000], "B": [10000, 30000, 50000]}
FEAT = ["DCA between the two daughter tracks [cm]", "cos of the pointing angle", "DCA of the positive daughter to the PV [cm]",
        "DCA of the negative daughter to the PV [cm]", "transverse decay radius [cm]", "m L / p, proper decay length with the K0S mass [cm]",
        "Armenteros ratio qT / |alpha|", "|m(p pi) - m_Lambda| [GeV] (mass under the Lambda hypothesis minus 1.11568)"]
MAX_TREES, MAX_DEPTH = 12, 4
_D = {}


def data():
    if not _D:
        D, tr, te = I.load()
        _D.update(D=D, tr=tr, te=te, yte=D["y"][te].astype(float))
    return _D


def cell(n, seed):
    """labels of one cell, split exactly as improve.gbm does: (F_train, y_train, F_val, y_val, F_all, y_all, tr, va)"""
    d = data()
    idx, y = I.sample(d["tr"], n, 1000 * seed + n)
    r = np.random.default_rng(seed); p = r.permutation(len(y)); nv = max(len(y) // 5, 20)
    va, tr = p[:nv], p[nv:]
    F = d["D"]["F"][idx]
    return F[tr], y[tr], F[va], y[va], F, y, tr, va


# ------------------------------------------------------------------ trees written by the LLM
def check(ens):
    trees = ens.get("trees", [])
    if not isinstance(trees, list) or len(trees) > MAX_TREES:
        raise ValueError(f"need a list of at most {MAX_TREES} trees")

    def walk(node, depth):
        if isinstance(node, (int, float)):
            if not np.isfinite(node) or abs(node) > 20:
                raise ValueError("leaf log-odds must be finite and |leaf| <= 20")
            return
        if depth >= MAX_DEPTH:
            raise ValueError(f"depth above {MAX_DEPTH}")
        if int(node["f"]) not in range(8):
            raise ValueError("feature index 0..7")
        float(node["t"])
        walk(node["lo"], depth + 1); walk(node["hi"], depth + 1)
    for t in trees:
        walk(t, 0)
    float(ens.get("bias", 0.0))


def score(ens, F):
    s = np.full(len(F), float(ens.get("bias", 0.0)))

    def walk(node, mask):
        if isinstance(node, (int, float)):
            s[mask] += float(node); return
        lo = F[:, int(node["f"])] < float(node["t"])
        walk(node["lo"], mask & lo); walk(node["hi"], mask & ~lo)
    for t in ens.get("trees", []):
        walk(t, np.ones(len(F), bool))
    return s


def boost(Ftr, ytr, Fva, yva, init_tr, init_va, seed):
    """improve.gbm recipe with the LLM score as init_score; returns the booster"""
    kw = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=max(5, min(20, len(ytr) // 50)),
              feature_fraction=1.0, verbose=-1, seed=seed)
    dtr = lgb.Dataset(Ftr, ytr, init_score=init_tr)
    dva = lgb.Dataset(Fva, yva, init_score=init_va, reference=dtr)
    return lgb.train(kw, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def evaluate(ens, n, seed):
    Ftr, ytr, Fva, yva, F, y, _, _ = cell(n, seed)
    d = data(); Fte = d["D"]["F"][d["te"]]; yte = d["yte"]
    s_tr, s_va, s_te = score(ens, Ftr), score(ens, Fva), score(ens, Fte)
    b = boost(Ftr, ytr, Fva, yva, s_tr, s_va, seed)
    bva = s_va + b.predict(Fva, raw_score=True); bte = s_te + b.predict(Fte, raw_score=True)
    auc = lambda yy, ss: float(roc_auc_score(yy, ss)) if len(set(ss.round(9))) > 1 else 0.5
    gain = b.feature_importance("gain"); gain = gain / max(gain.sum(), 1e-12)
    dump = b.dump_model()["tree_info"]
    splits = []
    for t in dump[:200]:
        def walk(nd):
            if "split_feature" in nd:
                splits.append((nd["split_gain"], nd["split_feature"], nd["threshold"])); walk(nd["left_child"]); walk(nd["right_child"])
        walk(t["tree_structure"])
    agg = {}
    for g, f, t in splits:                      # the same split recurs in many boosting trees: sum its gain
        k = (int(f), float(f"{t:.3g}")); agg[k] = agg.get(k, 0) + g
    tot = max(sum(agg.values()), 1e-12)
    splits = sorted(((g, f, t) for (f, t), g in agg.items()), reverse=True)
    nbv = int((yva == 0).sum())
    r95 = lambda yy, ss: (lambda r: r if r <= nbv else f"> {nbv} (no held-back background passes)")(float(I.rej(yy, ss, .95)))
    return dict(
        val=dict(trees_auc=auc(yva, s_va), trees_r95=r95(yva, s_va), trees_logloss=float(log_loss(yva, sig(s_va), labels=[0, 1])),
                 boost_auc=auc(yva, bva), boost_r95=r95(yva, bva), boost_logloss=float(log_loss(yva, sig(bva), labels=[0, 1])),
                 boost_trees_added=int(b.best_iteration or b.num_trees()),
                 boost_gain_share=[round(float(g), 3) for g in gain],
                 boost_top_splits=[dict(f=int(f), t=float(f"{t:.4g}"), gain_share=round(float(g / tot), 3))
                                   for g, f, t in splits[:8]]),
        test=dict(trees_auc=auc(yte, s_te), trees_r95=float(I.rej(yte, s_te, .95)),
                  boost_auc=auc(yte, bte), boost_r95=float(I.rej(yte, bte, .95))))


# ------------------------------------------------------------------ prompts
SYSTEM = open(os.path.join(HERE, "prompt_system.txt")).read()


def quant(x):
    return [float(f"{v:.4g}") for v in np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])]


def diag_start(n, seed):
    Ftr, ytr, *_ = cell(n, seed)
    S, B = Ftr[ytr == 1], Ftr[ytr == 0]
    return {f"{i} {FEAT[i]}": dict(K0S=quant(S[:, i]), background=quant(B[:, i])) for i in range(8)}


def diag_survivors(ens, n, seed):
    """training-part background that passes the threshold keeping 95 % of training K0S"""
    Ftr, ytr, *_ = cell(n, seed)
    s = score(ens, Ftr); cut = np.quantile(s[ytr == 1], 0.05)
    Bs = Ftr[(ytr == 0) & (s > cut)]
    out = dict(background_kept=f"{len(Bs)} of {(ytr == 0).sum()} training background at 95 % training K0S efficiency")
    if len(Bs) >= 3:
        out["kept_background_quantiles_5_25_50_75_95"] = {f"{i}": quant(Bs[:, i]) for i in range(8)}
    return out


def item_id(n, seed):
    return f"c_N{n}_s{seed}"


def state_path(n, seed):
    return os.path.join(OUT, "state", item_id(n, seed) + ".json")


def load_state(n, seed):
    p = state_path(n, seed)
    return json.load(open(p)) if os.path.exists(p) else dict(N=n, seed=seed, rounds=[])


def prep(R):
    os.makedirs(os.path.join(OUT, "batches"), exist_ok=True)
    for part, ns in PARTS.items():
        lines = [SYSTEM, f"\n# Round {R}. Items in this file: {len(ns) * len(SEEDS)}\n"]
        for n in ns:
            for seed in SEEDS:
                st = load_state(n, seed)
                ntr = n - max(n // 5, 20)
                it = dict(item=item_id(n, seed), labels=n, training_labels=ntr, held_back_labels=n - ntr,
                          training_quantiles_5_25_50_75_95=diag_start(n, seed))
                hist = []
                for r in st["rounds"]:
                    hist.append(dict(round=r["round"], ensemble=r["ensemble"], held_back=r["val"],
                                     training_survivors=r["survivors"]))
                if hist:
                    it["your_previous_rounds"] = hist
                lines.append(f"\n## {it['item']}\n```json\n{json.dumps(it, indent=None)}\n```\n")
        f = os.path.join(OUT, "batches", f"round{R}_{part}.md")
        open(f, "w").write("".join(lines))
        print("wrote", f, os.path.getsize(f), "bytes")


def ingest(R):
    for f in sorted(glob.glob(os.path.join(OUT, "answers", f"round{R}_*.json"))):
        A = json.load(open(f))
        for key, ens in A.items():
            n, seed = int(key.split("_")[1][1:]), int(key.split("_")[2][1:])
            st = load_state(n, seed)
            if any(r["round"] == R for r in st["rounds"]):
                continue
            try:
                check(ens); ev = evaluate(ens, n, seed); err = None
            except Exception as e:  # noqa: BLE001
                ens, ev, err = dict(bias=0.0, trees=[], note=f"invalid: {e}"), evaluate(dict(bias=0.0, trees=[]), n, seed), str(e)
            st["rounds"].append(dict(round=R, ensemble=ens, error=err, survivors=diag_survivors(ens, n, seed), **ev))
            os.makedirs(os.path.join(OUT, "state"), exist_ok=True)
            json.dump(st, open(state_path(n, seed), "w"), indent=1)
            print(key, R, "val logloss trees %.3f boost %.4f | test boost R95 %.0f (trees alone %.0f)" % (
                ev["val"]["trees_logloss"], ev["val"]["boost_logloss"], ev["test"]["boost_r95"], ev["test"]["trees_r95"]), err or "", flush=True)


def final():
    out = []
    for n in NS:
        for seed in SEEDS:
            st = load_state(n, seed)
            if not st["rounds"]:
                continue
            best = min(st["rounds"], key=lambda r: r["val"]["boost_logloss"])
            out.append(dict(N=n, seed=seed, chosen_round=best["round"], rounds=len(st["rounds"]),
                            test_auc=best["test"]["boost_auc"], test_r95=best["test"]["boost_r95"],
                            trees_only_test_auc=best["test"]["trees_auc"], trees_only_test_r95=best["test"]["trees_r95"]))
    json.dump(out, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    import pandas as pd
    print(pd.DataFrame(out).groupby("N")[["test_r95", "trees_only_test_r95", "chosen_round"]].median())


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"prep": lambda: prep(int(sys.argv[2])), "ingest": lambda: ingest(int(sys.argv[2])), "final": final}[cmd]()
