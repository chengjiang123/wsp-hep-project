"""Story 3, part A: recovering physics laws. Our method (an LLM reads the trained law-shaped network's weights)
against PySR, Operon and LLM-SR, all on the same seeded subsamples and scored by one rule.

Layout (under $S3_ROOT, default $SCRATCH/wsp_bench/story3):
  sets/<set>/data/<mode>_<law>.npz, sets/<set>/truth/<mode>.json     made by gen.py (set = real | b101 .. b105)
  sets/<set>/nets/<law>_n<N>_s<seed>.json                            law-shaped networks (what the reader sees)
  results/<method>/<set>__<law>__n<N>__s<seed>.json                  one result per method and instance
Scoring rule (the same for every method): recovered when the method's formula, evaluated OUTSIDE the training range,
meets the family threshold of the paper: kin normalised RMS <= 0.02, cons 1 - AUC <= 0.001, decay mean|P - P_true| <= 0.01.
"""
import glob
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("S3_ROOT", "/path/to/scratch/wsp_bench/story3")
SETS = ["real", "b101", "b102", "b103", "b104", "b105"]
NS = [100, 300, 1000, 3000, 10000]
THR = {"kin": 0.02, "cons": 0.001, "decay": 0.01}
sys.path.insert(0, HERE)


def mode_of(s):
    return "real" if s == "real" else "blind"


def laws_in(s):
    m = mode_of(s)
    return sorted(os.path.basename(f)[len(m) + 1:-4] for f in glob.glob(f"{ROOT}/sets/{s}/data/{m}_*.npz"))


def load(s, law):
    return np.load(f"{ROOT}/sets/{s}/data/{mode_of(s)}_{law}.npz")


def truth(s, law):
    return json.load(open(f"{ROOT}/sets/{s}/truth/{mode_of(s)}.json"))["laws"][law]


def rows(N, seed, pool=20000):
    return np.random.default_rng(7000 + seed).permutation(pool)[:N]


def metric(fam, pred, d, tag):
    from sklearn.metrics import roc_auc_score
    pred = np.nan_to_num(np.asarray(pred, float), nan=0.0, posinf=1e6, neginf=-1e6)
    y = d["y" + tag]
    if fam == "kin":
        return float(np.sqrt(np.mean((pred - y) ** 2) / np.mean(y ** 2)))
    if fam == "cons":
        try:
            return float(1 - roc_auc_score(y, pred))
        except ValueError:
            return 1.0
    return float(np.mean(np.abs(pred - d["P" + tag])))


def score(fam, f, d):
    """f: callable X -> prediction. Returns in-range and out-of-range metrics and the verdict."""
    r = {}
    for t in ("id", "ood"):
        try:
            r[t] = metric(fam, f(d["X" + t]), d, t)
        except Exception as e:  # a formula that cannot be evaluated is not recovered
            r[t], r[t + "_error"] = float("inf"), str(e)[:200]
    r["recovered"] = bool(r["ood"] <= THR[fam])
    return r


def save(method, s, law, N, seed, obj):
    p = f"{ROOT}/results/{method}/{s}__{law}__n{N}__s{seed}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p + ".tmp", "w"), indent=1, default=float)
    os.replace(p + ".tmp", p)


def done(method, s, law, N, seed):
    return os.path.exists(f"{ROOT}/results/{method}/{s}__{law}__n{N}__s{seed}.json")


# ---------------------------------------------------------------------------------- reader answers -> predictions
QN = 5


def answer_predictor(fam, a):
    """Turn the reader's structured answer into a function of the inputs (to score it like any formula)."""
    num = _num
    if fam == "kin":
        S = [int(num(i)) for i in a["S"]]
        s = np.array([num(x) for x in a["s"]])

        def f(X):
            V = X.reshape(len(X), 3, 4)
            tot = V[:, S, :].sum(1)
            return (tot ** 2 * s).sum(1)
        return f
    if fam == "cons":
        U = np.array([[num(x) for x in r] for r in a["U"]], float)

        def f(X):
            q = X.reshape(len(X), 4, QN)
            dq = q[:, 0] + q[:, 1] - q[:, 2] - q[:, 3]
            return -((dq @ U.T) ** 2).sum(1)
        return f
    e = [num(x) for x in a["exp"]]
    lam = num(a["lam"])

    def f(X):
        return -np.expm1(-np.exp(X @ np.array(e)) / lam)
    return f


def _num(x):
    from fractions import Fraction
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    m = re.match(r"^e\^\(?\s*([-+0-9.eE]+)\)?", s)
    if m:
        return float(np.exp(float(m.group(1))))
    m = re.match(r"^[-+]?\d+\s*/\s*\d+", s)
    if m:
        return float(Fraction(m.group(0).replace(" ", "")))
    m = re.match(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s)
    if m:
        return float(m.group(0))
    raise ValueError(f"not a number: {x!r}")


def symbolic_match(fam, a, t):
    """The paper's exact-reading rule (score.py)."""
    from scipy.linalg import subspace_angles
    try:
        if fam == "kin":
            S = sorted(int(_num(i)) for i in a["S"])
            s = np.array([_num(x) for x in a["s"]])
            s = s / max(abs(s).max(), 1e-12)
            return bool(S == sorted(t["S"]) and np.allclose(s, t["s"], atol=0.05))
        if fam == "cons":
            Ut, Ua = np.array(t["U"], float), np.array([[_num(x) for x in r] for r in a["U"]], float)
            if np.linalg.matrix_rank(Ua) != len(Ut):
                return False
            return bool(np.degrees(subspace_angles(Ut.T, Ua.T)).max() < 5)
        e, lam = [_num(x) for x in a["exp"]], _num(a["lam"])
        return bool(np.allclose(e, t["exp"], atol=0.05) and abs(lam / t["lam"] - 1) < 0.1)
    except Exception:
        return False


# ------------------------------------------------------------------------------- problem text for LLM-SR / Operon
def symbols(fam, feat):
    if fam == "kin":
        desc = {"E": "energy", "px": "x-momentum", "py": "y-momentum", "pz": "z-momentum"}
        d = [f"{desc[f[:-1]]} of particle {f[-1]} in GeV" for f in feat]
        out = "an unknown scalar function of the four-vectors of three particles, in GeV^2"
    elif fam == "cons":
        qd = {"Q": "electric charge", "B": "baryon number", "Le": "electron lepton number", "Lmu": "muon lepton number", "S": "strangeness"}
        slot = {"a": "incoming particle a", "b": "incoming particle b", "c": "outgoing particle c", "d": "outgoing particle d"}
        d = [f"{qd[f.split('_')[0]]} of {slot[f.split('_')[1]]}" for f in feat]
        out = "1 if the reaction a + b -> c + d is allowed, 0 if it is forbidden"
    else:
        d = ["natural log of the flight distance L in cm", "natural log of the momentum p in GeV", "natural log of the mass m in GeV"]
        out = "1 if the particle decayed before flying the distance L, 0 if not"
    names = [re.sub(r"[^A-Za-z0-9_]", "_", f) for f in feat]
    return names, d, out
