"""Direction 2: map the sensitivity of the written K0S net to physical effects.

For each effect, the written net (log layer, soft + smooth, fixed prior + syst widths) is tuned on 30k candidates
from a sample with only that effect switched on. Its readout (threshold shifts in units of dt, output weights
of cut and smooth units) minus the nominal readout is the effect's fingerprint. Seed spread on nominal = noise.

Inverse test: a mystery sample (two effects at 60% of their single-effect size) is decoded by non-negative
least squares on the fingerprints, in units of the noise.
"""
import json, numpy as np, torch
from scipy.optimize import nnls
import models as M, models_v2 as V
from improve import load, sample

torch.set_num_threads(1)
N = 30000
EFFECTS = [  # (key, label, file, weight function on (D, idx) -> (wS, wB) or None)
    ("ip", "impact-parameter resolution x1.5", "ip15.npz", None),
    ("mom", "momentum + angle resolution x2", "mom2.npz", None),
    ("disp", "worse resolution for displaced tracks", "disp4.npz", None),
    ("mult", "multiplicity x2 (10 overlaid events)", "nov10.npz", None),
    ("lam", "Lambda yield x2", "nom.npz", "lam"),
    ("pt", "harder K0S pT spectrum", "nom.npz", "pt"),
]
MYST = ("myst.npz", "impact-parameter x1.3 and multiplicity x1.6", {"ip": 0.6, "mult": 0.6})


def weights(D, tr, kind):
    if kind is None:
        return None, None
    if kind == "lam":
        return None, np.where(D["cat"][tr[1]] == 1, 2.0, 1.0)
    # harder spectrum: w = exp(pT / 1 GeV), mean pT rises by about 0.3 GeV
    return np.exp(D["pt"][tr[0]] / 1.0), None


def readout_vec(net):
    rows, smooth = V.read_units(net)
    return np.array([r["shift_dt"] for r in rows] + [r["out_weight"] for r in rows] + list(smooth.values()))


def labels():
    names = [u[0] for u in M.programmed_units()]
    return ([f"cut: {n}" for n in names] + [f"weight: {n}" for n in names] + [f"smooth: {f}" for f in M.FEATS])


def tune(f, kind, seed):
    D, tr, te = load(f, 2000, 2000)
    wS, wB = weights(D, tr, kind)
    idx, y = sample(tr, N, 77 + seed, wS, wB)
    net = V.Net("written", space="log", seed=seed); mu, sig = net.prior(syst=True)
    net = V.train(net, V.phys(D["F"][idx], "log"), y, mu=mu, sig=sig, mode="fixed", N0=1000)
    return readout_vec(net)


if __name__ == "__main__":
    nom = np.array([tune("nom.npz", None, s) for s in range(6)])
    res = dict(labels=labels(), nominal=nom.tolist(), effects={})
    for key, lab, f, kind in EFFECTS:
        v = np.array([tune(f, kind, s) for s in range(3)])
        res["effects"][key] = dict(label=lab, runs=v.tolist())
        print(key, np.round(v.mean(0) - nom.mean(0), 2), flush=True)
    my = np.array([tune(MYST[0], None, s) for s in range(3)])
    res["mystery"] = dict(label=MYST[1], truth=MYST[2], runs=my.tolist())
    # decode: whiten by the per-component noise of a 3-run mean
    noise = nom.std(0, ddof=1) * np.sqrt(1 / 3 + 1 / len(nom)) + 1e-3
    A = np.array([(np.mean(res["effects"][k]["runs"], 0) - nom.mean(0)) / noise for k, *_ in EFFECTS]).T
    b = (my.mean(0) - nom.mean(0)) / noise
    c, rnorm = nnls(A, b)
    # bootstrap over runs for an error on the coefficients
    rng = np.random.default_rng(0); boots = []
    for _ in range(200):
        nb = nom[rng.integers(len(nom), size=len(nom))].mean(0)
        Ab = np.array([(np.array(res["effects"][k]["runs"])[rng.integers(3, size=3)].mean(0) - nb) / noise
                       for k, *_ in EFFECTS]).T
        boots.append(nnls(Ab, (my[rng.integers(3, size=3)].mean(0) - nb) / noise)[0])
    res["decode"] = dict(keys=[k for k, *_ in EFFECTS], coef=c.tolist(), err=np.std(boots, 0).tolist(),
                         resid=float(rnorm), norm_b=float(np.linalg.norm(b)), noise=noise.tolist())
    print("decode", dict(zip(res["decode"]["keys"], np.round(c, 2))), "err", np.round(np.std(boots, 0), 2))
    json.dump(res, open("fingerprint.json", "w"), indent=1)
