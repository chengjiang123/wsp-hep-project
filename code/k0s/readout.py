"""Read the tuned weights back as topological cuts, and store what the plot needs."""
import json, numpy as np, torch
from sklearn.metrics import roc_auc_score, roc_curve
import models as M
from run_scan import load, sample, plain_cuts

torch.set_num_threads(2)
U = M.programmed_units()


def read_units(net):
    W = net.l1.weight.detach().numpy(); b = net.l1.bias.detach().numpy(); w2 = net.l2.weight.detach().numpy()[0]
    rows = []
    for j in range(len(b)):
        f = int(np.argmax(np.abs(W[j])))
        t = float(M.FC[f] - b[j] / W[j, f] * M.FS[f])
        r = dict(unit=j, feature=M.FEATS[f], purity=round(float(np.abs(W[j, f]) / np.linalg.norm(W[j])), 3),
                 threshold=round(t, 5), sharpness=round(float(abs(W[j, f]) / M.FS[f]), 3),
                 direction=">" if W[j, f] > 0 else "<", out_weight=round(float(w2[j]), 3))
        if j < len(U) and f == U[j][1]:
            r["shift_in_dt"] = round((t - U[j][2]) / U[j][5], 2)
        rows.append(r)
    return rows


def cut_at_eff(y, p, eff=0.9):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.searchsorted(tpr, eff)]


D, tr, te, ratio = load()
yte = D["y"][te]; Fte = M.std_feats(D["F"][te])
res = {"programmed_N0": read_units(M.FeatNet("prog"))}
scores = {"written": M.predict(M.FeatNet("prog"), Fte)}
for n in (1000, 30000):
    idx, y = sample(tr, n, 7 + n)
    F = M.std_feats(D["F"][idx])
    for tag in ("0.3", "syst"):
        net = M.FeatNet("prog"); mu = M.flat(net)
        sig = M.syst_sigma(net) if tag == "syst" else 0.3
        net = M.train_map(net, F, y, prior_mean=mu, prior_sigma=sig)
        res[f"programmed_sigma{tag}_N{n}"] = read_units(net)
        p = M.predict(net, Fte); res[f"auc_programmed_sigma{tag}_N{n}"] = roc_auc_score(yte, p)
        if n == 30000 and tag == "0.3":
            scores["tuned"] = p
    rnd = M.train_map(M.FeatNet("random"), F, y)
    res[f"random_N{n}"] = read_units(rnd)
    res[f"auc_random_N{n}"] = roc_auc_score(yte, M.predict(rnd, Fte))

res["single_feature_auc"] = {nm: round(max(a, 1 - a), 3) for nm, a in
                             ((nm, roc_auc_score(yte, D["F"][te][:, i])) for i, nm in enumerate(M.FEATS))}
# invariant-mass spectra on the test set at 90% signal efficiency, background reweighted to the natural mix
m = D["m"][te]; w = np.where(yte, 1.0, ratio)
mass = {"all": dict(m=m.tolist(), w=w.tolist(), y=yte.astype(int).tolist(), cat=D["cat"][te].tolist())}
for k, p in scores.items():
    mass[k] = (p >= cut_at_eff(yte, p)).astype(int).tolist()
mass["plain_cuts"] = plain_cuts(D["F"][te]).astype(int).tolist()
res["mass"] = mass
res["natural_bkg_per_sig"] = ratio
res["hist"] = {nm: dict(sig=D["F"][te][yte, i][:5000].tolist(), bkg=D["F"][te][~yte, i][:5000].tolist())
               for i, nm in enumerate(M.FEATS)}
json.dump(res, open("readout.json", "w"))
for k, v in res.items():
    if k in ("hist", "mass"):
        continue
    print(k)
    if isinstance(v, list):
        for r in v: print("   ", r)
    else:
        print("   ", v)
