"""Ablation 2.3: a scripted editor, no LLM. It sees the same edit set and the same kind of diagnostics as the LLM loop
(background survivors and K0S at the 95% working point, survivor categories) and makes one change per round by a
fixed rule. Same start (hand/r0.json), same budget, same held-back stop rule and patience as llm_loop.py.

Rule, each round:
  * survivors = background passing the 95% K0S working point on the edit set
  * if more than half of the survivors are true Lambda decays: strengthen the Lambda mass veto (w + 1, or add it)
  * otherwise, for every feature, the one-sided cut that loses 2% of the K0S (on the side away from the survivor
    median) is computed, and the feature whose cut removes the most survivors is chosen: an existing unit on that
    feature and side is moved to the new threshold with w + 0.5, else a new unit (k = 3, w = 1) is added
  * a change that would leave the spec as it is is skipped and the next best feature is taken
usage (same env as the loop: NEDIT, STOPFRAC, DATA, NTEST_S, NTEST_B):
  python scripted_edit.py --tag scripted_n10000 [--rounds 25] [--patience 5] [--lose 0.02]
  then: HE_OUT=llm_runs/<tag>/test.json python handedit.py test $(ls llm_runs/<tag>/r*.json | sort -V)"""
import argparse, copy, json, os, numpy as np
import models_v2 as V
import handedit as H

NAMES = ["dca_dau", "cos_pa", "dca_pos", "dca_neg", "radius", "mL/p", "arm", "dmLambda"]


def layer(F):
    return V.phys(F, "log").numpy() if hasattr(V.phys(F[:2], "log"), "numpy") else np.asarray(V.phys(F, "log"))


def propose(spec, F, y, cat, p, lose):
    s, b = y == 1, y == 0
    thr = np.quantile(p[s], 0.05); surv = b & (p >= thr)
    if surv.sum() == 0:
        return None, "no survivors"
    lam = np.mean(cat[surv] == 1)
    cands = []
    if lam > 0.5:
        cands.append(("lambda", 7, None, ">", f"{lam:.0%} of the {surv.sum()} survivors are Lambda: strengthen the Lambda veto"))
    for f in range(8):
        x = F[:, f] if f != 1 else 1 - F[:, 1]                   # work with 1 - cos so that '<' means tighter pointing
        side = "<" if np.median(x[surv]) > np.median(x[s]) else ">"
        t = np.quantile(x[s], 1 - lose) if side == "<" else np.quantile(x[s], lose)
        removed = np.mean(x[surv] > t) if side == "<" else np.mean(x[surv] < t)
        if f == 1:                                            # back to cos(pointing angle)
            t, side = 1 - t, (">" if side == "<" else "<")
        cands.append(("cut", f, float(t), side, f"{NAMES[f]} {side} {t:.4g} removes {removed:.0%} of {surv.sum()} survivors", removed))
    cuts = sorted([c for c in cands if c[0] == "cut"], key=lambda c: -c[5])
    return ([c for c in cands if c[0] == "lambda"] + cuts), None


def apply(spec, c):
    new = copy.deepcopy(spec); U = new["units"]
    kind, f, t, side, why = c[:5]
    if kind == "lambda":
        for u in U:
            if u["f"] == 7 and u["dir"] == ">":
                u["w"] = round(u["w"] + 1.0, 3); break
        else:
            U.append(dict(cut="dmLambda > 0.005", f=7, t=0.005, dir=">", k=2.0, w=1.0))
    else:
        if f != 6 and t <= 0:
            return None
        for u in U:
            if u["f"] == f and u["dir"] == side:
                if abs(np.log(max(t, 1e-9)) - np.log(max(u["t"], 1e-9))) < 0.01 and f != 6:
                    return None                               # nothing would change
                u["t"] = t; u["w"] = round(u["w"] + 0.5, 3); u["cut"] = f"{NAMES[f]} {side} {t:.4g}"; break
        else:
            if len(U) >= 20:
                return None
            U.append(dict(cut=f"{NAMES[f]} {side} {t:.4g}", f=f, t=t, dir=side, k=3.0, w=1.0)); new["bias"] -= 1.0
    new["note"] = why
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--start", default="hand/r0.json")
    ap.add_argument("--rounds", type=int, default=25); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lose", type=float, default=0.02, help="K0S share a new cut may lose on its own")
    a = ap.parse_args()
    out = f"llm_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    D, te, idx, y, _, _ = H.labels()
    F = D["F"][idx]; X = V.phys(F, "log"); cat = D["cat"][idx]
    spec = json.load(open(a.start)); json.dump(spec, open(f"{out}/r0.json", "w"), indent=1)
    stops = [H.stop_auc(spec)]; hist = [(0, None, spec.get("note", "start"))]
    for rnd in range(1, a.rounds + 1):
        p = V.predict(H.build(spec), X)
        cands, msg = propose(spec, F, y, cat, p, a.lose)
        if cands is None:
            print("round", rnd, msg); break
        for c in cands:
            new = apply(spec, c)
            if new is not None:
                break
        else:
            print("round", rnd, "no change possible"); break
        spec = new; json.dump(spec, open(f"{out}/r{rnd}.json", "w"), indent=1)
        stops.append(H.stop_auc(spec)); hist.append((rnd, None, spec["note"]))
        print(f"round {rnd}: stop AUC {stops[-1]:.4f}  {spec['note'][:90]}", flush=True)
        if rnd - int(np.argmax(stops)) >= a.patience:
            print("stop-set AUC has not improved for", a.patience, "rounds"); break
    chosen = int(np.argmax(stops))
    json.dump(dict(history=hist, stop_auc=stops, chosen=chosen, rule="best held-back stop-set AUC"), open(f"{out}/history.json", "w"), indent=1)
    print("chosen round:", chosen)


if __name__ == "__main__":
    main()
