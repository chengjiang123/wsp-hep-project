"""Study 1: how many evaluations does black-box search need to match the LLM hand edits, in the same weight space?

Same objective the LLM used to pick its final round: AUC on the 1000 labelled edit candidates.
Two parameter spaces:
  r0  the 8 written cuts: threshold (layer units, +-3 around written, i.e. +-1.5 decades), sharpness, output weight
  r7  the LLM's final structure (13 cut units + 4 free units): cut threshold/sharpness/weight, free-unit bias/weight
      (gives the optimizer the structure the LLM invented, so it isolates tuning from design)
Optimizers: random search, Optuna TPE, CMA-ES. 1000 evaluations, 3 seeds.
For every new best on the edit set, the test set is scored (reporting only, never used for selection).

usage: python search.py {r0,r7} {random,tpe,cmaes} seed
"""
import json, sys, numpy as np, torch, optuna
from sklearn.metrics import roc_auc_score, roc_curve
import models_v2 as V
from improve import load, sample
from handedit import build, layer_x

torch.set_num_threads(1)
optuna.logging.set_verbosity(optuna.logging.WARNING)
BUDGET = int(__import__("os").environ.get("BUDGET", 1000))
D, tr, te = load()
import handedit as _H                       # same labelled set and stop split as the hand edits (NEDIT, STOPFRAC)
_, _, idx, ye, sidx, ys = _H.labels()
Xe = V.phys(D["F"][idx], "log"); Xs = V.phys(D["F"][sidx], "log") if len(sidx) else None
Xt = V.phys(D["F"][te], "log"); yt = D["y"][te].astype(float)


def rej(y, p, eff):
    fpr, tpr, _ = roc_curve(y, p)
    return 1 / max(np.interp(eff, tpr, fpr), 1e-6)


def params_of(spec, neutral=False):
    """Box for each tunable number: (name, low, high, start)."""
    P = []
    for j, u in enumerate(spec["units"]):
        if "cut" in u:
            t0 = layer_x(u["f"], [1.0, 0.97, 0.06, 0.06, 0.5, 20.0, 0.2, 0.005][u["f"]]) if u["f"] != 4 or u["t"] < 10 else layer_x(4, 40.0)
            if neutral:     # r7 space: start from written thresholds, k 2, w 1 (not from the LLM's values)
                P += [(f"u{j}_t", t0 - 3, t0 + 3, t0), (f"u{j}_k", 0.3, 6.0, 2.0), (f"u{j}_w", 0.0, 4.0, 1.0)]
            else:
                P += [(f"u{j}_t", t0 - 3, t0 + 3, layer_x(u["f"], u["t"])), (f"u{j}_k", 0.3, 6.0, u["k"]), (f"u{j}_w", 0.0, 4.0, u["w"])]
        else:
            P += [(f"u{j}_b", u["b"] - 3, u["b"] + 3, u["b"]), (f"u{j}_w", 0.0, 4.0, 0.5 if neutral else u["w"])]
    return P


def spec_from(base, vals):
    s = json.loads(json.dumps(base))
    for j, u in enumerate(s["units"]):
        if "cut" in u:
            tl = vals[f"u{j}_t"]
            u["t"] = float(V.itf(u["f"], V.LOG_C[u["f"]] + tl * V.LOG_S[u["f"]]))
            u["k"] = vals[f"u{j}_k"]; u["w"] = vals[f"u{j}_w"]
        else:
            u["b"] = vals[f"u{j}_b"]; u["w"] = vals[f"u{j}_w"]
    return s


def main(space, opt, seed):
    base = json.load(open(f"hand/{space}.json"))
    P = params_of(base, neutral=(space == "r7"))
    hist, best, bstop = [], -1, -1
    def f(vals):
        nonlocal best, bstop
        s = spec_from(base, vals)
        p = V.predict(build(s), Xe)
        a = roc_auc_score(ye, p)
        row = dict(i=len(hist), edit_auc=a)
        if Xs is not None:                  # held-back stop set: recorded for model selection, never optimised
            row["stop_auc"] = roc_auc_score(ys, V.predict(build(s), Xs))
        better_stop = "stop_auc" in row and row["stop_auc"] > bstop
        if a > best or better_stop:
            best = max(best, a); bstop = max(bstop, row.get("stop_auc", -1))
            pt = V.predict(build(s), Xt)
            row.update(test_auc=roc_auc_score(yt, pt), test_rej=rej(yt, pt, .9), test_rej95=rej(yt, pt, .95))
        hist.append(row)
        return a
    start = {n: s0 for n, lo, hi, s0 in P}
    rng = np.random.default_rng(seed)
    if opt == "random":
        f(start)
        for _ in range(BUDGET - 1):
            f({n: rng.uniform(lo, hi) for n, lo, hi, s0 in P})
    elif opt == "tpe":
        st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        st.enqueue_trial(start)
        st.optimize(lambda t: f({n: t.suggest_float(n, lo, hi) for n, lo, hi, s0 in P}), n_trials=BUDGET)
    else:
        import cma
        lo = np.array([p[1] for p in P]); hi = np.array([p[2] for p in P]); x0 = np.array([p[3] for p in P])
        x0 = np.clip(x0, lo, hi)
        es = cma.CMAEvolutionStrategy(((x0 - lo) / (hi - lo)).tolist(), 0.15,
                                      dict(bounds=[0, 1], seed=seed + 1, verbose=-9, popsize=16))
        while len(hist) < BUDGET:
            xs = es.ask()
            fs = [-f({n: float(lo[i] + x[i] * (hi[i] - lo[i])) for i, (n, *_r) in enumerate(P)}) for x in xs]
            es.tell(xs, fs)
        del hist[BUDGET:]
    json.dump(dict(space=space, opt=opt, seed=seed, nparams=len(P), hist=hist), open(f"search/{space}_{opt}_{seed}{__import__('os').environ.get('TAG', '')}.json", "w"))
    b = max(range(len(hist)), key=lambda i: hist[i]["edit_auc"])
    last = [h for h in hist[:b + 1] if "test_auc" in h][-1]
    print(space, opt, seed, "params", len(P), "best edit AUC", round(hist[b]["edit_auc"], 5), "at eval", b,
          "test", round(last["test_auc"], 5), round(last["test_rej"]), round(last["test_rej95"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
