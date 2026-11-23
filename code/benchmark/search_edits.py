"""Black-box search on the LLM's own edit space of a pretrained tagger, at a matched evaluation budget (FT -> edit arm).

Same start (the cell's plain-FT network, results_llm-opus55sub/.../ft_state.pt), same ops (common.apply_patch), same
selection criterion (validation loss on the validation part of the label budget) and same test set as the LLM editor,
which gets 4 evaluations. The search vector (31 numbers) spans every op the LLM may use:
  8  pair_scale log-factors u_h in [-2.3, 2.3]           (factor e^u, 1 = untouched)
  1  head_mix alpha in [0, 1]                            (0 = keep the fine-tuned head)
  10 signal coefficients on the JetClass rows in [-1, 1]
  10 background coefficients on the JetClass rows in [-1, 1]
  1  head_scale log-factor in [-1.6, 1.6]
  1  head_bias delta in [-3, 3]
The incumbent "no edit" is not counted (the LLM's rounds do not include it either). Test metrics of the best-so-far
candidate (by validation loss) are recorded at budgets 1, 2, 4, 8, ..., MAX.

usage: python search_edits.py --cells cells_search.txt --line K --method cmaes|tpe|random [--budget 256] [--seed 0]
output: $WSP_ROOT/search_edits/<ds>/<model>/N<N>/s<seed>/<method>_r<seed>.json
"""
import argparse
import copy
import json
import math
import os
import time

import numpy as np

import common as C

LO = np.array([-2.3] * 8 + [0.0] + [-1.0] * 20 + [-1.6, -3.0])
HI = np.array([2.3] * 8 + [1.0] + [1.0] * 20 + [1.6, 3.0])
X0 = np.array([0.0] * 8 + [0.0] + [0.0] * 20 + [0.0, 0.0])          # = no edit


def patch_of(x):
    x = np.clip(np.asarray(x, float), LO, HI)
    p = [{"op": "pair_scale", "head": h, "factor": float(math.exp(x[h]))} for h in range(8) if abs(x[h]) > 1e-4]
    if x[8] > 1e-4:
        p.append({"op": "head_mix", "alpha": float(x[8]),
                  "signal": {r: float(c) for r, c in zip(C.JETCLASS, x[9:19])},
                  "background": {r: float(c) for r, c in zip(C.JETCLASS, x[19:29])}})
    if abs(x[29]) > 1e-4:
        p.append({"op": "head_scale", "factor": float(math.exp(x[29]))})
    if abs(x[30]) > 1e-4:
        p.append({"op": "head_bias", "delta": float(x[30])})
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--line", type=int, required=True)
    ap.add_argument("--method", required=True, choices=["cmaes", "tpe", "random"])
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="_llm-opus55sub")
    a = ap.parse_args()
    line = [l.split() for l in open(a.cells) if l.split()][a.line - 1]
    ds, kind, N, cseed = line[0], line[1], line[2], int(line[3])
    out = f"{C.ROOT}/search_edits/{ds}/{kind}/N{N}/s{cseed}"
    os.makedirs(out, exist_ok=True)
    fo = f"{out}/{a.method}_r{a.seed}.json"
    if os.path.exists(fo):
        print("exists", fo); return
    t0 = time.time()
    (trs, tri), (vas, vai) = C.splits_for(ds, N, cseed)
    va = C.Split(ds, vas, vai)
    te = C.Split(ds, "test", None)
    import torch
    base = C.build(kind, ds)
    st = torch.load(f"{C.ROOT}/results{a.tag}/{ds}/{kind}/N{N}/s{cseed}/ft_state.pt")
    base.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()})
    ft_val = C.evaluate(base, va)
    print(f"{ds}/{kind}/N{N}/s{cseed} {a.method}: val {len(va)} jets, FT val loss {ft_val['loss']:.4f} ({time.time() - t0:.0f} s)", flush=True)

    hist = []                               # (val loss, x)

    def f(x):
        m = copy.deepcopy(base)
        C.apply_patch(m, patch_of(x))
        v = C.evaluate(m, va)
        loss = v["loss"] if np.isfinite(v["loss"]) else 1e9
        hist.append((loss, np.clip(np.asarray(x, float), LO, HI).tolist(), v))
        return loss

    rng = np.random.default_rng(a.seed)
    if a.method == "random":
        for _ in range(a.budget):
            f(rng.uniform(LO, HI))
    elif a.method == "cmaes":
        import cma
        es = cma.CMAEvolutionStrategy(X0, 0.3, {"bounds": [LO.tolist(), HI.tolist()], "seed": a.seed + 1, "verbose": -9,
                                                "CMA_stds": ((HI - LO) / 4).tolist()})
        while len(hist) < a.budget:
            xs = es.ask()
            xs = xs[: a.budget - len(hist)]
            es.tell(xs, [f(x) for x in xs]) if len(xs) == es.popsize else [f(x) for x in xs]
    else:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        s = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=a.seed))
        s.optimize(lambda t: f(np.array([t.suggest_float(f"x{i}", LO[i], HI[i]) for i in range(len(LO))])), n_trials=a.budget)

    budgets = [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if b <= a.budget]
    recs, cache = [], {}
    for b in budgets:
        i = int(np.argmin([h[0] for h in hist[:b]]))
        if i not in cache:
            m = copy.deepcopy(base)
            C.apply_patch(m, patch_of(hist[i][1]))
            cache[i] = C.evaluate(m, te)
        recs.append(dict(budget=b, best_eval=i + 1, val=hist[i][2], test=cache[i], patch=patch_of(hist[i][1])))
        print(f"  budget {b:4d}: best eval {i + 1:4d} val loss {hist[i][0]:.4f}  test r30 {cache[i]['r30']:.1f} r50 {cache[i]['r50']:.1f}", flush=True)
    C.save_json(dict(ds=ds, model=kind, N=N, seed=cseed, method=a.method, search_seed=a.seed, n_val=len(va), ft_val=ft_val,
                     val_loss_history=[h[0] for h in hist], budgets=recs, seconds=time.time() - t0), fo)
    print("wrote", fo, f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
