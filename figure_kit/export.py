"""Turn the raw result JSONs into one tidy CSV per figure (one row per measurement), plus small summary tables.
Run after every refresh; the plot scripts read only these CSVs.

  python export.py [--sync ../sync]

Writes
  fig_laws/data.csv              method, set, law, family, N, seed, recovered, exact, err_in_range, err_out_of_range, threshold, network_pass
  fig_laws/summary.csv           method, N, laws, recovered_fraction (mean over seeds), sd_over_seeds
  fig_nscan/data.csv             task, model, N, seed, arm, test_auc, test_r30, test_r50, n_train, steps_run, auc_after_edit
  fig_nscan/summary.csv          task, model, N, arm, seeds, auc_mean, auc_se, d_auc_vs_ft_mean, d_auc_vs_ft_se
  fig_structure/pair_factors.csv task, model, N, seed, head, factor        (product of the LLM's pair_scale ops, A3 patch)
  fig_structure/displacement.csv task, model, N, seed, arm, tensor, layer_group, rel_change
  fig_structure/edit_alone.csv   task, model, N, seed, what, test_auc     (after the edit / LR head, before any training)
  fig_editors/data.csv           task, model, N, seed, editor, arm, test_auc/r30/r50, ft_*, script_*, best_round (Story 1 by LLM)
  fig_edit_search/data.csv       method, space, N, seed, evaluation, test_auc, test_r95  (K0S edits vs black-box search)
  fig_srbench/data.csv           method, dataset, seed, noise, monomial, r2_test, solved, exact   (our SRBench reruns)
  fig_srbench/summary.csv        source, method, subset, noise, seeds, solved_fraction, se_over_seeds  (ours + published)
"""
import argparse
import csv
import glob
import json
import math
import os
import re

import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
THR = {"kin": 0.02, "cons": 0.001, "decay": 0.01}
LAW_METHODS = {"reader_gptoss": "LLM reads weights (gpt-oss-120b)", "reader_sonnet55sub": "LLM reads weights (Sonnet 5.5)",
               "reader_opus55sub": "LLM reads weights (Opus 5.5)", "pysr": "PySR", "operon": "Operon",
               "network": "the trained network itself, used as the predictor"}
ARMS = ["ft", "ft_wsp", "wsp_ft", "lpft", "script_ft", "random_ft", "scratch"]


def write(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{os.path.relpath(path, HERE)}: {len(rows)} rows")


def fnum(x):
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def laws(sync):
    rows = []
    net = {}
    for f in glob.glob(f"{sync}/story3_results/network/*.json"):   # does the trained network itself pass (same rule)?
        d = json.load(open(f))
        net[(d["set"], d["law"], d["N"], d["seed"])] = int(bool(d["recovered"]))
    for m in LAW_METHODS:
        for f in glob.glob(f"{sync}/story3_results/{m}/*.json"):
            d = json.load(open(f))
            rows.append(dict(method=m, set=d["set"], law=d["law"], family=d["fam"], N=d["N"], seed=d["seed"],
                             recovered=int(bool(d["recovered"])), exact=int(bool(d.get("exact"))) if "exact" in d else "",
                             err_in_range=fnum(d.get("id")), err_out_of_range=fnum(d.get("ood")), threshold=THR[d["fam"]],
                             network_pass=net.get((d["set"], d["law"], d["N"], d["seed"]), "")))
    rows.sort(key=lambda r: (r["method"], r["N"], r["seed"], r["set"], r["law"]))
    write(f"{HERE}/fig_laws/data.csv", rows, list(rows[0]) if rows else ["method"])
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[(r["method"], r["N"])][r["seed"]].append(r["recovered"])
    summ = []
    for (m, N), seeds in sorted(agg.items()):
        fr = [sum(v) / len(v) for v in seeds.values()]
        mu = sum(fr) / len(fr)
        sd = (sum((x - mu) ** 2 for x in fr) / (len(fr) - 1)) ** 0.5 if len(fr) > 1 else ""
        summ.append(dict(method=m, N=N, laws=sum(len(v) for v in seeds.values()), seeds=len(fr),
                         recovered_fraction=round(mu, 4), sd_over_seeds=round(sd, 4) if sd != "" else ""))
    write(f"{HERE}/fig_laws/summary.csv", summ, ["method", "N", "laws", "seeds", "recovered_fraction", "sd_over_seeds"])


def group(model, k):
    if model == "part":
        if k.startswith("head."):
            return "head"
        if k.startswith(("mod.pair_embed", "pair_bn")):
            return "pair emb."
        if k.startswith("mod.embed"):
            return "particle emb."
        m = re.match(r"mod\.blocks\.(\d+)\.", k)
        if m:
            return f"block {int(m.group(1)) + 1}"
        return "class attn." if k.startswith("mod.cls_blocks") else ""
    if k.startswith("net.classifier.out"):
        return "head"
    if k.startswith("net.body.interaction"):
        return "pair emb."
    if k.startswith(("net.body.embed", "net.body.local_physics")):
        return "particle emb."
    m = re.match(r"net\.body\.in_blocks\.(\d+)\.", k)
    if m:
        return f"block {int(m.group(1)) + 1}"
    return "class attn." if k.startswith("net.classifier") else ""


def nscan(sync):
    rows, pf, disp, alone = [], [], [], []
    for f in glob.glob(f"{sync}/nscan/*/*/N*/s*/*.json"):
        arm = os.path.basename(f)[:-5]
        if arm not in ARMS:
            continue
        d = json.load(open(f))
        key = dict(task=d["ds"], model=d["model"], N=str(d["N"]), seed=d["seed"])
        t, tr = d.get("test", {}), d.get("train") or {}
        after = d.get("test_after_edit") or d.get("test_after_head") or {}
        rows.append(dict(**key, arm=arm, test_auc=t.get("auc"), test_r30=fnum(t.get("r30")), test_r50=fnum(t.get("r50")),
                         n_train=d.get("n_train"), steps_run=tr.get("steps_run", ""), auc_after_edit=after.get("auc", "")))
        for k, v in (tr.get("displacement") or {}).items():
            g = group(d["model"], k)
            if g:
                disp.append(dict(**key, arm=arm, tensor=k, layer_group=g, rel_change=v))
        if after:
            alone.append(dict(**key, what={"wsp_ft": "LLM edit", "script_ft": "script edit", "lpft": "LR head",
                                           "random_ft": "random patch"}.get(arm, arm), test_auc=after["auc"]))
        if arm == "wsp_ft":
            fac = {h: 1.0 for h in range(8)}
            for op in (d.get("llm") or {}).get("best_patch") or []:
                if op.get("op") == "pair_scale":
                    fac[int(op["head"])] *= float(op["factor"])
            pf += [dict(**key, head=h + 1, factor=fac[h]) for h in range(8)]
    order = lambda r: (r["task"], r["model"], 1e9 if r["N"] == "full" else int(r["N"]), r["seed"])
    for lst in (rows, pf, disp, alone):
        lst.sort(key=order)
    write(f"{HERE}/fig_nscan/data.csv", rows, ["task", "model", "N", "seed", "arm", "test_auc", "test_r30", "test_r50",
                                               "n_train", "steps_run", "auc_after_edit"])
    write(f"{HERE}/fig_structure/pair_factors.csv", pf, ["task", "model", "N", "seed", "head", "factor"])
    write(f"{HERE}/fig_structure/displacement.csv", disp, ["task", "model", "N", "seed", "arm", "tensor", "layer_group", "rel_change"])
    write(f"{HERE}/fig_structure/edit_alone.csv", alone, ["task", "model", "N", "seed", "what", "test_auc"])
    # summary: mean AUC per arm and paired difference to plain FT
    cell = defaultdict(dict)
    for r in rows:
        cell[(r["task"], r["model"], r["N"], r["seed"])][r["arm"]] = r["test_auc"]
    agg = defaultdict(lambda: {"auc": [], "d": []})
    for (ds, m, N, s), arms in cell.items():
        for a, v in arms.items():
            agg[(ds, m, N, a)]["auc"].append(v)
            if a != "ft" and "ft" in arms:
                agg[(ds, m, N, a)]["d"].append(v - arms["ft"])
    ms = lambda x: (sum(x) / len(x), (sum((y - sum(x) / len(x)) ** 2 for y in x) / (len(x) - 1)) ** 0.5 / len(x) ** 0.5 if len(x) > 1 else "")
    summ = []
    for (ds, m, N, a), v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], 1e9 if kv[0][2] == "full" else int(kv[0][2]), ARMS.index(kv[0][3]))):
        am, ase = ms(v["auc"])
        dm, dse = ms(v["d"]) if v["d"] else ("", "")
        summ.append(dict(task=ds, model=m, N=N, arm=a, seeds=len(v["auc"]), auc_mean=round(am, 5), auc_se=round(ase, 5) if ase != "" else "",
                         d_auc_vs_ft_mean=round(dm, 5) if dm != "" else "", d_auc_vs_ft_se=round(dse, 5) if dse != "" else ""))
    write(f"{HERE}/fig_nscan/summary.csv", summ, ["task", "model", "N", "arm", "seeds", "auc_mean", "auc_se", "d_auc_vs_ft_mean", "d_auc_vs_ft_se"])


SRB_METHODS = ["pysr", "operon2021"]   # the monomial reader/rounding runs (dropped) stay on Perlmutter


def srb_solved(d):
    """SRBench's rule (postprocessing/collate_groundtruth_results.py): error or ratio to the true model is a constant,
    and the simplified model is not missing, 0 or nan."""
    ok = any(bool(d.get(k)) for k in ("symbolic_error_is_zero", "symbolic_error_is_constant", "symbolic_fraction_is_constant"))
    sm = d.get("simplified_symbolic_model")
    return int(ok and sm is not None and str(sm) not in ("0", "nan"))


def srb_exact(side, true_model):
    """Monomial subset only: every exponent and the constant (to 0.1 %) equal the true law's (metadata formula)."""
    import sympy as sp
    e = sp.simplify(sp.sympify(true_model, locals={n: sp.Symbol(n) for n in side["names"]}))
    c, pw = sp.Integer(1), {}
    for fac in sp.Mul.make_args(e):
        if fac.is_number:
            c *= fac
        else:
            b, q = fac.as_base_exp()
            pw[str(b)] = q
    try:
        got = {n: sp.nsimplify(x, rational=True) for n, x in zip(side["names"], side["exp"])}
    except Exception:
        return 0
    same = all(got.get(n, 0) == pw.get(n, 0) for n in set(got) | set(pw))
    return int(same and abs(side["C_value"] / float(c) - 1) < 1e-3)


def srbench(sync):
    import pandas as pd
    mono_f = f"{sync}/srbench/monomials.txt"
    mono = set(open(mono_f).read().split()) if os.path.exists(mono_f) else set()
    truth_f = f"{HERE}/fig_srbench/true_models.csv"   # full precision (the verifier's .updated files round constants)
    truth = dict(csv.reader(open(truth_f))) if os.path.exists(truth_f) else {}
    rows = []
    for m in SRB_METHODS:
        for f in glob.glob(f"{sync}/srbench/results/{m}/noise*/*/*.json"):
            u = f + ".updated"
            d = json.load(open(u if os.path.exists(u) else f))
            ds = f.split("/")[-2]
            side = json.load(open(f + ".side")) if os.path.exists(f + ".side") else None
            rows.append(dict(method=m, dataset=ds, seed=int(d["random_state"]), noise=float(d["target_noise"]),
                             monomial=int(ds in mono), r2_test=fnum(d.get("r2_test")),
                             solved=srb_solved(d) if os.path.exists(u) else "",
                             exact=srb_exact(side, truth[ds]) if side and ds in truth else ""))
    rows.sort(key=lambda r: (r["method"], r["noise"], r["dataset"], r["seed"]))
    cols = ["method", "dataset", "seed", "noise", "monomial", "r2_test", "solved", "exact"]
    write(f"{HERE}/fig_srbench/data.csv", rows, cols)
    ours = pd.DataFrame(rows, columns=cols)
    ours = ours[ours.solved != ""].assign(source="ours")
    pub = pd.read_csv(f"{HERE}/fig_srbench/published.csv").assign(source="published")
    pub["monomial"] = pub.dataset.isin(mono).astype(int)
    pub = pub.rename(columns={"algorithm": "method"})
    allr = pd.concat([ours[["source", "method", "dataset", "seed", "noise", "monomial", "solved"]],
                      pub[["source", "method", "dataset", "seed", "noise", "monomial", "solved"]]])
    allr["solved"] = allr.solved.astype(float)
    summ = []
    for subset, sel in (("all", allr), ("monomial", allr[allr.monomial == 1])):
        for (src, m, nz), g in sel.groupby(["source", "method", "noise"]):
            fr = g.groupby("seed").solved.mean()          # solution rate per seed, then mean and SE over seeds
            se = fr.std(ddof=1) / len(fr) ** 0.5 if len(fr) > 1 else float("nan")
            summ.append(dict(source=src, method=m, subset=subset, noise=nz, seeds=len(fr), runs=len(g),
                             solved_fraction=round(fr.mean(), 4), se_over_seeds=round(se, 4) if se == se else ""))
    write(f"{HERE}/fig_srbench/summary.csv", summ, ["source", "method", "subset", "noise", "seeds", "runs", "solved_fraction", "se_over_seeds"])


EDITORS = {"nscan": "gpt-oss-120b", "nscan_opus55sub": "Opus 5.5", "nscan_sonnet55sub": "Sonnet 5.5"}


def editors(sync):
    """Story 1 by editing LLM: test AUC of FT->WSP and WSP->FT per editor, with the cell's plain FT and fixed-rule edit
    (both from the main scan) for pairing. Claude editors ran on N = 1000 (seeds 0-2), 100000 and full (seed 0)."""
    base = {}
    for f in glob.glob(f"{sync}/nscan/*/*/N*/s*/*.json"):
        arm = os.path.basename(f)[:-5]
        if arm in ("ft", "script_ft"):
            d = json.load(open(f))
            base[(d["ds"], d["model"], str(d["N"]), d["seed"], arm)] = d["test"]
    rows = []
    for folder, ed in EDITORS.items():
        for f in glob.glob(f"{sync}/{folder}/*/*/N*/s*/*.json"):
            arm = os.path.basename(f)[:-5]
            if arm not in ("ft_wsp", "wsp_ft"):
                continue
            d = json.load(open(f))
            k = (d["ds"], d["model"], str(d["N"]), d["seed"])
            if folder != "nscan" or os.path.exists(f"{sync}/nscan_opus55sub/{k[0]}/{k[1]}/N{k[2]}/s{k[3]}/{arm}.json"):
                ft, sc = base.get(k + ("ft",), {}), base.get(k + ("script_ft",), {})
                rows.append(dict(task=k[0], model=k[1], N=k[2], seed=k[3], editor=ed, arm=arm, test_auc=d["test"]["auc"],
                                 test_r30=fnum(d["test"].get("r30")), test_r50=fnum(d["test"].get("r50")),
                                 ft_auc=ft.get("auc", ""), ft_r30=fnum(ft.get("r30")), ft_r50=fnum(ft.get("r50")),
                                 script_auc=sc.get("auc", ""), script_r30=fnum(sc.get("r30")), script_r50=fnum(sc.get("r50")),
                                 best_round=(d.get("llm") or {}).get("best_round", "")))
    rows.sort(key=lambda r: (r["task"], r["model"], 1e9 if r["N"] == "full" else int(r["N"]), r["seed"], r["editor"], r["arm"]))
    write(f"{HERE}/fig_editors/data.csv", rows, ["task", "model", "N", "seed", "editor", "arm", "test_auc", "test_r30",
                                                 "test_r50", "ft_auc", "ft_r30", "ft_r50", "script_auc", "script_r30",
                                                 "script_r50", "best_round"])


RAW = os.environ.get("WSP_DATA", os.path.join(HERE, "..", "data"))   # K0S edit-vs-search runs (paper Figure 4)


def edit_search(raw=RAW):
    """K0S: direct LLM weight edits vs black-box search, on the scaled sample with N = 1e4 and 5e4 labels (30 % held back).
    One row per (method, space, N, seed, evaluation): test AUC / R95 of the best round so far, chosen on held-back labels.
    Methods: llm_interactive, llm_automated (Opus, 3 seeds, N = 1e4 only), cmaes, tpe, random; space r0 = the 24 weights of
    the 8 encoded cuts, r7 = the 41 weights of the structure the LLM built. refs.csv: gradient / tree baselines at the same N; refs_vs_N.csv: the same at N = 100 to 5e4, per seed."""
    D, CB = f"{raw}/v0_k0s/big", f"{raw}/cborg/llm_runs"
    if not os.path.isdir(D):
        print("fig_edit_search: raw data not found, skipped"); return
    rows = []

    def add(method, space, n, seed, hist):
        for i, (ta, tr) in enumerate(hist, 1):
            rows.append(dict(method=method, space=space, N=n, seed=seed, evaluation=i, test_auc=ta, test_r95=tr))

    for n in (10000, 50000):
        for f in sorted(glob.glob(f"{D}/search/r*_*_*_n{n}.json")):
            sp, opt, seed = os.path.basename(f).split("_")[:3]
            r, cur, t, h = json.load(open(f)), -1, None, []
            for x in r["hist"]:
                if x["stop_auc"] > cur and "test_auc" in x:
                    cur, t = x["stop_auc"], x
                h.append((t["test_auc"], t["test_rej95"]))
            add(opt, sp, n, int(seed), h)
        H, cur, t, h = json.load(open(f"{D}/hand_n{n}/test_all.json")), -1, None, []
        for x in H:
            if x["stop_auc"] > cur:
                cur, t = x["stop_auc"], x
            h.append((t["test_auc"], t["test_rej95"]))
        add("llm_interactive", "llm", n, 0, h)
        for rd in sorted(glob.glob(f"{CB}/opus55_n{n}_s*")):
            hh = json.load(open(f"{rd}/history.json"))
            T = {int(os.path.basename(x["spec"])[1:-5]): x for x in json.load(open(f"{rd}/test.json"))}
            cur, t, h = -1, None, []
            for (rr, _, _), st in zip(hh["history"], hh["stop_auc"]):
                if st > cur:
                    cur, t = st, T[rr]
                h.append((t["auc"], t["rej95"]))
            add("llm_automated", "llm", n, int(rd[-1]), h)
    write(f"{HERE}/fig_edit_search/data.csv", rows, ["method", "space", "N", "seed", "evaluation", "test_auc", "test_r95"])
    nets, trees = json.load(open(f"{D}/improve_nets.json")), json.load(open(f"{D}/improve_trees.json"))
    refs = []
    for model, src, lab in (("[log] written, prior max(N,N0) + syst*", nets, "encoded + gradient"),
                            ("[log] random init, 64 units", nets, "random start + gradient"),
                            ("boosted trees, from scratch", trees, "boosted trees"),
                            ("boosted trees on top of tuned written net*", trees, "boosted trees on encoded")):
        for n in (10000, 50000):
            v = [r for r in src if r["model"] == model and r["n"] == n]
            if v:
                refs.append(dict(reference=lab, N=n, test_auc=round(float(np.mean([r["auc"] for r in v])), 6),
                                 test_r95=round(float(np.mean([r["rej95"] for r in v])), 2)))
    write(f"{HERE}/fig_edit_search/refs.csv", refs, ["reference", "N", "test_auc", "test_r95"])
    # the same trained models at every label budget, one row per seed (panel c: rejection vs N)
    vn = []
    for model, src, lab in (("boosted trees, from scratch", trees, "boosted trees"),
                            ("[log] random init, 64 units", nets, "neural network, gradient from random weights"),
                            ("[log] written, prior max(N,N0) + syst*", nets, "encoded network + gradient")):
        for r in src:
            if r["model"] == model:
                vn.append(dict(reference=lab, N=r["n"], seed=r["seed"], test_auc=r["auc"], test_r95=r["rej95"]))
    lb = os.path.join(os.path.dirname(raw), "v0_k0s_llmbdt", "results.json")   # WSP + BDT: Opus 5.5 writes the trees, LightGBM boosts on top
    if os.path.exists(lb):
        for r in json.load(open(lb)):
            vn.append(dict(reference="LLM-written trees + boosting", N=r["N"], seed=r["seed"], test_auc=r["test_auc"], test_r95=r["test_r95"]))
    vn.sort(key=lambda r: (r["reference"], r["N"], r["seed"]))
    write(f"{HERE}/fig_edit_search/refs_vs_N.csv", vn, ["reference", "N", "seed", "test_auc", "test_r95"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", default=os.path.join(HERE, "..", "sync"))
    a = ap.parse_args()
    laws(a.sync)
    nscan(a.sync)
    srbench(a.sync)
    editors(a.sync)
    edit_search()
