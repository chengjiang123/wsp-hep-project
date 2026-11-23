"""Ablation 2.1 / 2.2: what does the written K0S network owe to the LLM?
Specs, all in the same architecture and scored exactly like write_k0s.py (as written, and as start + prior at N labels):
  alice_hard   the published ALICE K0S topological cuts (arXiv:2211.08936, Table A.1), near-hard (k = 6)
  alice_soft   the same cuts, soft (k = 2)
  alice_soft_arm  + Armenteros qT/|alpha| > 0.2: identical to the interactively LLM-written hand/r0.json
  random_XX    the same 8 cut directions with thresholds drawn log-uniformly over the preselected range, k = 2 (10 draws)
usage:  python ablation_write.py make            -> specs/ablation/*.json
        python ablation_write.py score [tags]    -> llm_runs/write_abl_<tag>/score.json (uses write_k0s.score)
        python ablation_write.py summary         -> table (and ablation_write.json)"""
import glob, json, os, sys, numpy as np

D = "specs/ablation"
ALICE = [("dca_dau < 1.0", 0, 1.0, "<"), ("cos_pa > 0.97", 1, 0.97, ">"), ("dca_pos > 0.06", 2, 0.06, ">"), ("dca_neg > 0.06", 3, 0.06, ">"),
         ("radius > 0.5", 4, 0.5, ">"), ("mL/p < 20", 5, 20.0, "<"), ("dmLambda > 0.005", 7, 0.005, ">")]
ARM = ("arm > 0.2", 6, 0.2, ">")
# ranges for random thresholds: inside the preselection (dca < 2, cos > 0.9, R > 0.2) and the physical range of each feature
RANGE = {0: (0.02, 2.0), 1: (0.9, 0.99999), 2: (0.005, 2.0), 3: (0.005, 2.0), 4: (0.2, 40.0), 5: (1.0, 100.0), 6: (0.02, 0.5), 7: (0.0005, 0.05)}


def spec(cuts, k, note):
    u = [dict(cut=c, f=f, t=t, dir=d, k=(1.2 if f == 6 else k), w=1.0) for c, f, t, d in cuts]
    return dict(note=note, bias=-float(len(u) - 1), units=u)


def make():
    os.makedirs(D, exist_ok=True)
    out = {"alice_hard": spec(ALICE, 6.0, "ALICE pp 13 TeV K0S cuts, near-hard"),
           "alice_soft": spec(ALICE, 2.0, "ALICE pp 13 TeV K0S cuts, soft"),
           "alice_soft_arm": spec(ALICE + [ARM], 2.0, "ALICE cuts soft + Armenteros (= hand/r0.json)")}
    r = np.random.default_rng(2026)
    for i in range(10):
        cuts = []
        for c, f, t, d in ALICE + [ARM]:
            lo, hi = RANGE[f]
            if f == 1:                                       # draw 1 - cos log-uniformly
                v = 1 - 10 ** r.uniform(np.log10(1 - hi), np.log10(1 - lo))
            elif f == 6:
                v = r.uniform(lo, hi)
            else:
                v = 10 ** r.uniform(np.log10(lo), np.log10(hi))
            cuts.append((f"{c.split()[0]} {d} {v:.4g}", f, float(v), d))
        out[f"random_{i:02d}"] = spec(cuts, 2.0, f"random thresholds, draw {i}")
    for k, s in out.items():
        json.dump(s, open(f"{D}/{k}.json", "w"), indent=1)
    print("wrote", len(out), "specs to", D)


def score(tags):
    from write_k0s import score as sc
    for f in sorted(glob.glob(f"{D}/*.json")):
        tag = os.path.basename(f)[:-5]
        if tags and tag not in tags:
            continue
        out = f"llm_runs/write_abl_{tag}"; os.makedirs(out, exist_ok=True)
        if os.path.exists(f"{out}/score.json"):
            print("skip", tag); continue
        print("=====", tag, flush=True); sc(json.load(open(f)), out, f"abl_{tag}")


def summary():
    rows = {}
    for f in glob.glob("llm_runs/write_*/score.json"):
        r = json.load(open(f))
        if isinstance(r, list):
            rows[f.split("/")[1][len("write_"):]] = r
    group = lambda t: "random thresholds (10 draws)" if t.startswith("abl_random") else t
    G = {}
    for t, r in rows.items():
        G.setdefault(group(t), []).extend(r)
    res = {}
    print(f"{'spec':34s} {'R95 as written':>15s} " + " ".join(f"{'N=' + str(n):>15s}" for n in (100, 1000, 10000)))
    for g in sorted(G):
        r = G[g]; line = {}
        a = [x["rej95"] for x in r if x["n"] == 0 and x["model"] == "written, as is"]
        line["asis"] = (float(np.mean(a)), float(np.std(a))) if a else None
        for n in (100, 1000, 10000):
            v = [x["rej95"] for x in r if x["n"] == n]
            line[n] = (float(np.mean(v)), float(np.std(v))) if v else None
        res[g] = line
        f = lambda m: f"{m[0]:7.1f} ± {m[1]:5.1f}" if m else f"{'-':>15s}"
        print(f"{g:34s} {f(line['asis'])} " + " ".join(f(line[n]) for n in (100, 1000, 10000)))
    json.dump(res, open("ablation_write.json", "w"), indent=1)


if __name__ == "__main__":
    {"make": lambda: make(), "score": lambda: score(sys.argv[2:]), "summary": lambda: summary()}[sys.argv[1]]()
