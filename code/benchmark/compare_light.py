"""Story 1, light comparison on the cells of cells_light.txt: test AUC of plain FT, FT->WSP and WSP->FT with the LLM edit
from gpt-oss (main scan, results/) and from Claude (results<tag>/), plus the fixed-rule edit (script_ft), paired by cell.
usage: python compare_light.py [--tag _llm-opus55sub]"""
import argparse
import json
import os

import numpy as np

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_llm-opus55sub")
ap.add_argument("--tag2", default="_llm-sonnet55sub")
a = ap.parse_args()


def auc(root, ds, m, N, s, arm):
    f = f"{C.ROOT}/{root}/{ds}/{m}/N{N}/s{s}/{arm}.json"
    return json.load(open(f))["test"]["auc"] if os.path.exists(f) else np.nan


rows = []
for line in open("cells_light.txt"):
    ds, m, N, s = line.split()
    r = dict(cell=f"{ds}/{m}/N{N}/s{s}", ft=auc("results", ds, m, N, s, "ft"), script_ft=auc("results", ds, m, N, s, "script_ft"),
             gpt_ft_wsp=auc("results", ds, m, N, s, "ft_wsp"), gpt_wsp_ft=auc("results", ds, m, N, s, "wsp_ft"),
             op_ft_wsp=auc(f"results{a.tag}", ds, m, N, s, "ft_wsp"), op_wsp_ft=auc(f"results{a.tag}", ds, m, N, s, "wsp_ft"),
             sn_ft_wsp=auc(f"results{a.tag2}", ds, m, N, s, "ft_wsp"), sn_wsp_ft=auc(f"results{a.tag2}", ds, m, N, s, "wsp_ft"))
    rows.append(r)
cols = ["ft", "script_ft", "gpt_ft_wsp", "gpt_wsp_ft", "op_ft_wsp", "op_wsp_ft", "sn_ft_wsp", "sn_wsp_ft"]
print("gpt = gpt-oss-120b, op = Opus 5.5, sn = Sonnet 5.5 (subagents); test AUC")
print(f"{'cell':26s} " + " ".join(f"{c:>10s}" for c in cols))
for r in rows:
    print(f"{r['cell']:26s} " + " ".join(f"{r[c]:10.4f}" for c in cols))
print("\npaired difference to plain FT, mean (SE) over cells with both:")
for c in cols[1:]:
    d = np.array([r[c] - r["ft"] for r in rows if np.isfinite(r[c])])
    if len(d):
        print(f"  {c:12s} {d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.4f})  n={len(d)}")
