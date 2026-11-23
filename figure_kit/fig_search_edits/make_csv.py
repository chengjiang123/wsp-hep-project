"""E1: black-box search vs LLM editor on the pretrained-tagger edit space (FT -> edit arm). Writes data.csv:
one row per (ds, model, N, seed, method, evaluations): test AUC / R30 / R50 of the best-by-validation candidate, the
cell's plain-FT test values, and the gain R/R_FT - 1 at the task's working point (R30 top, R50 qg).
LLM rows (method llm_opus, llm_sonnet): the round chosen by validation loss, evaluations = number of rounds (4)."""
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "..", "..", "sync")
WP = {"top": "r30", "qg": "r50"}
rows = []


def ft_of(ds, model, N, seed):
    for t in ("opus55sub", "sonnet55sub"):
        f = f"{S}/results_llm-{t}/{ds}/{model}/N{N}/s{seed}/ft.json"
        if os.path.exists(f):
            return json.load(open(f))["test"]


for f in glob.glob(f"{S}/search_edits/*/*/N*/s*/*_r*.json"):
    d = json.load(open(f))
    ft = ft_of(d["ds"], d["model"], d["N"], d["seed"])
    for b in d["budgets"]:
        rows.append(dict(ds=d["ds"], model=d["model"], N=d["N"], seed=d["seed"], method=d["method"], evaluations=b["budget"],
                         test_auc=b["test"]["auc"], test_r30=b["test"]["r30"], test_r50=b["test"]["r50"],
                         ft_auc=ft["auc"], ft_r30=ft["r30"], ft_r50=ft["r50"], n_val=d["n_val"]))
for t, name in (("opus55sub", "llm_opus"), ("sonnet55sub", "llm_sonnet")):
    for f in glob.glob(f"{S}/results_llm-{t}/*/*/N*/s*/ft_wsp.json"):
        d = json.load(open(f))
        ft = json.load(open(f.replace("ft_wsp.json", "ft.json")))["test"]
        rows.append(dict(ds=d["ds"], model=d["model"], N=d["N"], seed=d["seed"], method=name, evaluations=len(d["llm"]["rounds"]),
                         test_auc=d["test"]["auc"], test_r30=d["test"]["r30"], test_r50=d["test"]["r50"],
                         ft_auc=ft["auc"], ft_r30=ft["r30"], ft_r50=ft["r50"], n_val=d["n_val"]))
D = pd.DataFrame(rows)
D["gain"] = [100 * (r[f"test_{WP[r.ds]}"] / r[f"ft_{WP[r.ds]}"] - 1) for r in D.itertuples()]
D.sort_values(["ds", "model", "N", "seed", "method", "evaluations"]).to_csv(os.path.join(HERE, "data.csv"), index=False)
print(len(D), "rows ->", os.path.join(HERE, "data.csv"))
