"""One cell of the scan: dataset x model x label budget N x seed. Runs every arm and control, one JSON per arm.

Arms   A1 ft          pretrained + new head -> fine-tune
       A2 ft_wsp      A1's network -> LLM weight edit (no training afterwards)
       A3 wsp_ft      pretrained + new head -> LLM weight edit -> fine-tune      (also scored right after the edit: wsp0)
Ctrl   C1 lpft        logistic-regression head on frozen features -> fine-tune
       C2 script_ft   fixed-rule edit (common.script_patch) -> fine-tune         (also scored after the edit: script0)
       C3 random_ft   random patch on the same tensors as A3's, same size -> fine-tune
       C0 scratch     random weights, published from-scratch recipe
Usage: python run_cell.py --ds top --model part --N 1000 --seed 0 [--arms ft,ft_wsp,...] [--llm claude-opus-5-5]
Results: $WSP_ROOT/results/<ds>/<model>/N<N>/s<seed>/<arm>.json   (existing files are skipped)
"""
import argparse
import copy
import os
import time

import torch

import common as C
import llm_edit as L

ALL = ["ft", "ft_wsp", "wsp_ft", "lpft", "script_ft", "random_ft", "scratch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["top", "qg"])
    ap.add_argument("--model", required=True, choices=["part", "omni"])
    ap.add_argument("--N", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ALL))
    ap.add_argument("--llm", default=os.environ.get("LLM_MODEL", "lbl/gpt-oss-120b"))
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None, help="probe only: cap the fine-tuning steps")
    ap.add_argument("--ntest", type=int, default=0, help="probe only: first NTEST test jets (0 = all)")
    ap.add_argument("--tag", default="", help="results subfolder suffix, e.g. _llm-<model> for the appendix")
    a = ap.parse_args()
    arms = a.arms.split(",")
    out = f"{C.ROOT}/results{a.tag}/{a.ds}/{a.model}/N{a.N}/s{a.seed}"
    os.makedirs(out, exist_ok=True)
    have = lambda arm: os.path.exists(f"{out}/{arm}.json")
    if all(have(x) for x in arms):
        print("all done", out)
        return

    t0 = time.time()
    (trs, tri), (vas, vai) = C.splits_for(a.ds, a.N, a.seed)
    tr, va = C.Split(a.ds, trs, tri), C.Split(a.ds, vas, vai)
    te = C.Split(a.ds, "test", None if not a.ntest else range(a.ntest))
    print(f"data: train {len(tr)}, val {len(va)}, test {len(te)}  ({time.time() - t0:.0f} s)", flush=True)
    budget = len(tr) + len(va)
    ft = lambda m, mode="ft": C.finetune(m, tr, va, mode=mode, seed=a.seed, max_steps=a.max_steps)
    test = lambda m: C.evaluate(m, te)
    base = dict(ds=a.ds, model=a.model, N=a.N, seed=a.seed, n_train=len(tr), n_val=len(va))
    save = lambda arm, d: C.save_json({**base, "arm": arm, **d}, f"{out}/{arm}.json")
    keep_state = a.N in ("1000", "10000") and a.seed == 0  # Story 2 figure

    def state(m, name):
        if keep_state:
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in m.state_dict().items()}, f"{out}/{name}.pt")

    # A1 (needed by A2)
    a1 = None
    if "ft" in arms or ("ft_wsp" in arms and not have("ft_wsp")):
        a1 = C.build(a.model, a.ds)
        if have("ft") and os.path.exists(f"{out}/ft_state.pt"):
            a1.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in torch.load(f"{out}/ft_state.pt").items()})
        else:
            r = ft(a1)
            save("ft", {"test": test(a1), "train": r})
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in a1.state_dict().items()}, f"{out}/ft_state.pt")
        print("ft done", flush=True)

    if "ft_wsp" in arms and not have("ft_wsp"):
        m, rec = L.edit_loop(a1, a.ds, "ft", tr, va, a.llm, rounds=a.rounds, budget=budget, log_path=f"{out}/llm_ft_wsp.json")
        save("ft_wsp", {"test": test(m), "llm": rec})
        state(m, "ft_wsp_state")
        print("ft_wsp done", flush=True)

    a3_patch = None
    if ("wsp_ft" in arms and not have("wsp_ft")) or ("random_ft" in arms and not have("random_ft")):
        if have("wsp_ft"):
            import json
            a3_patch = json.load(open(f"{out}/wsp_ft.json"))["llm"]["best_patch"]
        else:
            m0 = C.build(a.model, a.ds)
            m, rec = L.edit_loop(m0, a.ds, "pre", tr, va, a.llm, rounds=a.rounds, budget=budget, log_path=f"{out}/llm_wsp_ft.json")
            a3_patch = rec["best_patch"]
            zero = test(m)
            state(m, "wsp0_state")
            r = ft(m)
            save("wsp_ft", {"test": test(m), "test_after_edit": zero, "train": r, "llm": rec})
            state(m, "wsp_ft_state")
            print("wsp_ft done", flush=True)

    if "lpft" in arms and not have("lpft"):
        m = C.build(a.model, a.ds)
        C.lp_head(m, tr)
        zero = test(m)
        r = ft(m)
        save("lpft", {"test": test(m), "test_after_head": zero, "train": r})
        state(m, "lpft_state")
        print("lpft done", flush=True)

    if "script_ft" in arms and not have("script_ft"):
        m = C.build(a.model, a.ds)
        p = C.apply_patch(m, C.script_patch(a.ds))
        zero = test(m)
        r = ft(m)
        save("script_ft", {"test": test(m), "test_after_edit": zero, "patch": p, "train": r})
        print("script_ft done", flush=True)

    if "random_ft" in arms and not have("random_ft"):
        m = C.build(a.model, a.ds)
        p = C.random_patch_like(m, a3_patch, seed=10_000 + a.seed)
        zero = test(m)
        r = ft(m)
        save("random_ft", {"test": test(m), "test_after_edit": zero, "patch": p, "train": r})
        print("random_ft done", flush=True)

    if "scratch" in arms and not have("scratch"):
        m = C.build(a.model, a.ds, pretrained=False)
        r = ft(m, mode="scratch")
        save("scratch", {"test": test(m), "train": r})
        print("scratch done", flush=True)
    print(f"cell done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
