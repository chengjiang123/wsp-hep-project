"""The LLM edit loop of llm_edit.py, split into steps so the LLM can answer outside the GPU job (e.g. Claude subagents
on a subscription). Same prompt (llm_edit.prompt / SYSTEM), same ops, same diagnostics on the training split, same
choice of round by validation loss, same fine-tuning and test as run_cell.py. Only the transport differs: the prompt
of round r is written to a file, the reply is read from a file, and the model is rebuilt by replaying the patches.

  python handoff.py step   --cells cells_light.txt --tag _llm-opus55sub --round R   (R = 1..rounds+1)
      R <= rounds: replay replies 1..R-1 (score the newest one on val), write req<R>.json for every cell and arm
      R = rounds+1: pick the best round by val loss, write the arm result like run_cell.py (wsp_ft: then fine-tune)
  python handoff.py ingest --tag _llm-opus55sub --round R answers.json [answers2.json ...]
      answers: {"<ds>/<model>/N<N>/s<seed>/<arm>": {"reasoning": ..., "patch": [...]}, ...} -> resp<R>.txt

Layout: $WSP_ROOT/handoff<tag>/<ds>/<model>/N<N>/s<seed>/<arm>/{req<R>.json, resp<R>.txt, after<R>.json}
Starts: ft_wsp starts from the cell's plain-FT network (results<tag>/.../ft_state.pt, copied from the main scan);
        wsp_ft starts from the pretrained network with a new 2-row head, seeded by the cell seed so replays match."""
import argparse
import copy
import glob
import json
import os

import torch

import common as C
import llm_edit as L

ARMS = {"ft_wsp": "ft", "wsp_ft": "pre"}


def cells(path, only=None):
    """only: 1-based line number (one Slurm array task per cell)."""
    for i, line in enumerate((l for l in open(path) if l.split()), 1):
        if only is None or i == only:
            p = line.split()
            yield p[0], p[1], p[2], int(p[3])


def start_model(ds, kind, N, seed, start, tag):
    if start == "pre":
        torch.manual_seed(seed)
        return C.build(kind, ds)
    m = C.build(kind, ds)
    st = torch.load(f"{C.ROOT}/results{tag}/{ds}/{kind}/N{N}/s{seed}/ft_state.pt")
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()})
    return m


def replay(ds, kind, N, seed, arm, tag, upto, tr, va):
    """Model after the valid replies 1..upto, the history the prompt shows, and the per-round records."""
    H = f"{C.ROOT}/handoff{tag}/{ds}/{kind}/N{N}/s{seed}/{arm}"
    cur = start_model(ds, kind, N, seed, ARMS[arm], tag)
    history, rounds = [], []
    for r in range(1, upto + 1):
        entry = {"round": r}
        try:
            obj = L.parse(open(f"{H}/resp{r}.txt").read())
            trial = copy.deepcopy(cur)
            applied = C.apply_patch(trial, obj["patch"])
        except Exception as e:   # as in edit_loop: an unusable reply is a failed round, not shown in the history
            entry.update(failed=True, errors=[str(e)[:300]])
            rounds.append(entry)
            continue
        cur = trial
        af = f"{H}/after{r}.json"
        if os.path.exists(af):
            after = json.load(open(af))
        else:
            after = {"val": C.evaluate(cur, va), "diag": C.diagnostics(cur, tr)}
            C.save_json(after, af)
        history.append({"patch": applied, "diag": after["diag"]})
        entry.update(reasoning=obj.get("reasoning", ""), patch=applied, val=after["val"], train_after=after["diag"]["current"])
        rounds.append(entry)
    return cur, history, rounds, H


def step(a):
    for ds, kind, N, seed in cells(a.cells, a.line):
        (trs, tri), (vas, vai) = C.splits_for(ds, N, seed)
        tr, va = C.Split(ds, trs, tri), C.Split(ds, vas, vai)
        budget = len(tr) + len(va)
        for arm, start in ARMS.items():
            out = f"{C.ROOT}/results{a.tag}/{ds}/{kind}/N{N}/s{seed}"
            if os.path.exists(f"{out}/{arm}.json"):
                continue
            cur, history, rounds, H = replay(ds, kind, N, seed, arm, a.tag, a.round - 1, tr, va)
            os.makedirs(H, exist_ok=True)
            if a.round <= a.rounds:
                diag = C.diagnostics(cur, tr)
                C.save_json({"system": L.SYSTEM, "user": L.prompt(ds, cur.kind, start, diag, history, budget)}, f"{H}/req{a.round}.json")
                print("req", a.round, H, flush=True)
                continue
            ok = [e for e in rounds if not e.get("failed")]
            if not ok:
                print("no valid reply in any round:", H, flush=True)
                continue
            best = min(ok, key=lambda e: e["val"]["loss"])["round"]
            m, _, _, _ = replay(ds, kind, N, seed, arm, a.tag, best, tr, va)
            rec = {"model_id": a.model_id, "start": start, "transport": "handoff files", "rounds": rounds, "best_round": best,
                   "best_patch": [p for e in rounds[:best] if not e.get("failed") for p in e["patch"]]}
            te = C.Split(ds, "test", None)
            base = dict(ds=ds, model=kind, N=N, seed=seed, n_train=len(tr), n_val=len(va), arm=arm)
            if arm == "ft_wsp":
                C.save_json({**base, "test": C.evaluate(m, te), "llm": rec}, f"{out}/{arm}.json")
            else:
                zero = C.evaluate(m, te)
                r = C.finetune(m, tr, va, mode="ft", seed=seed)
                C.save_json({**base, "test": C.evaluate(m, te), "test_after_edit": zero, "train": r, "llm": rec}, f"{out}/{arm}.json")
            print("done", arm, out, "best round", best, flush=True)


def ingest(a):
    n = 0
    for f in a.answers:
        for key, obj in json.load(open(f)).items():
            H = f"{C.ROOT}/handoff{a.tag}/{key}"
            if not os.path.isdir(H):
                print("unknown key", key)
                continue
            open(f"{H}/resp{a.round}.txt", "w").write(obj if isinstance(obj, str) else json.dumps(obj))
            n += 1
    missing = [os.path.dirname(q) for q in glob.glob(f"{C.ROOT}/handoff{a.tag}/*/*/*/*/*/req{a.round}.json")
               if not os.path.exists(os.path.join(os.path.dirname(q), f"resp{a.round}.txt"))]
    print(n, "replies written;", len(missing), "requests without a reply:", missing[:5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["step", "ingest"])
    ap.add_argument("answers", nargs="*")
    ap.add_argument("--cells")
    ap.add_argument("--line", type=int, default=None, help="only this line of --cells (Slurm array task id)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--model-id", default="claude-opus-5-5 (subagent)")
    a = ap.parse_args()
    step(a) if a.cmd == "step" else ingest(a)
