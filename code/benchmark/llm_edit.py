"""The LLM edits the weights of an existing network, in rounds, from diagnostics on the labelled jets of the budget.

Each round: the LLM gets the task, the editable tensors, the diagnostics (training split of the budget only) and
the history of its own patches; it returns a JSON patch that `common.apply_patch` writes into the weights.
No gradient step happens here. The round kept is the one with the lowest loss on the validation split of the budget
(rounds 1..R only: at least one edit is always applied).
Endpoint: model ids starting with "claude-" go to the Anthropic API (claude_llm.py, key in ANTHROPIC_API_KEY); any other
id goes to an OpenAI-compatible server, default CBorg (key in CBORG_API_KEY).
"""
import copy
import json
import os
import re
import time

import requests

import common as C

URL = os.environ.get("LLM_URL", "https://api.cborg.lbl.gov") + "/v1/chat/completions"

TASK = {
    "top": "Hadronic top-quark jets (signal, t -> b q q') against QCD light-quark and gluon jets (background). "
           "Anti-kT R=0.8, pT 550-650 GeV, |eta|<2, 14 TeV pp, Delphes (TopLandscape).",
    "qg": "Quark jets (signal) against gluon jets (background). Anti-kT R=0.4, pT 500-550 GeV, |y|<1.7, 14 TeV pp, "
          "Pythia 8 (QuarkGluon dataset, particle IDs available).",
}
MODEL_TXT = {
    "part": "Particle Transformer (ParT, 2.1M parameters) pretrained on JetClass (100M jets, 10 classes). "
            "Its pretrained output layer has one weight row per JetClass class: " + ", ".join(C.JETCLASS) + ". "
            "Each of its 8 attention heads adds a pairwise bias U_h(i,j), computed from (ln kT, ln z, ln DeltaR, ln m^2) of "
            "particle pair (i,j), to the attention logits of all 8 particle-attention layers.",
    "omni": "OmniLearned PET2-small, a jet foundation model pretrained on about 1 billion jets from several datasets with a "
            "210-row output layer. Rows 2-11 are the JetClass classes " + ", ".join(C.JETCLASS) + "; rows 0-1 are an ATLAS "
            "top-tagging set; other rows are other datasets' classes (named rowK). Each of its 8 attention heads adds a "
            "pairwise bias U_h(i,j) from pair kinematics to the attention logits.",
}

SYSTEM = (
    "You are a particle physicist who edits neural-network weights directly. You never train the network. "
    "You return a JSON patch that is written into the weights as is. Reply with one JSON object only: "
    '{"reasoning": "<= 120 words", "patch": [ ... ops ... ]}.'
)

OPS = """Allowed ops (each one edits weights in place, in order):
- {"op": "head_from_rows", "signal": {ROW: coef, ...}, "background": {ROW: coef, ...}}
    sets the 2-row output layer (weights and biases) to linear combinations of pretrained output rows.
- {"op": "head_mix", "alpha": a, "signal": {...}, "background": {...}}
    output layer = (1-a) * current + a * that combination (0 < a <= 1).
- {"op": "head_scale", "factor": f}      multiply the output weights and biases by f (sharpens or softens the scores).
- {"op": "head_bias", "delta": d}        add d to the signal output bias.
- {"op": "pair_scale", "head": h, "factor": f}   scale pairwise-interaction head h (0-7) by f in [0, 10]:
    f = 0 switches the head off, 0 < f < 1 weakens it, f > 1 strengthens it.
ROW is a row name as listed in the diagnostics (e.g. "Tbqq", "QCD", "row57")."""


def prompt(ds, kind, start, diag, history, budget):
    where = ("The network is the pretrained model with a freshly initialised 2-row output layer (random weights); "
             "after your edit it will be fine-tuned on the same labelled jets." if start == "pre" else
             "The network has already been fine-tuned on these labelled jets; your edit is final (no training afterwards).")
    hist = "\n".join(f"Round {i + 1} patch: {json.dumps(h['patch'])}\n  -> diagnostics after it: {json.dumps(h['diag']['current'])}"
                     for i, h in enumerate(history)) or "(none yet)"
    return f"""Task: {TASK[ds]}
Network: {MODEL_TXT[kind]}
{where}
Label budget: {budget} labelled jets in total; the diagnostics below use only the training part of it.
Metric to improve: separation of signal from background (AUC, background rejection at 50% signal efficiency) and the
cross-entropy loss of the 2-row output (row 1 = signal, row 0 = background).

{OPS}

Current diagnostics (JSON):
{json.dumps(diag)}

Your previous rounds:
{hist}

Return the next patch."""


def call(model_id, messages, tries=4, max_tokens=4000, max_wait=3600):
    """CBorg allows 10 parallel requests per key; a 429 (or a 5xx / network error) is waited out with jittered backoff
    for up to max_wait seconds. Other errors (bad model id, budget) fail after `tries` attempts."""
    if model_id.startswith("claude-"):   # your Anthropic API key, official SDK (claude_llm.py)
        import claude_llm
        return claude_llm.call(model_id, messages)
    import random
    key = os.environ["CBORG_API_KEY"]
    last, bad, t0 = None, 0, time.time()
    while True:
        transient = True
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=600,
                              json={"model": model_id, "messages": messages, "max_tokens": max_tokens})
            if r.status_code == 200:
                j = r.json()
                return j["choices"][0]["message"]["content"] or "", j.get("usage", {})
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            transient = r.status_code == 429 or r.status_code >= 500
        except Exception as e:  # network hiccup
            last = repr(e)
        if transient:
            if time.time() - t0 > max_wait:
                break
            time.sleep(random.uniform(15, 60))
        else:
            bad += 1
            if bad >= tries:
                break
            time.sleep(10 * bad)
    raise RuntimeError(f"LLM call failed: {last}")


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    obj = json.loads(m.group(0))
    if not isinstance(obj.get("patch"), list):
        raise ValueError("reply has no 'patch' list")
    return obj


def edit_loop(model, ds, start, tr, va, model_id, rounds=4, budget=None, log_path=None):
    """Returns (best edited model, record). `model` is not modified."""
    cur = copy.deepcopy(model)
    history, rec = [], {"model_id": model_id, "start": start, "rounds": []}
    best, best_loss = None, float("inf")
    for r in range(rounds):
        diag = C.diagnostics(cur, tr)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(ds, cur.kind, start, diag, history, budget)}]
        entry = {"round": r + 1}
        for attempt in range(3):
            text, usage = call(model_id, msgs)
            try:
                obj = parse(text)
                trial = copy.deepcopy(cur)
                applied = C.apply_patch(trial, obj["patch"])
                break
            except Exception as e:
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That patch could not be applied ({e}). Return a valid JSON object with allowed ops only."}]
                entry.setdefault("errors", []).append(str(e))
        else:
            entry["failed"] = True
            rec["rounds"].append(entry)
            continue
        cur = trial
        v = C.evaluate(cur, va)
        after = C.diagnostics(cur, tr)
        history.append({"patch": applied, "diag": after})
        entry.update({"reasoning": obj.get("reasoning", ""), "patch": applied, "usage": usage, "val": v, "train_after": after["current"]})
        rec["rounds"].append(entry)
        if v["loss"] < best_loss:
            best_loss, best, rec["best_round"] = v["loss"], copy.deepcopy(cur), r + 1
        if log_path:
            C.save_json(rec, log_path)
    if best is None:
        raise RuntimeError("no valid LLM patch in any round")
    rec["best_patch"] = [p for h in history[: rec["best_round"]] for p in h["patch"]]
    return best, rec
