"""Our method: the LLM reads the law-shaped network's weights (exactly what show.py prints) and states the law.
It never sees data or truth. Same prompt, answer format and one repair turn as laws/read_cborg.py (the paper's reader).
usage: python reader.py <set> <N> <seed> --model claude-opus-5-5 --tag opus
Result per law: results/reader_<tag>/...json with the answer, the paper's exact-match verdict and the shared
out-of-range verdict."""
import argparse
import json
import os
import subprocess
import sys
import time

import requests

import s3

FORMAT = {
    "kin": 'Inputs are the four-vectors (E, px, py, pz) of particles 1, 2, 3 in GeV, smeared by 1%. The network was trained '
           'to regress an unknown scalar function of them. State the function. Answer JSON: {"S": [0-based indices of the '
           'particles whose four-vectors are summed], "s": [coefficients of E^2, px^2, py^2, pz^2 of the summed vector], "law": "<formula and physics meaning>"}',
    "cons": 'Inputs are quantum numbers (Q, B, Le, Lmu, S) of the four particles of a reaction a + b -> c + d. The network was '
            'trained to tell real reactions from forbidden ones. State which linear combinations of (Q, B, Le, Lmu, S) are '
            'conserved. Answer JSON: {"U": [[integer coefficients of (Q, B, Le, Lmu, S)], ...] one row per independent conserved quantity, "law": "<statement>"}',
    "decay": 'Inputs are log L (cm), log p (GeV), log m (GeV) of a particle that flew a distance L. The network gives the '
             'probability that it decayed before L, P = 1 - exp(-rate). State the rate as rate = L^alpha p^beta m^gamma / lam. '
             'Answer JSON: {"exp": [alpha, beta, gamma] (simple rational numbers), "lam": <lam in the units of the inputs>, "law": "<statement>"}',
}
KEYS = {"kin": ("S", "s"), "cons": ("U",), "decay": ("exp", "lam")}
SYSTEM = "You read small neural networks and state the physical law they implement. Reply with one JSON object only."
URL = os.environ.get("LLM_URL", "https://api.cborg.lbl.gov") + "/v1/chat/completions"


def extract(txt, keys):
    import ast
    import re
    cands = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S)
    st = []
    for i, ch in enumerate(txt):
        if ch == "{":
            st.append(i)
        elif ch == "}" and st:
            j = st.pop()
            if not st:
                cands.append(txt[j:i + 1])
    for c in reversed(cands):
        for f in (json.loads, lambda x: json.loads(re.sub(r",\s*([}\]])", r"\1", x)),
                  lambda x: ast.literal_eval(re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", x))))):
            try:
                d = f(c)
            except Exception:
                continue
            if isinstance(d, dict) and all(k in d for k in keys):
                return d
    raise ValueError("no JSON object with keys " + ", ".join(keys))


def ask(model, prompt, history=None):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}] + (history or [])
    if model.startswith("claude-"):   # your Anthropic API key, official SDK (wsp_bench/code/claude_llm.py)
        sys.path.insert(0, os.environ.get("WSP_CODE", "/path/to/scratch/wsp_bench/code"))
        import claude_llm
        return claude_llm.call(model, msgs)
    for attempt in range(8):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {os.environ['CBORG_API_KEY']}"}, timeout=900,
                              json={"model": model, "messages": msgs, "max_tokens": 16000})
            if r.status_code == 200:
                j = r.json()
                return j["choices"][0]["message"].get("content") or "", j.get("usage", {})
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except requests.RequestException:
            pass
        time.sleep(min(30 * 2 ** attempt, 600))
    raise RuntimeError("LLM unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("N", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--tag", default="opus")
    a = ap.parse_args()
    method = f"reader_{a.tag}"
    for law in s3.laws_in(a.set):
        if s3.done(method, a.set, law, a.N, a.seed):
            continue
        fam = law.split("_")[0]
        nf = f"{s3.ROOT}/sets/{a.set}/nets/{law}_n{a.N}_s{a.seed}.json"
        view = subprocess.run([sys.executable, f"{s3.HERE}/show.py", nf], capture_output=True, text=True).stdout
        prompt = FORMAT[fam] + "\n\nNetwork:\n" + view
        t0, raws, usage = time.time(), [], []
        try:
            txt, u = ask(a.model, prompt)
            raws.append(txt); usage.append(u)
            try:
                ans = extract(txt, KEYS[fam])
            except ValueError as e:
                txt, u = ask(a.model, prompt, [{"role": "assistant", "content": txt[-6000:]}, {"role": "user", "content":
                             f"Your answer could not be read ({e}). Reply with exactly one JSON object in the format asked, nothing else."}])
                raws.append(txt); usage.append(u)
                ans = extract(txt, KEYS[fam])
                ans["repaired"] = True
        except Exception as e:
            ans = {"error": str(e)[:300]}
        d = s3.load(a.set, law)
        t = s3.truth(a.set, law)
        if "error" in ans:
            r = {"id": float("inf"), "ood": float("inf"), "recovered": False}
            exact = False
        else:
            try:
                r = s3.score(fam, s3.answer_predictor(fam, ans), d)
            except Exception as e:
                r = {"id": float("inf"), "ood": float("inf"), "recovered": False, "error": str(e)[:200]}
            exact = s3.symbolic_match(fam, ans, t)
        s3.save(method, a.set, law, a.N, a.seed, dict(set=a.set, law=law, fam=fam, N=a.N, seed=a.seed, method=method, model=a.model,
                                                      answer=ans, exact=exact, raw=raws, usage=usage, seconds=time.time() - t0, **r))
        print(f"{a.set} {law} N={a.N} s{a.seed}: exact={exact} ood={r['ood']:.3g} recovered={r['recovered']}", flush=True)


if __name__ == "__main__":
    main()
