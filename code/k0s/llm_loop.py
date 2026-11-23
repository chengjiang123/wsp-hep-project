"""Open-weight LLM in the hand-edit loop (K0S weight space). UNTESTED here (no GPU); written for NERSC.

The model replaces me: each round it receives the diagnostics of the current spec (from handedit.py diag, which use
only the 1000 edit candidates), the history of edit-set AUCs, and returns a new spec as JSON.
It never sees the test set. After the loop, `python handedit.py test llm_runs/<tag>/r*.json` scores every round once.

Talks to any OpenAI-compatible server: CBorg (LLM_URL=https://api.cborg.lbl.gov, key in CBORG_API_KEY) or a local vLLM:
    vllm serve <model> --port 8000 --tensor-parallel-size 4
usage:
    NEDIT=10000 STOPFRAC=0.3 python llm_loop.py --model <model> --rounds 25 --tag qwen32b [--url ...] [--start hand/r0.json]
    The model sees diagnostics on the edit part only. With STOPFRAC>0 the held-back part picks the final round
    and ends the loop after --patience rounds without improvement.
"""
import argparse, contextlib, io, json, os, re, sys, time, urllib.error, urllib.request
import handedit as H

SYSTEM = """You are a particle physicist tuning a K0S -> pi+ pi- selection by editing neural-network weights by hand.
No gradient training is allowed. The network is a sum of tanh units on log-scaled topological features,
score = sum_j w_j * tanh(...) + bias. Each unit is either
  a cut unit:   {"cut": name, "f": feature index, "t": threshold in physical units, "dir": ">" or "<", "k": sharpness (0.3-6), "w": output weight (0-4)}
  a free unit:  {"name": name, "lin": {"feature index": coefficient in layer units}, "b": bias, "w": output weight}
Feature indices: 0 DCA between daughters [cm], 1 cos(pointing angle), 2 DCA of positive daughter to PV [cm],
3 DCA of negative daughter to PV [cm], 4 decay radius [cm], 5 mL/p [cm], 6 Armenteros qT/|alpha|, 7 |m(p pi) - m_Lambda| [GeV].
Layer units: x = (log10(feature) - log10(written threshold)) / 0.5 for all features except 6 (x = (value - 0.2)/0.2)
and 1 (x uses log10(1 - cos)).  Tracks start within 40 cm of the beam.
Goal: maximise separation (AUC, and background rejection at 90-95% K0S efficiency) on the edit set,
in a way that will generalise: prefer physically motivated edits over fitting individual candidates.
Answer with a short reasoning paragraph, then ONE JSON object {"note": ..., "bias": ..., "units": [...]} in a ```json block."""
_P = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts", "k0s_edit_system.txt")
if os.path.exists(_P):                                   # the frozen prompt file is the reference copy
    SYSTEM = open(_P).read()


def log_usage(model, usage, seconds):
    """one JSON line per call to $LLM_USAGE_LOG (tokens and wall time, for the cost column of the benchmark)"""
    f = os.environ.get("LLM_USAGE_LOG")
    if f:
        with open(f, "a") as fh:
            fh.write(json.dumps(dict(model=model, prompt_tokens=usage.get("prompt_tokens", 0),
                                     completion_tokens=usage.get("completion_tokens", 0), seconds=round(seconds, 1))) + "\n")


def chat(url, model, messages, temperature=0.3, max_tokens=None):
    """OpenAI-compatible call. Retries on 429 / 5xx with backoff. Reasoning models (gpt-oss "high") spend many tokens
    thinking before the answer, so the budget is large (MAX_TOKENS env, default 16000)."""
    max_tokens = max_tokens or int(os.environ.get("MAX_TOKENS", "16000"))
    hdr = {"Content-Type": "application/json"}
    key = os.environ.get("CBORG_API_KEY") or os.environ.get("OPENAI_API_KEY")    # never stored in files
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    body = json.dumps(dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)).encode()
    for attempt in range(8):
        req = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", method="POST", headers=hdr, data=body)
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=900) as r:
                resp = json.loads(r.read())
            ch = resp["choices"][0]
            log_usage(model, resp.get("usage") or {}, time.time() - t0)
            txt = ch["message"].get("content") or ""
            if not txt.strip():
                txt = f"[empty answer, finish_reason={ch.get('finish_reason')}]"
            return txt
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise SystemExit(f"model '{model}' not found at {url} (HTTP 404): check the exact id with nersc/cborg_models.sh")
            if (e.code == 429 or 500 <= e.code < 600) and attempt < 7:      # incl. gateway timeouts 520-529
                time.sleep(min(30 * 2 ** attempt, 600)); continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:   # dropped connection or read timeout
            if attempt < 7:
                time.sleep(min(30 * 2 ** attempt, 600)); continue
            raise


def diag_text(spec):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        H.diag(spec); print(); H.diag2(spec)
    return buf.getvalue()


def edit_auc(text):
    m = re.search(r"AUC ([0-9.]+)", text)
    return float(m.group(1)) if m else float("nan")


def parse_spec(text):
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{.*\})", text, re.S)
    spec = json.loads(m[-1])
    assert isinstance(spec.get("units"), list) and spec["units"], "no units"
    for u in spec["units"]:
        assert "w" in u and (("cut" in u and {"f", "t", "dir", "k"} <= set(u)) or ("lin" in u and "b" in u)), f"bad unit {u}"
    spec.setdefault("bias", 0.0)
    H.build(spec)                     # raises if the spec cannot be built
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default=os.environ.get("LLM_URL", "http://localhost:8000/v1"),
                    help="OpenAI-compatible endpoint, e.g. https://api.cborg.lbl.gov (key from CBORG_API_KEY) or a local vLLM")
    ap.add_argument("--rounds", type=int, default=8); ap.add_argument("--tag", required=True)
    ap.add_argument("--start", default="hand/r0.json"); ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=5, help="stop after this many rounds without a better stop-set AUC (needs STOPFRAC>0)")
    a = ap.parse_args()
    out = f"llm_runs/{a.tag}"; os.makedirs(out, exist_ok=True)
    spec = json.load(open(a.start)); json.dump(spec, open(f"{out}/r0.json", "w"), indent=1)
    d = diag_text(spec); hist = [(0, edit_auc(d), spec.get("note", "start"))]
    stops = [H.stop_auc(spec)]                    # held-back AUC per round, never shown to the model
    log = open(f"{out}/log.txt", "w")
    for rnd in range(1, a.rounds + 1):
        msg = (f"Round {rnd}. Current spec:\n```json\n{json.dumps(spec, indent=1)}\n```\n\nDiagnostics on the edit set:\n{d}\n"
               "History (round, edit-set AUC, note):\n" + "\n".join(f"{r}: {auc:.4f}  {n}" for r, auc, n in hist))
        for attempt in range(3):
            t0 = time.time()
            reply = chat(a.url, a.model, [dict(role="system", content=SYSTEM), dict(role="user", content=msg)], a.temperature)
            try:
                new = parse_spec(reply); break
            except Exception as e:                     # ask again with the error, and keep the bad answer for inspection
                log.write(f"=== round {rnd} attempt {attempt}: unusable answer ({e})\n{reply[-4000:]}\n\n"); log.flush()
                msg += f"\n\nYour last answer could not be used ({str(e)[:300]}). Return one valid JSON spec in a ```json block."
        else:
            print("round", rnd, "failed to produce a valid spec"); break
        spec = new; json.dump(spec, open(f"{out}/r{rnd}.json", "w"), indent=1)
        d = diag_text(spec); hist.append((rnd, edit_auc(d), spec.get("note", ""))); stops.append(H.stop_auc(spec))
        log.write(f"=== round {rnd} ({time.time() - t0:.0f}s)\n{reply}\n\n{d}\n"); log.flush()
        print(f"round {rnd}: edit AUC {hist[-1][1]:.4f}  stop AUC {stops[-1]:.4f}  {hist[-1][2][:80]}", flush=True)
        if H.STOPFRAC > 0 and rnd - int(max(range(len(stops)), key=lambda i: stops[i])) >= a.patience:
            print("stop-set AUC has not improved for", a.patience, "rounds"); break
    if H.STOPFRAC > 0:
        chosen, rule = int(max(range(len(stops)), key=lambda i: stops[i])), "best held-back stop-set AUC"
    else:
        chosen, rule = max(hist, key=lambda h: h[1])[0], "best edit-set AUC"
    json.dump(dict(history=hist, stop_auc=stops, chosen=chosen, rule=rule), open(f"{out}/history.json", "w"), indent=1)
    print("chosen round:", chosen, f"({rule})")


if __name__ == "__main__":
    main()
