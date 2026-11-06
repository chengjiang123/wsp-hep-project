"""Write task for the automated benchmark: an LLM writes the K0S network from physics alone (prompts/k0s_write.txt,
no data, no diagnostics), then the written network is scored on the test set as it is, and as start + prior for
training on N labelled candidates (prior width 0.3 around the written weights, strength 1/max(N, 1000), as the
"written, prior max(N,N0)" net of the paper).

usage (run from the K0S data folder, e.g. v0_k0s/big, with the same env as the edit loop):
  python write_k0s.py --model <id> --tag <name>          ask the model, then score  -> llm_runs/write_<tag>/
  python write_k0s.py --spec hand/r0.json --tag ref      score an existing spec (the interactively written one)
  python write_k0s.py --baselines --tag base             random-start network (16 units) and boosted trees, same N and seeds
env: NS_WRITE (default 100,1000,10000), SEEDS_WRITE (default 3), NTEST_S, NTEST_B, DATA, LLM_URL, CBORG_API_KEY
"""
import argparse, json, os, time, numpy as np, torch
from sklearn.metrics import roc_auc_score
import models_v2 as V
import handedit as H
from improve import load, sample, rej, gbm
from llm_loop import chat

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
HERE = os.path.dirname(os.path.realpath(__file__))
NS = [int(x) for x in os.environ.get("NS_WRITE", "100,1000,10000").split(",")]
SEEDS = int(os.environ.get("SEEDS_WRITE", "3"))


def parse(text):
    import re
    m = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{.*\})", text, re.S)
    spec = json.loads(m[-1])
    units = spec.get("units")
    assert isinstance(units, list) and 1 <= len(units) <= 30, "units must be a list of 1-30 cut units"
    for u in units:
        assert "cut" in u and {"f", "t", "dir", "k", "w"} <= set(u), f"bad unit {u}"
        u["f"] = int(u["f"]); u["t"] = float(u["t"]); u["k"] = float(u["k"]); u["w"] = float(u["w"])
        assert 0 <= u["f"] < 8 and u["dir"] in (">", "<"), f"bad unit {u}"
        assert u["f"] == 6 or u["t"] > 0, f"threshold must be positive: {u}"
        assert u["f"] != 1 or u["t"] < 1, f"cos(pointing angle) threshold must be below 1: {u}"
    spec["bias"] = float(spec.get("bias", 0.0))
    H.build(spec)
    return spec


def score(spec, out, tag):
    D, tr, te = load()
    yte = D["y"][te].astype(float); Xte = V.phys(D["F"][te], "log")
    res = []

    def rec(name, n, seed, p):
        r = dict(model=name, tag=tag, n=n, seed=seed, auc=float(roc_auc_score(yte, p)), rej90=float(rej(yte, p, 0.9)), rej95=float(rej(yte, p, 0.95)))
        res.append(r); print(f"{name:34s} N={n:6d} seed {seed}: AUC {r['auc']:.5f}  R90 {r['rej90']:8.1f}  R95 {r['rej95']:7.1f}", flush=True)
    if spec is not None:
        rec("written, as is", 0, 0, V.predict(H.build(spec), Xte))
    for n in NS:
        for seed in range(SEEDS):
            idx, y = sample(tr, n, 1000 * seed + n)
            X = V.phys(D["F"][idx], "log")
            if spec is not None:
                net = H.build(spec); mu = [p.detach().clone() for p in net.parameters()]
                net = V.train(net, X, y, mu=mu, sig=0.3, mode="max")
                rec("written, trained with prior", n, seed, V.predict(net, Xte))
            else:
                net = V.train(V.Net("random", space="log", H=16, seed=seed), X, y)
                rec("random start, 16 units", n, seed, V.predict(net, Xte))
                rec("boosted trees", n, seed, gbm(D["F"][idx], y, seed, Fte=D["F"][te]))
    json.dump(res, open(os.path.join(out, "score.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--tag", required=True); ap.add_argument("--spec"); ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--url", default=os.environ.get("LLM_URL", "https://api.cborg.lbl.gov"))
    ap.add_argument("--temperature", type=float, default=0.3)
    a = ap.parse_args()
    out = f"llm_runs/write_{a.tag}"; os.makedirs(out, exist_ok=True)
    if a.baselines:
        return score(None, out, a.tag)
    if a.spec:
        spec = json.load(open(a.spec))
    else:
        prompt = open(os.path.join(HERE, "prompts", "k0s_write.txt")).read()
        msgs = [dict(role="user", content=prompt)]; log = open(f"{out}/log.txt", "w")
        for attempt in range(3):
            t0 = time.time(); reply = chat(a.url, a.model, msgs, a.temperature)
            log.write(f"=== attempt {attempt} ({time.time() - t0:.0f}s)\n{reply}\n\n"); log.flush()
            try:
                spec = parse(reply); break
            except Exception as e:
                msgs = msgs[:1] + [dict(role="assistant", content=reply[-6000:]),
                                   dict(role="user", content=f"Your answer could not be used ({str(e)[:300]}). Return one valid JSON object in a ```json block.")]
        else:
            json.dump(dict(error="no valid spec after 3 attempts"), open(f"{out}/score.json", "w")); raise SystemExit("no valid spec")
    json.dump(spec, open(f"{out}/spec.json", "w"), indent=1)
    score(spec, out, a.tag)


if __name__ == "__main__":
    main()
