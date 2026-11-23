"""Symbolic-regression baselines on the law instances, same rows and same scoring as the reader.
  pysr    the paper's PySR setup (laws/sr_baseline.py: family-specific unary operators, 5 min, multithreaded)
  operon  SRBench's Operon estimator (srbench/experiment/methods/operon/regressor.py), unchanged
  llmsr   LLM-SR, official searcher from LLM-SRBench (methods/llmsr), backbone via CBORG, 1000 sampled equations
usage: python sr_laws.py <method> <set> <N> <seed> [--law LAW] [--minutes 5] [--llm gpt-4o-mini]
"""
import argparse
import copy
import importlib.util
import os
import sys
import time

import numpy as np

import s3

SR = os.environ.get("SR_DIR", "/path/to/scratch/wsp_bench/sr")


def fit_pysr(fam, X, y, minutes, seed):
    from pysr import PySRRegressor
    import inspect
    un = {"kin": ["square"], "cons": ["square", "abs"], "decay": ["exp", "log", "square"]}[fam]
    kw = dict(niterations=10000, timeout_in_seconds=int(60 * minutes), maxsize=40, populations=30,
              binary_operators=["+", "-", "*", "/"], unary_operators=un, model_selection="best",
              random_state=seed, deterministic=False, parallelism="multithreading", progress=False, verbosity=0,
              temp_equation_file=True)
    ok = inspect.signature(PySRRegressor.__init__).parameters
    m = PySRRegressor(**{k: v for k, v in kw.items() if k in ok})
    m.fit(X, y)
    return m.predict, str(m.sympy())


def fit_operon(fam, X, y, minutes, seed):
    spec = importlib.util.spec_from_file_location("srb_operon", f"{SR}/srbench/experiment/methods/operon/regressor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    est = copy.deepcopy(mod.est)
    est.random_state = seed
    if os.environ.get("OPERON_MAX_TIME"):  # smoke tests only; full runs keep SRBench's 3600 s
        t = int(os.environ["OPERON_MAX_TIME"])
        est.max_time = t
        est.fixed_params = dict(est.fixed_params, max_time=max(5, t // 4))
    est.fit(X, y)
    return est.predict, est.model


def fit_llmsr(fam, X, y, feat, llm, seed, n_samples=1000, log_dir=None):
    sys.path.insert(0, f"{SR}/llm-srbench")
    sys.path.insert(0, f"{SR}/llm-srbench/methods")
    from bench.dataclasses import SEDTask
    import methods.llmsr.searcher as srch_mod
    from methods.llmsr.searcher import LLMSRSearcher
    from methods.llmsr import config, sampler, profile
    # Adaptations for our laws (documented in README): LLM-SR fits at most MAX_NPARAMS = 10 constants, fewer than its own
    # starting skeleton needs for 12 (kin) or 20 (cons) inputs; and its TensorBoard text log crashes on a failed sample.
    names, descs, out_desc = s3.symbols(fam, feat)
    task = SEDTask(name=f"law_{fam}", symbols=["y"] + names, symbol_descs=[out_desc] + descs,
                   symbol_properties=["O"] + ["V"] * len(names), samples=np.column_stack([y, X]))
    return fit_llmsr_task(task, X.shape[1], llm, seed, n_samples, log_dir)


def fit_llmsr_task(task, n_inputs, llm, seed, n_samples, log_dir):
    """LLM-SR on any LLM-SRBench SEDTask (our laws or a benchmark problem). Same adaptations as above; for <= 3 inputs
    the parameter cap stays at the published 10."""
    sys.path.insert(0, f"{SR}/llm-srbench")
    sys.path.insert(0, f"{SR}/llm-srbench/methods")
    import methods.llmsr.searcher as srch_mod
    from methods.llmsr.searcher import LLMSRSearcher
    from methods.llmsr import config, sampler, profile
    npar = max(10, 2 * n_inputs + 4)
    for k in ("eval_spec", "optimization_spec", "execution_spec"):
        setattr(srch_mod, k, getattr(srch_mod, k).replace("MAX_NPARAMS = 10", f"MAX_NPARAMS = {npar}"))
    for name in dir(profile):
        obj = getattr(profile, name)
        if isinstance(obj, type) and hasattr(obj, "_write_tensorboard"):
            obj._write_tensorboard = lambda self: None
    from openai import AsyncOpenAI
    cfg = config.Config(experience_buffer=config.ExperienceBufferConfig(num_islands=10), use_api=True, api_model=llm,
                        samples_per_prompt=4)

    client = AsyncOpenAI(base_url=os.environ.get("LLM_URL", "https://api.cborg.lbl.gov") + "/v1/",
                         api_key=os.environ["CBORG_API_KEY"], timeout=300)
    reasoning = any(k in llm for k in ("gpt-oss", "qwen", "glm", "gemma-4-thinking"))

    class _Completions:  # LLM-SR asks for max_tokens=1024; a reasoning model spends that on thinking and returns no text
        async def create(self, **kw):
            if reasoning:
                kw["max_tokens"] = 6000
                kw["extra_body"] = {"reasoning_effort": "low"}
            r = await client.chat.completions.create(**kw)
            if r.choices[0].message.content is None:
                r.choices[0].message.content = ""
            return r

    class _Client:
        chat = type("C", (), {"completions": _Completions()})()

    def make(samples_per_prompt):
        s = sampler.LocalLLM(samples_per_prompt=samples_per_prompt, api_url="https://api.openai.com/v1/", api_key="x")
        s.client = _Client()
        return s
    os.makedirs(log_dir, exist_ok=True)
    np.random.seed(seed)
    srch = LLMSRSearcher(f"llmsr_{llm}", cfg, make, global_max_sample_num=n_samples, log_path=log_dir)
    res = srch.discover(task)
    eq = res[0].equation if isinstance(res, list) else res.equation if hasattr(res, "equation") else res
    return (lambda Z: np.asarray(eq.lambda_format(np.asarray(Z, float)), float)), eq.program_format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["pysr", "operon", "llmsr"])
    ap.add_argument("set")
    ap.add_argument("N", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--law", default=None)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--llm", default="gpt-4o-mini")
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args()
    method = a.method if a.method != "llmsr" else "llmsr_" + a.llm.replace("/", "-")
    for law in ([a.law] if a.law else s3.laws_in(a.set)):
        if s3.done(method, a.set, law, a.N, a.seed):
            continue
        fam = law.split("_")[0]
        d = s3.load(a.set, law)
        r = s3.rows(a.N, a.seed)
        X, y = d["Xtr"][r], d["ytr"][r]
        t0 = time.time()
        try:
            if a.method == "pysr":
                f, expr = fit_pysr(fam, X, y, a.minutes, a.seed)
            elif a.method == "operon":
                f, expr = fit_operon(fam, X, y, a.minutes, a.seed)
            else:
                f, expr = fit_llmsr(fam, X, y, [str(x) for x in d["feat"]], a.llm, a.seed, a.samples,
                                    log_dir=f"{s3.ROOT}/logs/{method}/{a.set}__{law}__n{a.N}__s{a.seed}")
            res = s3.score(fam, f, d)
        except Exception as e:
            expr, res = None, {"id": float("inf"), "ood": float("inf"), "recovered": False, "error": repr(e)[:300]}
        s3.save(method, a.set, law, a.N, a.seed, dict(set=a.set, law=law, fam=fam, N=a.N, seed=a.seed, method=method,
                                                      expr=expr, seconds=time.time() - t0, **res))
        print(f"{method} {a.set} {law} N={a.N} s{a.seed}: ood={res['ood']:.3g} recovered={res['recovered']} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
