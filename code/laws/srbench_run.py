"""SRBench ground-truth track, rerun with SRBench's own code (experiment/evaluate_model.py + assess_symbolic_model.py).
  python srbench_run.py fetch                         133 Feynman + Strogatz datasets from PMLB (+ metadata.yaml with the formula)
  python srbench_run.py tasks <method> > list.txt     one line per (dataset, seed, noise): the 10 SRBench seeds x 4 noise levels
  python srbench_run.py run <method> <dataset> <seed> <noise>
  python srbench_run.py assess <method>               symbolic assessment of every result (SRBench verifier)
Methods: operon2021  the published (2021 ground-truth table) Operon: tuned/params/_operonregressor.py + tuned overrides,
                     rebuilt on the current pyoperon (renamed parameters mapped, see OPERON_MAP)
         pysr        SRBench's current PySR config (experiment/methods/pysr/regressor.py), see estimator() for run-level changes
         reader_gptoss  ours on the 35 monomial laws (monomials()): the LLM reads a fitted multiplicative unit
         rounding       the same unit read by a fixed rule, no LLM (control)
Results: $S3_ROOT/srbench/results/<method>/<dataset>/<dataset>_<method>_<seed>[_target-noiseX].json (SRBench layout)
"""
import glob
import importlib.util
import os
import subprocess
import sys
import urllib.request

ROOT = os.environ.get("S3_ROOT", "/path/to/scratch/wsp_bench/story3")
SRB = os.environ.get("SRBENCH", "/path/to/scratch/wsp_bench/sr/srbench")
DATA = f"{ROOT}/srbench/pmlb"
RES = f"{ROOT}/srbench/results"
NOISE = [0.0, 0.001, 0.01, 0.1]
sys.path.insert(0, f"{SRB}/experiment")


def names():
    """The 130 ground-truth datasets of the published table (cached in a text file, so any env can read it)."""
    cache = f"{ROOT}/srbench/names.txt"
    if os.path.exists(cache):
        return open(cache).read().split()
    import pandas as pd
    d = pd.read_feather(f"{SRB}/results/ground-truth_results.feather")
    n = sorted(d.dataset.unique())
    open(cache, "w").write("\n".join(n) + "\n")
    return n


def fetch():
    """Straight from the PMLB GitHub repository (the pip package no longer lists the Feynman/Strogatz sets)."""
    base = "https://github.com/EpistasisLab/pmlb/raw/master/datasets"
    for n in names():
        os.makedirs(f"{DATA}/{n}", exist_ok=True)
        if not os.path.exists(f"{DATA}/{n}/{n}.tsv.gz"):
            urllib.request.urlretrieve(f"{base}/{n}/{n}.tsv.gz", f"{DATA}/{n}/{n}.tsv.gz.part")
            os.replace(f"{DATA}/{n}/{n}.tsv.gz.part", f"{DATA}/{n}/{n}.tsv.gz")
        if not os.path.exists(f"{DATA}/{n}/metadata.yaml"):
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/EpistasisLab/pmlb/master/datasets/{n}/metadata.yaml",
                                       f"{DATA}/{n}/metadata.yaml")
    print(len(glob.glob(f"{DATA}/*/*.tsv.gz")), "datasets in", DATA)


OPERON_MAP = {"local_iterations": "optimizer_iterations", "time_limit": "max_time"}  # 2021 name -> current pyoperon name


def estimator(method):
    if method in MONO:
        return mono_estimator(method)
    if method == "pysr":
        spec = importlib.util.spec_from_file_location("srb_pysr", f"{SRB}/experiment/methods/pysr/regressor.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # SRBench sets turbo=True; with the current SymbolicRegression.jl the LoopVectorization fallback lets sin(+-Inf)
        # throw a DomainError that kills the run (4719 of 5200 runs). turbo only changes evaluation speed, not the search.
        mod.est.set_params(turbo=False)
        # PySR rejects feature names that are sympy names (Feynman uses I for current, also E, S, N, Q, ...): fit on
        # neutral names x0..xk and map the formula back to the dataset's names for SRBench's symbolic verifier.
        import copy
        import numpy as np
        import sympy
        from sklearn.base import BaseEstimator, RegressorMixin
        base = mod.est
        if os.environ.get("SRB_SMOKE"):  # smoke test only: tiny budget
            base.set_params(timeout_in_seconds=60, niterations=3)
        if os.environ.get("SRB_SERIAL"):
            # SRBench's procs=1 "multiprocessing" is one Julia worker evolving both populations plus a Python/Julia head
            # that busy-polls it; with 128 fits per node the head never stops the worker at the timeout. Serial mode runs
            # the same search (same populations, cycles, operators, budget) in the one process.
            for k, v in (("multithreading", None), ("procs", None), ("parallelism", "serial")):  # the old names are kept as plain attributes
                setattr(base, k, v)
        if os.environ.get("SRB_TIMEOUT"):  # debugging only: shorter wall-clock budget, search unchanged
            base.set_params(timeout_in_seconds=int(os.environ["SRB_TIMEOUT"]))

        class PySRWrap(BaseEstimator, RegressorMixin):
            def __init__(self, random_state=None):
                self.random_state = random_state

            def fit(self, X, y):
                self.cols_ = X.columns.tolist() if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
                self.est_ = copy.deepcopy(base)
                if self.random_state is not None:
                    self.est_.set_params(random_state=self.random_state)
                self.est_.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).ravel(),
                              variable_names=[f"x{i}" for i in range(len(self.cols_))])
                return self

            def predict(self, X):
                return self.est_.predict(np.asarray(X, dtype=np.float64))

        def model(w, X=None):
            e = w.est_.sympy()
            return str(e.subs({sympy.Symbol(f"x{i}"): sympy.Symbol(c) for i, c in enumerate(w.cols_)}, simultaneous=True))
        mod.est, mod.model = PySRWrap(), model
        return mod.est, mod.model, mod
    if method == "operon2021":
        from pyoperon.sklearn import SymbolicRegressor
        import inspect
        spec = importlib.util.spec_from_file_location("p", f"{SRB}/experiment/methods/tuned/params/_operonregressor.py")
        p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p)
        params = dict(p.params)
        params.update(allowed_symbols="add,sub,mul,div,exp,log,sin,cos,sqrt,square,constant,variable",
                      max_evaluations=1_000_000, generations=100_000, time_limit=8 * 60 * 60)   # the tuned/ overrides
        if os.environ.get("SRB_SMOKE"):
            params["max_evaluations"] = 20000
        ok = inspect.signature(SymbolicRegressor.__init__).parameters
        kw, dropped = {}, []
        for k, v in params.items():
            k2 = OPERON_MAP.get(k, k)
            (kw.__setitem__(k2, v) if k2 in ok else dropped.append(k))
        if dropped:
            print("operon2021: parameters not in current pyoperon, left at default:", dropped)
        import numpy as np
        from sklearn.base import BaseEstimator, RegressorMixin

        class OperonWrap(BaseEstimator, RegressorMixin):
            """current pyoperon wants C-contiguous float64 arrays; SRBench passes scaled arrays / frames"""
            def __init__(self, random_state=None):
                self.random_state = random_state

            def fit(self, X, y):
                self.cols_ = X.columns.tolist() if hasattr(X, "columns") else None
                self.est_ = SymbolicRegressor(**dict(kw, random_state=self.random_state if self.random_state is not None else kw.get("random_state")))
                self.est_.fit(np.ascontiguousarray(np.asarray(X, dtype=np.float64)), np.asarray(y, dtype=np.float64).ravel())
                return self

            def predict(self, X):
                return self.est_.predict(np.ascontiguousarray(np.asarray(X, dtype=np.float64)))

        est = OperonWrap()

        class M:
            pass
        mod = M()

        def model(w, X=None):
            e = w.est_
            names_ = X.columns.tolist() if X is not None and hasattr(X, "columns") else (w.cols_ or list(e.variables_.values()))
            return e.get_model_string(e.model_, 6, names_).replace("^", "**")
        mod.est, mod.model = est, model
        return est, model, mod
    raise ValueError(method)


MONO = ("reader_gptoss", "rounding")


def monomials():
    """The ground-truth laws one multiplicative unit can express, fixed by rule from the metadata formula before any run:
    y = C * prod_i x_i^p_i with C > 0 a number and every p_i rational. 35 of the 130; all have positive inputs."""
    cache = f"{ROOT}/srbench/monomials.txt"
    if os.path.exists(cache):
        return open(cache).read().split()
    import pandas as pd
    import sympy as sp
    from symbolic_utils import get_sym_model
    keep = []
    for n in names():
        ds = f"{DATA}/{n}/{n}.tsv.gz"
        feats = [c for c in pd.read_csv(ds, sep="\t", nrows=5).columns if c != "target"]
        e = sp.sympify(get_sym_model(ds), locals={f: sp.Symbol(f) for f in feats})
        ok = True
        for fac in sp.Mul.make_args(e):
            if fac.is_number:
                ok &= bool(float(fac) > 0)
            else:
                b, p = fac.as_base_exp()
                ok &= bool(b.is_Symbol and p.is_Rational)
        if ok:
            keep.append(n)
    open(cache, "w").write("\n".join(keep) + "\n")
    return keep


def fit_unit(X, y):
    """The law-shaped network for this subset, one multiplicative unit y = exp(b) * prod_i x_i^w_i: log-linear least
    squares on the positive targets as a start, then least squares on y itself (the noise is added to y)."""
    import numpy as np
    from scipy.optimize import least_squares
    L = np.log(X)
    m = y > 0
    th = np.linalg.lstsq(np.c_[L[m], np.ones(m.sum())], np.log(y[m]), rcond=None)[0]
    s = np.sqrt(np.mean(y ** 2))
    r = least_squares(lambda t: (np.exp(np.clip(L @ t[:-1] + t[-1], -700, 700)) - y) / s, th)
    return r.x[:-1], float(r.x[-1])


UNIT_PROMPT = ("A single multiplicative unit y = exp(b) * x1^w1 * x2^w2 * ... * x{k}^w{k} was fitted by least squares to "
               "noisy measurements of an unknown physical law of the positive inputs x1..x{k}. Fitted weights:\n"
               "w = {w}\nb = {b:.6f}   (exp(b) = {eb:.6g})\n\n"
               "State the law. The exponents are simple rational numbers; the constant is a simple closed form (a rational "
               "number, possibly times a power of pi). Answer JSON: {{\"exp\": [exponent of x1, ..., exponent of x{k}], "
               "\"C\": \"<the constant in python syntax, e.g. 1/(4*pi)>\", \"law\": \"<formula>\"}}")


def closed_form(c, tol=0.02, qmax=12, pmax=48, kmax=2):
    """The rule-based reading of a constant: among r * pi^k (r = p/q, p <= pmax, q <= qmax, |k| <= kmax) within a relative
    tol of c, the simplest one (smallest p + q + |k|, then closest); if none is that close, the closest. The same family
    the reader is told about."""
    import math
    import sympy as sp
    best = None
    for k in range(-kmax, kmax + 1):
        for q in range(1, qmax + 1):
            for p in range(1, pmax + 1):
                if math.gcd(p, q) != 1:
                    continue
                d = abs(math.log(c) - math.log(p / q * math.pi ** k))
                key = (0, p + q + abs(k), d) if d <= math.log1p(tol) else (1, d, 0)
                if best is None or key < best[0]:
                    best = (key, sp.Rational(p, q) * sp.pi ** k)
    return best[1]


def mono_estimator(method):
    """reader_gptoss: the LLM reads the fitted unit (neutral names x1..xk, so it cannot recognise a Feynman law by its
    variable names) and states exponents and constant. rounding: the same unit, no LLM: exponents to the nearest fraction
    with denominator <= 3, constant by closed_form() (nearest r * pi^k). Both go through evaluate_model and the
    SRBench verifier like every other method."""
    import json as _json
    from fractions import Fraction
    import numpy as np
    import sympy as sp
    from sklearn.base import BaseEstimator, RegressorMixin

    class MonoWrap(BaseEstimator, RegressorMixin):
        def __init__(self, random_state=None):
            self.random_state = random_state

        def fit(self, X, y):
            self.cols_ = X.columns.tolist() if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
            X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).ravel()
            self.w_, self.b_ = fit_unit(X, y)
            k, self.raw_ = len(self.w_), []
            try:
                if method == "rounding":
                    p = [sp.Rational(Fraction(float(v)).limit_denominator(3)) for v in self.w_]
                    C = closed_form(float(np.exp(self.b_)))
                    self.answer_ = {"exp": [str(v) for v in p], "C": str(C)}
                else:
                    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                    import reader as rd
                    prompt = UNIT_PROMPT.format(k=k, w=np.array2string(self.w_, precision=4, separator=", "), b=self.b_,
                                                eb=float(np.exp(self.b_)))
                    txt, _ = rd.ask(os.environ.get("LLM_MODEL", "lbl/gpt-oss-120b"), prompt)
                    self.raw_.append(txt)
                    try:
                        ans = rd.extract(txt, ("exp", "C"))
                    except ValueError as e:
                        txt, _ = rd.ask(os.environ.get("LLM_MODEL", "lbl/gpt-oss-120b"), prompt,
                                        [{"role": "assistant", "content": txt[-6000:]}, {"role": "user", "content":
                                         f"Your answer could not be read ({e}). Reply with exactly one JSON object in the format asked."}])
                        self.raw_.append(txt)
                        ans = rd.extract(txt, ("exp", "C"))
                    self.answer_ = ans
                    p = [sp.nsimplify(str(v), rational=True) for v in ans["exp"]]
                    C = sp.sympify(str(ans["C"]), locals={"pi": sp.pi})
                    assert len(p) == k and C.is_number
            except Exception as e:   # an unreadable answer is a failed recovery, kept in the record
                self.answer_ = getattr(self, "answer_", {}) or {}
                self.answer_["error"] = str(e)[:300]
                p, C = [sp.Integer(0)] * k, sp.Integer(0)
            self.C_, self.p_ = C, p
            xs = [sp.Symbol(f"x{i}") for i in range(k)]
            self.expr_ = sp.Float(sp.N(C, 15)) * sp.Mul(*[x ** q for x, q in zip(xs, p)])
            self.f_ = sp.lambdify(xs, self.expr_, "numpy")
            return self

        def predict(self, X):
            X = np.asarray(X, dtype=np.float64)
            return np.broadcast_to(np.asarray(self.f_(*X.T), dtype=np.float64), (len(X),)).copy()

        def side(self):
            return _json.dumps({"names": self.cols_, "w": list(map(float, self.w_)), "b": self.b_, "answer": self.answer_,
                                "exp": [str(q) for q in self.p_], "C": str(self.C_), "C_value": float(sp.N(self.C_)),
                                "raw": self.raw_})

    class M:
        pass
    mod = M()

    def model(w, X=None):
        return str(w.expr_.subs({sp.Symbol(f"x{i}"): sp.Symbol(c) for i, c in enumerate(w.cols_)}, simultaneous=True))
    mod.est, mod.model = MonoWrap(), model
    return mod.est, model, mod


def tasks(method, nseeds=10):
    """nseeds: the first nseeds SRBench seeds. Seed-major order, so the first seeds of every dataset finish first."""
    from seeds import SEEDS
    for s in SEEDS[:int(nseeds)]:
        for n in (monomials() if method in MONO else names()):
            for tn in NOISE:
                print(n, s, tn)


FIT_TIME = {"pysr": 3600, "operon2021": 8 * 3600, "reader_gptoss": 3600, "rounding": 3600}   # evaluate_model's alarm; each method's own budget stops it first


def run(method, n, seed, noise):
    """The published ground-truth protocol (SRBench v2.0 experiment/evaluate_model.py, the code behind the 2021 table):
    with -sym_data it sets scale_x = scale_y = False, and it trains on 10,000 rows drawn WITH replacement
    (np.random.choice right after np.random.seed(random_state)) from the 75 % split. The current SRBench master dropped the
    scaling switch and draws without replacement; both are restored here, so the training rows equal the published ones."""
    os.chdir(f"{SRB}/experiment")
    import numpy as np
    out = f"{RES}/{method}/noise{float(noise)}/{n}/"   # SRBench names results dataset_method_seed: one folder per noise level
    f = f"{out}{n}_{method}_{seed}.json"
    if os.path.exists(f):   # before importing anything heavy, so resubmitting is cheap
        return
    os.makedirs(out, exist_ok=True)
    from evaluate_model import evaluate_model
    est, model, mod = estimator(method)
    choice = np.random.choice

    def choice_v20(a, size=None, replace=True, p=None):   # first call only: the training subsample
        np.random.choice = choice
        return choice(a, size=size, replace=True, p=p)
    np.random.choice = choice_v20
    evaluate_model(dataset=f"{DATA}/{n}/{n}.tsv.gz", results_path=out, random_state=int(seed), est_name=method, est=est,
                   model=model, algorithm=mod, target_noise=float(noise), sym_data=True, scale_x=False, scale_y=False,
                   max_train_samples=10000, fit_time_limit=FIT_TIME[method])
    if hasattr(est, "side"):   # the fitted unit and the reader's answer (".side", so assess() does not pick it up)
        open(f + ".side", "w").write(est.side())


def assess(method):
    os.chdir(f"{SRB}/experiment")
    for f in sorted(glob.glob(f"{RES}/{method}/noise*/*/*.json")):
        if os.path.exists(f + ".updated"):
            continue
        n = f.split("/")[-2]
        subprocess.run([sys.executable, "assess_symbolic_model.py", f"{DATA}/{n}/{n}.tsv.gz", "-json_file", f, "-sym_data"], check=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fetch":
        fetch()
    elif cmd == "tasks":
        tasks(sys.argv[2], *(sys.argv[3:4]))
    elif cmd == "run":
        run(*sys.argv[2:6])
    elif cmd == "assess":
        assess(sys.argv[2])
