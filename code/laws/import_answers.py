"""Score readings made outside reader.py (e.g. Claude subagents answering the prompts of build_reader_prompts.py) with
exactly reader.py's rule: the shared out-of-range recovery verdict (s3.score) and the paper's exact-match verdict.
usage: python import_answers.py --tag opus55sub --model "claude-opus-5-5 (subagent)" answers1.json [answers2.json ...]
answers: {"<set>__<law>__n<N>__s<seed>": {answer object in the family's format}, ...}
Results: results/reader_<tag>/<set>__<law>__n<N>__s<seed>.json, same fields as reader.py (raw = the answer as given)."""
import argparse
import json

import reader as R
import s3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    method = f"reader_{a.tag}"
    n = 0
    for f in a.answers:
        for key, given in json.load(open(f)).items():
            st, law, nn, ss = key.split("__")
            N, seed, fam = int(nn[1:]), int(ss[1:]), law.split("_")[0]
            raw = given if isinstance(given, str) else json.dumps(given)
            try:
                ans = R.extract(raw, R.KEYS[fam])
            except ValueError as e:
                ans = {"error": str(e)[:300]}
            d, t = s3.load(st, law), s3.truth(st, law)
            if "error" in ans:
                r, exact = {"id": float("inf"), "ood": float("inf"), "recovered": False}, False
            else:
                try:
                    r = s3.score(fam, s3.answer_predictor(fam, ans), d)
                except Exception as e:
                    r = {"id": float("inf"), "ood": float("inf"), "recovered": False, "error": str(e)[:200]}
                exact = s3.symbolic_match(fam, ans, t)
            s3.save(method, st, law, N, seed, dict(set=st, law=law, fam=fam, N=N, seed=seed, method=method, model=a.model,
                                                   answer=ans, exact=exact, raw=[raw], usage=[], seconds=None, **r))
            n += 1
    print(n, "readings scored ->", method)


if __name__ == "__main__":
    main()
