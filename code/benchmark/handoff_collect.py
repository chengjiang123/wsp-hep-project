"""Collect every req<R>.json of a handoff tag into one JSON {key: {system, user}} (key = <ds>/<model>/N<N>/s<seed>/<arm>),
skipping loops that already have a reply for round R.   usage: python handoff_collect.py --tag _llm-opus55sub --round R out.json"""
import argparse
import glob
import json
import os

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--tag", required=True)
ap.add_argument("--round", type=int, required=True)
a = ap.parse_args()
base = f"{C.ROOT}/handoff{a.tag}/"
out = {}
for q in sorted(glob.glob(f"{base}*/*/*/*/*/req{a.round}.json")):
    H = os.path.dirname(q)
    if not os.path.exists(f"{H}/resp{a.round}.txt"):
        out[H[len(base):]] = json.load(open(q))
json.dump(out, open(a.out, "w"), indent=1)
print(len(out), "prompts ->", a.out)
