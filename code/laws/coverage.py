"""Coverage table: each method's recovery on all instances and on the instances where the trained network itself
passes the shared out-of-range rule (results/network, from net_fidelity.py).
usage: python coverage.py [--light]   (--light: seed 0, N = 100, 1000, 10000, the instances the Claude readers read)"""
import collections
import glob
import json
import sys

import s3

R = f"{s3.ROOT}/results"
METHODS = ["reader_gptoss", "reader_sonnet55sub", "reader_opus55sub", "pysr", "operon"]


def load(m):
    out = {}
    for f in glob.glob(f"{R}/{m}/*.json"):
        d = json.load(open(f))
        out[(d["set"], d["law"], d["N"], d["seed"])] = d
    return out


net = load("network")
T = collections.defaultdict(lambda: [0, 0])
for k, d in net.items():
    T[(d["fam"], k[2])][0] += 1
    T[(d["fam"], k[2])][1] += d["recovered"]
print("network itself passes (family, N):", {f"{f}@{N}": f"{v[1]}/{v[0]}" for (f, N), v in sorted(T.items())})
keys = {k for k in net if k[3] == 0 and k[2] in (100, 1000, 10000)} if "--light" in sys.argv else set(net)
print(f"\n{len(keys)} instances. method: all | network passes | by family where network passes")
for m in METHODS:
    D = load(m)
    ks = [k for k in keys if k in D]
    ok = [k for k in ks if net[k]["recovered"]]
    F = collections.defaultdict(lambda: [0, 0])
    for k in ok:
        F[D[k]["fam"]][0] += 1
        F[D[k]["fam"]][1] += D[k]["recovered"]
    a = sum(D[k]["recovered"] for k in ks)
    b = sum(D[k]["recovered"] for k in ok)
    print(f"  {m:20s} {a}/{len(ks)} | {b}/{len(ok)} ({100 * b / max(len(ok), 1):.0f}%) |",
          {f: f"{v[1]}/{v[0]}" for f, v in sorted(F.items())})
