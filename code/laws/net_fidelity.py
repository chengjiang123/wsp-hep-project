"""Does the trained law-shaped network itself capture the law? Each network is used as a predictor and scored with the
shared rule (s3.score: family threshold, outside the training range). This is the reader-independent filter for the
coverage subset: a reader can only read what is in the weights, so instances whose network fails are out of scope for
every method alike. Results: results/network/<set>__<law>__n<N>__s<seed>.json
usage: python net_fidelity.py"""
import json

import nets
import s3

n = 0
for st in s3.SETS:
    for law in s3.laws_in(st):
        fam, d = law.split("_")[0], s3.load(st, law)
        for N in s3.NS:
            for seed in (0, 1, 2):
                net = json.load(open(f"{s3.ROOT}/sets/{st}/nets/{law}_n{N}_s{seed}.json"))
                r = s3.score(fam, lambda X: nets.predict(net, X), d)
                s3.save("network", st, law, N, seed, dict(set=st, law=law, fam=fam, N=N, seed=seed, method="network", **r))
                n += 1
print(n, "networks scored")
