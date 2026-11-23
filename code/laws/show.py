"""What the reader sees: one network's weights, plus the quadratic form they define (pure algebra on the weights, no data).
usage: python show.py nets/<file>.json"""
import json, sys, numpy as np

np.set_printoptions(linewidth=260, suppress=True, precision=3)
n = json.load(open(sys.argv[1])); W = np.array(n["W"]); a = np.array(n["a"]); f = n["feat"]
print("architecture:", n["arch"]); print("features:", f)
print("a =", a);
if "b" in n:
    print("b =", np.array(n["b"]))
if n["fam"] == "decay":
    print("W =\n", W)
    print("per unit: log(a_j) + b_j =", np.log(a) + np.array(n["b"]))
    sys.exit()
Q = (W.T * a) @ W; s = np.abs(Q).max()
print(f"Q = W^T diag(a) W, divided by {s:.4g}:")
print("        " + " ".join(f"{x:>6s}" for x in f))
for i, row in enumerate(Q / s):
    print(f"{f[i]:>7s} " + " ".join(f"{x:6.2f}" for x in row))
ev, V = np.linalg.eigh(Q)
keep = np.abs(ev) > 0.02 * np.abs(ev).max()
print("eigenvalues / max:", np.round(ev / np.abs(ev).max(), 3))
for e, v in zip(ev[keep], V[:, keep].T):
    v = v / np.abs(v).max()
    print(f"  eig {e / s:+.3f}:", " ".join(f"{f[i]}:{v[i]:+.2f}" for i in np.where(np.abs(v) > 0.05)[0]))
