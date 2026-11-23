"""Physics laws in weight space: data for three law families.

  kin    kinematic quadratic invariants of 3 particles   y = sum_mu s_mu (sum_{i in S} p_i^mu)^2      (regression)
  cons   conservation laws in 2 -> 2 reactions            real reaction iff Delta(u . q) = 0 for every conserved u (classification)
  decay  decay-in-flight law                              P(decayed before L) = 1 - exp(-L^a p^b m^g / lam)  (classification)

usage:
  python gen.py real          -> data/real_*.npz, truth/real.json    (the true physics laws, named)
  python gen.py blind [seed]  -> data/blind_*.npz, truth/blind.json  (hidden random laws; nothing about them is printed)
Each npz: Xtr, ytr (pool of 20000 labelled examples), Xid, yid (in range), Xood, yood (outside the training range),
feature names; for decay also Pid, Pood (true probabilities, used only for scoring).
"""
import json, os, sys, itertools, zlib, numpy as np

os.makedirs("data", exist_ok=True); os.makedirs("truth", exist_ok=True)
MASS = np.array([0.13957, 0.49368, 0.93827])          # pi, K, p
COMP = ["E", "px", "py", "pz"]
KIN_FEAT = [f"{c}{i}" for i in (1, 2, 3) for c in COMP]

# ---------------------------------------------------------------- kinematics
def particles(r, n, ood):
    m = MASS[r.integers(0, 3, (n, 3))]
    p = r.uniform(3, 30, (n, 3)) if ood else np.minimum(r.exponential(1.0, (n, 3)), 3.0)
    ct = r.uniform(-1, 1, (n, 3)); ph = r.uniform(0, 2 * np.pi, (n, 3)); st = np.sqrt(1 - ct ** 2)
    P = np.stack([p * st * np.cos(ph), p * st * np.sin(ph), p * ct], -1)          # n,3,3
    return m, P

def four(m, P):
    E = np.sqrt((P ** 2).sum(-1) + m ** 2)
    return np.concatenate([E[..., None], P], -1)                                   # n,3,4

def kin_target(V, S, s):
    tot = V[:, list(S), :].sum(1)                                                  # n,4
    return (tot ** 2 * np.array(s)).sum(1)

def kin_data(r, law, n=20000):
    out = {}
    for tag, ood in (("tr", False), ("id", False), ("ood", True)):
        m, P = particles(r, n, ood)
        y = kin_target(four(m, P), law["S"], law["s"])
        Pm = P * (1 + 0.01 * r.standard_normal(P.shape))                           # 1% momentum resolution
        out["X" + tag] = four(m, Pm).reshape(n, 12); out["y" + tag] = y
    out["feat"] = np.array(KIN_FEAT)
    return out

# ---------------------------------------------------------------- conservation
QN = ["Q", "B", "Le", "Lmu", "S"]
TABLE = {  # Q B Le Lmu S
    "pi+": (1, 0, 0, 0, 0), "pi-": (-1, 0, 0, 0, 0), "pi0": (0, 0, 0, 0, 0), "K+": (1, 0, 0, 0, 1), "K-": (-1, 0, 0, 0, -1),
    "K0": (0, 0, 0, 0, 1), "K0bar": (0, 0, 0, 0, -1), "p": (1, 1, 0, 0, 0), "pbar": (-1, -1, 0, 0, 0), "n": (0, 1, 0, 0, 0),
    "nbar": (0, -1, 0, 0, 0), "Lambda": (0, 1, 0, 0, -1), "Lambdabar": (0, -1, 0, 0, 1), "e-": (-1, 0, 1, 0, 0),
    "e+": (1, 0, -1, 0, 0), "nu_e": (0, 0, 1, 0, 0), "nubar_e": (0, 0, -1, 0, 0), "mu-": (-1, 0, 0, 1, 0), "mu+": (1, 0, 0, -1, 0),
    "nu_mu": (0, 0, 0, 1, 0), "nubar_mu": (0, 0, 0, -1, 0), "gamma": (0, 0, 0, 0, 0),
    # only in the out-of-range test: multi-strange and charged hyperons
    "Sigma+": (1, 1, 0, 0, -1), "Sigma-": (-1, 1, 0, 0, -1), "Xi-": (-1, 1, 0, 0, -2), "Xi0": (0, 1, 0, 0, -2), "Omega-": (-1, 1, 0, 0, -3),
}
OOD_P = {"Sigma+", "Sigma-", "Xi-", "Xi0", "Omega-"}
CONS_FEAT = [f"{q}_{slot}" for slot in ("a", "b", "c", "d") for q in QN]       # a + b -> c + d

def cons_data(r, law, n=20000):
    names = list(TABLE); q = np.array([TABLE[k] for k in names], float)
    U = np.array(law["U"], float)                                                # conserved combinations (k x 5)
    idx = np.array(list(itertools.product(range(len(names)), repeat=4)))
    ood_mask = np.isin(np.array(names)[idx], list(OOD_P)).any(1)
    X = q[idx].reshape(len(idx), 20)
    delta = q[idx[:, 0]] + q[idx[:, 1]] - q[idx[:, 2]] - q[idx[:, 3]]
    real = (np.abs(delta @ U.T) < 1e-9).all(1)
    out = {}
    half = r.random(len(idx)) < 0.5                                              # train and in-range test use disjoint reactions
    for tag, pool in (("tr", ~ood_mask & half), ("id", ~ood_mask & ~half), ("ood", ood_mask)):
        pos = np.where(pool & real)[0]; neg = np.where(pool & ~real)[0]
        i = np.concatenate([r.choice(pos, n // 2), r.choice(neg, n // 2)]); r.shuffle(i)
        out["X" + tag] = X[i]; out["y" + tag] = real[i].astype(float)
    out["feat"] = np.array(CONS_FEAT)
    return out

# ---------------------------------------------------------------- decay in flight
DEC_FEAT = ["logL_cm", "logp_GeV", "logm_GeV"]

def decay_data(r, law, n=20000):
    out = {}
    for tag, ood in (("tr", False), ("id", False), ("ood", True)):
        if ood:   # momenta and distances beyond the training range
            lL = r.uniform(np.log(100), np.log(2000), n); lp = r.uniform(np.log(10), np.log(200), n)
        else:
            lL = r.uniform(np.log(0.5), np.log(100), n); lp = r.uniform(np.log(0.3), np.log(10), n)
        lm = r.uniform(np.log(0.1), np.log(2.0), n)
        a, b, g = law["exp"]
        P = 1 - np.exp(-np.exp(a * lL + b * lp + g * lm) / law["lam"])
        out["X" + tag] = np.stack([lL, lp, lm], 1); out["y" + tag] = (r.random(n) < P).astype(float); out["P" + tag] = P
    out["feat"] = np.array(DEC_FEAT)
    return out

GEN = {"kin": kin_data, "cons": cons_data, "decay": decay_data}

REAL = {
    "kin_mass12": dict(fam="kin", S=[0, 1], s=[1, -1, -1, -1], text="invariant mass squared of particles 1 and 2, m12^2 = (E1+E2)^2 - |p1+p2|^2"),
    "kin_pt123": dict(fam="kin", S=[0, 1, 2], s=[0, 1, 1, 0], text="transverse momentum squared of the 3-particle system, (px1+px2+px3)^2 + (py1+py2+py3)^2"),
    "cons_strong": dict(fam="cons", U=np.eye(5).tolist(), text="strong interaction: Q, B, Le, Lmu and S all conserved"),
    "cons_weak": dict(fam="cons", U=np.eye(5)[:4].tolist(), text="weak interaction: Q, B, Le, Lmu conserved, strangeness not"),
    "decay_k0s": dict(fam="decay", exp=[1, -1, 1], lam=2.6844, text="K0S decay in flight, P = 1 - exp(-L m / (p c tau)), c tau = 2.6844 cm"),
}

MENU_S = {"(+,-,-,-)": [1, -1, -1, -1], "(0,+,+,0)": [0, 1, 1, 0], "(0,+,+,+)": [0, 1, 1, 1], "(+,0,0,-)": [1, 0, 0, -1],
          "(0,+,-,0)": [0, 1, -1, 0], "(+,-,0,0)": [1, -1, 0, 0]}
MENU_SUB = [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
MENU_U = {"Q": [1, 0, 0, 0, 0], "B": [0, 1, 0, 0, 0], "Le": [0, 0, 1, 0, 0], "Lmu": [0, 0, 0, 1, 0], "S": [0, 0, 0, 0, 1],
          "B+S": [0, 1, 0, 0, 1], "Le+Lmu": [0, 0, 1, 1, 0], "Le-Lmu": [0, 0, 1, -1, 0], "Q-S": [1, 0, 0, 0, -1], "B-Le-Lmu": [0, 1, -1, -1, 0]}
MENU_EXP = [[1, -1, 1], [2, -1, 1], [1, -2, 2], [1, -1, 0], [0.5, -1, 1], [1, -0.5, 0.5], [1, -2, 1]]


def blind(seed):
    r = np.random.default_rng(seed); laws = {}
    for i in range(4):
        laws[f"kin_{i}"] = dict(fam="kin", S=MENU_SUB[r.integers(len(MENU_SUB))], s=list(MENU_S.values())[r.integers(len(MENU_S))])
    for i in range(4):
        while True:
            k = int(r.integers(1, 4)); pick = r.choice(list(MENU_U), k, replace=False)
            U = np.array([MENU_U[p] for p in pick])
            if np.linalg.matrix_rank(U) == k:
                break
        laws[f"cons_{i}"] = dict(fam="cons", U=U.tolist(), names=list(pick))
    for i in range(4):
        laws[f"decay_{i}"] = dict(fam="decay", exp=MENU_EXP[r.integers(len(MENU_EXP))], lam=float(np.exp(r.uniform(np.log(1), np.log(30)))))
    return laws


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "real":
        laws, seed = REAL, 1
    else:
        seed = int(sys.argv[2]) if len(sys.argv) > 2 else int.from_bytes(os.urandom(4), "little")
        laws = blind(seed)
    json.dump(dict(seed=seed, laws=laws), open(f"truth/{mode}.json", "w"), indent=1)
    for k, law in laws.items():
        r = np.random.default_rng(zlib.crc32(k.encode()) + seed)
        np.savez(f"data/{mode}_{k}.npz", **GEN[law["fam"]](r, law))
    print(f"{mode}: wrote {len(laws)} instances (truth in truth/{mode}.json, not printed)")
