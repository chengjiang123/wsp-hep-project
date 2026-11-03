"""Stage 2: reconstruction of V0 candidates from stored truth tracks, with knobs for
detector effects, event overlay and an injected long-lived particle (for anomaly detection).

Knobs (defaults are the nominal sample, same as gen.py):
  ip     scale on the position resolution (impact-parameter resolution)
  mom    scale on the momentum and angle resolution
  disp   position resolution grows as (1 + R/disp cm), smaller disp = worse for displaced tracks
  nov    minimum-bias events overlaid per event (multiplicity)
  mwin   m_pipi window kept
  inj    probability per overlaid event to add an anomaly S -> pi+ pi- (mass inj_m, c*tau inj_ctau cm),
         with the K0S momentum spectrum from the truth files

usage: python reco.py out.npz [key=value ...]
"""
import sys, numpy as np
from gen import Tracks, v0_pairs, features, MPI

DEF = dict(ip=1.0, mom=1.0, disp=10.0, nov=5, mlo=0.42, mhi=0.58, inj=0.0, inj_m=0.7, inj_ctau=5.0, seed=1,
           files="truth_a.npz,truth_b.npz", nmax=0)


def smear(rng, r0, p, c):
    pt = np.hypot(p[:, 0], p[:, 1]); pp = np.linalg.norm(p, axis=1)
    Rp = np.hypot(r0[:, 0], r0[:, 1])
    spt = c["mom"] * np.hypot(0.01, 0.005 * pt); sang = c["mom"] * np.hypot(5e-4, 2e-3 / pp)
    spos = c["ip"] * np.hypot(20e-4, 60e-4 / pt) * (1 + Rp / c["disp"])
    pt2 = pt * (1 + spt * rng.standard_normal(len(pt)))
    phi = np.arctan2(p[:, 1], p[:, 0]) + sang * rng.standard_normal(len(pt))
    th = np.arctan2(pt, p[:, 2]) + sang * rng.standard_normal(len(pt))
    p2 = np.stack([pt2 * np.cos(phi), pt2 * np.sin(phi), pt2 / np.tan(th)], 1)
    return r0 + spos[:, None] * rng.standard_normal(r0.shape), p2


def decay_S(rng, P, m, ctau):
    """S with 3-momentum P decays to pi+ pi- after an exponential flight. Returns 2 truth tracks."""
    E = np.sqrt(P @ P + m * m); bg = np.linalg.norm(P) / m
    V = P / np.linalg.norm(P) * bg * ctau * rng.exponential()
    ps = np.sqrt(m * m / 4 - MPI**2)
    u = rng.normal(size=3); u /= np.linalg.norm(u)
    out = []
    for s in (+1, -1):
        e, pv = m / 2, s * ps * u
        b = P / E; g = E / m; bp = b @ pv
        plab = pv + ((g - 1) * bp / (b @ b) + g * e) * b
        out.append((*V, *plab, float(s), 211 * s, -7, 9900))
    return np.array(out)


def run(c):
    rng = np.random.default_rng(int(c["seed"]))
    T, off, K0 = [], [], []
    for f in c["files"].split(","):
        d = np.load(f); n0 = sum(len(t) for t in T)
        T.append(d["T"]); off.append(d["off"][:-1] + n0); K0.append(d["K0"])
    T = np.concatenate(T); off = np.concatenate(off + [[len(T)]]); K0 = np.concatenate(K0)
    nov = int(c["nov"]); nev = (len(off) - 1) // nov
    if c["nmax"]:
        nev = min(nev, int(c["nmax"]))
    keep = {k: [] for k in ("F", "raw", "m", "y", "cat", "pt")}
    for ie in range(nev):
        parts = []
        for iov in range(nov):
            k = ie * nov + iov
            t = T[off[k]:off[k + 1]].copy()
            t[:, 8] += 100000 * iov
            parts.append(t)
        if c["inj"] > 0 and rng.random() < c["inj"]:
            s = decay_S(rng, K0[rng.integers(len(K0))], c["inj_m"], c["inj_ctau"])
            pt = np.hypot(s[:, 3], s[:, 4]); eta = np.arcsinh(s[:, 5] / pt); R = np.hypot(s[:, 0], s[:, 1])
            parts.append(s[(pt > 0.15) & (np.abs(eta) < 0.8) & (R < 40)])
        t = np.concatenate(parts)
        if len(t) < 2:
            continue
        r0, p, q, pid, mid, mpdg = t[:, :3], t[:, 3:6], t[:, 6], t[:, 7], t[:, 8], t[:, 9]
        rm, pm = smear(rng, r0, p, c)
        Tr = Tracks(rm, pm, q)
        dpv, raw = Tr.dca_pv()
        pos, neg = np.where(q > 0)[0], np.where(q < 0)[0]
        pos, neg = pos[dpv[pos] > 0.02], neg[dpv[neg] > 0.02]
        if len(pos) == 0 or len(neg) == 0:
            continue
        ip, ineg = [a.ravel() for a in np.meshgrid(pos, neg, indexing="ij")]
        dcad, V, pa, pb = v0_pairs(Tr, ip, ineg)
        ok = np.isfinite(dcad) & (dcad < 2.0)
        if not ok.any():
            continue
        ip, ineg, dcad, V, pa, pb = ip[ok], ineg[ok], dcad[ok], V[ok], pa[ok], pb[ok]
        F, mpp = features(dcad, V, pa, pb, dpv[ip], dpv[ineg])
        sel = (F[:, 1] > 0.9) & (F[:, 4] > 0.2) & (mpp > c["mlo"]) & (mpp < c["mhi"])
        if not sel.any():
            continue
        ip, ineg, F, mpp, pa, pb = ip[sel], ineg[sel], F[sel], mpp[sel], pa[sel], pb[sel]
        same = mid[ip] == mid[ineg]
        y = same & (mpdg[ip] == 310) & (np.abs(pid[ip]) == 211) & (np.abs(pid[ineg]) == 211)
        # 0 = K0S, 1 = Lambda pair, 2 = other true pair, 3 = combinatorial, 4 = injected anomaly
        cat = np.where(y, 0, np.where(same & (mpdg[ip] == 9900), 4, np.where(same & (np.abs(mpdg[ip]) == 3122), 1,
                                                                            np.where(same, 2, 3))))
        keep["F"].append(F); keep["raw"].append(np.concatenate([raw[ip], raw[ineg]], 1))
        keep["m"].append(mpp); keep["y"].append(y); keep["cat"].append(cat)
        keep["pt"].append(np.hypot((pa + pb)[:, 0], (pa + pb)[:, 1]))
    return {k: np.concatenate(v) for k, v in keep.items()}, nev


if __name__ == "__main__":
    c = dict(DEF)
    for a in sys.argv[2:]:
        k, v = a.split("=")
        c[k] = v if k == "files" else float(v)
    D, nev = run(c)
    np.savez_compressed(sys.argv[1], nev=nev, cfg=str(c), **D)
    print(sys.argv[1], "events", nev, "cands", len(D["y"]), "cat", np.bincount(D["cat"], minlength=5))
