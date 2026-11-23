"""K0S -> pi+ pi- candidates from Pythia8 minimum-bias pp at 13.6 TeV with an ALICE-like tracking toy.

Detector toy (not a full simulation):
  * charged final-state particles, |eta| < 0.8, pT > 0.15 GeV, production radius < 40 cm
  * helix tracks in a uniform B = 0.5 T along z
  * smearing at the production point: sigma_pT/pT = 1% (+) 0.5% pT,
    angles 0.5 mrad (+) 2 mrad/p, position 20 um (+) 60 um/pT, scaled by (1 + R_prod/10 cm)
  * no material, so no photon conversions and no hadronic interactions

V0 finder: every opposite-sign pair, decay vertex from the transverse circle-circle intersection
(or closest approach), loose preselection like the ALICE offline finder.

To get a high-multiplicity sample, NOV minimum-bias events are overlaid at a common vertex
(dNch/deta about 5x minimum bias). Combinatorial background then grows about NOV^2.

usage: python gen.py n_events seed out.npz   (n_events counts overlaid events)
"""
import sys, numpy as np, pythia8mc

B = 0.5
RC = 100.0 / (0.3 * B)        # helix radius [cm] per GeV of pT
MPI, MP, MK0, ML = 0.13957, 0.93827, 0.497611, 1.115683
MWIN = (0.42, 0.58)           # m_pipi window kept
NOV = 5                       # minimum-bias events per overlaid event


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Tracks:
    """Measured helices. r0: reference point [cm], p: momentum at r0 [GeV], q: charge."""
    def __init__(self, r0, p, q):
        self.r0, self.p, self.q = r0, p, q
        self.pt = np.hypot(p[:, 0], p[:, 1])
        self.R = RC * self.pt
        perp = np.stack([p[:, 1], -p[:, 0]], 1) / self.pt[:, None]
        self.c = r0[:, :2] + (q * self.R)[:, None] * perp
        d0v = r0[:, :2] - self.c
        self.phi0 = np.arctan2(d0v[:, 1], d0v[:, 0])
        self.tl = p[:, 2] / self.pt

    def at(self, i, P):
        """z and momentum of track(s) i at transverse point(s) P on their circle."""
        d = P - self.c[i]
        s = -self.q[i] * self.R[i] * wrap(np.arctan2(d[:, 1], d[:, 0]) - self.phi0[i])
        z = self.r0[i, 2] + s * self.tl[i]
        t = self.q[i][:, None] * np.stack([d[:, 1], -d[:, 0]], 1) / self.R[i][:, None]
        pxy = t * self.pt[i][:, None]
        return z, np.concatenate([pxy, (self.tl[i] * self.pt[i])[:, None]], 1)

    def dca_pv(self):
        cn = np.linalg.norm(self.c, axis=1)
        P = self.c * (1 - self.R / cn)[:, None]
        z, pm = self.at(np.arange(len(self.q)), P)
        dxy = np.abs(cn - self.R)
        # raw helix parameters at the PCA, inputs for the generic network
        sgn = np.sign(P[:, 0] * pm[:, 1] - P[:, 1] * pm[:, 0])
        phi = np.arctan2(pm[:, 1], pm[:, 0])
        raw = np.stack([sgn * dxy, z, np.cos(phi), np.sin(phi), self.tl, self.q / self.pt], 1)
        return np.hypot(dxy, z), raw


def v0_pairs(T, ip, ineg):
    """Vertex fit for pairs (ip[k], ineg[k]). Returns dca between daughters, vertex, daughter momenta."""
    c1, c2, R1, R2 = T.c[ip], T.c[ineg], T.R[ip], T.R[ineg]
    dv = c2 - c1; d = np.linalg.norm(dv, axis=1); u = dv / d[:, None]
    cand = []
    # two intersections
    a = (d**2 + R1**2 - R2**2) / (2 * d)
    h2 = R1**2 - a**2
    inter = h2 > 0
    h = np.sqrt(np.clip(h2, 0, None))
    base = c1 + a[:, None] * u
    nrm = np.stack([-u[:, 1], u[:, 0]], 1)
    for sg in (+1, -1):
        P = base + sg * h[:, None] * nrm
        cand.append((P, P))
    # no intersection: closest approach along the line of centres
    sep = d >= R1 + R2
    P1 = np.where(sep[:, None], c1 + R1[:, None] * u, np.where((R1 > R2)[:, None], c1 + R1[:, None] * u, c1 - R1[:, None] * u))
    P2 = np.where(sep[:, None], c2 - R2[:, None] * u, np.where((R1 > R2)[:, None], c2 + R2[:, None] * u, c2 - R2[:, None] * u))
    best = None
    for k, (Pa, Pb) in enumerate(cand + [(P1, P2)]):
        za, pa = T.at(ip, Pa); zb, pb = T.at(ineg, Pb)
        dca = np.sqrt(((Pa - Pb)**2).sum(1) + (za - zb)**2)
        valid = inter if k < 2 else ~inter
        dca = np.where(valid, dca, np.inf)
        V = np.concatenate([(Pa + Pb) / 2, ((za + zb) / 2)[:, None]], 1)
        if best is None:
            best = [dca, V, pa, pb]
        else:
            m = dca < best[0]
            best = [np.where(m, dca, best[0]), np.where(m[:, None], V, best[1]),
                     np.where(m[:, None], pa, best[2]), np.where(m[:, None], pb, best[3])]
    return best


def minv(pa, pb, ma, mb):
    ea = np.sqrt((pa**2).sum(1) + ma**2); eb = np.sqrt((pb**2).sum(1) + mb**2)
    return np.sqrt(np.clip((ea + eb)**2 - ((pa + pb)**2).sum(1), 0, None))


def features(dcad, V, pa, pb, dpos, dneg):
    P = pa + pb; pabs = np.linalg.norm(P, axis=1); L = np.linalg.norm(V, axis=1)
    cospa = (V * P).sum(1) / (L * pabs)
    R = np.hypot(V[:, 0], V[:, 1])
    ctau = MK0 * L / pabs
    ph = P / pabs[:, None]
    pla, plb = (pa * ph).sum(1), (pb * ph).sum(1)
    qt = np.linalg.norm(pa - pla[:, None] * ph, axis=1)
    alpha = (pla - plb) / (pla + plb)
    arm = np.clip(qt / np.maximum(np.abs(alpha), 1e-3), 0, 2)
    dml = np.minimum(np.abs(minv(pa, pb, MP, MPI) - ML), np.abs(minv(pa, pb, MPI, MP) - ML))
    F = np.stack([dcad, cospa, dpos, dneg, R, ctau, arm, np.clip(dml, 0, 0.1)], 1)
    return F, minv(pa, pb, MPI, MPI)


def smear(rng, r0, p):
    pt = np.hypot(p[:, 0], p[:, 1]); pp = np.linalg.norm(p, axis=1)
    Rp = np.hypot(r0[:, 0], r0[:, 1])
    spt = np.hypot(0.01, 0.005 * pt); sang = np.hypot(5e-4, 2e-3 / pp)
    spos = np.hypot(20e-4, 60e-4 / pt) * (1 + Rp / 10.0)
    pt2 = pt * (1 + spt * rng.standard_normal(len(pt)))
    phi = np.arctan2(p[:, 1], p[:, 0]) + sang * rng.standard_normal(len(pt))
    th = np.arctan2(pt, p[:, 2]) + sang * rng.standard_normal(len(pt))
    p2 = np.stack([pt2 * np.cos(phi), pt2 * np.sin(phi), pt2 / np.tan(th)], 1)
    return r0 + spos[:, None] * rng.standard_normal(r0.shape), p2


def main(nev, seed, out):
    py = pythia8mc.Pythia("", False)
    for s in ("Beams:eCM = 13600", "SoftQCD:inelastic = on", "ParticleDecays:limitTau0 = on",
              "ParticleDecays:tau0Max = 1000", "Random:setSeed = on", f"Random:seed = {seed}", "Next:numberCount = 0"):
        py.readString(s)
    py.init()
    rng = np.random.default_rng(seed)
    keep = {k: [] for k in ("F", "raw", "m", "y", "cat")}
    nsig = 0
    for iev in range(nev):
        r0, p, q, mid, mpdg, pid = [], [], [], [], [], []
        for iov in range(NOV):
            while not py.next():
                pass
            ev = py.event
            for i in range(ev.size()):
                x = ev[i]
                if not (x.isFinal() and x.isCharged()) or x.pT() < 0.15 or abs(x.eta()) > 0.8:
                    continue
                if np.hypot(x.xProd(), x.yProd()) > 400.0:      # mm
                    continue
                r0.append((x.xProd() / 10, x.yProd() / 10, x.zProd() / 10)); p.append((x.px(), x.py(), x.pz()))
                q.append(np.sign(x.charge())); pid.append(x.id())
                m = x.mother1(); mid.append(100000 * iov + m); mpdg.append(ev[m].id())
        if len(q) < 2:
            continue
        r0, p, q = np.array(r0), np.array(p), np.array(q, float)
        mid, mpdg, pid = np.array(mid), np.array(mpdg), np.array(pid)
        rm, pm = smear(rng, r0, p)
        T = Tracks(rm, pm, q)
        dpv, raw = T.dca_pv()
        pos, neg = np.where(q > 0)[0], np.where(q < 0)[0]
        pos, neg = pos[dpv[pos] > 0.02], neg[dpv[neg] > 0.02]
        if len(pos) == 0 or len(neg) == 0:
            continue
        ip, ineg = [a.ravel() for a in np.meshgrid(pos, neg, indexing="ij")]
        dcad, V, pa, pb = v0_pairs(T, ip, ineg)
        ok = np.isfinite(dcad) & (dcad < 2.0)
        if not ok.any():
            continue
        ip, ineg, dcad, V, pa, pb = ip[ok], ineg[ok], dcad[ok], V[ok], pa[ok], pb[ok]
        F, mpp = features(dcad, V, pa, pb, dpv[ip], dpv[ineg])
        sel = (F[:, 1] > 0.9) & (F[:, 4] > 0.2) & (mpp > MWIN[0]) & (mpp < MWIN[1])
        if not sel.any():
            continue
        ip, ineg, F, mpp = ip[sel], ineg[sel], F[sel], mpp[sel]
        same = mid[ip] == mid[ineg]
        y = same & (mpdg[ip] == 310) & (np.abs(pid[ip]) == 211) & (np.abs(pid[ineg]) == 211)
        # background category: 1 = true Lambda/anti-Lambda pair, 2 = other true pair, 3 = combinatorial
        cat = np.where(y, 0, np.where(same & (np.abs(mpdg[ip]) == 3122), 1, np.where(same, 2, 3)))
        keep["F"].append(F); keep["raw"].append(np.concatenate([raw[ip], raw[ineg]], 1))
        keep["m"].append(mpp); keep["y"].append(y); keep["cat"].append(cat)
        nsig += y.sum()
        if iev % 4000 == 0:
            print(seed, iev, nsig, sum(len(a) for a in keep["y"]), flush=True)
    np.savez_compressed(out, **{k: np.concatenate(v) for k, v in keep.items()})


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
