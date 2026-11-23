"""Stage 1: Pythia8 minimum-bias pp at 13.6 TeV, store true charged tracks per event (no smearing).
Reconstruction, overlay and anomaly injection happen in reco.py, so variations need no new Pythia run.

usage: python truth.py n_mb_events seed out.npz
"""
import sys, numpy as np, pythia8mc


def main(nev, seed, out):
    py = pythia8mc.Pythia("", False)
    for s in ("Beams:eCM = 13600", "SoftQCD:inelastic = on", "ParticleDecays:limitTau0 = on",
              "ParticleDecays:tau0Max = 1000", "Random:setSeed = on", f"Random:seed = {seed}", "Next:numberCount = 0"):
        py.readString(s)
    py.init()
    T, off, K = [], [0], []
    for iev in range(nev):
        while not py.next():
            pass
        ev = py.event
        for i in range(ev.size()):
            x = ev[i]
            if x.id() == 310 and abs(x.eta()) < 0.8:
                K.append((x.px(), x.py(), x.pz()))
            if not (x.isFinal() and x.isCharged()) or x.pT() < 0.15 or abs(x.eta()) > 0.8:
                continue
            if np.hypot(x.xProd(), x.yProd()) > 400.0:      # mm
                continue
            m = x.mother1()
            T.append((x.xProd() / 10, x.yProd() / 10, x.zProd() / 10, x.px(), x.py(), x.pz(),
                      np.sign(x.charge()), x.id(), m, ev[m].id()))
        off.append(len(T))
    np.savez_compressed(out, T=np.array(T), off=np.array(off), K0=np.array(K))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
