"""Toptag winner on K0S: soft cuts + one smooth unit per feature, with the prior variants,
on a linear or a log-scaled physics layer.

Physics layer ("space"):
  lin  standardised features as in models.py
  log  log10 of every heavy-tailed feature (DCAs, 1 - cos PA, radius, mL/p, |dm_Lambda|), Armenteros linear.
       Cuts stay cuts (the map is monotone), and thresholds read back in cm/GeV.

Written net (LLM-written*): cut units, then one smooth unit tanh(0.5 x) per feature with output weight 0
at the start, then optional extra random units (residual capacity, prior mean 0).

Prior: ||(theta - theta_prog) / sigma||^2 / (2 * scale)
  scale = N           ("1/N", usual MAP)
  scale = N0          ("fixed")
  scale = max(N, N0)  ("max": fixed at small N, 1/N at large N)
sigma: 0.3 everywhere, optionally sigma_b = k * dt on the written cut biases ("syst"),
where dt is the cut's systematic variation mapped into the physics layer.
"""
import numpy as np
import torch
import torch.nn as nn
import models as M

SOFT = 5.0
NF = len(M.FEATS)
# transform per feature for the log space: (forward, inverse, increasing?)
EPS = [1e-4, 1e-7, 0, 0, 0, 1e-3, None, 1e-4]


def tf(f, x):
    x = np.asarray(x, float)
    if f == 6:
        return x
    if f == 1:
        return np.log10(np.clip(1 - x, 0, None) + EPS[1])
    return np.log10(np.clip(x, 1e-12, None) + EPS[f])


def itf(f, g):
    if f == 6:
        return g
    if f == 1:
        return 1 - (10**g - EPS[1])
    return 10**g - EPS[f]


INC = [True, False, True, True, True, True, True, True]
LOG_C = np.array([tf(u[1], u[2]) for u in M.programmed_units()])       # centre on the written thresholds
LOG_S = np.array([0.5] * 6 + [0.2, 0.5])                                # half a decade, Armenteros 0.2


def phys(F, space="log"):
    if space == "lin":
        return M.std_feats(F)
    G = np.stack([tf(f, F[:, f]) for f in range(NF)], 1)
    return ((G - LOG_C) / LOG_S).astype(np.float32)


def units(space="log", soft=SOFT):
    """Written cuts in the physics layer: (name, f, t_layer, k_layer, sign_layer, dt_layer, t_phys, dt_phys)."""
    out = []
    for name, f, t, k, sg, dt in M.programmed_units():
        if space == "lin":
            out.append((name, f, (t - M.FC[f]) / M.FS[f], k * M.FS[f] / soft, sg, dt / M.FS[f], t, dt))
        else:
            tl = (tf(f, t) - LOG_C[f]) / LOG_S[f]
            dtl = abs(tf(f, t + dt) - tf(f, t)) / LOG_S[f]
            # soft edges: about 0.25 decade for log features (4 per decade), Armenteros k/SOFT = 6 per unit
            kl = 4.0 * LOG_S[f] if f != 6 else (k / soft) * LOG_S[f]
            sl = sg if INC[f] else -sg
            out.append((name, f, tl, kl, sl, dtl, t, dt))
    return out


class Net(nn.Module):
    def __init__(self, init="written", space="log", feats=None, extra=0, seed=0, H=None):
        """feats: indices into M.FEATS used as inputs (default all). Written cuts on other features are dropped."""
        super().__init__()
        self.space = space
        self.feats = list(range(NF)) if feats is None else list(feats)
        nf = len(self.feats)
        self.U = [u for u in units(space) if u[1] in self.feats]
        self.ncut, self.extra = len(self.U), extra
        H = H or (self.ncut + nf + extra)
        g = torch.Generator().manual_seed(seed)
        self.l1 = nn.Linear(nf, H); self.l2 = nn.Linear(H, 1)
        with torch.no_grad():
            if init == "random":
                self.U, self.ncut = [], 0
                for p in self.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * (1 / np.sqrt(p.shape[-1]) if p.dim() > 1 else 0.1))
                return
            W1 = np.zeros((H, nf)); b1 = np.zeros(H); w2 = np.zeros(H)
            for j, (_, f, tl, kl, sl, *_r) in enumerate(self.U):
                fi = self.feats.index(f)
                W1[j, fi] = sl * kl; b1[j] = -sl * kl * tl; w2[j] = 1.0
            for i in range(nf):
                W1[self.ncut + i, i] = 0.5
            W1 = torch.tensor(W1, dtype=torch.float32); b1 = torch.tensor(b1, dtype=torch.float32)
            if extra:
                W1[-extra:] = torch.randn(extra, nf, generator=g) / np.sqrt(nf)
                b1[-extra:] = 0.1 * torch.randn(extra, generator=g)
            self.l1.weight.copy_(W1); self.l1.bias.copy_(b1)
            self.l2.weight.copy_(torch.tensor(w2, dtype=torch.float32)[None])
            self.l2.bias.fill_(-(self.ncut - 1.0))

    def inp(self, F):
        return phys(F, self.space)[:, self.feats]

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x))).squeeze(-1)

    def prior(self, syst=False, base=0.3):
        mu = [p.detach().clone() for p in self.parameters()]
        sig = [torch.full_like(p, base) for p in self.parameters()]
        if self.extra:
            mu[0][-self.extra:] = 0; mu[1][-self.extra:] = 0; mu[2][:, -self.extra:] = 0
        if syst:
            for j, u in enumerate(self.U):
                sig[1][j] = max(u[3] * u[5], 0.05)
        return mu, sig


def train(model, X, y, w=None, mu=None, sig=0.3, mode="1/N", N0=1000, freeze_l1=False, steps=1500, lr=5e-3):
    X = torch.tensor(X); y = torch.tensor(y, dtype=torch.float32)
    w = torch.ones_like(y) if w is None else torch.tensor(w, dtype=torch.float32)
    w = w / w.mean()
    for p in model.l1.parameters():
        p.requires_grad_(not freeze_l1)
    ps = list(model.parameters())
    if mu is None:
        mu = [torch.zeros_like(p) for p in ps]
    if not isinstance(sig, list):
        sig = [torch.full_like(p, sig) for p in ps]
    N = len(y)
    scale = {"1/N": N, "fixed": N0, "max": max(N, N0)}[mode]
    opt = torch.optim.Adam([p for p in ps if p.requires_grad], lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(steps):
        opt.zero_grad()
        pen = sum((((p - m) / s)**2).sum() for p, m, s in zip(ps, mu, sig) if p.requires_grad)
        loss = (w * bce(model(X), y)).mean() + pen / (2 * scale)
        loss.backward(); opt.step()
    for p in model.l1.parameters():
        p.requires_grad_(True)
    return model


def read_units(net):
    """Written cut units read back as cuts in physical units, plus smooth-unit output weights."""
    W = net.l1.weight.detach().numpy(); b = net.l1.bias.detach().numpy(); w2 = net.l2.weight.detach().numpy()[0]
    rows = []
    for j, (name, f, tl, kl, sl, dtl, t0, dt) in enumerate(net.U):
        fi = net.feats.index(f)
        g = -b[j] / W[j, fi]                                          # threshold in the layer
        if net.space == "lin":
            t = M.FC[f] + g * M.FS[f]
        else:
            t = float(itf(f, LOG_C[f] + g * LOG_S[f]))
        rows.append(dict(cut=name, feature=M.FEATS[f], threshold=float(t), shift_dt=float((t - t0) / dt),
                         shift_layer=float((g - tl) / dtl), purity=float(abs(W[j, fi]) / np.linalg.norm(W[j])),
                         out_weight=float(w2[j])))
    smooth = {M.FEATS[f]: float(w2[net.ncut + i]) for i, f in enumerate(net.feats)}
    return rows, smooth


@torch.no_grad()
def predict(model, X, bs=16384):
    X = torch.tensor(X)
    return np.concatenate([model(X[i:i + bs]).numpy() for i in range(0, len(X), bs)])
