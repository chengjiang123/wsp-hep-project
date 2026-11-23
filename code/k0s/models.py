"""K0S selection: hand-written topological cuts as tanh units, physics-prior MAP training,
and a generic MLP on raw helix parameters as the 'learn everything' baseline.
Training code is the same as toptag/models.py."""
import numpy as np
import torch
import torch.nn as nn

FEATS = ["dca_dau", "cos_pa", "dca_pos", "dca_neg", "radius", "mL/p", "qT/|alpha|", "|m_pipr - m_Lambda|"]
UNITS_OF = ["cm", "", "cm", "cm", "cm", "cm", "", "GeV"]
# fixed, hand-chosen standardisation (not fitted)
FC = np.array([0.5, 0.97, 0.1, 0.1, 5.0, 5.0, 0.2, 0.01])
FS = np.array([0.5, 0.03, 0.1, 0.1, 5.0, 5.0, 0.2, 0.01])
H = 10


def std_feats(F):
    return ((F - FC) / FS).astype(np.float32)


def programmed_units():
    """(name, feature, threshold, sharpness k, sign, systematic variation dt).
    Thresholds: typical ALICE K0S topological selection (pp strangeness analyses).
    dt: an illustrative +-range of the kind analysis notes vary each cut by for systematics.
    It is not taken from a specific note."""
    return [
        ("dca_dau < 1.0 cm",      0, 1.0,   5.0,  -1, 0.25),
        ("cos_pa > 0.97",         1, 0.97,  300., +1, 0.01),
        ("dca_pos > 0.06 cm",     2, 0.06,  100., +1, 0.02),
        ("dca_neg > 0.06 cm",     3, 0.06,  100., +1, 0.02),
        ("radius > 0.5 cm",       4, 0.5,   5.0,  +1, 0.2),
        ("mL/p < 20 cm",          5, 20.0,  0.5,  -1, 5.0),
        ("qT > 0.2|alpha|",       6, 0.2,   30.,  +1, 0.05),
        ("|m_Lambda| > 5 MeV",    7, 0.005, 1000., +1, 0.002),
    ]


def handwritten_weights():
    W1 = np.zeros((H, len(FEATS))); b1 = np.zeros(H)
    for j, (_, f, t, k, sg, _) in enumerate(programmed_units()):
        W1[j, f] = sg * k * FS[f]
        b1[j] = sg * k * (FC[f] - t)
    w2 = np.zeros(H); w2[:8] = 1.0
    b2 = -7.0                         # all 8 pass -> +1, one fails -> -1
    return W1, b1, w2, b2


class FeatNet(nn.Module):
    def __init__(self, init="random", seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = nn.Linear(len(FEATS), H); self.l2 = nn.Linear(H, 1)
        with torch.no_grad():
            if init == "random":
                for p in self.parameters():
                    p.copy_(torch.randn(p.shape, generator=g) * (1 / np.sqrt(p.shape[-1]) if p.dim() > 1 else 0.1))
            else:
                W1, b1, w2, b2 = [torch.tensor(np.asarray(a), dtype=torch.float32) for a in handwritten_weights()]
                W1[8:] = 0.1 * torch.randn(W1[8:].shape, generator=g)
                self.l1.weight.copy_(W1); self.l1.bias.copy_(b1)
                self.l2.weight.copy_(w2[None]); self.l2.bias.fill_(float(b2))

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x))).squeeze(-1)


def syst_sigma(net, base=0.3):
    """Per-parameter prior width. Written biases get sigma_b = k * dt, so the prior lets each
    threshold move by about its systematic variation. Everything else gets `base`."""
    sig = [torch.full_like(p, base) for p in net.parameters()]
    for j, (_, f, t, k, sg, dt) in enumerate(programmed_units()):
        sig[1][j] = k * dt            # l1.bias
    return torch.cat([s.reshape(-1) for s in sig])


class RawMLP(nn.Module):
    """Generic network on raw helix parameters of both tracks (12 inputs)."""
    def __init__(self, seed=0, w=64):
        super().__init__()
        torch.manual_seed(seed)
        self.f = nn.Sequential(nn.Linear(12, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def flat(m):
    return torch.cat([p.reshape(-1) for p in m.parameters()])


def train_map(model, X, y, prior_mean=None, prior_sigma=1.0, steps=1500, lr=5e-3, batch=None, seed=0):
    """MAP with Gaussian prior N(prior_mean, prior_sigma^2). prior_sigma may be a per-parameter tensor."""
    X = torch.tensor(X); y = torch.tensor(y, dtype=torch.float32)
    mu = torch.zeros_like(flat(model)) if prior_mean is None else prior_mean.detach().clone()
    sig = prior_sigma if torch.is_tensor(prior_sigma) else torch.full_like(mu, prior_sigma)
    N = len(y); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g) if batch and batch < N else slice(None)
        opt.zero_grad()
        loss = bce(model(X[idx]), y[idx]) + (((flat(model) - mu) / sig)**2).sum() / (2 * N)
        loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, X, bs=8192):
    X = torch.tensor(X)
    return np.concatenate([model(X[i:i + bs]).numpy() for i in range(0, len(X), bs)])
