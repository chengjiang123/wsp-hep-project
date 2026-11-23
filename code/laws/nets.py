"""Readable architectures for the three law families, a generic MLP for comparison, training and evaluation.

kin    y      = sum_j a_j (w_j . x)^2                         (square units, no bias)
cons   logit  = b - sum_j c_j (w_j . x)^2,  c_j >= 0          (a reaction scores high when every learned w_j . x is 0)
decay  rate   = sum_j c_j exp(w_j . x + b_j),  c_j >= 0       (x = log L, log p, log m)
       P      = 1 - exp(-rate)
A network is stored as JSON {"fam", "feat", "W" [H x D], "a" [H], "b" (cons: scalar, decay: [H])}.
"""
import json, numpy as np, torch
from sklearn.metrics import roc_auc_score

torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
ARCH = {
    "kin": "y = sum_j a[j] * (W[j] . x)^2",
    "cons": "logit = b - sum_j a[j] * (W[j] . x)^2 ; a[j] >= 0 ; P(real reaction) = sigmoid(logit)",
    "decay": "rate = sum_j a[j] * exp(W[j] . x + b[j]) ; a[j] >= 0 ; P(decayed before L) = 1 - exp(-rate)",
}


def forward(fam, p, X):
    """p: dict of tensors W, a, b (a already >= 0 where required). Returns prediction (kin) or probability."""
    z = X @ p["W"].T
    if fam == "kin":
        return (z ** 2 * p["a"]).sum(1)
    if fam == "cons":
        return torch.sigmoid(p["b"] - (z ** 2 * p["a"]).sum(1))
    lograte = torch.logsumexp(torch.log(p["a"] + 1e-300) + z + p["b"], 1)
    return -torch.expm1(-torch.exp(lograte))


def predict(net, X):
    p = {k: torch.tensor(np.array(net[k], float)) for k in ("W", "a", "b") if k in net}
    if "b" not in p:
        p["b"] = torch.tensor(0.0)
    with torch.no_grad():
        return forward(net["fam"], p, torch.tensor(X)).numpy()


def loss_fn(fam, out, y, ymean2):
    if fam == "kin":
        return ((out - y) ** 2).mean() / ymean2
    out = out.clamp(1e-12, 1 - 1e-12)
    return -(y * torch.log(out) + (1 - y) * torch.log(1 - out)).mean()


def train_readable(fam, X, y, H=None, seed=0, steps=3000, l1=1e-4, init=None, prior=0.0, n0=100):
    """init: a stored network (e.g. the weights written from the law) to start from instead of random weights.
    prior > 0: Gaussian prior around init, loss + |theta - theta0|^2 / (2 prior^2 max(N, n0)), as for the K0S written net"""
    H = H or {"kin": 8, "cons": 8, "decay": 1}[fam]
    g = torch.Generator().manual_seed(seed); D = X.shape[1]
    X, y = torch.tensor(X), torch.tensor(y); ym2 = (y ** 2).mean()
    if init is None:
        W = (0.3 * torch.randn(H, D, generator=g)).requires_grad_()
        a = (0.3 * torch.randn(H, generator=g) if fam == "kin" else torch.zeros(H)).requires_grad_()
        b = (torch.zeros(()) if fam == "cons" else torch.zeros(H)).requires_grad_()
    else:
        W = torch.tensor(np.array(init["W"], float)).requires_grad_()
        a0 = torch.tensor(np.array(init["a"], float))
        a = (a0 if fam == "kin" else torch.log(torch.expm1(a0.clamp(min=1e-6)))).requires_grad_()   # inverse softplus
        b = torch.tensor(np.array(init.get("b", 0.0), float)).requires_grad_()
    th0 = [t.detach().clone() for t in (W, a, b)]
    nprior = max(len(y), n0)
    params = [W, a] + ([b] if fam != "kin" else [])
    pos = (lambda t: t) if fam == "kin" else torch.nn.functional.softplus

    def L():
        out = loss_fn(fam, forward(fam, {"W": W, "a": pos(a), "b": b}, X), y, ym2)
        if init is not None and prior > 0:
            return out + sum(((t - t0) ** 2).sum() for t, t0 in zip((W, a, b), th0)) / (2 * prior ** 2 * nprior)
        return out + l1 * W.abs().mean()
    opt = torch.optim.Adam(params, lr=0.01)
    for _ in range(steps):
        opt.zero_grad(); l = L(); l.backward(); opt.step()
    opt = torch.optim.LBFGS(params, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); l = L(); l.backward(); return l
    opt.step(closure)
    net = dict(fam=fam, W=W.detach().numpy().tolist(), a=pos(a).detach().numpy().tolist())
    if fam != "kin":
        net["b"] = b.detach().numpy().tolist()
    return net


class MLP(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(D, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.f(x)[:, 0]


def train_mlp(fam, X, y, seed=0, steps=3000):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    ym, ys = (y.mean(), y.std() + 1e-9) if fam == "kin" else (0.0, 1.0)
    m = MLP(X.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
    Xt = torch.tensor((X - mu) / sd); yt = torch.tensor((y - ym) / ys) if fam == "kin" else torch.tensor(y)
    for _ in range(steps):
        opt.zero_grad(); o = m(Xt)
        l = ((o - yt) ** 2).mean() if fam == "kin" else torch.nn.functional.binary_cross_entropy_with_logits(o, yt)
        l.backward(); opt.step()

    def pred(Xn):
        with torch.no_grad():
            o = m(torch.tensor((Xn - mu) / sd)).numpy()
        return o * ys + ym if fam == "kin" else 1 / (1 + np.exp(-o))
    return pred


def metric(fam, pred, d, tag):
    """kin: normalised RMS error; cons: 1 - AUC; decay: mean |P_pred - P_true|. Lower is better for all."""
    y = d["y" + tag]
    if fam == "kin":
        return float(np.sqrt(np.mean((pred - y) ** 2) / np.mean(y ** 2)))
    if fam == "cons":
        return float(1 - roc_auc_score(y, pred))
    return float(np.mean(np.abs(pred - d["P" + tag])))


def train_readable_best(fam, X, y, seed=0, restarts=3):
    """A few random restarts, keep the one with the lowest training loss (no test data involved)."""
    best, bl = None, np.inf
    for r in range(restarts):
        net = train_readable(fam, X, y, seed=100 * seed + r)
        p = predict(net, X)
        l = np.mean((p - y) ** 2) / np.mean(y ** 2) if fam == "kin" else -np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))
        if l < bl:
            best, bl = net, l
    return best
