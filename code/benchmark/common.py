"""Shared pieces of the benchmark: data, input features, the two pretrained models, fine-tuning, metrics, and
the weight-patch editor that the LLM (and the script / random controls) use.

Golden rule: an "edit" is a patch applied directly to tensors of an existing network, with no gradient step.
"""
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.environ.get("WSP_ROOT", "/path/to/scratch/wsp_bench")
PART_DIR = os.environ.get("PART_DIR", "/path/to/scratch/hf_alice/weaver/particle_transformer/models")
OMNI_CKPT = os.environ.get("OMNI_CKPT", "/path/to/scratch/hf_alice/omni_ckpt/best_model_pretrain_s.pt")
P = 128
JETCLASS = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------- data


class Split:
    """Jets of one split held as numpy (CPU); `batch(idx)` returns GPU tensors."""

    def __init__(self, ds, split, idx=None):
        base = f"{ROOT}/cache/{ds}/{split}"
        load = lambda k: np.load(f"{base}_{k}.npy", mmap_mode="r")
        self.has_pid = os.path.exists(f"{base}_pid.npy")
        y = load("y")
        idx = np.arange(len(y)) if idx is None else np.sort(np.asarray(idx))
        self.idx = idx
        self.vec = np.ascontiguousarray(load("vec")[idx])
        self.n = np.asarray(load("n")[idx]).astype(np.int64)
        self.jet = np.asarray(load("jet")[idx])
        self.y = np.asarray(y[idx]).astype(np.int64)
        self.pid = np.ascontiguousarray(load("pid")[idx]) if self.has_pid else None

    def __len__(self):
        return len(self.y)

    def batch(self, rows):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[rows])).to(DEV, non_blocking=True)
        b = {"vec": t(self.vec).float(), "n": t(self.n), "jet": t(self.jet).float(), "y": t(self.y)}
        if self.pid is not None:
            b["pid"] = t(self.pid).long()
        return b


def splits_for(ds, N, seed, n_val_full=100_000):
    """Nested label budgets: for a seed, the first N jets of one fixed permutation of the pool.
    N < full: 80% train / 20% validation, both inside the budget.  N = 'full': the whole pool (top) or 88.89% (qg)
    to train, and the official validation file (top) or the rest of the pool (qg) to validate, as the ParT scripts."""
    n_pool = len(np.load(f"{ROOT}/cache/{ds}/pool_y.npy", mmap_mode="r"))
    perm = np.random.default_rng(1000 + seed).permutation(n_pool)
    if N == "full":
        if ds == "top":
            n_val = len(np.load(f"{ROOT}/cache/top/val_y.npy", mmap_mode="r"))
            va = np.random.default_rng(seed).permutation(n_val)[:n_val_full]
            return ("pool", perm), ("val", va)
        cut = int(round(0.8889 * n_pool))
        return ("pool", perm[:cut]), ("pool", perm[cut:][:n_val_full])
    N = int(N)
    ntr = int(round(0.8 * N))
    return ("pool", perm[:ntr]), ("pool", perm[ntr:N])


# ------------------------------------------------------------------------------------------------------------- features


def part_inputs(b, use_pid):
    """ParT inputs exactly as data/TopLandscape/top_kin.yaml and data/QuarkGluon/qg_kinpid.yaml (manual standardisation)."""
    px, py, pz, E, deta, dphi = b["vec"].unbind(-1)
    mask = (torch.arange(P, device=px.device)[None] < b["n"][:, None]).float()
    lg = lambda a: torch.log(a.clamp(min=1e-8))
    pt = torch.hypot(px, py)
    jpt, jE = b["jet"][:, 0:1], b["jet"][:, 2:3]
    std = lambda a, c, s: ((a - c) * s).clamp(-5, 5)
    f = [std(lg(pt), 1.7, 0.7), std(lg(E), 2.0, 0.7), std(lg(pt / jpt), -4.7, 0.7), std(lg(E / jE), -4.7, 0.7),
         std(torch.hypot(deta, dphi), 0.2, 4.0)]
    if use_pid:
        f += [b["pid"][..., k].float() for k in range(6)]
    f += [deta.clamp(-5, 5), dphi.clamp(-5, 5)]
    x = torch.stack(f, 1) * mask[:, None]
    v = torch.stack([px, py, pz, E], 1) * mask[:, None]
    return x, v, mask[:, None]


def omni_inputs(b, use_pid):
    """OmniLearned layout: (deta, dphi, ln pT, ln E) in GeV, zero rows = padding (mask is feature 2 != 0)."""
    px, py, pz, E, deta, dphi = b["vec"].unbind(-1)
    mask = (torch.arange(P, device=px.device)[None] < b["n"][:, None])
    lpt = torch.log(torch.hypot(px, py).clamp(min=1e-8))
    lpt = torch.where(lpt.abs() < 1e-6, torch.full_like(lpt, 1e-6), lpt)  # ln pT = 0 would read as padding
    X = torch.stack([deta, dphi, lpt, torch.log(E.clamp(min=1e-8))], -1) * mask[..., None]
    pid = b["pid"][..., 6] * mask if use_pid else None
    return X, pid


# --------------------------------------------------------------------------------------------------------------- models


class ParTModel(nn.Module):
    """Official ParT (weaver ParticleTransformer, config of networks/example_ParticleTransformer_finetune.py) + new head."""
    kind = "part"
    head_lr_mult = 50.0  # --optimizer-option lr_mult ("fc.*",50)

    def __init__(self, use_pid, pretrained=True, n_out=2):
        super().__init__()
        from weaver.nn.model.ParticleTransformer import ParticleTransformer
        self.use_pid = use_pid
        self.mod = ParticleTransformer(
            input_dim=13 if use_pid else 7, num_classes=None, pair_input_dim=4, use_pre_activation_pair=False,
            embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8, num_cls_layers=2,
            block_params=None, cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
            fc_params=None, activation="gelu", trim=True, for_inference=False)
        self.head = nn.Linear(128, n_out)
        name = "ParT_kinpid.pt" if use_pid else "ParT_kin.pt"
        sd = torch.load(f"{PART_DIR}/{name}", map_location="cpu")
        self.pre_W, self.pre_b = sd["mod.fc.0.weight"].float().clone(), sd["mod.fc.0.bias"].float().clone()
        self.row_names = list(JETCLASS)
        if pretrained:
            r = self.mod.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mod.") and not k.startswith("mod.fc.")}, strict=True)
        # U_h = GELU(BN_h(conv)): scaling gamma_h, beta_h scales head h before its GELU; factor 0 switches it off exactly
        self.pair_bn = [m for m in self.mod.pair_embed.embed if isinstance(m, nn.BatchNorm1d)][-1]

    def no_decay(self, name, p):
        return p.ndim < 2 or "cls_token" in name

    def is_head(self, name):
        return name.startswith("head.")

    def embed(self, b):
        x, v, m = part_inputs(b, self.use_pid)
        return self.mod(x, v=v, mask=m)

    def forward(self, b):
        return self.head(self.embed(b))

    def pair_scale_(self, h, f):
        with torch.no_grad():
            self.pair_bn.weight[h] *= f
            self.pair_bn.bias[h] *= f

    def pair_state(self):
        return {"gamma": self.pair_bn.weight.detach().cpu().tolist(), "beta": self.pair_bn.bias.detach().cpu().tolist()}


class OmniModel(nn.Module):
    """OmniLearned PET2 'small', built with the pretrained interaction and local-interaction blocks, new 2-class output."""
    kind = "omni"
    head_lr_mult = 5.0  # --lr-factor 5.0 for new layers in the published small-model fine-tunes
    NEW = ("net.classifier.out", "net.body.cond", "net.body.add_embed", "net.body.local_physics", "net.body.embed", "net.body.interaction")

    def __init__(self, use_pid, pretrained=True, n_out=2):
        super().__init__()
        from omnilearned.network import PET2
        from omnilearned.utils import get_model_parameters
        self.use_pid = use_pid
        self.net = PET2(input_dim=4, use_int=True, local_int=True, pid=use_pid, pid_dim=9, add_info=False,
                        mode="classifier", num_classes=n_out, **get_model_parameters("small"))
        ck = torch.load(OMNI_CKPT, map_location="cpu", weights_only=False)
        self.pre_W = ck["classifier_head"]["out.weight"].float().clone()
        self.pre_b = ck["classifier_head"]["out.bias"].float().clone()
        names = [f"row{i}" for i in range(len(self.pre_b))]
        for i, n in enumerate(JETCLASS):
            names[2 + i] = n  # JetClass label offset 2 in the pretraining label map
        names[0], names[1] = "ATLAS_0", "ATLAS_1"
        self.row_names = names
        if pretrained:
            own = self.net.body.state_dict()
            body = {k: v for k, v in ck["body"].items() if k in own and own[k].shape == v.shape}
            self.net.body.load_state_dict(body, strict=False)
            head = {k: v for k, v in ck["classifier_head"].items() if not k.startswith("out.")}
            self.net.classifier.load_state_dict(head, strict=False)
            self.loaded = sorted(body)
        self._nwd = set(self.net.no_weight_decay()) if hasattr(self.net, "no_weight_decay") else set()

    @property
    def head(self):
        return self.net.classifier.out

    def no_decay(self, name, p):
        return p.ndim < 2 or any(k in name for k in self._nwd)

    def is_head(self, name):
        return name.startswith(self.NEW)

    def embed(self, b):
        X, pid = omni_inputs(b, self.use_pid)
        body = self.net.body(X, None, pid, None, torch.zeros(X.shape[0], device=X.device))
        c = self.net.classifier
        mask = body[:, c.num_tokens:, 2:3] != 0
        x = body
        for blk in c.in_blocks:
            x = blk(x, mask=mask)
        return c.fc(x[:, :c.num_tokens].reshape(len(x), -1))

    def forward(self, b):
        return self.head(self.embed(b))

    def pair_scale_(self, h, f):
        fc2 = self.net.body.interaction.mlp.fc2
        with torch.no_grad():
            fc2.weight[h] *= f
            fc2.bias[h] *= f

    def pair_state(self):
        fc2 = self.net.body.interaction.mlp.fc2
        return {"row_norm": fc2.weight.detach().norm(dim=1).cpu().tolist(), "bias": fc2.bias.detach().cpu().tolist()}


def build(kind, ds, pretrained=True):
    use_pid = ds == "qg"
    m = ParTModel(use_pid, pretrained) if kind == "part" else OmniModel(use_pid, pretrained)
    return m.to(DEV)


# -------------------------------------------------------------------------------------------------------------- metrics


def rej(score, y, eff):
    s, b = score[y == 1], score[y == 0]
    thr = np.quantile(s, 1 - eff)
    eb = (b >= thr).mean()
    return float(1 / eb) if eb > 0 else float("inf")


def metrics(score, y):
    from sklearn.metrics import roc_auc_score
    return {"auc": float(roc_auc_score(y, score)), "r30": rej(score, y, 0.3), "r50": rej(score, y, 0.5), "n": int(len(y))}


@torch.no_grad()
def predict(model, data, bs=None, rows=None):
    model.eval()
    bs = bs or (256 if model.kind == "omni" else 1024)  # OmniLearned's pair MLP holds B x P x P x 256 activations
    rows = np.arange(len(data)) if rows is None else rows
    out = []
    for i in range(0, len(rows), bs):
        b = data.batch(rows[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            out.append(model(b).float())
    return torch.cat(out)


def evaluate(model, data, rows=None):
    lg = predict(model, data, rows=rows)
    y = data.y if rows is None else data.y[rows]
    loss = float(F.cross_entropy(lg, torch.as_tensor(y, device=lg.device)))
    m = metrics((lg[:, 1] - lg[:, 0]).cpu().numpy(), y)
    m["loss"] = loss
    return m


# ---------------------------------------------------------------------------------------------------------- fine-tuning


class Lookahead(torch.optim.Optimizer):
    """Ranger = RAdam + Lookahead(k=6, alpha=0.5), as weaver's --optimizer ranger."""

    def __init__(self, base, k=6, alpha=0.5):
        self.base, self.k, self.alpha, self._n = base, k, alpha, 0
        self.param_groups = base.param_groups
        self.defaults = base.defaults
        self.state = base.state
        self.slow = [[p.detach().clone() for p in g["params"]] for g in base.param_groups]

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        self.base.step()
        self._n += 1
        if self._n % self.k == 0:
            for g, sl in zip(self.base.param_groups, self.slow):
                for p, s in zip(g["params"], sl):
                    s.add_(p - s, alpha=self.alpha)
                    p.copy_(s)


RECIPES = {
    # published fine-tune settings (ParT: train_TopLandscape.sh / train_QuarkGluon.sh; OmniLearned: experiment_commands.md, *_s)
    ("part", "ft"): dict(opt="ranger", lr=1e-4, wd=0.01, batch=512, epochs=20, sched="none"),
    ("part", "scratch"): dict(opt="ranger", lr=1e-3, wd=0.01, batch=512, epochs=20, sched="flat+decay"),
    ("omni", "ft"): dict(opt="lion", lr=5e-6, wd=0.1, batch=64, epochs=10, sched="cosine", warmup=0.1),
    ("omni", "scratch"): dict(opt="lion", lr=5e-4, wd=0.5, batch=64, epochs=15, sched="cosine", warmup=0.1),
}


def make_opt(model, rc, head_mult):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (model.is_head(n), model.no_decay(n, p))
        groups.setdefault(key, []).append(p)
    pg = [{"params": ps, "lr": rc["lr"] * (head_mult if h else 1.0), "weight_decay": 0.0 if nd else rc["wd"]}
          for (h, nd), ps in groups.items()]
    for g in pg:
        g["initial_lr"] = g["lr"]
    if rc["opt"] == "ranger":
        return Lookahead(torch.optim.RAdam(pg, lr=rc["lr"], decoupled_weight_decay=True))
    from pytorch_optimizer import Lion
    return Lion(pg, lr=rc["lr"], betas=(0.95, 0.98))


def lr_factor(rc, step, total):
    if rc["sched"] == "none":
        return 1.0
    if rc["sched"] == "flat+decay":  # weaver default: flat, then exponential decay to 1% over the last 30%
        start = int(0.7 * total)
        return 1.0 if step < start else 0.01 ** ((step - start) / max(1, total - start))
    w = int(rc.get("warmup", 0) * total)
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, total - w)))


def finetune(model, tr, va, mode="ft", seed=0, min_steps=600, max_steps=None, patience=8, log=None, freeze_body=False):
    """Fine-tune with the published recipe; the step budget is the published number of epochs over the training jets,
    with a floor of `min_steps` for small budgets. Early stopping and model selection on the validation jets."""
    rc = dict(RECIPES[(model.kind, mode)])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bs = min(rc["batch"], len(tr))
    total = max(min_steps, int(math.ceil(rc["epochs"] * len(tr) / bs)))
    if max_steps:
        total = min(total, max_steps)
    every = max(25, total // 40)
    head_mult = model.head_lr_mult if mode == "ft" else 1.0
    if freeze_body:
        for n, p in model.named_parameters():
            p.requires_grad_(model.is_head(n) and n.startswith(("head.", "net.classifier.out")))
    opt = make_opt(model, rc, head_mult)
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best, best_loss, bad, hist = None, float("inf"), 0, []
    order, pos = rng.permutation(len(tr)), 0
    t0 = time.time()
    for step in range(total):
        model.train()
        if pos + bs > len(order):
            order, pos = rng.permutation(len(tr)), 0
        rows = order[pos:pos + bs]
        pos += bs
        f = lr_factor(rc, step, total)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * f
        b = tr.batch(rows)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            loss = F.cross_entropy(model(b).float(), b["y"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if (step + 1) % every == 0 or step + 1 == total:
            v = evaluate(model, va)
            hist.append({"step": step + 1, "train_loss": float(loss), "val_loss": v["loss"], "val_auc": v["auc"], "t": time.time() - t0})
            if v["loss"] < best_loss - 1e-5:
                best_loss, bad = v["loss"], 0
                best = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience and step + 1 >= min_steps:
                    break
            if log:
                log(hist[-1])
    if best is not None:
        model.load_state_dict(best)
    for p in model.parameters():
        p.requires_grad_(True)
    disp = displacement(start, model.state_dict())
    return {"steps_budget": total, "steps_run": hist[-1]["step"] if hist else 0, "best_val_loss": best_loss,
            "time_s": time.time() - t0, "hist": hist, "recipe": rc, "displacement": disp}


def displacement(a, b):
    """Relative weight change per tensor, ||b - a|| / ||a|| (Story 2)."""
    out = {}
    for k in a:
        if a[k].dtype.is_floating_point and a[k].numel() > 1 and "running" not in k:
            na = float(a[k].float().norm())
            out[k] = float((b[k].float() - a[k].float()).norm()) / (na + 1e-12)
    return out


def lp_head(model, tr, wd=1e-3, max_jets=200_000):
    """C1 (LP-FT): logistic regression on frozen features, written into the head."""
    rows = np.arange(len(tr))[:max_jets]
    Z = []
    model.eval()
    with torch.no_grad():
        bs = 256 if model.kind == "omni" else 1024
        for i in range(0, len(rows), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                Z.append(model.embed(tr.batch(rows[i:i + bs])).float())
    Z = torch.cat(Z)
    y = torch.as_tensor(tr.y[rows], device=Z.device)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    A = (Z - mu) / sd
    W = torch.zeros(2, Z.shape[1], device=Z.device, requires_grad=True)
    c = torch.zeros(2, device=Z.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, c], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = F.cross_entropy(A @ W.T + c, y) + wd * (W ** 2).sum()
        l.backward()
        return l
    opt.step(closure)
    with torch.no_grad():
        model.head.weight.copy_(W / sd)
        model.head.bias.copy_(c - (W / sd) @ mu)


# ------------------------------------------------------------------------------------------------------- weight patches


def resolve(model, spec):
    """{'Tbqq': 1.0, 'QCD': -0.2} or {'row57': 1} -> weight row and bias as linear combinations of pretrained output rows."""
    w = torch.zeros_like(model.pre_W[0])
    b = torch.zeros(())
    for name, c in spec.items():
        k = model.row_names.index(name) if name in model.row_names else int(str(name).replace("row", ""))
        w += float(c) * model.pre_W[k]
        b += float(c) * model.pre_b[k]
    return w.to(DEV), b.to(DEV)


def apply_patch(model, patch):
    """Apply a list of weight edits in place. Returns a log of what changed. Allowed ops (all direct weight edits):
      head_from_rows {signal: {row: coef}, background: {row: coef}}   output weights = combinations of pretrained rows
      head_mix       {alpha, signal, background}                      output = (1-alpha) current + alpha combination
      head_scale     {factor}                                         multiply output weights and biases
      head_bias      {delta}                                          add delta to the signal output bias
      pair_scale     {head, factor in [0, 10]}                        scale pair-interaction head h (0 = off)
    """
    done = []
    H = model.head
    with torch.no_grad():
        for op in patch:
            kind = op.get("op")
            if kind in ("head_from_rows", "head_mix"):
                ws, bs = resolve(model, op["signal"])
                wb, bb = resolve(model, op["background"])
                a = float(op.get("alpha", 1.0)) if kind == "head_mix" else 1.0
                newW = torch.stack([wb, ws])
                newb = torch.stack([bb, bs]).to(DEV)
                H.weight.copy_((1 - a) * H.weight + a * newW)
                H.bias.copy_((1 - a) * H.bias + a * newb)
            elif kind == "head_scale":
                f = float(op["factor"])
                assert 0 < abs(f) <= 100
                H.weight.mul_(f)
                H.bias.mul_(f)
            elif kind == "head_bias":
                H.bias[1] += float(op["delta"])
            elif kind == "pair_scale":
                h, f = int(op["head"]), float(op["factor"])
                assert 0 <= h < 8 and 0 <= f <= 10
                model.pair_scale_(h, f)
            else:
                raise ValueError(f"unknown op {kind}")
            done.append(op)
    return done


def script_patch(ds):
    """C2: the obvious edit, fixed before any run."""
    if ds == "top":
        return [{"op": "head_from_rows", "signal": {"Tbqq": 1.0}, "background": {"QCD": 1.0}}]
    return [{"op": "head_from_rows", "signal": {"Zqq": 1.0}, "background": {"Hgg": 1.0}}]


def random_patch_like(model, llm_patch, seed):
    """C3: same tensors as the LLM's patch, same size, random direction / random head and sign."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    out = []
    H = model.head
    for op in llm_patch:
        if op["op"] in ("head_from_rows", "head_mix"):
            tmp = copy.deepcopy(model)
            apply_patch(tmp, [op])
            with torch.no_grad():
                for r in range(2):
                    w = torch.randn(H.weight.shape[1], generator=g).to(DEV)
                    H.weight[r] = w / w.norm() * tmp.head.weight[r].norm()
                H.bias.copy_(tmp.head.bias)
            del tmp
            out.append({"op": "random_head", "from": op["op"]})
        elif op["op"] == "pair_scale":
            f = float(op["factor"])
            h = int(rng.integers(8))
            if f == 0:  # switch off a random head instead
                g2 = 0.0
            else:
                g2 = min(10.0, math.exp(abs(math.log(f)) * rng.choice([-1, 1])))
            model.pair_scale_(h, g2)
            out.append({"op": "pair_scale", "head": h, "factor": g2})
        else:
            apply_patch(model, [op])
            out.append(op)
    return out


# ------------------------------------------------------------------------------------------------------------ diagnostics


@torch.no_grad()
def diagnostics(model, data, max_jets=4000, top_rows=12):
    """What the LLM is shown: computed only on the given (training-split) jets."""
    rows = np.arange(len(data))[:max_jets]
    model.eval()
    Z = []
    bs = 256 if model.kind == "omni" else 1024
    for i in range(0, len(rows), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            Z.append(model.embed(data.batch(rows[i:i + bs])).float())
    Z = torch.cat(Z)
    y = data.y[rows]
    lg = model.head(Z)
    cur = metrics((lg[:, 1] - lg[:, 0]).cpu().numpy(), y)
    cur["loss"] = float(F.cross_entropy(lg, torch.as_tensor(y, device=lg.device)))
    pre = (Z @ model.pre_W.T.to(DEV) + model.pre_b.to(DEV)).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    rows_info = []
    for k in range(pre.shape[1]):
        a = roc_auc_score(y, pre[:, k])
        rows_info.append({"row": model.row_names[k], "auc_alone": round(float(a), 4),
                          "mean_logit_signal": round(float(pre[y == 1, k].mean()), 3),
                          "mean_logit_background": round(float(pre[y == 0, k].mean()), 3)})
    jc = [r for r in rows_info if r["row"] in JETCLASS]
    others = sorted([r for r in rows_info if r["row"] not in JETCLASS], key=lambda r: -abs(r["auc_alone"] - 0.5))[:top_rows]
    wdiff = (model.head.weight[1] - model.head.weight[0]).float()
    cos = {}
    for r in jc:
        k = model.row_names.index(r["row"])
        cos[r["row"]] = round(float(F.cosine_similarity(wdiff, model.pre_W[k].to(DEV).float(), 0)), 3)
    # effect of switching off each pair-interaction head on these jets
    abl = []
    sub = rows[: min(len(rows), 2000)]
    base_auc = metrics(_score(model, data, sub), y[: len(sub)])["auc"]
    for h in range(8):
        m2 = copy.deepcopy(model)
        m2.pair_scale_(h, 0.0)
        abl.append(round(metrics(_score(m2, data, sub), y[: len(sub)])["auc"] - base_auc, 4))
        del m2
    return {"n_jets": int(len(rows)), "signal_fraction": round(float(y.mean()), 3), "current": {k: round(v, 4) for k, v in cur.items()},
            "jetclass_rows": jc, "other_rows_most_separating": others, "cos_head_direction_vs_rows": cos,
            "pair_heads": {"state": model.pair_state(), "delta_auc_if_switched_off": abl}}


def _score(model, data, rows):
    lg = predict(model, data, rows=rows)
    return (lg[:, 1] - lg[:, 0]).cpu().numpy()


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)
