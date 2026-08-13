"""The real federated training engine.

This replaces the supplied `train_eval.py`, in which the global model was a
random 100-vector, "gradients" were random draws, and every reported metric was
assigned by formula -- including an explicit penalty applied to the baselines by
name (PROJECT_SPEC.md §2).

Here each round performs genuine local SGD on real traffic windows, and every
metric is measured from the resulting predictions.

The three ANFIS antecedents, computed server-side each round:

  x1  validation F1  -- macro-F1 of the congestion classes obtained by applying
                        this client's update to the global model and evaluating
                        on the server's small clean root set. Measured, not
                        reported by the client: a dishonest client would lie.
  x2  directional similarity -- cosine between the client's update and the
                        geometric-median consensus, mapped to [0,1].
  x3  cleanliness ratio -- 1 - anomaly ratio, where the anomaly ratio is the
                        client's update norm relative to the cohort median,
                        squashed to [0,1]. Inflated or vanishing updates both
                        read as unclean.

The online teacher for ANFIS learning is derived from evidence the server can
observe, never from a maliciousness label (which does not exist at deployment
and would make the evaluation circular).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

from . import attacks as A
from .baselines import (LOCAL_VARIANTS, NEEDS_SERVER_UPDATE, TRUST_RULES,
                        AggContext, FLAggregator)
from .datasets import (Client, N_CHANNELS, server_root_set, speed_to_class)
from .metrics import (classification_metrics, regression_metrics, trust_metrics)
from .models import build_model, geometric_median, get_flat, n_params, set_flat
from .provenance import source_fingerprint


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dev(a, device, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=device)


def local_train(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                cfg: C.RunConfig, global_flat: torch.Tensor,
                gen: torch.Generator,
                c_local: torch.Tensor | None = None,
                c_global: torch.Tensor | None = None,
                min_steps: int | None = None) -> tuple[torch.Tensor, int]:
    """E epochs of genuine local SGD. Returns (flat weights, steps taken).

    `min_steps` exists for FLTrust: its root set is far smaller than a client's
    shard, so at equal EPOCHS the server takes an order of magnitude fewer
    optimisation steps. Since FLTrust rescales every client update to the server
    update's norm, an undertrained server shrinks all honest updates and the
    method fails for a reason unrelated to its merit.
    """
    model.train()
    opt = (torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
           if cfg.algorithm == "SCAFFOLD"
           else torch.optim.Adam(model.parameters(), lr=cfg.lr))

    n = len(x)
    per_epoch = max(1, int(np.ceil(n / cfg.batch_size)))
    n_epochs = cfg.local_epochs
    if min_steps is not None:
        n_epochs = max(n_epochs, int(np.ceil(min_steps / per_epoch)))

    steps = 0
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=x.device, generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad()
            loss = F.l1_loss(model(x[idx]), y[idx])

            if cfg.algorithm == "FedProx":
                prox = torch.zeros((), device=x.device)
                j = 0
                for p in model.parameters():
                    m = p.numel()
                    prox = prox + ((p.reshape(-1) - global_flat[j:j + m]) ** 2).sum()
                    j += m
                loss = loss + 0.5 * cfg.fedprox_mu * prox

            loss.backward()
            opt.step()

            if cfg.algorithm == "SCAFFOLD" and c_local is not None:
                with torch.no_grad():
                    set_flat(model, get_flat(model) - cfg.lr * (c_global - c_local))
            steps += 1
    return get_flat(model), steps


@torch.no_grad()
def evaluate(model: nn.Module, flat: torch.Tensor, clients: list[Client],
             split: str, device: str) -> dict:
    """Pooled evaluation in physical mph, denormalised per client."""
    set_flat(model, flat)
    model.eval()
    yt, yp = [], []
    for c in clients:
        x, y = c.split(split)
        if len(x) == 0:
            continue
        pred = model(_dev(x, device)).cpu().numpy()
        yt.append(c.denorm(y))
        yp.append(c.denorm(pred))
    if not yt:
        return {}
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    out = regression_metrics(yt, yp)
    out.update(classification_metrics(yt, yp))
    return out


@torch.no_grad()
def _root_f1(model: nn.Module, flat: torch.Tensor, rx: torch.Tensor,
             ry_mph: np.ndarray, mu: float, sd: float) -> float:
    """Macro-F1 of congestion classes on the server root set, in mph space."""
    from sklearn.metrics import f1_score
    set_flat(model, flat)
    model.eval()
    pred = model(rx).cpu().numpy() * sd + mu
    yt, yp = speed_to_class(ry_mph), speed_to_class(pred)
    return float(f1_score(yt, yp, average="macro", labels=list(range(5)),
                          zero_division=0))


@torch.no_grad()
def trust_features(updates: torch.Tensor, global_flat: torch.Tensor,
                   work: nn.Module, rx: torch.Tensor, ry_mph: np.ndarray,
                   mu: float, sd: float) -> np.ndarray:
    """The three ANFIS antecedents, all measured server-side. Returns (K,3)."""
    k = len(updates)
    feats = np.zeros((k, 3))

    # x1 -- validation F1 of theta_global + delta_k on the clean root set.
    for i in range(k):
        feats[i, 0] = _root_f1(work, global_flat + updates[i], rx, ry_mph, mu, sd)
    set_flat(work, global_flat)

    # x2 -- directional similarity to the ROBUST consensus, mapped to [0,1].
    consensus = geometric_median(updates)
    cn = consensus.norm().clamp_min(1e-12)
    un = updates.norm(dim=1).clamp_min(1e-12)
    cos = ((updates @ consensus) / (un * cn)).cpu().numpy()
    feats[:, 1] = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)

    # x3 -- cleanliness. Deviation of the update norm from the cohort median, in
    # log space so that a 10x inflation and a 10x shrinkage are equally anomalous.
    norms = un.cpu().numpy()
    med = float(np.median(norms)) + 1e-12
    dev = np.abs(np.log(np.clip(norms / med, 1e-6, None)))
    feats[:, 2] = np.clip(1.0 - dev / 3.0, 0.0, 1.0)

    return feats


def online_teacher(feats: np.ndarray) -> np.ndarray:
    """Server-side supervision target for the ANFIS.

    At deployment the server has no ground-truth maliciousness label; assuming
    one would make the whole evaluation circular. The target is instead derived
    from observable evidence: an update is desirable to the extent that it
    improves the root-set F1 AND agrees with the consensus AND looks clean.

    The F1 channel is rank-normalised because on a small root set only the
    ORDERING of clients is trustworthy, not the absolute value.

    WEIGHTING, and why it is not uniform. x1 dominates at 0.85 because it is the
    only channel measured against clean data the adversary cannot touch, and the
    only one that cannot be gamed by collusion. x2 is actively misleading under a
    colluding adversary -- measured across all three datasets, attackers scored
    HIGHER directional similarity than honest clients (0.78-0.85 against
    0.63-0.66), because clients performing the same corruption agree with one
    another while honest clients under non-IID data disagree in diverse ways.
    Giving x2 and x3 a combined 0.15 keeps them as context without letting an
    inverted channel drag the target.

    The earlier 0.5/0.3/0.2 split compressed the teacher's own range: malicious
    clients whose root-set F1 was 0.13 against 0.93 for honest ones still
    received targets only ~0.31 apart, and the ANFIS reproduced an even narrower
    gap (0.10-0.19), which is what starved the gate downstream.
    """
    f1 = feats[:, 0]
    rank = np.argsort(np.argsort(f1)) / max(len(f1) - 1, 1)
    context = 0.5 * (feats[:, 1] + feats[:, 2])
    return np.clip(0.85 * rank + 0.15 * context, 0.0, 1.0)


def run(clients: list[Client], cfg: C.RunConfig, data_cfg: C.DataConfig,
        verbose: bool = False) -> dict:
    _seed_all(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    k = len(clients)
    t0 = time.time()

    gen = torch.Generator(device=device); gen.manual_seed(cfg.seed)
    tgen = torch.Generator(device=device); tgen.manual_seed(cfg.seed + 7)
    rng = np.random.default_rng(cfg.seed)

    malicious = A.select_malicious(k, cfg.malicious_frac, cfg.seed)
    is_mal = np.zeros(k, bool)
    if cfg.attack != "No_Attack":
        is_mal[malicious] = True

    # ---- client data, with data-level poisoning applied to attackers ----
    xs, ys = [], []
    for i, c in enumerate(clients):
        y = c.y_train.copy()
        if is_mal[i] and cfg.attack in A.POISONS_DATA:
            y = A.poison_targets(y, cfg.attack, rng)
        xs.append(_dev(c.x_train, device))
        ys.append(_dev(y, device))

    # ---- server root set ------------------------------------------------
    rx_np, ry_np = server_root_set(clients, data_cfg.root_set_size, seed=cfg.seed)
    rx, ry = _dev(rx_np, device), _dev(ry_np, device)
    mu = float(np.mean([c.mean for c in clients]))
    sd = float(np.mean([c.std for c in clients]))
    ry_mph = ry_np * sd + mu

    # ---- models ---------------------------------------------------------
    global_model = build_model(N_CHANNELS, cfg.hidden).to(device)
    work = build_model(N_CHANNELS, cfg.hidden).to(device)
    global_flat = get_flat(global_model).clone()
    d = n_params(global_model)

    c_global = torch.zeros(d, device=device)
    c_local = [torch.zeros(d, device=device) for _ in range(k)]
    c_delta = [torch.zeros(d, device=device) for _ in range(k)]

    agg = FLAggregator(cfg.algorithm, cfg.anfis_lr_premise,
                       cfg.anfis_lr_consequent, cfg.seed)

    logs, trust_rows = [], []
    weights_np = np.full(k, 1.0 / k)
    trust_np = np.full(k, 0.5)

    for r in range(1, cfg.rounds + 1):
        updates = torch.zeros(k, d, device=device)
        sizes = torch.zeros(k, device=device)
        benign: list[torch.Tensor] = []

        # ---- local training ---------------------------------------------
        for i in range(k):
            set_flat(work, global_flat)
            new_flat, steps = local_train(
                work, xs[i], ys[i], cfg, global_flat, gen,
                c_local[i] if cfg.algorithm == "SCAFFOLD" else None,
                c_global if cfg.algorithm == "SCAFFOLD" else None)
            upd = new_flat - global_flat

            if cfg.algorithm == "SCAFFOLD":
                c_new = c_local[i] - c_global - upd / max(steps * cfg.lr, 1e-12)
                c_delta[i] = c_new - c_local[i]
                c_local[i] = c_new

            updates[i] = upd
            sizes[i] = float(len(xs[i]))
            if not is_mal[i]:
                benign.append(upd)

        # ---- update-level poisoning -------------------------------------
        for i in range(k):
            if is_mal[i] and cfg.attack in A.POISONS_UPDATE:
                updates[i] = A.poison_update(updates[i], cfg.attack,
                                             benign=benign, gen=tgen)

        # ---- server reference update (FLTrust) --------------------------
        server_update = None
        if cfg.algorithm in NEEDS_SERVER_UPDATE:
            set_flat(work, global_flat)
            matched = int(np.median([
                max(1, int(np.ceil(len(xs[i]) / cfg.batch_size))) * cfg.local_epochs
                for i in range(k)]))
            su, _ = local_train(work, rx, ry, cfg, global_flat, gen,
                                min_steps=matched)
            server_update = su - global_flat

        # ---- trust features ---------------------------------------------
        feats = None
        teacher = None
        if cfg.algorithm in TRUST_RULES:
            feats = trust_features(updates, global_flat, work, rx, ry_mph, mu, sd)
            teacher = online_teacher(feats)

        ctx = AggContext(updates=updates, sizes=sizes, trust_features=feats,
                         server_update=server_update, teacher=teacher,
                         tau=cfg.zt_tau, n_byz=cfg.n_byz_assumed,
                         beta=cfg.mamdani_beta)
        delta, weights_np, trust_np = agg.aggregate(ctx)
        global_flat = global_flat + delta

        if cfg.algorithm == "SCAFFOLD":
            c_global = c_global + torch.stack(c_delta).mean(0)

        for i in range(k):
            row = {"round": r, "client": i, "is_malicious": bool(is_mal[i]),
                   "trust": float(trust_np[i]), "weight": float(weights_np[i])}
            if feats is not None:
                row.update(x1_val_f1=feats[i, 0], x2_similarity=feats[i, 1],
                           x3_cleanliness=feats[i, 2])
            trust_rows.append(row)

        if r % cfg.eval_every == 0 or r == cfg.rounds:
            m = evaluate(global_model, global_flat, clients, "val", device)
            tau_now = agg.tau_history[-1] if getattr(agg, "tau_history", None) else cfg.zt_tau
            tm = trust_metrics(trust_np, is_mal, weights_np, tau_now)
            logs.append({"round": r, **{f"val_{a}": b for a, b in m.items()
                                        if not isinstance(b, list)},
                         **{f"sec_{a}": b for a, b in tm.items()}})
            if verbose and (r % 10 == 0 or r == 1):
                print(f"    r{r:02d} MAE={m['mae']:.3f} Acc={m['accuracy']:.4f} "
                      f"F1={m['macro_f1']:.4f}")

    test = evaluate(global_model, global_flat, clients, "test", device)
    val = evaluate(global_model, global_flat, clients, "val", device)
    tau_final = agg.tau_history[-1] if getattr(agg, "tau_history", None) else cfg.zt_tau
    sec = trust_metrics(trust_np, is_mal, weights_np, tau_final)
    sec["tau_used"] = float(tau_final)

    out = {
        "config": {"dataset": cfg.dataset, "algorithm": cfg.algorithm,
                   "attack": cfg.attack, "seed": cfg.seed, "rounds": cfg.rounds,
                   "n_clients": k, "n_params": d,
                   "malicious_frac": cfg.malicious_frac, "zt_tau": cfg.zt_tau},
        "malicious_clients": malicious.tolist(),
        "test": test, "val": val, "security": sec,
        "logs": logs, "trust_rows": trust_rows,
        "wall_time_s": time.time() - t0,
        # Which code produced this number. The cache key covers RunConfig only,
        # so without this a constant edited inside baselines.py or this file
        # leaves cached results that look valid and are not.
        "source_fingerprint": source_fingerprint(),
    }
    if agg.anfis is not None:
        out["anfis_rules"] = agg.anfis.describe_rules()
        out["anfis_final_c"] = agg.anfis.c.tolist()
        out["anfis_final_sigma"] = agg.anfis.sigma.tolist()
    return out
