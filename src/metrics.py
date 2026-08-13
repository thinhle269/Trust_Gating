"""Evaluation metrics, all measured from predictions.

In the supplied code F1 was assigned by formula and MAE/RMSE/Accuracy were
algebraic transforms of it. Here every quantity is computed from the model's
actual predictions on the held-out test split, in physical units (mph).

On AUC for an ordered 5-class problem driven by a regressor: the model emits a
scalar speed, not a class distribution, so a conventional one-vs-rest AUC would
require inventing class probabilities. The headline is therefore the ORDINAL
AUC -- for each of the four congestion boundaries the predicted speed ranks the
binary question "is the true class above this boundary?", macro-averaged. It is
threshold-free, invents nothing, and is exactly the discrimination a routing
layer needs.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .datasets import CLASS_EDGES_MPH, N_CLASSES, speed_to_class


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    denom = np.maximum(np.abs(y_true), 1.0)      # 1 mph floor; MAPE is unstable near 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / denom) * 100.0),
    }


def ordinal_auc(y_cls: np.ndarray, score: np.ndarray) -> tuple[float, list]:
    aucs = []
    for c in range(1, N_CLASSES):
        b = (y_cls >= c).astype(int)
        aucs.append(float(roc_auc_score(b, score)) if len(np.unique(b)) > 1 else float("nan"))
    ok = [a for a in aucs if np.isfinite(a)]
    return (float(np.mean(ok)) if ok else float("nan")), aucs


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt, yp = speed_to_class(y_true), speed_to_class(y_pred)
    labels = list(range(N_CLASSES))
    macro_auc, per_boundary = ordinal_auc(yt, y_pred)
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "macro_recall": float(recall_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "macro_precision": float(precision_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)),
        "auc": macro_auc,
        "auc_per_boundary": per_boundary,
        "confusion_matrix": confusion_matrix(yt, yp, labels=labels).tolist(),
        "support": np.bincount(yt, minlength=N_CLASSES).tolist(),
        "per_class_recall": recall_score(yt, yp, average=None, labels=labels, zero_division=0).tolist(),
        "per_class_f1": f1_score(yt, yp, average=None, labels=labels, zero_division=0).tolist(),
    }


def trust_metrics(trust: np.ndarray, is_malicious: np.ndarray,
                  weights: np.ndarray | None = None, tau: float = 0.35) -> dict:
    """How well a rule's own trust/weighting separates honest from malicious.

    `trust_auc` is rank-based, so it is invariant to the arbitrary scale each
    rule uses and is the only quantity comparable across rules.

    `malicious_weight_mass` -- the share of aggregation weight that reached
    attackers -- needs no threshold at all and is what actually predicts damage.
    """
    is_mal = np.asarray(is_malicious, bool)
    t = np.asarray(trust, float)

    out = {"trust_auc": float("nan"), "malicious_weight_mass": float("nan"),
           "tdr": float("nan"), "fpr": float("nan"),
           "mean_trust_benign": float("nan"), "mean_trust_malicious": float("nan"),
           "trust_margin": float("nan")}

    if (~is_mal).any():
        out["mean_trust_benign"] = float(t[~is_mal].mean())
    if is_mal.any():
        out["mean_trust_malicious"] = float(t[is_mal].mean())
    if weights is not None:
        w = np.asarray(weights, float)
        tot = np.abs(w).sum()
        out["malicious_weight_mass"] = float(np.abs(w[is_mal]).sum() / tot) if tot > 0 else 0.0

    if is_mal.all() or not is_mal.any():
        return out

    out["trust_auc"] = float(roc_auc_score(is_mal.astype(int), -t))
    out["trust_margin"] = out["mean_trust_benign"] - out["mean_trust_malicious"]
    blocked = t < tau
    out["tdr"] = float((blocked & is_mal).sum() / max(is_mal.sum(), 1))
    out["fpr"] = float((blocked & ~is_mal).sum() / max((~is_mal).sum(), 1))
    return out


def aggregate_seeds(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[f"{k}_mean"] = out[f"{k}_std"] = float("nan")
            out[f"{k}_n"] = 0
        else:
            out[f"{k}_mean"] = float(v.mean())
            out[f"{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            out[f"{k}_n"] = int(len(v))
    return out


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon + Cohen's d across matched seeds.

    NOTE on power: with n=5 the smallest attainable two-sided p is 2/2^5 =
    0.0625, so significance at 0.05 is impossible below 6 seeds.
    """
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2:
        return {"n": int(len(a)), "wilcoxon_p": float("nan"),
                "cohens_d": float("nan"), "mean_diff": float("nan")}
    d = a - b
    try:
        p = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    sd = d.std(ddof=1)
    return {"n": int(len(a)), "wilcoxon_p": p,
            "cohens_d": float(d.mean() / sd) if sd > 1e-12 else 0.0,
            "mean_diff": float(d.mean())}
