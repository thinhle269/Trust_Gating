"""Local spatiotemporal forecaster and flat-parameter utilities.

The supplied code represented the "global model" as `np.random.normal(0,0.1,100)`
and its "gradients" as further random draws, so nothing was ever learned. This
module provides a real model: a GRU over the 12-step input window, trained by
gradient descent on real traffic windows.

Aggregation rules operate on one flat vector per client, so the helpers that
move between module state and R^d live here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpatioTemporalGRU(nn.Module):
    """Single-layer GRU over (W, C) windows predicting the next-step speed.

    The head is linear. Normalised speed is routinely negative (below the
    client's own mean), so a ReLU head could not represent half the target
    range.
    """

    def __init__(self, n_channels: int = 4, hidden: int = 48) -> None:
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


def build_model(n_channels: int = 4, hidden: int = 48) -> nn.Module:
    return SpatioTemporalGRU(n_channels, hidden)


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model: nn.Module, flat: torch.Tensor) -> None:
    total = sum(p.numel() for p in model.parameters())
    if flat.numel() != total:
        # Check before writing: a partial copy leaves the model in an
        # inconsistent state that is very hard to trace later.
        raise ValueError(f"expected {total} parameters, got {flat.numel()}")
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[i:i + n].view_as(p))
            i += n


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def geometric_median(points: torch.Tensor, n_iter: int = 80,
                     eps: float = 1e-7) -> torch.Tensor:
    """Weiszfeld iteration over (K, d) points.

    Used as the robust consensus direction that directional similarity is
    measured against. The arithmetic mean would be the wrong reference: it is
    exactly the quantity a coalition is trying to drag, so a large enough
    coalition could define "normal" and then score itself as conforming to it.
    The geometric median has breakdown point 1/2.
    """
    med = points.mean(0)
    for _ in range(n_iter):
        dist = torch.norm(points - med.unsqueeze(0), dim=1).clamp_min(eps)
        w = 1.0 / dist
        new = (w.unsqueeze(1) * points).sum(0) / w.sum()
        if torch.norm(new - med) < eps:
            return new
        med = new
    return med
