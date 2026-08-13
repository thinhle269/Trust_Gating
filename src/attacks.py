"""Traffic threat models, per description.docx section 1.

Three attacks on compromised RSUs / roadside sensors:

  Byzantine_Noise   a faulty or hijacked sensor submits a maximally noisy weight
                    matrix, deflecting global convergence
  Label_Flipping    the sensor inverts the traffic state it reports
                    (free-flow <-> severe congestion)
  Sybil_Poisoning   a spoofed edge device reports a fixed, false speed to
                    corrupt signal-control and routing decisions

IMPORTANT DESIGN NOTE. Label_Flipping and Sybil_Poisoning poison the DATA, so
the attacker trains on corrupted targets and its update genuinely points toward
a poisoned optimum. Byzantine_Noise corrupts the UPDATE directly. This
distinction matters: an update that is merely geometrically odd but carries no
harmful content tends to be averaged away, because the next round's honest local
training corrects it. Damage requires harmful content, so two of the three
attacks are applied at the data level rather than as post-hoc vector noise.

Adversary model: controls a fixed fraction of clients; knows its own data and
the broadcast model; cannot corrupt the server or read honest clients' raw data.
"""
from __future__ import annotations

import numpy as np
import torch

ATTACKS = ("No_Attack", "Byzantine_Noise", "Label_Flipping", "Sybil_Poisoning")

# Attacks whose attacker trains on POISONED DATA.
POISONS_DATA = frozenset({"Label_Flipping", "Sybil_Poisoning"})
# Attacks that transform the submitted update.
POISONS_UPDATE = frozenset({"Byzantine_Noise"})

DESCRIPTIONS = {
    "No_Attack": "All clients honest.",
    "Byzantine_Noise": "Compromised RSU submits a maximally noisy weight update.",
    "Label_Flipping": "Sensor inverts reported traffic state (free-flow <-> congested).",
    "Sybil_Poisoning": "Spoofed device reports a fixed false speed to skew control decisions.",
}


def select_malicious(n_clients: int, frac: float, seed: int) -> np.ndarray:
    n_bad = int(np.floor(frac * n_clients + 0.5))
    if n_bad <= 0:
        return np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed + 991)
    return np.sort(rng.choice(n_clients, min(n_bad, n_clients), replace=False))


def poison_targets(y: np.ndarray, attack: str, rng: np.random.Generator) -> np.ndarray:
    """Data-space poisoning of a client's NORMALISED training targets."""
    if attack == "Label_Flipping":
        # Negating a normalised target reflects it about the client's own mean
        # speed: free-flow becomes congested and vice versa. This is the
        # regression analogue of label flipping and attacks the downstream
        # congestion classification directly.
        return (-y).astype(np.float32)

    if attack == "Sybil_Poisoning":
        # A spoofed device reports one fixed, low speed regardless of reality.
        # In normalised units a strongly negative constant is well below any
        # client's mean, i.e. permanent fake congestion.
        return np.full_like(y, -1.5, dtype=np.float32)

    return y


def poison_update(update: torch.Tensor, attack: str,
                  benign: list[torch.Tensor] | None = None,
                  boost: float = 10.0,
                  gen: torch.Generator | None = None) -> torch.Tensor:
    """Model-space poisoning of `update = theta_local - theta_global`."""
    if attack != "Byzantine_Noise":
        return update

    ref = (torch.stack([u.norm() for u in benign]).median()
           if benign else update.norm().clamp_min(1e-12))
    noise = torch.randn(update.shape, device=update.device,
                        dtype=update.dtype, generator=gen)
    # Maximal-noise submission: random direction at a magnitude far above the
    # honest median, which is what "maximally noisy weight matrix" means.
    return noise / noise.norm().clamp_min(1e-12) * ref * boost
