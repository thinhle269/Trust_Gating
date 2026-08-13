"""Real spatiotemporal traffic data: METR-LA, PEMS-BAY, PEMS04.

This replaces the supplied `datasets.py`, which generated synthetic sine-wave
data and never opened a dataset file (PROJECT_SPEC.md §2). Everything here reads
the actual benchmark files.

Properties that make the downstream numbers trustworthy:

* **No leakage.** Splits are chronological and taken per sensor BEFORE any
  statistic is computed. The scaler is fit on training targets only. Validation
  and test windows may read input history from up to `input_steps` before their
  boundary -- ordinary autoregressive lead-in -- but no target ever crosses a
  boundary. `verify_no_leakage()` asserts this instead of trusting it.

* **Missing data is masked, not imputed as zero.** In METR-LA 8.1% of entries
  are exactly 0.0, meaning the loop detector was offline, not that traffic was
  stationary. Treating those as real speeds corrupts both training and metrics.

* **Real heterogeneity.** Clients keep their sensors as separate series.
  Collapsing a client's sensors into one mean curve destroys the within-client
  variation that makes the federated problem non-trivial.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config as C

STEPS_PER_DAY = 288

# Congestion classes on physical speed (mph). US highway free-flow is ~65 mph.
CLASS_EDGES_MPH = (25.0, 40.0, 50.0, 60.0)
CLASS_NAMES = ("Severe", "Heavy", "Moderate", "Light", "Free")
N_CLASSES = len(CLASS_NAMES)

N_CHANNELS = 4  # speed, sin(tod), cos(tod), validity


def speed_to_class(speed_mph: np.ndarray) -> np.ndarray:
    """0 = Severe (slowest) ... 4 = Free. Monotone increasing in speed."""
    return np.digitize(speed_mph, CLASS_EDGES_MPH).astype(np.int64)


@dataclass
class Client:
    cid: int
    sensors: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: float                 # train-only scaler, physical units
    std: float
    t_train: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    t_val: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    t_test: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    def denorm(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

    def split(self, name: str):
        return getattr(self, f"x_{name}"), getattr(self, f"y_{name}")


# --------------------------------------------------------------- raw loading
def load_raw(dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (speed matrix (T, S) in physical units, coords (S,2) or None)."""
    meta = C.DATASETS[dataset]
    path = meta["file"]
    if not path.exists():
        raise FileNotFoundError(f"{dataset}: {path} not found")

    if meta["kind"] == "h5":
        import h5py
        with h5py.File(path, "r") as f:
            speed = np.asarray(f["data"]["block0_values"], dtype=np.float64)

    elif meta["kind"] == "csv":
        df = pd.read_csv(path, index_col=0)
        speed = df.to_numpy(dtype=np.float64)

    elif meta["kind"] == "npz":
        arr = np.load(path)["data"]                 # (T, S, F)
        speed = np.asarray(arr[:, :, meta["target_feature"]], dtype=np.float64)

    else:
        raise ValueError(f"unknown kind {meta['kind']}")

    coords = None
    if meta.get("coords") is not None and meta["coords"].exists():
        cdf = pd.read_csv(meta["coords"])
        if {"latitude", "longitude"} <= set(cdf.columns):
            coords = cdf[["latitude", "longitude"]].to_numpy(float)
            if len(coords) != speed.shape[1]:
                coords = None
    return speed, coords


def load_features(dataset: str) -> np.ndarray | None:
    """Full (T, S, F) tensor for multi-feature datasets; None otherwise.

    Used to derive a genuine occupancy channel for the fuzzy congestion index on
    PEMS04, rather than the dispersion proxy the speed-only datasets require.
    """
    meta = C.DATASETS[dataset]
    if meta["kind"] != "npz":
        return None
    return np.asarray(np.load(meta["file"])["data"], dtype=np.float64)


# --------------------------------------------------------------- windowing
def _windows(speed: np.ndarray, valid: np.ndarray, lo: int, hi: int,
             cfg: C.DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows whose TARGET lies in [lo, hi). History may precede `lo`."""
    W = cfg.input_steps
    targets = np.arange(max(lo, W), hi, cfg.stride, dtype=np.int64)
    if len(targets) == 0:
        return (np.zeros((0, W, N_CHANNELS), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int64))

    idx = targets[:, None] - np.arange(W, 0, -1)[None, :]
    hist, hist_ok = speed[idx], valid[idx]

    keep = valid[targets] & (hist_ok.mean(axis=1) >= (1.0 - cfg.max_missing_frac))
    if not keep.any():
        return (np.zeros((0, W, N_CHANNELS), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.int64))

    idx, hist, hist_ok, targets = idx[keep], hist[keep], hist_ok[keep], targets[keep]

    h = hist.astype(np.float64).copy()
    h[~hist_ok] = np.nan
    h = pd.DataFrame(h).ffill(axis=1).bfill(axis=1).to_numpy()
    h = np.nan_to_num(h, nan=float(speed[valid].mean()) if valid.any() else 0.0)

    tod = idx % STEPS_PER_DAY
    x = np.stack([h,
                  np.sin(2 * np.pi * tod / STEPS_PER_DAY),
                  np.cos(2 * np.pi * tod / STEPS_PER_DAY),
                  hist_ok.astype(np.float64)], axis=-1).astype(np.float32)
    return x, speed[targets].astype(np.float32), targets


def load(dataset: str, cfg: C.DataConfig | None = None,
         verbose: bool = True) -> tuple[list[Client], dict]:
    cfg = cfg or C.DataConfig()
    speed, coords = load_raw(dataset)
    T, S = speed.shape

    # Zeros denote an offline detector in these benchmarks, not zero speed.
    valid = speed > 0.0
    t_tr, t_va = int(T * cfg.train_frac), int(T * (cfg.train_frac + cfg.val_frac))

    # Client partition: geographic when coordinates exist, otherwise contiguous
    # blocks of sensor index, which in PeMS files follow highway ordering and so
    # remain spatially coherent (and therefore genuinely non-IID).
    rng = np.random.default_rng(cfg.seed)
    if coords is not None:
        from sklearn.cluster import KMeans
        z = (coords - coords.mean(0)) / coords.std(0)
        labels = KMeans(n_clusters=cfg.n_clients, n_init=10,
                        random_state=cfg.seed).fit_predict(z)
    else:
        labels = np.array_split(np.arange(S), cfg.n_clients)
        lab = np.zeros(S, dtype=int)
        for i, blk in enumerate(labels):
            lab[blk] = i
        labels = lab

    clients: list[Client] = []
    for k in range(cfg.n_clients):
        members = np.where(labels == k)[0]
        if len(members) == 0:
            continue

        parts: dict[str, list] = {"train": [], "val": [], "test": []}
        for s in members:
            for name, lo, hi in (("train", 0, t_tr), ("val", t_tr, t_va),
                                 ("test", t_va, T)):
                x, y, t = _windows(speed[:, s], valid[:, s], lo, hi, cfg)
                if len(x):
                    parts[name].append((x, y, t))
        if not parts["train"]:
            continue

        def cat(name):
            if not parts[name]:
                return (np.zeros((0, cfg.input_steps, N_CHANNELS), np.float32),
                        np.zeros(0, np.float32), np.zeros(0, np.int64))
            xs, ys, ts = zip(*parts[name])
            return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)

        xtr, ytr, ttr = cat("train")
        xva, yva, tva = cat("val")
        xte, yte, tte = cat("test")

        mu, sd = float(ytr.mean()), float(ytr.std()) or 1.0

        def prep(x, y, t, cap):
            if cap and len(x) > cap:
                sel = np.sort(rng.choice(len(x), cap, replace=False))
                x, y, t = x[sel], y[sel], t[sel]
            xn = x.copy()
            xn[..., 0] = (xn[..., 0] - mu) / sd
            return xn, ((y - mu) / sd).astype(np.float32), t

        xtr, ytr, ttr = prep(xtr, ytr, ttr, cfg.max_train_per_client)
        xva, yva, tva = prep(xva, yva, tva, cfg.max_eval_per_client)
        xte, yte, tte = prep(xte, yte, tte, cfg.max_eval_per_client)

        clients.append(Client(len(clients), members, xtr, ytr, xva, yva,
                              xte, yte, mu, sd, ttr, tva, tte))

    info = {
        "dataset": dataset, "n_sensors": int(S), "n_timesteps": int(T),
        "missing_frac": float((~valid).mean()), "n_clients": len(clients),
        "input_steps": cfg.input_steps, "stride": cfg.stride,
        "t_train_end": t_tr, "t_val_end": t_va,
        "class_edges_mph": list(CLASS_EDGES_MPH), "class_names": list(CLASS_NAMES),
        "partition": "kmeans-geographic" if coords is not None else "contiguous-index",
        "fingerprint": fingerprint(clients),
    }
    if verbose:
        print(f"[{dataset}] {S} sensors x {T} steps  missing={info['missing_frac']:.4f}  "
              f"partition={info['partition']}")
        tot = sum(c.n_train for c in clients)
        print(f"[{dataset}] {len(clients)} clients, {tot} train windows, "
              f"speed range {speed[valid].min():.1f}-{speed[valid].max():.1f}")
    return clients, info


def verify_no_leakage(clients: list[Client], info: dict) -> None:
    """Assert splits are disjoint in TARGET time. Raises on violation."""
    t_tr, t_va = info["t_train_end"], info["t_val_end"]
    for c in clients:
        assert (c.t_train < t_tr).all(), f"client {c.cid}: train target >= t_train"
        assert (c.t_val >= t_tr).all() and (c.t_val < t_va).all(), \
            f"client {c.cid}: val target outside its split"
        assert (c.t_test >= t_va).all(), f"client {c.cid}: test target < t_val"
        assert len(np.intersect1d(c.t_train, c.t_val)) == 0
        assert len(np.intersect1d(c.t_val, c.t_test)) == 0
        assert len(np.intersect1d(c.t_train, c.t_test)) == 0


def server_root_set(clients: list[Client], n: int, seed: int = 42):
    """Small clean set the server owns (the FLTrust assumption).

    Drawn from TRAINING windows only, so it is disjoint from val and test. The
    honest statement in the paper is the FLTrust assumption -- the server holds a
    small clean dataset -- not that no raw data ever leaves a client.
    """
    rng = np.random.default_rng(seed)
    per = max(1, n // max(1, len(clients)))
    xs, ys = [], []
    for c in clients:
        if c.n_train == 0:
            continue
        sel = rng.choice(c.n_train, min(per, c.n_train), replace=False)
        xs.append(c.x_train[sel])
        ys.append(c.y_train[sel])
    return np.concatenate(xs), np.concatenate(ys)


def fingerprint(clients: list[Client]) -> str:
    h = hashlib.sha256()
    for c in clients:
        h.update(np.ascontiguousarray(c.sensors).tobytes())
        h.update(np.array([c.mean, c.std, c.n_train,
                           len(c.x_val), len(c.x_test)]).tobytes())
    return h.hexdigest()[:16]
