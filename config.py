 
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
CSV = RESULTS / "csv"
FIGS = RESULTS / "figures"
EXCEL = RESULTS / "excel"

for _d in (RESULTS, RAW, CSV, FIGS, EXCEL):
    _d.mkdir(parents=True, exist_ok=True)

 
DATASETS = {
    "METR-LA": {
        "file": DATA / "metr_la.h5",
        "kind": "h5",
        "n_sensors": 207,
        "features": ["speed"],
        "freq_min": 5,
        "coords": DATA / "sensor_locations_la.csv",
    },
    "PEMS-BAY": {
        "file": DATA / "PEMS-BAY.csv",
        "kind": "csv",
        "n_sensors": 325,
        "features": ["speed"],
        "freq_min": 5,
        "coords": None,
    },
  
    "PEMS04": {
        "file": DATA / "PEMS04.npz",
        "kind": "npz",
        "n_sensors": 307,
        "features": ["flow", "occupancy", "speed"],
        "freq_min": 5,
        "coords": None,
        "target_feature": 2,      # speed is the forecasting target
    },
}

BASELINES = [
    "FedAvg", "FedProx", "SCAFFOLD", "FLTrust",
    "Multi-Krum", "Static-Mamdani-FL", "ZeroTrust-ANFIS-FL",
]

ATTACKS = ["No_Attack", "Byzantine_Noise", "Label_Flipping", "Sybil_Poisoning"]


@dataclass
class DataConfig:
    n_clients: int = 10
    input_steps: int = 12          # 12 x 5 min = 1 hour of history
    pred_steps: int = 1
    stride: int = 6
    train_frac: float = 0.70
    val_frac: float = 0.15
    max_train_per_client: int = 4000
    max_eval_per_client: int = 2000
    root_set_size: int = 200       # server clean set (FLTrust assumption)
    max_missing_frac: float = 0.25
    seed: int = 42


@dataclass
class RunConfig:
    """One federated experiment."""

    dataset: str = "METR-LA"
    algorithm: str = "FedAvg"
    attack: str = "No_Attack"
    malicious_frac: float = 0.3
    seed: int = 42

    rounds: int = 40
    local_epochs: int = 3
    lr: float = 5e-3
    batch_size: int = 64
    hidden: int = 48

    # --- algorithm-specific -------------------------------------------
    fedprox_mu: float = 0.01
    n_byz_assumed: int = 3         # fixed, non-oracle; shared by Multi-Krum
    zt_tau: float = 0.35           # Zero-Trust gate threshold
    mamdani_beta: float = 2.0
    anfis_lr_premise: float = 0.01
    anfis_lr_consequent: float = 0.01

    device: str = "cuda"
    eval_every: int = 1

    def key(self) -> str:
        d = {k: v for k, v in asdict(self).items() if k not in ("device", "eval_every")}
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
        return (f"{self.dataset}__{self.algorithm}__{self.attack}"
                f"__s{self.seed}__{h}")


DEFAULT_DATA = DataConfig()


# ---------------------------------------------------------------- compute
def auto_workers(vram_per_worker_gb: float = 0.45, plateau: int = 12) -> int:
    
    try:
        import torch
    except ImportError:
        return 1
    cpus = os.cpu_count() or 2
    if not torch.cuda.is_available():
        return max(1, cpus // 2)
    try:
        free, _ = torch.cuda.mem_get_info()
        by_vram = max(1, int((free / 1e9) // vram_per_worker_gb))
    except Exception:
        by_vram = plateau
    return max(1, min(by_vram, max(1, cpus - 2), plateau))


def environment() -> dict:
    env = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env.update(cuda=torch.version.cuda, gpu=p.name,
                       vram_gb=round(p.total_memory / 1e9, 2),
                       compute_capability=f"{p.major}.{p.minor}")
    except ImportError:
        env["torch"] = None
    env["workers"] = auto_workers()
    return env


def verify() -> bool:
    """Check imports and datasets. Returns True when everything is ready."""
    ok = True
    print("=" * 70)
    print("ZeroTrust-ANFIS-FL  --  configuration check")
    print("=" * 70)

    print("\n[1] packages")
    for mod in ("numpy", "pandas", "torch", "sklearn", "scipy",
                "matplotlib", "h5py", "openpyxl"):
        try:
            m = __import__(mod)
            print(f"    ok    {mod:12s} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"    MISS  {mod}")
            ok = False

    print("\n[2] datasets")
    for name, meta in DATASETS.items():
        f = meta["file"]
        if f.exists():
            print(f"    ok    {name:10s} {f.name:20s} "
                  f"{f.stat().st_size / 1e6:8.1f} MB  "
                  f"{meta['n_sensors']:4d} sensors  {meta['features']}")
        else:
            print(f"    MISS  {name:10s} expected at {f}")
            ok = False

    print("\n[3] compute")
    env = environment()
    for k, v in env.items():
        print(f"    {k:18s} {v}")

    with open(RESULTS / "environment.json", "w") as fh:
        json.dump(env, fh, indent=2)

    print("\n" + "=" * 70)
    print("READY" if ok else "NOT READY - resolve the items marked MISS above")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
