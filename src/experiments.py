"""Experiment matrix with a parallel worker pool.

The forecaster has ~8k parameters, so a single run leaves the GPU around 8%
utilised -- the workload is latency-bound (kernel launch overhead on small
batches), not throughput-bound. The matrix is embarrassingly parallel across
(dataset x attack x algorithm x seed), so the lever is CONCURRENCY, not a larger
batch. Measured: batch 512 is 6x faster per round than batch 64 but costs 1.3%
MAE, whereas 12 worker processes give a ~7x throughput gain at zero cost to the
numbers.

Each run is cached to results/raw/<key>.json and skipped if present, so an
interrupted sweep resumes for free. The cache key hashes every field that
changes a number.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

import config as C

_G: dict = {}

HEADLINE = ["mae", "rmse", "mape", "accuracy", "macro_f1", "macro_recall",
            "macro_precision", "auc"]
SEC = ["trust_auc", "malicious_weight_mass", "tdr", "fpr",
       "mean_trust_benign", "mean_trust_malicious", "trust_margin"]


def _init_worker(data_cfg: C.DataConfig, threads: int):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = True     # fixed shapes -> cache algorithms
    _G["data_cfg"] = data_cfg
    _G["datasets"] = {}


def _dataset(name: str):
    """Datasets are loaded once per worker and reused across runs."""
    if name not in _G["datasets"]:
        from src.datasets import load, verify_no_leakage
        clients, info = load(name, _G["data_cfg"], verbose=False)
        verify_no_leakage(clients, info)
        _G["datasets"][name] = (clients, info)
    return _G["datasets"][name]


def _run_one(cfg: C.RunConfig):
    from src.train_eval import run
    path = C.RAW / f"{cfg.key()}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            path.unlink()                     # truncated by an interrupt; redo
    clients, _info = _dataset(cfg.dataset)
    out = run(clients, cfg, _G["data_cfg"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f)
    tmp.replace(path)                          # atomic: never a half-written file
    return out


def flatten(out: dict) -> dict:
    row = dict(out["config"])
    for k, v in out["test"].items():
        if not isinstance(v, list):
            row[f"test_{k}"] = v
    for k, v in out["val"].items():
        if not isinstance(v, list):
            row[f"val_{k}"] = v
    for k, v in out["security"].items():
        row[f"sec_{k}"] = v
    row["n_malicious"] = len(out["malicious_clients"])
    row["wall_time_s"] = out["wall_time_s"]
    return row


def execute(configs: list[C.RunConfig], workers: int,
            data_cfg: C.DataConfig, label: str = "") -> list[dict]:
    todo = [c for c in configs if not (C.RAW / f"{c.key()}.json").exists()]
    print(f"[{label}] {len(configs)} runs, {len(todo)} to compute, "
          f"{len(configs) - len(todo)} cached, {workers} workers")
    if not configs:
        return []

    t0 = time.time()
    threads = max(1, 20 // max(workers, 1))
    if workers <= 1:
        _init_worker(data_cfg, threads)
        return [_run_one(c) for c in configs]

    ctx = mp.get_context("spawn")
    outs = []
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(data_cfg, threads)) as pool:
        for i, o in enumerate(pool.imap(_run_one, configs, chunksize=1)):
            outs.append(o)
            if (i + 1) % 10 == 0 or i + 1 == len(configs):
                el = time.time() - t0
                rate = (i + 1) / max(el, 1e-9)
                eta = (len(configs) - i - 1) / max(rate, 1e-9) / 60
                print(f"  {i+1}/{len(configs)}  {el/60:.1f} min  "
                      f"({rate*60:.1f} runs/min, eta {eta:.0f} min)")
    return outs


def build_matrix(seeds: list[int], datasets: list[str] | None = None,
                 rounds: int = 40, batch: int = 64) -> list[C.RunConfig]:
    datasets = datasets or list(C.DATASETS)
    cfgs = []
    for ds in datasets:
        for atk in C.ATTACKS:
            for algo in C.BASELINES:
                for s in seeds:
                    cfgs.append(C.RunConfig(dataset=ds, algorithm=algo, attack=atk,
                                            seed=s, rounds=rounds, batch_size=batch))
    return cfgs


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    from src.metrics import aggregate_seeds
    keys = [f"test_{k}" for k in HEADLINE] + [f"sec_{k}" for k in SEC]
    recs = []
    for (ds, atk, algo), sub in df.groupby(["dataset", "attack", "algorithm"]):
        rec = {"Dataset": ds, "Attack": atk, "Algorithm": algo,
               "n_seeds": len(sub)}
        rec.update(aggregate_seeds(sub.to_dict("records"), keys))
        recs.append(rec)
    res = pd.DataFrame(recs)
    res.to_csv(C.CSV / "summary_results.csv", index=False)
    return res


def significance(df: pd.DataFrame, ours="ZeroTrust-ANFIS-FL") -> pd.DataFrame:
    from src.metrics import paired_test
    recs = []
    for (ds, atk), sub in df.groupby(["dataset", "attack"]):
        if ours not in set(sub["algorithm"]):
            continue
        piv = sub.pivot_table(index="seed", columns="algorithm",
                              values=["test_mae", "test_macro_f1", "test_accuracy"])
        for algo in sub["algorithm"].unique():
            if algo == ours:
                continue
            for metric in ("test_mae", "test_macro_f1", "test_accuracy"):
                try:
                    a = piv[(metric, ours)].to_numpy()
                    b = piv[(metric, algo)].to_numpy()
                except KeyError:
                    continue
                recs.append({"Dataset": ds, "Attack": atk, "Metric": metric,
                             "Baseline": algo, **paired_test(a, b)})
    res = pd.DataFrame(recs)
    res.to_csv(C.CSV / "significance_tests.csv", index=False)
    return res


def save_details(outs, cfgs):
    rounds, trust, cms = [], [], []
    for o, c in zip(outs, cfgs):
        if o is None:
            continue
        tag = dict(Dataset=c.dataset, Attack=c.attack, Algorithm=c.algorithm, seed=c.seed)
        for r in o["logs"]:
            rounds.append({**tag, **r})
        for r in o["trust_rows"]:
            trust.append({**tag, **r})
        if c.seed == cfgs[0].seed:
            cm = np.asarray(o["test"]["confusion_matrix"])
            for i, row in enumerate(cm):
                cms.append({**tag, "true_class": i, **{f"pred_{j}": int(v)
                                                       for j, v in enumerate(row)}})
    if rounds:
        pd.DataFrame(rounds).to_csv(C.CSV / "rounds_logs.csv", index=False)
    if trust:
        pd.DataFrame(trust).to_csv(C.CSV / "trust_evolution.csv", index=False)
    if cms:
        pd.DataFrame(cms).to_csv(C.CSV / "confusion_matrices.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = detect from this machine at run time")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    workers = args.workers or C.auto_workers()
    seeds = [42 + i for i in range(args.seeds)]
    data_cfg = C.DataConfig()

    cfgs = build_matrix(seeds, args.datasets, args.rounds, args.batch)
    print(f"matrix: {len(C.DATASETS if not args.datasets else args.datasets)} datasets "
          f"x {len(C.ATTACKS)} attacks x {len(C.BASELINES)} algorithms "
          f"x {len(seeds)} seeds = {len(cfgs)} runs")

    outs = execute(cfgs, workers, data_cfg, label="main")
    good = [(o, c) for o, c in zip(outs, cfgs) if o]
    df = pd.DataFrame([flatten(o) for o, _ in good])
    df.to_csv(C.CSV / "all_runs_raw.csv", index=False)

    summarise(df)
    significance(df)
    save_details([o for o, _ in good], [c for _, c in good])
    print(f"\nwrote {C.CSV}/summary_results.csv and companions")


if __name__ == "__main__":
    main()
