"""Recompute every trust-rule run from scratch and diff against the cache.

Reviewer's point: TRUST_RULES = {Static-Mamdani-FL, ZeroTrust-ANFIS-FL} are the
only rules that consume trust_features / online_teacher / adaptive_gate, and
those live in files edited after some of the runs were cached. Since the cache
key hashes RunConfig fields only, an edited constant would not invalidate it.

    2 methods x 3 datasets x 4 scenarios x 6 seeds = 144 runs

Rather than delete and recompute -- which destroys the evidence -- this
recomputes all 144 with the current source and compares each against its cached
counterpart. Agreement is the reproducibility claim the reviewer is asking for;
disagreement identifies exactly which cells were computed under superseded code.

    PYTHONPATH=. python src/rerun_trust.py --workers 0

Writes results/csv/trust_rerun_audit.csv.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import pandas as pd

import config as C
from src.experiments import _init_worker, _dataset

TRUST_RULES = ["Static-Mamdani-FL", "ZeroTrust-ANFIS-FL"]
FIELDS = ["mae", "rmse", "mape", "accuracy", "macro_f1", "macro_recall", "auc"]
SEC_FIELDS = ["trust_auc", "malicious_weight_mass", "tdr", "fpr"]
TOL = 1e-9          # runs are deterministic; anything above this is a real change


def _recompute(cfg: C.RunConfig) -> dict:
    """Run fresh, ignoring the cache, and diff against the cached result."""
    from src.train_eval import run

    path = C.RAW / f"{cfg.key()}.json"
    old = None
    if path.exists():
        try:
            with open(path) as f:
                old = json.load(f)
        except json.JSONDecodeError:
            old = None

    clients, _ = _dataset(cfg.dataset)
    new = run(clients, cfg, _G_DATA["data_cfg"])

    rec = {"Dataset": cfg.dataset, "Algorithm": cfg.algorithm,
           "Attack": cfg.attack, "seed": cfg.seed,
           "had_cache": old is not None}
    worst = 0.0
    for k in FIELDS:
        nv = new["test"].get(k)
        if nv is None:
            continue
        rec[f"new_test_{k}"] = float(nv)
        if old is not None and k in old["test"]:
            d = abs(float(old["test"][k]) - float(nv))
            rec[f"d_test_{k}"] = d
            worst = max(worst, d)
    for k in SEC_FIELDS:
        nv = new["security"].get(k)
        if nv is None:
            continue
        rec[f"new_sec_{k}"] = float(nv)
        if old is not None and k in old["security"]:
            d = abs(float(old["security"][k]) - float(nv))
            rec[f"d_sec_{k}"] = d
            worst = max(worst, d)
    rec["worst_delta"] = worst
    rec["changed"] = bool(old is not None and worst > TOL)

    # The recomputed run is what the paper should cite from here on.
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(new, f)
    tmp.replace(path)
    return rec


_G_DATA: dict = {}


def _init(data_cfg, threads):
    _init_worker(data_cfg, threads)
    _G_DATA["data_cfg"] = data_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    workers = args.workers or C.auto_workers()
    seeds = [42 + i for i in range(args.seeds)]
    data_cfg = C.DataConfig()

    cfgs = [C.RunConfig(dataset=ds, algorithm=algo, attack=atk, seed=s,
                        rounds=args.rounds, batch_size=args.batch)
            for ds in C.DATASETS
            for atk in C.ATTACKS
            for algo in TRUST_RULES
            for s in seeds]

    print(f"{len(TRUST_RULES)} methods x {len(C.DATASETS)} datasets x "
          f"{len(C.ATTACKS)} scenarios x {len(seeds)} seeds = {len(cfgs)} runs")
    print(f"recomputing all of them with the current source, {workers} workers\n")

    t0 = time.time()
    threads = max(1, 20 // max(workers, 1))
    recs = []
    if workers <= 1:
        _init(data_cfg, threads)
        for i, c in enumerate(cfgs):
            recs.append(_recompute(c))
            print(f"  {i+1}/{len(cfgs)}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init,
                      initargs=(data_cfg, threads)) as pool:
            for i, r in enumerate(pool.imap_unordered(_recompute, cfgs, chunksize=1)):
                recs.append(r)
                if (i + 1) % 12 == 0 or i + 1 == len(cfgs):
                    el = time.time() - t0
                    rate = (i + 1) / max(el, 1e-9)
                    print(f"  {i+1}/{len(cfgs)}  {el/60:.1f} min  "
                          f"(eta {(len(cfgs)-i-1)/max(rate,1e-9)/60:.0f} min)")

    df = pd.DataFrame(recs).sort_values(
        ["Algorithm", "Dataset", "Attack", "seed"])
    df.to_csv(C.CSV / "trust_rerun_audit.csv", index=False)

    print("\n" + "=" * 70)
    print(f"recomputed {len(df)} runs in {(time.time()-t0)/60:.1f} min")
    no_cache = int((~df.had_cache).sum())
    if no_cache:
        print(f"  {no_cache} had no cached counterpart (nothing to compare)")
    comp = df[df.had_cache]
    changed = comp[comp.changed]
    print(f"  {len(comp)} compared against cache")
    print(f"  {len(changed)} differ beyond {TOL:g}")
    if len(changed):
        print("\n  cells whose numbers changed:")
        for _, r in changed.iterrows():
            print(f"    {r.Algorithm:20s} {r.Dataset:9s} {r.Attack:16s} "
                  f"s{int(r.seed)}  worst delta {r.worst_delta:.3e}")
        print("\n  -> regenerate CSVs, tables, figures, and re-check the prose.")
    else:
        print("\n  Every cached trust-rule run reproduces exactly under the")
        print("  current source. The published numbers stand as they are.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
