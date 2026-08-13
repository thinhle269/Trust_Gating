"""Retire cached runs for named algorithms so the next sweep recomputes them.

The cache key covers RunConfig fields only, so edits to constants inside the
source (min_gap, teacher weights, trust_features) leave stale entries that look
valid. This moves the affected entries aside -- it never deletes them, so a
comparison against the superseded numbers stays possible.

    PYTHONPATH=. python src/invalidate.py --algos Static-Mamdani-FL ZeroTrust-ANFIS-FL

Then rerun the full matrix; untouched algorithms reload from cache.
"""
from __future__ import annotations

import argparse
import shutil

import config as C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", required=True)
    ap.add_argument("--tag", default="superseded",
                    help="subdirectory name under results/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unknown = [a for a in args.algos if a not in C.BASELINES]
    if unknown:
        print(f"unknown algorithm(s): {unknown}\nknown: {C.BASELINES}")
        return 1

    dest = C.RESULTS / f"raw_{args.tag}"
    hits = []
    for p in sorted(C.RAW.glob("*.json")):
        # key format: <dataset>__<algorithm>__<attack>__s<seed>__<hash>
        parts = p.name.split("__")
        if len(parts) >= 2 and parts[1] in args.algos:
            hits.append(p)

    print(f"matched {len(hits)} cached runs for {args.algos}")
    if args.dry_run:
        for p in hits[:5]:
            print(f"  would move {p.name}")
        print("  ...") if len(hits) > 5 else None
        return 0

    if not hits:
        print("nothing to do")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    for p in hits:
        shutil.move(str(p), str(dest / p.name))
    print(f"moved to {dest}")
    print(f"remaining in cache: {len(list(C.RAW.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
