"""Old cached numbers against freshly recomputed ones, side by side.

Reads the audit written by rerun_trust.py plus the pre-rerun snapshot, and
states plainly whether anything moved and in which direction. The decision it
supports is deliberately narrow: if nothing moved there is nothing to adopt or
revert, and if something moved the recomputed value is the one the current
source actually produces.

    PYTHONPATH=. python src/compare_rerun.py
"""
from __future__ import annotations

import pandas as pd

import config as C

TRUST = ["Static-Mamdani-FL", "ZeroTrust-ANFIS-FL"]
SNAP = C.RESULTS / "snapshot_before_rerun"
LOWER_BETTER = {"test_mae", "test_rmse", "test_mape", "sec_malicious_weight_mass",
                "sec_fpr"}


def main() -> int:
    audit_p = C.CSV / "trust_rerun_audit.csv"
    if not audit_p.exists():
        raise SystemExit("no audit yet -- run src/rerun_trust.py first")
    a = pd.read_csv(audit_p)

    print("=" * 72)
    print(f"RECOMPUTED {len(a)} trust-rule runs "
          f"(2 methods x 3 datasets x 4 scenarios x 6 seeds)")
    print("=" * 72)
    comp = a[a.had_cache] if "had_cache" in a else a
    changed = comp[comp.changed] if "changed" in comp else comp.iloc[0:0]
    print(f"  compared against cache : {len(comp)}")
    print(f"  differing              : {len(changed)}")
    print(f"  worst delta anywhere   : {comp.worst_delta.max():.3e}")

    if len(changed) == 0:
        print("\n  Every run reproduces exactly. The cached numbers were")
        print("  already correct under the current source, so there is")
        print("  nothing to adopt and nothing to revert.")
        print("=" * 72)
        return 0

    # Something moved: show old vs new per cell, using the snapshot as OLD.
    old_p = SNAP / "all_runs_raw.csv"
    if not old_p.exists():
        print("\n  snapshot missing; cannot show old values")
        return 1
    old = pd.read_csv(old_p)
    old = old[old.algorithm.isin(TRUST)]

    print("\n  cell means, OLD vs NEW (test MAE):")
    print(f"  {'algorithm':22s} {'dataset':10s} {'attack':17s} "
          f"{'old':>8s} {'new':>8s} {'delta':>9s}")
    for algo in TRUST:
        for ds in old.dataset.unique():
            for atk in old.attack.unique():
                o = old[(old.algorithm == algo) & (old.dataset == ds)
                        & (old.attack == atk)].test_mae
                n = a[(a.Algorithm == algo) & (a.Dataset == ds)
                      & (a.Attack == atk)].new_test_mae
                if o.empty or n.empty:
                    continue
                om, nm = o.mean(), n.mean()
                if abs(om - nm) < 1e-9:
                    continue
                flag = "better" if nm < om else "worse"
                print(f"  {algo:22s} {ds:10s} {atk:17s} "
                      f"{om:8.3f} {nm:8.3f} {nm-om:+9.3f}  {flag}")

    for algo in TRUST:
        o = old[old.algorithm == algo].test_mae.mean()
        n = a[a.Algorithm == algo].new_test_mae.mean()
        print(f"\n  {algo}: overall mean MAE {o:.4f} -> {n:.4f} ({n-o:+.4f})")

    print("\n  The recomputed values are what the current source produces.")
    print("  Keeping the old ones means shipping numbers no longer")
    print("  reproducible from this code.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
