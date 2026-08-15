 

    python run_all.py                      # full study, 6 seeds, 40 rounds
    python run_all.py --seeds 3            # quicker pass
    python run_all.py --skip-train         # regenerate outputs from cached runs
 thout recomputing anything.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def step(n: int, total: int, title: str) -> None:
    print("\n" + "=" * 72)
    print(f"[STEP {n}/{total}]  {title}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6,
                    help="6 is the minimum at which a two-sided Wilcoxon test "
                         "can reach p<0.05")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 72)
    print("  ZeroTrust-ANFIS-FL  --  complete research pipeline")
    print("=" * 72)

    total = 5

    step(1, total, "Configuration and dataset verification")
    import config as C
    if not C.verify():
        print("\nconfiguration check failed; nothing was run")
        return 1

    step(2, total, "Invariant tests (ANFIS gradients, learning, structure)")
    r = subprocess.run([sys.executable, "-m", "tests.test_anfis_gradient"],
                       cwd=ROOT)
    if r.returncode != 0:
        print("\ninvariant tests failed; refusing to train on unverified code")
        return 1

    if not args.skip_train:
        step(3, total, f"Federated training  ({args.seeds} seeds x {args.rounds} rounds)")
        cmd = [sys.executable, "-m", "src.experiments",
               "--seeds", str(args.seeds), "--rounds", str(args.rounds),
               "--batch", str(args.batch), "--workers", str(args.workers)]
        if args.datasets:
            cmd += ["--datasets", *args.datasets]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print("\ntraining failed")
            return 1
    else:
        step(3, total, "Federated training  [SKIPPED]")

    step(4, total, "Excel export")
    subprocess.run([sys.executable, "-m", "src.export_excel"], cwd=ROOT)

    step(5, total, "Figure export")
    subprocess.run([sys.executable, "-m", "src.export_plots"], cwd=ROOT)

    mins = (time.time() - t0) / 60
    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETE in {mins:.1f} min")
    print(f"  tables  -> {C.CSV}")
    print(f"  excel   -> {C.EXCEL}")
    print(f"  figures -> {C.FIGS}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
