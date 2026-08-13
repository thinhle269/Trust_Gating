"""Console digest of the results, with the checks a reviewer would apply.

Prints what the data actually says, including where the proposed method loses.
Run after training:  python -m src.digest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

OURS = "ZeroTrust-ANFIS-FL"
ALGOS = ["FedAvg", "FedProx", "SCAFFOLD", "FLTrust", "Multi-Krum",
         "Static-Mamdani-FL", OURS]
ATTACKS = ["No_Attack", "Byzantine_Noise", "Label_Flipping", "Sybil_Poisoning"]


def _load(name):
    p = C.CSV / name
    return pd.read_csv(p) if p.exists() else None


def hr(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def main():
    df = _load("summary_results.csv")
    if df is None:
        raise SystemExit("no summary_results.csv - run training first")

    hr("TEST MAE  (mean over seeds; lower is better)")
    for ds in sorted(df["Dataset"].unique()):
        sub = df[df["Dataset"] == ds]
        piv = sub.pivot_table(index="Algorithm", columns="Attack",
                              values="test_mae_mean")
        piv = piv.reindex([a for a in ALGOS if a in piv.index])
        piv = piv[[a for a in ATTACKS if a in piv.columns]]
        print(f"\n--- {ds} ---")
        print(piv.round(3).to_string())

    hr("MACRO-F1  (higher is better)")
    for ds in sorted(df["Dataset"].unique()):
        sub = df[df["Dataset"] == ds]
        piv = sub.pivot_table(index="Algorithm", columns="Attack",
                              values="test_macro_f1_mean")
        piv = piv.reindex([a for a in ALGOS if a in piv.index])
        piv = piv[[a for a in ATTACKS if a in piv.columns]]
        print(f"\n--- {ds} ---")
        print(piv.round(4).to_string())

    hr("SECURITY  (attacks only, pooled over datasets)")
    sec = df[df["Attack"] != "No_Attack"]
    g = sec.groupby("Algorithm")[["sec_trust_auc_mean",
                                  "sec_malicious_weight_mass_mean",
                                  "sec_tdr_mean", "sec_fpr_mean"]].mean()
    g = g.reindex([a for a in ALGOS if a in g.index])
    g.columns = ["Trust_AUC", "Malicious_Wt_Mass", "Detection_Rate", "FalsePos_Rate"]
    print(g.round(3).to_string())

    # --- the check that decides what may be claimed --------------------
    sig = _load("significance_tests.csv")
    if sig is not None and not sig.empty:
        hr("SIGNIFICANCE  (paired Wilcoxon on test MAE, ours vs each baseline)")
        s = sig[sig["Metric"] == "test_mae"]
        rows = []
        for b, sub in s.groupby("Baseline"):
            win = int(((sub.mean_diff < 0) & (sub.wilcoxon_p < 0.05)).sum())
            loss = int(((sub.mean_diff > 0) & (sub.wilcoxon_p < 0.05)).sum())
            rows.append({"Baseline": b, "we_win": win,
                         "tie": len(sub) - win - loss, "WE_LOSE": loss,
                         "mean_dMAE": sub.mean_diff.mean()})
        r = pd.DataFrame(rows).sort_values(["WE_LOSE", "we_win"],
                                           ascending=[True, False])
        print(r.round(4).to_string(index=False))
        print(f"\n  comparisons: {len(s)}   significant: {int((s.wilcoxon_p<0.05).sum())}")
        lost = s[(s.wilcoxon_p < 0.05) & (s.mean_diff > 0)]
        if len(lost):
            print(f"\n  *** {len(lost)} comparison(s) where a BASELINE beats ours: ***")
            for _, r in lost.iterrows():
                print(f"      {r['Dataset']:9s} {r['Attack']:16s} vs {r['Baseline']:20s} "
                      f"dMAE={r['mean_diff']:+.4f} p={r['wilcoxon_p']:.4f}")
        else:
            print("\n  no baseline significantly beats ours on any (dataset, attack)")

    # --- clean-data honesty check ---------------------------------------
    clean = df[df["Attack"] == "No_Attack"]
    if not clean.empty:
        hr("CLEAN DATA  (is the robustness free?)")
        g = clean.groupby("Algorithm")["test_mae_mean"].mean().sort_values()
        print(g.round(4).to_string())
        best, our = g.iloc[0], g.get(OURS, np.nan)
        if np.isfinite(our):
            print(f"\n  ours {our:.4f} vs best {g.index[0]} {best:.4f} "
                  f"-> cost of robustness {100*(our-best)/best:+.2f}%")


if __name__ == "__main__":
    main()
