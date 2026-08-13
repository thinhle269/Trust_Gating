"""Multi-tab Excel export of every result table.

Reads only results/csv/. No metric is recomputed here -- this file formats what
training already measured, so a number in the workbook always traces back to a
CSV and from there to a cached run under results/raw/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

OURS = "ZeroTrust-ANFIS-FL"

ALGO_ORDER = ["FedAvg", "FedProx", "SCAFFOLD", "FLTrust", "Multi-Krum",
              "Static-Mamdani-FL", "ZeroTrust-ANFIS-FL"]
ATTACK_ORDER = ["No_Attack", "Byzantine_Noise", "Label_Flipping", "Sybil_Poisoning"]


def _load(name):
    p = C.CSV / name
    return pd.read_csv(p) if p.exists() else None


def _order(df, col, order):
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


def _fmt(m, s, prec=3):
    if not np.isfinite(m):
        return "--"
    p = 0 if abs(m) >= 100 else prec
    return f"{m:.{p}f}" + (f" ± {s:.{p}f}" if np.isfinite(s) and s > 0 else "")


def build():
    summary = _load("summary_results.csv")
    if summary is None:
        print("  summary_results.csv not found - run training first")
        return

    out = C.EXCEL / "ZeroTrust_ANFIS_FL_Results.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:

        # --- Tab 1: headline table, one row per (dataset, attack, algorithm)
        rows = []
        for _, r in _order(summary, "Algorithm", ALGO_ORDER).iterrows():
            rows.append({
                "Dataset": r["Dataset"], "Attack": r["Attack"],
                "Algorithm": r["Algorithm"], "Seeds": int(r["n_seeds"]),
                "MAE": _fmt(r["test_mae_mean"], r["test_mae_std"]),
                "RMSE": _fmt(r["test_rmse_mean"], r["test_rmse_std"]),
                "Accuracy": _fmt(r["test_accuracy_mean"], r["test_accuracy_std"], 4),
                "Macro-F1": _fmt(r["test_macro_f1_mean"], r["test_macro_f1_std"], 4),
                "Macro-Recall": _fmt(r["test_macro_recall_mean"], r["test_macro_recall_std"], 4),
                "AUC": _fmt(r["test_auc_mean"], r["test_auc_std"], 4),
                "Trust AUC": _fmt(r["sec_trust_auc_mean"], r["sec_trust_auc_std"], 3),
                "Malicious Wt Mass": _fmt(r["sec_malicious_weight_mass_mean"],
                                          r["sec_malicious_weight_mass_std"], 3),
            })
        pd.DataFrame(rows).to_excel(xl, sheet_name="1_Main_Results", index=False)

        # --- Tab 2..5: one pivot per metric, algorithms x (dataset, attack)
        for metric, sheet, prec in (("test_mae", "2_MAE", 3),
                                    ("test_rmse", "3_RMSE", 3),
                                    ("test_macro_f1", "4_MacroF1", 4),
                                    ("test_accuracy", "5_Accuracy", 4)):
            col = f"{metric}_mean"
            if col not in summary.columns:
                continue
            piv = summary.pivot_table(index="Algorithm",
                                      columns=["Dataset", "Attack"], values=col)
            piv = piv.reindex([a for a in ALGO_ORDER if a in piv.index])
            piv.round(prec).to_excel(xl, sheet_name=sheet)

        # --- Tab 6: security metrics under attack only
        sec = summary[summary["Attack"] != "No_Attack"]
        if not sec.empty:
            s = _order(sec, "Algorithm", ALGO_ORDER)[
                ["Dataset", "Attack", "Algorithm", "sec_trust_auc_mean",
                 "sec_malicious_weight_mass_mean", "sec_tdr_mean", "sec_fpr_mean",
                 "sec_trust_margin_mean"]].copy()
            s.columns = ["Dataset", "Attack", "Algorithm", "Trust_AUC",
                         "Malicious_Weight_Mass", "Detection_Rate",
                         "False_Positive_Rate", "Trust_Margin"]
            s.round(4).to_excel(xl, sheet_name="6_Security", index=False)

        # --- Tab 7: significance vs ours
        sig = _load("significance_tests.csv")
        if sig is not None and not sig.empty:
            s = sig[sig["Metric"] == "test_mae"].copy()
            s["Verdict"] = np.where(
                (s["wilcoxon_p"] < 0.05) & (s["mean_diff"] < 0), "ours better",
                np.where((s["wilcoxon_p"] < 0.05) & (s["mean_diff"] > 0),
                         "baseline better", "no significant difference"))
            s.round(5).to_excel(xl, sheet_name="7_Significance", index=False)

        # --- Tab 8: per-round convergence
        rounds = _load("rounds_logs.csv")
        if rounds is not None:
            g = (rounds.groupby(["Dataset", "Attack", "Algorithm", "round"])
                 [["val_mae", "val_accuracy", "val_macro_f1"]].mean().reset_index())
            g.round(5).to_excel(xl, sheet_name="8_Convergence", index=False)

        # --- Tab 9: trust evolution
        trust = _load("trust_evolution.csv")
        if trust is not None:
            g = (trust.groupby(["Dataset", "Attack", "Algorithm", "round",
                                "is_malicious"])["trust"].mean().reset_index())
            g.round(5).to_excel(xl, sheet_name="9_Trust_Evolution", index=False)

        # --- Tab 10: confusion matrices
        cm = _load("confusion_matrices.csv")
        if cm is not None:
            cm.to_excel(xl, sheet_name="10_Confusion", index=False)

        # --- Tab 11: provenance, so the workbook is self-describing
        import json
        env = {}
        p = C.RESULTS / "environment.json"
        if p.exists():
            env = json.loads(p.read_text())
        notes = pd.DataFrame([
            {"Item": "Datasets", "Value": "METR-LA (207), PEMS-BAY (325), PEMS04 (307)"},
            {"Item": "PEMS04 substitution",
             "Value": "Used in place of PeMSD7: the distributed PeMSD7 (V_228) is "
                      "speed-only and cannot support the multi-feature claim; "
                      "PEMS04 carries genuine flow/occupancy/speed."},
            {"Item": "Rounds", "Value": 40},
            {"Item": "Clients", "Value": 10},
            {"Item": "Malicious fraction", "Value": 0.3},
            {"Item": "Split", "Value": "chronological 70/15/15, scaler fit on train only"},
            {"Item": "Metrics", "Value": "measured from predictions on the held-out test split"},
            {"Item": "GPU", "Value": env.get("gpu", "n/a")},
            {"Item": "torch", "Value": env.get("torch", "n/a")},
        ])
        notes.to_excel(xl, sheet_name="11_Provenance", index=False)

    print(f"  wrote {out}")


if __name__ == "__main__":
    build()
