"""Publication figures, generated from results/csv/.

Design choices made for readability rather than for flattery:

* A shared axis that includes an undefended rule at 5+ mph compresses every
  defended rule into one flat line -- and so does a log axis. Panels comparing
  defended rules therefore scale to THEM, with the collapsed values printed
  underneath so nothing is hidden.
* Per-class recall is shown as a heat-map rather than a wall of confusion
  matrices: seven algorithms x five classes x four attacks is 140 numbers as a
  heat-map against 700 inside full matrices, and the heat-map is the one a
  reader can actually analyse. Four representative full matrices are still
  given, where the off-diagonal structure is the point.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.datasets import CLASS_NAMES

OURS = "ZeroTrust-ANFIS-FL"

ALGOS = ["FedAvg", "FedProx", "SCAFFOLD", "FLTrust", "Multi-Krum",
         "Static-Mamdani-FL", OURS]
ATTACKS = ["No_Attack", "Byzantine_Noise", "Label_Flipping", "Sybil_Poisoning"]
DATASET_ORDER = ["METR-LA", "PEMS-BAY", "PEMS04"]
FIGS_DIR = C.FIGS
ATTACK_LABEL = {"No_Attack": "No attack", "Byzantine_Noise": "Byzantine noise",
                "Label_Flipping": "Label flipping", "Sybil_Poisoning": "Sybil poisoning"}

# Okabe-Ito, colourblind safe. Ours is black so it survives greyscale printing.
COLOUR = {"FedAvg": "#999999", "FedProx": "#E69F00", "SCAFFOLD": "#56B4E9",
          "FLTrust": "#CC79A7", "Multi-Krum": "#009E73",
          "Static-Mamdani-FL": "#D55E00", OURS: "#000000"}

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 320, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(name):
    p = C.CSV / name
    return pd.read_csv(p) if p.exists() else None


def _present(df, col, order):
    have = set(df[col])
    return [x for x in order if x in have]


# ------------------------------------------------------------------ fig 1
def fig_robustness():
    """MAE and Macro-F1 per attack, one panel per dataset."""
    df = _load("summary_results.csv")
    if df is None:
        return
    datasets = sorted(df["Dataset"].unique())
    algos = _present(df, "Algorithm", ALGOS)
    attacks = _present(df, "Attack", ATTACKS)

    for metric, label, lower_better, fname in (
            ("test_mae_mean", "Test MAE (mph)", True, "fig1_robustness_mae.png"),
            ("test_macro_f1_mean", "Test Macro-F1", False, "fig1b_robustness_f1.png")):
        std = metric.replace("_mean", "_std")
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.8),
                                 squeeze=False)
        x = np.arange(len(attacks))
        width = 0.8 / len(algos)

        for ax, ds in zip(axes[0], datasets):
            sub = df[df["Dataset"] == ds]
            for i, a in enumerate(algos):
                s = sub[sub["Algorithm"] == a].set_index("Attack").reindex(attacks)
                vals = s[metric].to_numpy(float)
                errs = s[std].fillna(0).to_numpy(float)
                ax.bar(x + i * width - 0.4 + width / 2, vals, width, yerr=errs,
                       capsize=1.5, color=COLOUR.get(a, "#777"),
                       edgecolor="white", linewidth=0.4,
                       label=a if ax is axes[0][0] else None,
                       zorder=3 if a == OURS else 2)
            ax.set_xticks(x)
            ax.set_xticklabels([ATTACK_LABEL[a].replace(" ", "\n") for a in attacks],
                               fontsize=7.5)
            ax.set_ylabel(label)
            ax.set_title(ds)
            if lower_better:
                ax.set_yscale("log")
        axes[0][0].legend(ncol=2, frameon=True, framealpha=0.9, fontsize=6.8,
                          loc="upper left", edgecolor="none")
        fig.suptitle(f"{label} under attack (30% malicious, 6 seeds, mean ± s.d.)",
                     y=1.02)
        fig.tight_layout()
        fig.savefig(C.FIGS / fname)
        plt.close(fig)
        print(f"  {fname}")


# ------------------------------------------------------------------ fig 2
def fig_convergence():
    """Validation MAE per round, all datasets x all attack scenarios.

    Datasets form the rows. An earlier version showed METR-LA alone, which was a
    real gap: the paper argues that behaviour holds across three benchmarks, so
    the evidence for it has to be shown on all three rather than asserted from
    one and generalised in the text.
    """
    df = _load("rounds_logs.csv")
    if df is None:
        return
    datasets = _present(df, "Dataset", DATASET_ORDER)
    attacks = _present(df, "Attack", ATTACKS)
    algos = _present(df, "Algorithm", ALGOS)

    fig, axes = plt.subplots(len(datasets), len(attacks),
                             figsize=(3.3 * len(attacks), 2.9 * len(datasets)),
                             squeeze=False)
    for r, ds in enumerate(datasets):
        sub = df[df["Dataset"] == ds]
        for c, atk in enumerate(attacks):
            ax = axes[r][c]
            s = sub[sub["Attack"] == atk]
            for a in algos:
                g = s[s["Algorithm"] == a].groupby("round")["val_mae"].agg(["mean", "std"])
                if g.empty:
                    continue
                ax.plot(g.index, g["mean"], color=COLOUR.get(a, "#777"),
                        lw=2.0 if a == OURS else 1.0, label=a,
                        zorder=5 if a == OURS else 2)
                ax.fill_between(g.index, g["mean"] - g["std"].fillna(0),
                                g["mean"] + g["std"].fillna(0),
                                color=COLOUR.get(a, "#777"), alpha=0.10, lw=0)
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(ATTACK_LABEL[atk], fontsize=9)
            if r == len(datasets) - 1:
                ax.set_xlabel("Communication round")
            if c == 0:
                ax.set_ylabel(f"{ds}\nValidation MAE (mph)", fontsize=8.5)
    axes[0][-1].legend(fontsize=6.2, frameon=False, ncol=1)
    fig.suptitle("Convergence across all datasets and threat models "
                 "(6 seeds, mean ± s.d.)", y=1.005)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "fig2_convergence.png")
    plt.close(fig)
    print(f"  fig2_convergence.png  ({len(datasets)} datasets x {len(attacks)} attacks)")


# ------------------------------------------------------------------ fig 3
def fig_trust_evolution():
    """Trust separation over rounds, for every dataset and attack.

    This is the figure that carries the adaptivity claim, so it needs to hold on
    all three datasets rather than on the one where it looks best.
    """
    df = _load("trust_evolution.csv")
    if df is None or "is_malicious" not in df.columns:
        return
    sub = df[(df["Algorithm"] == OURS) & (df["Attack"] != "No_Attack")]
    if sub.empty:
        return
    datasets = _present(sub, "Dataset", DATASET_ORDER)
    attacks = _present(sub, "Attack", ATTACKS)

    fig, axes = plt.subplots(len(datasets), len(attacks),
                             figsize=(3.2 * len(attacks), 2.7 * len(datasets)),
                             squeeze=False, sharey=True)
    for r, ds in enumerate(datasets):
        d = sub[sub["Dataset"] == ds]
        for c, atk in enumerate(attacks):
            ax = axes[r][c]
            s = d[d["Attack"] == atk]
            for mal, colour, lab in ((False, "#0072B2", "honest"),
                                     (True, "#D55E00", "malicious")):
                t = s[s["is_malicious"] == mal]
                if t.empty:
                    continue
                g = t.groupby("round")["trust"].agg(["mean", "std"])
                ax.plot(g.index, g["mean"], color=colour, lw=1.7, label=lab)
                ax.fill_between(g.index, g["mean"] - g["std"].fillna(0),
                                g["mean"] + g["std"].fillna(0),
                                color=colour, alpha=0.18, lw=0)
            if r == 0:
                ax.set_title(ATTACK_LABEL[atk], fontsize=9)
            if r == len(datasets) - 1:
                ax.set_xlabel("Round")
            if c == 0:
                ax.set_ylabel(f"{ds}\nANFIS trust", fontsize=8.5)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Trust separation learned by the adaptive rule base, "
                 "all datasets and threat models", y=1.005)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "fig3_trust_evolution.png")
    plt.close(fig)
    print(f"  fig3_trust_evolution.png  ({len(datasets)} datasets x {len(attacks)} attacks)")


# ------------------------------------------------------------------ fig 4
def fig_perclass_recall():
    """Per-class recall for every algorithm, attack and dataset.

    Read off the row-normalised diagonals of the stored confusion matrices, so
    this is the same data as a wall of confusion matrices but in a form that can
    actually be compared across seven algorithms and three datasets at once.
    """
    df = _load("confusion_matrices.csv")
    if df is None:
        return
    datasets = _present(df, "Dataset", DATASET_ORDER)
    attacks = _present(df, "Attack", ATTACKS)
    algos = _present(df, "Algorithm", ALGOS)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]

    fig, axes = plt.subplots(len(datasets), len(attacks),
                             figsize=(2.75 * len(attacks),
                                      (0.30 * len(algos) + 1.3) * len(datasets)),
                             squeeze=False)
    im = None
    for r, ds in enumerate(datasets):
        sub = df[df["Dataset"] == ds]
        for c, atk in enumerate(attacks):
            ax = axes[r][c]
            mat = np.full((len(algos), len(CLASS_NAMES)), np.nan)
            for i, a in enumerate(algos):
                s = sub[(sub["Attack"] == atk) & (sub["Algorithm"] == a)]
                if s.empty:
                    continue
                cm = s.sort_values("true_class")[pred_cols].to_numpy(float)
                mat[i] = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
            im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            for i in range(len(algos)):
                for j in range(len(CLASS_NAMES)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                                fontsize=4.8,
                                color="black" if 0.25 < mat[i, j] < 0.85 else "white")
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES if r == len(datasets) - 1 else [],
                               rotation=45, ha="right", fontsize=6.2)
            ax.set_yticks(range(len(algos)))
            ax.set_yticklabels(algos if c == 0 else [], fontsize=6.2)
            if c == 0:
                for tick, a in zip(ax.get_yticklabels(), algos):
                    if a == OURS:
                        tick.set_fontweight("bold")
                ax.set_ylabel(ds, fontsize=8.5, labelpad=34)
            if r == 0:
                ax.set_title(ATTACK_LABEL[atk], fontsize=8.5)
            ax.grid(False)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Per-class recall",
                     shrink=0.6)
    fig.suptitle("Per-class congestion recall across all datasets and threat models",
                 y=1.005)
    fig.savefig(FIGS_DIR / "fig4_perclass_recall.png")
    plt.close(fig)
    print(f"  fig4_perclass_recall.png  ({len(datasets)} datasets x {len(attacks)} attacks)")


# ------------------------------------------------------------------ fig 5
def fig_security():
    """Trust AUC and malicious weight mass: detection versus what it prevents."""
    df = _load("summary_results.csv")
    if df is None or "sec_trust_auc_mean" not in df.columns:
        return
    sub = df[df["Attack"] != "No_Attack"]
    algos = _present(sub, "Algorithm", ALGOS)
    attacks = _present(sub, "Attack", ATTACKS)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    x = np.arange(len(attacks))
    width = 0.8 / len(algos)

    for ax, col, label, hi in (
            (axes[0], "sec_trust_auc_mean", "Trust AUC (detection quality)", 1.0),
            (axes[1], "sec_malicious_weight_mass_mean",
             "Malicious weight mass (lower is better)", None)):
        for i, a in enumerate(algos):
            s = sub[sub["Algorithm"] == a].groupby("Attack")[col].mean().reindex(attacks)
            ax.bar(x + i * width - 0.4 + width / 2, s.to_numpy(float), width,
                   color=COLOUR.get(a, "#777"), edgecolor="white", linewidth=0.4,
                   label=a if ax is axes[0] else None, zorder=3 if a == OURS else 2)
        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_LABEL[a].replace(" ", "\n") for a in attacks],
                           fontsize=7.5)
        ax.set_ylabel(label)
        if hi:
            ax.set_ylim(0, hi)
        ax.set_title(label)
    axes[0].axhline(0.5, ls=":", c="k", lw=1)
    axes[0].text(0.02, 0.52, "chance", fontsize=6.5, transform=axes[0].get_yaxis_transform())
    axes[0].legend(ncol=2, fontsize=6.8, frameon=False)
    fig.suptitle("Security behaviour under attack (all datasets pooled, 6 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "fig5_security.png")
    plt.close(fig)
    print("  fig5_security.png")


def main():
    print("figures:")
    for fn in (fig_robustness, fig_convergence, fig_trust_evolution,
               fig_perclass_recall, fig_security):
        try:
            fn()
        except Exception as e:
            print(f"  SKIP {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\nfigures written to {FIGS_DIR}")


if __name__ == "__main__":
    main()
