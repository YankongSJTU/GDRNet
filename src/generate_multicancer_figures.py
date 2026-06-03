# -*- coding: utf-8 -*-
"""
Generate Multi-Cancer Comparison Figures for GDRNet Paper
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = Path("/export/home/kongyan/project/Organoid")
TABLES = ROOT / "results/tables"
FIG_DIR = ROOT / "results/figures/paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    "CRC":  "#4C72B0",  # blue
    "PDAC": "#DD8452",  # orange
    "BLCA": "#55A868",  # green
}
MARKERS = {"CRC": "o", "PDAC": "s", "BLCA": "^"}


def load_results():
    """Load all LOOCV results."""
    dfs = {}
    for ctype in ["crc", "pdac", "blca"]:
        path = TABLES / f"{ctype}_loocv_predictions.csv"
        if path.exists():
            dfs[ctype] = pd.read_csv(path)
        else:
            print(f"  [WARN] Missing: {path}")

    # Per-organoid
    per_org_path = TABLES / "multicancer_per_organoid.csv"
    per_org = pd.read_csv(per_org_path) if per_org_path.exists() else pd.DataFrame()

    # Per-drug
    per_drug_paths = {}
    for ctype in ["crc", "pdac", "blca"]:
        p = TABLES / f"{ctype}_loocv_per_drug.csv"
        if p.exists():
            per_drug_paths[ctype] = pd.read_csv(p)

    # Comparison table
    cmp_path = TABLES / "multicancer_loocv_comparison.csv"
    cmp = pd.read_csv(cmp_path) if cmp_path.exists() else pd.DataFrame()

    return dfs, per_org, per_drug_paths, cmp


def fig_multicancer_performance(cmp_df, save=True):
    """Fig: Multi-cancer performance comparison bar chart."""
    if cmp_df.empty:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    metrics = ["Pearson", "R2", "RMSE", "AUROC"]

    for ax, metric in zip(axes, metrics):
        cancers = cmp_df["Cancer"].tolist()
        vals = cmp_df[metric].tolist()
        colors = [COLORS.get(c, "#888") for c in cancers]

        bars = ax.bar(cancers, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylim(0, min(1.15, max(vals) * 1.2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=10)

    plt.suptitle("GDRNet Cross-Cancer Organoid LOOCV Performance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_multicancer_performance.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_per_cancer_scatter(dfs, save=True):
    """Fig: True vs Predicted scatter for each cancer type."""
    n_panels = len(dfs)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))

    if n_panels == 1:
        axes = [axes]

    response_cols = {"crc": "response", "pdac": "response", "blca": "response"}
    titles = {"crc": "CRC Organoids", "pdac": "PDAC Organoids", "blca": "Bladder Organoids"}

    for ax, (ctype, df) in zip(axes, dfs.items()):
        cancer = ctype.upper()
        color = COLORS.get(cancer, "#888")

        y_true = df["response"].values
        y_pred = df["pred_ensemble"].values

        ax.scatter(y_true, y_pred, alpha=0.4, s=20, color=color, edgecolors="none")

        # Diagonal
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, linewidth=1)

        # Stats
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f"Pearson = {pearson:.3f}\nR² = {r2:.3f}\nn = {len(y_true)}",
                transform=ax.transAxes, fontsize=11, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        xlabel = "1-AUC (Sensitivity)" if ctype == "pdac" else "LogIC50"
        ax.set_xlabel(f"True {xlabel}", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(titles.get(ctype, cancer), fontsize=13, fontweight="bold", color=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("GDRNet LOOCV: True vs Predicted Drug Response", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_multicancer_scatter.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_per_organoid_bar(per_org_df, save=True):
    """Fig: Per-organoid Pearson correlation, colored by cancer type."""
    if per_org_df.empty:
        return

    # Sort by Pearson
    df = per_org_df.sort_values("Pearson", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.35), 6))

    colors = [COLORS.get(row["cancer_type"].upper(), "#888") for _, row in df.iterrows()]
    bars = ax.barh(range(len(df)), df["Pearson"], color=colors, edgecolor="black", linewidth=0.3, height=0.7)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["organoid"], fontsize=8)
    ax.set_xlabel("Pearson Correlation", fontsize=12)
    ax.set_title("Per-Organoid Prediction Quality (LOOCV)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mean line
    mean_p = df["Pearson"].mean()
    ax.axvline(mean_p, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax.text(mean_p + 0.01, len(df) - 0.5, f"Mean={mean_p:.3f}", color="red", fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[c], edgecolor="black", label=c)
                       for c in COLORS if c in df["cancer_type"].str.upper().unique()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_multicancer_per_organoid.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_drug_overlap_analysis(dfs, per_drug_dfs, save=True):
    """Fig: Drug generalization analysis - GDSC overlap vs novel drugs (PDAC)."""
    if "pdac" not in dfs:
        return

    df = dfs["pdac"]
    drug_map = pd.read_csv(ROOT / "data/processed/expanded/drug_id_map.csv")
    gdsc_drugs = set(drug_map["DRUG_NAME"])

    # Classify drugs
    pdac_drugs = df["drug_name"].unique()
    overlap_drugs = [d for d in pdac_drugs if d in gdsc_drugs]
    novel_drugs = [d for d in pdac_drugs if d not in gdsc_drugs]

    # Compute per-drug Pearson
    drug_pearsons = {}
    for drug in pdac_drugs:
        mask = df["drug_name"] == drug
        if mask.sum() >= 5:
            y_t = df.loc[mask, "response"].values
            y_p = df.loc[mask, "pred_ensemble"].values
            drug_pearsons[drug] = np.corrcoef(y_t, y_p)[0, 1]

    overlap_p = [drug_pearsons[d] for d in overlap_drugs if d in drug_pearsons]
    novel_p = [drug_pearsons[d] for d in novel_drugs if d in drug_pearsons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Venn-style bar
    ax = axes[0]
    categories = ["GDSC-Overlap\n(known drugs)", "PDAC-Only\n(novel drugs)"]
    counts = [len(overlap_drugs), len(novel_drugs)]
    ax.bar(categories, counts, color=["#4C72B0", "#DD8452"], edgecolor="black", width=0.5)
    for i, (c, v) in enumerate(zip(categories, counts)):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Drugs", fontsize=12)
    ax.set_title("Drug Composition", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Box plot of per-drug Pearson
    ax = axes[1]
    data_to_plot = []
    labels = []
    if overlap_p:
        data_to_plot.append(overlap_p)
        labels.append(f"Known\n(n={len(overlap_p)})")
    if novel_p:
        data_to_plot.append(novel_p)
        labels.append(f"Novel\n(n={len(novel_p)})")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add mean markers
        for i, d in enumerate(data_to_plot):
            ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-0.05, 0.05, len(d)),
                       d, alpha=0.5, s=20, color="black", zorder=3)

    ax.set_ylabel("Per-Drug Pearson Correlation", fontsize=12)
    ax.set_title("Predictability by Drug Origin", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Top 10 and Bottom 10 drugs
    ax = axes[2]
    sorted_drugs = sorted(drug_pearsons.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_drugs[:10]
    bot10 = sorted_drugs[-10:]

    show_drugs = bot10 + top10
    names = [d[0][:15] for d in show_drugs]
    vals = [d[1] for d in show_drugs]
    bar_colors = ["#DD8452" if d[0] in gdsc_drugs else "#DD8452" for d in show_drugs]
    # Actually color by category
    bar_colors = ["#4C72B0" if d[0] in gdsc_drugs else "#DD8452" for d in show_drugs]

    ax.barh(range(len(show_drugs)), vals, color=bar_colors, edgecolor="black", linewidth=0.3, height=0.7)
    ax.set_yticks(range(len(show_drugs)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Pearson Correlation", fontsize=11)
    ax.set_title("Drug Predictability Ranking", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("PDAC Drug Generalization Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_pdac_drug_generalization.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_pdac_heatmap(dfs, save=True):
    """Fig: Drug sensitivity heatmap for PDAC (true vs predicted)."""
    if "pdac" not in dfs:
        return

    df = dfs["pdac"]
    pivot_true = df.pivot_table(index="drug_name", columns="organoid_id", values="response", aggfunc="mean")
    pivot_pred = df.pivot_table(index="drug_name", columns="organoid_id", values="pred_ensemble", aggfunc="mean")

    # Select top 15 most variable drugs
    drug_var = pivot_true.var(axis=1).sort_values(ascending=False)
    top_drugs = drug_var.head(15).index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, pivot, title in [(ax1, pivot_true, "True (1-AUC)"),
                              (ax2, pivot_pred, "Predicted")]:
        sub = pivot.loc[pivot.index.isin(top_drugs)]
        sub = sub.reindex(top_drugs)

        im = ax.imshow(sub.values, aspect="auto", cmap="RdYlBu_r")
        ax.set_yticks(range(len(top_drugs)))
        ax.set_yticklabels([d[:20] for d in top_drugs], fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Organoid", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("PDAC Drug Sensitivity: True vs Predicted", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_pdac_heatmap.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_blca_heatmap(dfs, save=True):
    """Fig: Drug sensitivity heatmap for BLCA."""
    if "blca" not in dfs:
        return

    df = dfs["blca"]
    pivot_true = df.pivot_table(index="drug_name", columns="organoid_id", values="response", aggfunc="mean")
    pivot_pred = df.pivot_table(index="drug_name", columns="organoid_id", values="pred_ensemble", aggfunc="mean")

    # Sort drugs by variance
    drug_var = pivot_true.var(axis=1).sort_values(ascending=False)
    top_drugs = drug_var.head(20).index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    for ax, pivot, title in [(ax1, pivot_true, "True (LogIC50)"),
                              (ax2, pivot_pred, "Predicted")]:
        sub = pivot.loc[pivot.index.isin(top_drugs)]
        sub = sub.reindex(top_drugs)

        im = ax.imshow(sub.values, aspect="auto", cmap="RdYlBu_r")
        ax.set_yticks(range(len(top_drugs)))
        ax.set_yticklabels(top_drugs, fontsize=8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Organoid", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Bladder Cancer Drug Sensitivity: True vs Predicted", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_blca_heatmap.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_multicancer_roc(dfs, save=True):
    """Fig: ROC curves for each cancer type."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6, 6))

    for ctype, df in dfs.items():
        cancer = ctype.upper()
        y_true = df["response"].values
        y_pred = df["pred_ensemble"].values
        thr = np.percentile(y_true, 30)
        sensitive = (y_true <= thr).astype(int)

        try:
            fpr, tpr, _ = roc_curve(sensitive, -y_pred)
            auroc = roc_auc_score(sensitive, -y_pred)
            ax.plot(fpr, tpr, color=COLORS.get(cancer, "#888"), linewidth=2,
                    label=f"{cancer} (AUROC={auroc:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC: Drug Sensitivity Classification", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_multicancer_roc.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def fig_dataset_overview(cmp_df, dfs, save=True):
    """Fig: Dataset overview / summary figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Dataset statistics
    ax = axes[0]
    ax.axis("off")
    stats = []
    for _, row in cmp_df.iterrows():
        ctype = row["Cancer"]
        n_org = dfs[ctype.lower()]["organoid_id"].nunique() if ctype.lower() in dfs else "?"
        n_drug = dfs[ctype.lower()]["drug_name"].nunique() if ctype.lower() in dfs else "?"
        n_pairs = len(dfs[ctype.lower()]) if ctype.lower() in dfs else "?"
        stats.append([ctype, str(n_org), str(n_drug), str(n_pairs),
                      f"{row['Pearson']:.3f}", f"{row['AUROC']:.3f}"])

    table = ax.table(
        cellText=stats,
        colLabels=["Cancer", "Organoids", "Drugs", "Pairs", "Pearson", "AUROC"],
        cellLoc="center",
        loc="center",
        colColours=["#E8E8E8"] * 6,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("black")
        if key[0] == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title("Dataset Summary & LOOCV Results", fontsize=13, fontweight="bold", pad=20)

    # Panel B: Performance radar-like comparison
    ax = axes[1]
    metrics = ["Pearson", "R2", "AUROC"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (_, row) in enumerate(cmp_df.iterrows()):
        cancer = row["Cancer"]
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=cancer,
               color=COLORS.get(cancer, "#888"), edgecolor="black", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Cross-Cancer Performance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save:
        for fmt in ["pdf", "png"]:
            fig.savefig(FIG_DIR / f"fig_multicancer_overview.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def main():
    print("=" * 60)
    print("  Generating Multi-Cancer Comparison Figures")
    print("=" * 60)

    dfs, per_org, per_drug, cmp = load_results()

    print(f"  Loaded data for: {list(dfs.keys())}")
    print(f"  Comparison table:\n{cmp.to_string(index=False)}")

    if not dfs:
        print("  [ERROR] No LOOCV results found. Run finetune_multicancer.py first.")
        return

    # Generate all figures
    print("\n  [1] Multi-cancer performance bar chart...")
    fig_multicancer_performance(cmp)

    print("  [2] True vs Predicted scatter...")
    fig_per_cancer_scatter(dfs)

    if not per_org.empty:
        print("  [3] Per-organoid correlation bar chart...")
        fig_per_organoid_bar(per_org)

    print("  [4] Drug generalization analysis (PDAC)...")
    fig_drug_overlap_analysis(dfs, per_drug)

    print("  [5] PDAC heatmap...")
    fig_pdac_heatmap(dfs)

    if "blca" in dfs:
        print("  [6] BLCA heatmap...")
        fig_blca_heatmap(dfs)

    print("  [7] Multi-cancer ROC curves...")
    fig_multicancer_roc(dfs)

    print("  [8] Dataset overview...")
    fig_dataset_overview(cmp, dfs)

    print(f"\n  All figures saved to {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
