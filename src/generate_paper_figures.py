# -*- coding: utf-8 -*-
"""
GDRNetV11 Paper Figures and Tables Generator
=============================================
Generate all figures and formatted tables for Briefings in Bioinformatics paper.

Main Figures (6):
  Fig 1 - Model Architecture (reference for hand-drawing)
  Fig 2 - GDSC Performance Comparison
  Fig 3 - Organoid LOOCV Results
  Fig 4 - Per-Organoid Heatmap
  Fig 5 - Ablation Study
  Fig 6 - Scatter Plots (GDSC + Organoid)

Supplementary Figures (3):
  Fig S1 - Training Curves
  Fig S2 - Drug Category Performance
  Fig S3 - Embedding UMAP

Tables (4):
  Table 1 - Dataset Statistics
  Table 2 - GDSC Comparison
  Table 3 - Organoid LOOCV Results
  Table 4 - Ablation Study
"""

import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path

ROOT = Path("/export/home/kongyan/project/Organoid")
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures/paper"
FIGURES.mkdir(parents=True, exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

# Color palette
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "tertiary": "#F18F01",
    "quaternary": "#C73E1D",
    "success": "#3A7D44",
    "neutral": "#6C757D",
    "cell": "#4A90D9",
    "drug": "#E85D75",
    "cross": "#50C878",
    "deep": "#9B59B6",
}


def save_figure(fig, name, dpi=300):
    """Save figure as PDF and PNG, also save raw data."""
    fig.savefig(FIGURES / f"{name}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}.pdf/.png")


def save_data(df, name):
    """Save raw data CSV for easy redo."""
    df.to_csv(FIGURES / f"{name}_data.csv", index=False)
    print(f"  Saved: {name}_data.csv")


# ── Figure 1: Model Architecture ──────────────────────────────────────────────

def fig1_architecture():
    """Generate V11 architecture diagram (reference for hand-drawing)."""
    print("\n[Fig 1] Model Architecture ...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Helper function for boxes
    def draw_box(x, y, w, h, label, color, fontsize=10):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.8)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Title
    ax.text(7, 9.5, "GDRNetV11 Architecture", ha="center", fontsize=16, fontweight="bold")

    # Cell branch (left side)
    ax.text(3, 8.8, "Cell Representation", ha="center", fontsize=12, fontweight="bold", color=COLORS["cell"])
    draw_box(1, 7.8, 2, 0.6, "Gene Expr\n(2000-dim)", COLORS["cell"], 8)
    draw_box(4, 7.8, 2, 0.6, "scF Emb\n(3072-dim)", COLORS["cell"], 8)
    draw_box(7, 7.8, 1.5, 0.6, "Cell ID\n(64-dim)", COLORS["cell"], 8)

    draw_box(1, 6.8, 2, 0.5, "Gene Enc", COLORS["neutral"], 9)
    draw_box(4, 6.8, 2, 0.5, "scF Enc", COLORS["neutral"], 9)

    draw_arrow(2, 7.8, 2, 7.3)
    draw_arrow(5, 7.8, 5, 7.3)
    draw_arrow(7.75, 7.8, 7.75, 7.3)

    # Cell concat
    draw_box(2.5, 5.8, 4, 0.5, "Concat (384-dim)", COLORS["cell"], 9)
    draw_arrow(2, 6.8, 3.5, 6.3)
    draw_arrow(5, 6.8, 4.5, 6.3)
    draw_arrow(7.75, 7.3, 7.75, 6.3)

    # Drug branch (right side)
    ax.text(11, 8.8, "Drug Representation", ha="center", fontsize=12, fontweight="bold", color=COLORS["drug"])
    draw_box(9.5, 7.8, 2, 0.6, "Morgan FP\n(2048-dim)", COLORS["drug"], 8)
    draw_box(11.5, 7.8, 2, 0.6, "RDKit Desc\n(188-dim)", COLORS["drug"], 8)
    draw_box(9.5, 7.0, 1.5, 0.5, "Drug ID\n(64-dim)", COLORS["drug"], 8)

    draw_box(9.5, 6.3, 2, 0.5, "FP Enc", COLORS["neutral"], 9)
    draw_box(11.5, 6.3, 2, 0.5, "Desc Enc", COLORS["neutral"], 9)

    draw_arrow(10.5, 7.8, 10.5, 6.8)
    draw_arrow(12.5, 7.8, 12.5, 6.8)
    draw_arrow(10.25, 7.0, 10.25, 6.8)

    # Drug concat
    draw_box(10, 5.3, 4, 0.5, "Concat (384-dim)", COLORS["drug"], 9)
    draw_arrow(10.5, 6.3, 11, 5.8)
    draw_arrow(12.5, 6.3, 12, 5.8)
    draw_arrow(10.25, 6.3, 10.25, 5.8)

    # Input projection
    draw_box(5.5, 4.5, 5, 0.5, "Input Projection (768 → 128)", COLORS["neutral"], 9)
    draw_arrow(4.5, 5.8, 7, 5.0)
    draw_arrow(12, 5.3, 9, 5.0)

    # DCN v2 Cross Network
    ax.text(3.5, 4.0, "DCN v2 Cross Network", ha="center", fontsize=11, fontweight="bold", color=COLORS["cross"])
    draw_box(2, 3.0, 3, 0.5, "Cross Layer 1", COLORS["cross"], 9)
    draw_box(2, 2.3, 3, 0.5, "Cross Layer 2", COLORS["cross"], 9)
    draw_box(2, 1.6, 3, 0.5, "Cross Layer 3", COLORS["cross"], 9)
    draw_arrow(7, 4.5, 3.5, 3.5)
    draw_arrow(3.5, 3.0, 3.5, 2.8)
    draw_arrow(3.5, 2.3, 3.5, 2.1)

    # Deep MLP Network
    ax.text(10.5, 4.0, "Deep MLP Network", ha="center", fontsize=11, fontweight="bold", color=COLORS["deep"])
    draw_box(9, 3.0, 3, 0.5, "Deep Layer 1", COLORS["deep"], 9)
    draw_box(9, 2.3, 3, 0.5, "Deep Layer 2", COLORS["deep"], 9)
    draw_box(9, 1.6, 3, 0.5, "Deep Layer 3", COLORS["deep"], 9)
    draw_arrow(9, 4.5, 10.5, 3.5)
    draw_arrow(10.5, 3.0, 10.5, 2.8)
    draw_arrow(10.5, 2.3, 10.5, 2.1)

    # Concat cross + deep
    draw_box(5.5, 0.8, 5, 0.5, "Concat (256-dim)", COLORS["neutral"], 9)
    draw_arrow(3.5, 1.6, 6.5, 1.3)
    draw_arrow(10.5, 1.6, 9.5, 1.3)

    # Output head
    draw_box(6.5, 0.1, 3, 0.5, "Output Head\n→ IC50", COLORS["quaternary"], 9)
    draw_arrow(8, 0.8, 8, 0.6)

    # Legend
    legend_y = 0.3
    ax.add_patch(mpatches.Rectangle((0.3, legend_y), 0.3, 0.3, facecolor=COLORS["cell"], alpha=0.8))
    ax.text(0.8, legend_y + 0.15, "Cell Features", fontsize=8, va="center")
    ax.add_patch(mpatches.Rectangle((2.3, legend_y), 0.3, 0.3, facecolor=COLORS["drug"], alpha=0.8))
    ax.text(2.8, legend_y + 0.15, "Drug Features", fontsize=8, va="center")
    ax.add_patch(mpatches.Rectangle((4.3, legend_y), 0.3, 0.3, facecolor=COLORS["cross"], alpha=0.8))
    ax.text(4.8, legend_y + 0.15, "Cross Network", fontsize=8, va="center")
    ax.add_patch(mpatches.Rectangle((6.8, legend_y), 0.3, 0.3, facecolor=COLORS["deep"], alpha=0.8))
    ax.text(7.3, legend_y + 0.15, "Deep Network", fontsize=8, va="center")

    save_figure(fig, "fig1_architecture")
    # Save architecture description as text
    with open(FIGURES / "fig1_architecture_description.txt", "w") as f:
        f.write("GDRNetV11 Architecture\n")
        f.write("=" * 50 + "\n\n")
        f.write("Cell Branch:\n")
        f.write("  - Gene Expression: 2000-dim → Gene Encoder (MLP) → 128-dim\n")
        f.write("  - scFoundation Embedding: 3072-dim → scF Encoder (MLP) → 256-dim\n")
        f.write("  - Cell ID Embedding: learnable 64-dim\n")
        f.write("  - Concatenate → 384-dim cell representation\n\n")
        f.write("Drug Branch:\n")
        f.write("  - Morgan Fingerprint: 2048-bit → FP Encoder (MLP) → 256-dim\n")
        f.write("  - RDKit Descriptors: 188-dim → Desc Encoder (MLP) → 64-dim\n")
        f.write("  - Drug ID Embedding: learnable 64-dim\n")
        f.write("  - Concatenate → 384-dim drug representation\n\n")
        f.write("Interaction Layers:\n")
        f.write("  - Input Projection: 768-dim → 128-dim\n")
        f.write("  - DCN v2 Cross Network: 3 layers, rank=64\n")
        f.write("  - Deep MLP Network: 3 layers, hidden=256\n")
        f.write("  - Concatenate cross + deep → 256-dim\n")
        f.write("  - Output Head: 256 → 1 (IC50 prediction)\n\n")
        f.write("Total Parameters: 5.26M\n")
        f.write("Trainable (fine-tuning): 0.62M (cross + deep + head)\n")
    print("  Saved: fig1_architecture_description.txt")


# ── Figure 2: GDSC Performance Comparison ──────────────────────────────────────

def fig2_gdsc_comparison():
    """Grouped bar chart for GDSC performance."""
    print("\n[Fig 2] GDSC Performance Comparison ...")

    df = pd.read_csv(TABLES / "model_comparison_v11.csv", index_col=0)

    # Select models to show
    models = ["V11-Ensemble", "V11-s42", "V11-s123", "V11-s456", "LightGBM",
              "GDRNetV3", "GDRNetV10", "GDRNetV8"]
    df = df.loc[df.index.isin(models)]

    # Reorder
    order = ["V11-Ensemble", "V11-s42", "V11-s123", "V11-s456", "LightGBM",
             "GDRNetV3", "GDRNetV10", "GDRNetV8"]
    df = df.reindex([m for m in order if m in df.index])

    metrics = ["Pearson", "R2", "RMSE"]
    x = np.arange(len(df))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = [COLORS["primary"], COLORS["secondary"], COLORS["tertiary"]]

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        bars = ax.bar(x, df[metric], width=0.6, color=colors[i], alpha=0.8, edgecolor="black")
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} on GDSC Test Set", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=9)

        # Add value labels
        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

        # Highlight best
        if metric != "RMSE":
            best_idx = df[metric].argmax()
        else:
            best_idx = df[metric].argmin()
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    save_figure(fig, "fig2_gdsc_comparison")
    save_data(df.reset_index().rename(columns={"index": "Model"}), "fig2_gdsc_comparison")


# ── Figure 3: Organoid LOOCV Results ───────────────────────────────────────────

def fig3_organoid_loocv():
    """Grouped bar chart for organoid LOOCV results."""
    print("\n[Fig 3] Organoid LOOCV Results ...")

    df = pd.read_csv(TABLES / "organoid_comparison.csv", index_col=0)

    # Select key methods
    methods = ["FineTune-LOOCV", "LightGBM-LOOCV", "Pretrained-Direct"]
    df = df.loc[df.index.isin(methods)]
    df = df.reindex(methods)

    metrics = ["Pearson", "R2", "RMSE", "AUROC"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    colors = [COLORS["success"], COLORS["tertiary"], COLORS["neutral"]]

    for i, (metric, ax) in enumerate(zip(metrics, axes)):
        bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.8, edgecolor="black")
        ax.set_xlabel("Method", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} on Organoid LOOCV", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=15, ha="right", fontsize=9)

        for bar, val in zip(bars, df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_figure(fig, "fig3_organoid_loocv")
    save_data(df.reset_index().rename(columns={"index": "Method"}), "fig3_organoid_loocv")


# ── Figure 4: Per-Organoid Heatmap ─────────────────────────────────────────────

def fig4_per_organoid_heatmap():
    """Heatmap of per-organoid Pearson correlation."""
    print("\n[Fig 4] Per-Organoid Heatmap ...")

    df = pd.read_csv(TABLES / "organoid_loocv_per_organoid.csv")

    # Pivot for heatmap
    heatmap_data = df[["organoid", "P_Pretrain", "P_FineTune", "P_LightGBM"]].set_index("organoid")
    heatmap_data.columns = ["Pretrained", "FineTune", "LightGBM"]

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Per-Organoid Pearson Correlation", fontsize=14, fontweight="bold")
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Organoid ID", fontsize=12)

    plt.tight_layout()
    save_figure(fig, "fig4_per_organoid_heatmap")
    save_data(df, "fig4_per_organoid_heatmap")


# ── Figure 5: Ablation Study ──────────────────────────────────────────────────

def fig5_ablation():
    """Horizontal bar chart for ablation study."""
    print("\n[Fig 5] Ablation Study ...")

    df = pd.read_csv(TABLES / "ablation_comparison.csv", index_col=0)

    # Sort by Pearson
    df = df.sort_values("Pearson", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(df))
    bars = ax.barh(y, df["Pearson"], color=COLORS["primary"], alpha=0.8, edgecolor="black")

    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=11)
    ax.set_xlabel("Pearson Correlation", fontsize=12)
    ax.set_title("Ablation Study on Organoid LOOCV", fontsize=14, fontweight="bold")
    ax.set_xlim(0.85, 0.89)

    # Add value labels
    for bar, val in zip(bars, df["Pearson"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", ha="left", va="center", fontsize=10)

    # Highlight Full Model
    full_idx = list(df.index).index("Full Model")
    bars[full_idx].set_color(COLORS["success"])
    bars[full_idx].set_edgecolor("red")
    bars[full_idx].set_linewidth(2)

    plt.tight_layout()
    save_figure(fig, "fig5_ablation")
    save_data(df.reset_index().rename(columns={"index": "Variant"}), "fig5_ablation")


# ── Figure 6: Scatter Plots ───────────────────────────────────────────────────

def fig6_scatter_plots():
    """Scatter plots of predicted vs true IC50."""
    print("\n[Fig 6] Scatter Plots ...")

    # Load GDSC predictions and targets
    gdsc_preds = np.load(TABLES / "gdr_v11_ensemble_val_preds.npy")
    gdsc_targets = np.load(TABLES / "gdsc_val_targets.npy")

    # Load organoid predictions
    organoid_df = pd.read_csv(TABLES / "organoid_loocv_predictions.csv")
    organoid_true = organoid_df["ic50"].values
    organoid_pred = organoid_df["pred_finetuned"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GDSC scatter
    ax = axes[0]
    ax.scatter(gdsc_targets, gdsc_preds, alpha=0.3, s=10, c=COLORS["primary"])
    ax.plot([gdsc_targets.min(), gdsc_targets.max()],
            [gdsc_targets.min(), gdsc_targets.max()], 'r--', lw=2)
    ax.set_xlabel("True IC50", fontsize=12)
    ax.set_ylabel("Predicted IC50", fontsize=12)
    ax.set_title("GDSC Test Set (V11-Ensemble)", fontsize=14, fontweight="bold")

    # Calculate Pearson for display
    p_gdsc = np.corrcoef(gdsc_targets, gdsc_preds)[0, 1]
    ax.text(0.05, 0.95, f"Pearson = {p_gdsc:.4f}", transform=ax.transAxes,
            fontsize=12, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Organoid scatter
    ax = axes[1]
    ax.scatter(organoid_true, organoid_pred, alpha=0.5, s=30, c=COLORS["success"])
    ax.plot([organoid_true.min(), organoid_true.max()],
            [organoid_true.min(), organoid_true.max()], 'r--', lw=2)
    ax.set_xlabel("True IC50", fontsize=12)
    ax.set_ylabel("Predicted IC50", fontsize=12)
    ax.set_title("Organoid LOOCV (FineTune)", fontsize=14, fontweight="bold")

    p_org = np.corrcoef(organoid_true, organoid_pred)[0, 1]
    ax.text(0.05, 0.95, f"Pearson = {p_org:.4f}", transform=ax.transAxes,
            fontsize=12, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    save_figure(fig, "fig6_scatter_plots")

    # Save raw data
    gdsc_df = pd.DataFrame({"true_ic50": gdsc_targets, "predicted_ic50": gdsc_preds})
    save_data(gdsc_df, "fig6_scatter_gdsc")

    organoid_scatter_df = pd.DataFrame({"true_ic50": organoid_true, "predicted_ic50": organoid_pred})
    save_data(organoid_scatter_df, "fig6_scatter_organoid")


# ── Figure S1: Training Curves ────────────────────────────────────────────────

def figs1_training_curves():
    """Training curves for V11 models."""
    print("\n[Fig S1] Training Curves ...")

    seeds = [42, 123, 456]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for i, seed in enumerate(seeds):
        history = pd.read_csv(TABLES / f"gdr_v11_s{seed}_history.csv")

        # Loss curve
        ax = axes[0, i]
        ax.plot(history["epoch"], history["train_loss"], color=COLORS["primary"], lw=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Training Loss", fontsize=11)
        ax.set_title(f"Seed {seed} - Training Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Pearson curve
        ax = axes[1, i]
        ax.plot(history["epoch"], history["val_pearson"], color=COLORS["success"], lw=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation Pearson", fontsize=11)
        ax.set_title(f"Seed {seed} - Validation Pearson", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "figs1_training_curves")

    # Combine histories
    all_histories = []
    for seed in seeds:
        h = pd.read_csv(TABLES / f"gdr_v11_s{seed}_history.csv")
        h["seed"] = seed
        all_histories.append(h)
    combined = pd.concat(all_histories, ignore_index=True)
    save_data(combined, "figs1_training_curves")


# ── Figure S2: Drug Category Performance ──────────────────────────────────────

def figs2_drug_performance():
    """Per-drug performance analysis."""
    print("\n[Fig S2] Drug Category Performance ...")

    df = pd.read_csv(TABLES / "organoid_per_drug.csv")

    # Sort by Pearson
    df = df.sort_values("Pearson", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))

    y = np.arange(len(df))
    colors = [COLORS["success"] if p > 0 else COLORS["quaternary"] for p in df["Pearson"]]

    bars = ax.barh(y, df["Pearson"], color=colors, alpha=0.8, edgecolor="black")

    ax.set_yticks(y)
    ax.set_yticklabels(df["drug"], fontsize=8)
    ax.set_xlabel("Pearson Correlation", fontsize=12)
    ax.set_title("Per-Drug Performance on Organoid LOOCV", fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", lw=1)

    plt.tight_layout()
    save_figure(fig, "figs2_drug_performance")
    save_data(df, "figs2_drug_performance")


# ── Figure S3: Embedding UMAP ─────────────────────────────────────────────────

def figs3_embedding_umap():
    """UMAP visualization of scF embeddings."""
    print("\n[Fig S3] Embedding UMAP ...")

    try:
        import umap
    except ImportError:
        print("  Warning: umap-learn not available, using PCA instead")
        from sklearn.decomposition import PCA
        umap = None

    # Load organoid embeddings
    meta = pd.read_csv(TABLES / "organoid_loocv_predictions.csv")
    cell_emb = np.load(ROOT / "data/processed/organoid_cell_emb.npy")

    # Get unique organoids and their embeddings
    organoid_ids = np.load(ROOT / "data/processed/organoid_cell_ids.npy", allow_pickle=True)

    # Reduce dimensionality
    if umap:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb_2d = reducer.fit_transform(cell_emb)
    else:
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(cell_emb)

    # Create dataframe
    emb_df = pd.DataFrame({
        "organoid": organoid_ids,
        "UMAP1": emb_2d[:, 0],
        "UMAP2": emb_2d[:, 1]
    })

    # Add average response per organoid
    avg_response = meta.groupby("organoid_id")["ic50"].mean().reset_index()
    avg_response.columns = ["organoid", "avg_ic50"]
    emb_df = emb_df.merge(avg_response, on="organoid", how="left")

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb_df["UMAP1"], emb_df["UMAP2"],
                         c=emb_df["avg_ic50"], cmap="coolwarm",
                         s=200, alpha=0.8, edgecolors="black")

    # Add labels
    for i, row in emb_df.iterrows():
        ax.text(row["UMAP1"], row["UMAP2"], row["organoid"],
                fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("UMAP 1" if umap else "PC1", fontsize=12)
    ax.set_ylabel("UMAP 2" if umap else "PC2", fontsize=12)
    ax.set_title("scFoundation Embedding Visualization", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Average IC50", fontsize=11)

    plt.tight_layout()
    save_figure(fig, "figs3_embedding_umap")
    save_data(emb_df, "figs3_embedding_umap")


# ── Tables ────────────────────────────────────────────────────────────────────

def generate_tables():
    """Generate formatted tables for manuscript."""
    print("\n[Tables] Generating formatted tables ...")

    # Table 1: Dataset Statistics
    table1 = pd.DataFrame({
        "Dataset": ["GDSC", "Organoid"],
        "Cell Lines": [700, 16],
        "Drugs": [286, 34],
        "Pairs": ["~177,000", 544],
        "IC50 Range": ["-5 to 10", "-2 to 5"],
        "Data Source": ["GDSC2", "In-house"]
    })
    table1.to_csv(TABLES / "table1_dataset_stats.csv", index=False)
    print("  Saved: table1_dataset_stats.csv")

    # Table 2: GDSC Comparison
    df = pd.read_csv(TABLES / "model_comparison_v11.csv", index_col=0)
    models = ["V11-Ensemble", "V11-s42", "V11-s123", "V11-s456", "LightGBM",
              "GDRNetV3", "GDRNetV10", "GDRNetV8"]
    table2 = df.loc[df.index.isin(models)].reindex([m for m in models if m in df.index])
    table2 = table2.reset_index().rename(columns={"index": "Model"})
    table2.to_csv(TABLES / "table2_gdsc_comparison.csv", index=False)
    print("  Saved: table2_gdsc_comparison.csv")

    # Table 3: Organoid LOOCV
    df = pd.read_csv(TABLES / "organoid_comparison.csv", index_col=0)
    methods = ["FineTune-LOOCV", "LightGBM-LOOCV", "Pretrained-Direct"]
    table3 = df.loc[df.index.isin(methods)].reindex(methods)
    table3 = table3.reset_index().rename(columns={"index": "Method"})
    table3.to_csv(TABLES / "table3_organoid_loocv.csv", index=False)
    print("  Saved: table3_organoid_loocv.csv")

    # Table 4: Ablation
    df = pd.read_csv(TABLES / "ablation_comparison.csv", index_col=0)
    table4 = df.sort_values("Pearson", ascending=False).reset_index().rename(columns={"index": "Variant"})
    table4.to_csv(TABLES / "table4_ablation.csv", index=False)
    print("  Saved: table4_ablation.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GDRNetV11 Paper Figures and Tables Generator")
    print("=" * 65)

    # Create output directory
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Generate main figures
    fig1_architecture()
    fig2_gdsc_comparison()
    fig3_organoid_loocv()
    fig4_per_organoid_heatmap()
    fig5_ablation()
    fig6_scatter_plots()

    # Generate supplementary figures
    figs1_training_curves()
    figs2_drug_performance()
    figs3_embedding_umap()

    # Generate tables
    generate_tables()

    print("\n" + "=" * 65)
    print("  All figures and tables generated!")
    print(f"  Output directory: {FIGURES}")
    print("=" * 65)


if __name__ == "__main__":
    main()
