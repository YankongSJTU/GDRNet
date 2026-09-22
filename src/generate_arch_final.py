#!/usr/bin/env python3
"""
Figure 8 Architecture Diagram — Final polish
=============================================
Strategy:
  1. Load AI-generated architecture image as visual base
  2. Cover inaccurate text areas with colored rectangles (matching zone colors)
  3. Overlay accurate vector labels, title, subtitle, legend, and training strategy

This ensures the diagram looks polished with correct scientific terminology
while retaining the visual quality of the AI-generated illustration.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results/figures/paper"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load AI-generated architecture image
# ---------------------------------------------------------------------------
img_path = FIG / "fig8A_workflow_7793767.png"
img = Image.open(img_path)
img_w, img_h = img.size  # 2752 x 1536
print(f"Loaded image: {img_w}x{img_h}")

# Convert to numpy array for matplotlib
img_arr = np.array(img)

# ---------------------------------------------------------------------------
# Color palette (Nature/Cell style)
# ---------------------------------------------------------------------------
COL = {
    "teal":    "#007C83",  # cell branch
    "vermilion": "#D55E00",  # drug branch
    "emerald": "#009E73",  # cross network
    "purple":  "#7030A0",  # deep network
    "orange":  "#E69F00",  # output
    "slate":   "#34495E",  # fusion and merge
    "white":   "#FFFFFF",
    "light_gray": "#E5E5E5",
    "text_dark": "#333333",
    "text_light": "#FFFFFF",
}

# ---------------------------------------------------------------------------
# Step 1: Cover inaccurate text areas with colored rectangles
# ---------------------------------------------------------------------------
# The AI image has text at various positions that may be inaccurate.
# We cover these with rectangles matching the background colors of each zone.
# Coordinates are in normalized (0-1) figure coordinates.

def cover_text_region(ax, x0, y0, x1, y1, color="#FFFFFF", alpha=1.0):
    """Cover a text region with a colored rectangle."""
    rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=color, edgecolor='none',
                               alpha=alpha, zorder=10)
    ax.add_patch(rect)

# Create figure with the image as background
fig = plt.figure(figsize=(24.0, 10.0), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img_arr, extent=[0, 1, 0, 1], aspect='auto')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis("off")

# Cover regions with inaccurate text
# Top title area (AI text is often garbled)
cover_text_region(ax, 0.0, 0.88, 1.0, 0.98, COL["white"], alpha=0.95)

# Bottom training strategy area
cover_text_region(ax, 0.0, 0.0, 1.0, 0.08, COL["white"], alpha=0.95)

# Cover scattered label areas throughout the diagram
# These are rough estimates based on typical AI-generated diagram layouts
cover_text_region(ax, 0.02, 0.45, 0.20, 0.55, COL["white"], alpha=0.85)   # cell labels
cover_text_region(ax, 0.25, 0.45, 0.45, 0.55, COL["white"], alpha=0.85)   # drug labels
cover_text_region(ax, 0.40, 0.30, 0.60, 0.45, COL["white"], alpha=0.85)   # fusion labels
cover_text_region(ax, 0.55, 0.15, 0.75, 0.30, COL["white"], alpha=0.85)   # output labels
cover_text_region(ax, 0.50, 0.55, 0.70, 0.70, COL["white"], alpha=0.85)   # cross/deep labels

# ---------------------------------------------------------------------------
# Step 2: Overlay accurate vector labels
# ---------------------------------------------------------------------------

def overlay_text(ax, x, y, text, fontsize=10, color=COL["text_dark"],
                 weight="normal", ha="center", va="center",
                 bbox=None, zorder=20):
    """Overlay text on the image."""
    txt = ax.text(x, y, text, fontsize=fontsize, color=color,
                  fontweight=weight, ha=ha, va=va,
                  bbox=bbox, zorder=zorder,
                  transform=ax.transAxes)
    return txt

def overlay_label_box(ax, x, y, text, fontsize=9,
                      box_color=COL["teal"], text_color=COL["text_light"],
                      pad=0.01, zorder=20):
    """Overlay a small colored label box with text."""
    bbox = dict(boxstyle=f"round,pad={pad}", facecolor=box_color,
                edgecolor=box_color, alpha=0.90)
    overlay_text(ax, x, y, text, fontsize=fontsize,
                 color=text_color, weight="bold",
                 bbox=bbox, zorder=zorder)

# ---- Title and subtitle ----
overlay_text(ax, 0.50, 0.93, "GDRNet Architecture",
             fontsize=18, color=COL["text_dark"], weight="bold",
             ha="center", va="center", zorder=20)
overlay_text(ax, 0.50, 0.91, "Deep Cross-Network for Drug Sensitivity Prediction",
             fontsize=12, color=COL["text_dark"], weight="normal",
             ha="center", va="center", zorder=20)

# ---- Zone 1: INPUTS ----
# Cell stream labels
overlay_label_box(ax, 0.10, 0.78, "Cell profile", box_color=COL["teal"])
overlay_text(ax, 0.10, 0.74, "Gene expression\nscFoundation\nCell ID",
             fontsize=7, color=COL["text_dark"], ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.005", facecolor=COL["light_gray"],
                      edgecolor=COL["teal"], alpha=0.9, linewidth=1.0), zorder=20)
overlay_label_box(ax, 0.10, 0.68, "Cell encoder", box_color=COL["teal"])

# Drug stream labels
overlay_label_box(ax, 0.28, 0.78, "Drug profile", box_color=COL["vermilion"])
overlay_text(ax, 0.28, 0.74, "Morgan fingerprint\nDrug ID",
             fontsize=7, color=COL["text_dark"], ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.005", facecolor=COL["light_gray"],
                      edgecolor=COL["vermilion"], alpha=0.9, linewidth=1.0), zorder=20)
overlay_label_box(ax, 0.28, 0.68, "Drug encoder", box_color=COL["vermilion"])

# Dimension badges for inputs
overlay_text(ax, 0.14, 0.82, "2000", fontsize=7, color=COL["teal"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["teal"], alpha=0.8, linewidth=0.5), zorder=20)
overlay_text(ax, 0.14, 0.79, "3072", fontsize=7, color=COL["teal"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["teal"], alpha=0.8, linewidth=0.5), zorder=20)
overlay_text(ax, 0.14, 0.76, "64", fontsize=7, color=COL["teal"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["teal"], alpha=0.8, linewidth=0.5), zorder=20)

overlay_text(ax, 0.32, 0.82, "2048", fontsize=7, color=COL["vermilion"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["vermilion"], alpha=0.8, linewidth=0.5), zorder=20)
overlay_text(ax, 0.32, 0.79, "320", fontsize=7, color=COL["vermilion"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["vermilion"], alpha=0.8, linewidth=0.5), zorder=20)

# ---- Zone 2: CORE MODEL ----
# Fusion node
overlay_label_box(ax, 0.50, 0.62, "Multimodal fusion", box_color=COL["slate"])
overlay_text(ax, 0.50, 0.59, "896-d", fontsize=8, color=COL["text_light"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.004", facecolor=COL["slate"],
                      edgecolor=COL["slate"], alpha=0.85, linewidth=0.5), zorder=20)

# Cross network tower
overlay_label_box(ax, 0.42, 0.42, "Cross network", box_color=COL["emerald"])
overlay_text(ax, 0.42, 0.36, "x₀  512-d", fontsize=7, color=COL["emerald"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["emerald"], alpha=0.8, linewidth=0.5), zorder=20)
overlay_text(ax, 0.42, 0.33, "xᴄ  512-d", fontsize=7, color=COL["emerald"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["emerald"], alpha=0.8, linewidth=0.5), zorder=20)
overlay_text(ax, 0.42, 0.30, "xᴅ  64-d", fontsize=7, color=COL["emerald"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["emerald"], alpha=0.8, linewidth=0.5), zorder=20)

# Deep network tower
overlay_label_box(ax, 0.58, 0.42, "Deep network", box_color=COL["purple"])
overlay_text(ax, 0.58, 0.36, "Residual MLP", fontsize=7, color=COL["purple"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor="white",
                      edgecolor=COL["purple"], alpha=0.8, linewidth=0.5), zorder=20)

# Merge node
overlay_label_box(ax, 0.50, 0.28, "Merge", box_color=COL["slate"])
overlay_text(ax, 0.50, 0.25, "576-d", fontsize=8, color=COL["text_light"],
             weight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.004", facecolor=COL["slate"],
                      edgecolor=COL["slate"], alpha=0.85, linewidth=0.5), zorder=20)

# ---- Zone 3: OUTPUT ----
overlay_label_box(ax, 0.65, 0.15, "Predicted lnIC₅₀", box_color=COL["orange"])
overlay_text(ax, 0.65, 0.12, "lower = more sensitive", fontsize=7,
             color=COL["text_dark"], ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.003", facecolor=COL["light_gray"],
                      edgecolor=COL["orange"], alpha=0.9, linewidth=0.5), zorder=20)

# ---- Training strategy timeline (bottom) ----
overlay_text(ax, 0.50, 0.05, "Training strategy",
             fontsize=10, color=COL["text_dark"], weight="bold",
             ha="center", va="center", zorder=20)

# Step 1
overlay_text(ax, 0.25, 0.03, "① GDSC pretraining\n700 cell lines × 229 drugs",
             fontsize=7, color=COL["text_dark"], ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.005", facecolor=COL["light_gray"],
                      edgecolor=COL["teal"], alpha=0.9, linewidth=0.8), zorder=20)

# Step 2
overlay_text(ax, 0.55, 0.03, "② Organoid LOOCV fine-tuning\nFrozen encoders; CRC / PDAC / BLCA",
             fontsize=7, color=COL["text_dark"], ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.005", facecolor=COL["light_gray"],
                      edgecolor=COL["emerald"], alpha=0.9, linewidth=0.8), zorder=20)

# Bracket between training and model
overlay_text(ax, 0.50, 0.07, "↑", fontsize=12, color=COL["slate"],
             ha="center", va="center", zorder=20)

# ---- Legend (bottom right corner) ----
legend_items = [
    (COL["teal"], "Cell"),
    (COL["vermilion"], "Drug"),
    (COL["emerald"], "Cross"),
    (COL["purple"], "Deep"),
    (COL["orange"], "Output"),
]

legend_x = 0.82
legend_y = 0.06
for i, (color, label) in enumerate(legend_items):
    y_pos = legend_y - i * 0.025
    # Color chip
    chip = mpatches.Rectangle((legend_x - 0.02, y_pos - 0.008), 0.015, 0.016,
                               facecolor=color, edgecolor=color,
                               alpha=0.9, zorder=20)
    ax.add_patch(chip)
    # Label
    overlay_text(ax, legend_x + 0.005, y_pos, label,
                 fontsize=7, color=COL["text_dark"], ha="left", va="center",
                 zorder=20)

# ---- Arrow flow annotations ----
# Cell encoder -> Fusion
overlay_text(ax, 0.20, 0.64, "h_c  R^576", fontsize=7, color=COL["teal"],
             weight="bold", ha="center", va="center", zorder=20)

# Drug encoder -> Fusion
overlay_text(ax, 0.35, 0.64, "h_d  R^320", fontsize=7, color=COL["vermilion"],
             weight="bold", ha="center", va="center", zorder=20)

# Fusion -> Cross/Deep
overlay_text(ax, 0.48, 0.52, "896-d", fontsize=7, color=COL["slate"],
             weight="bold", ha="center", va="center", zorder=20)

# Cross/Deep -> Merge
overlay_text(ax, 0.50, 0.36, "512-d", fontsize=7, color=COL["emerald"],
             weight="bold", ha="center", va="center", zorder=20)

# Merge -> Output
overlay_text(ax, 0.58, 0.22, "576-d", fontsize=7, color=COL["slate"],
             weight="bold", ha="center", va="center", zorder=20)

# ---- Equation reference tags ----
overlay_text(ax, 0.48, 0.55, "Eq.(1)-(15)", fontsize=6,
             color="#888888", fontstyle="italic", ha="center", va="center", zorder=20)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
for fmt in ["pdf", "png"]:
    out_path = FIG / f"fig8A_architecture_final.{fmt}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.05)
    print(f"Saved: {out_path}")

plt.close(fig)
print("Done!")
