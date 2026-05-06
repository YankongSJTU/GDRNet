# -*- coding: utf-8 -*-
"""
GDRNet Demo — Quick Start
==========================
Train and evaluate GDRNet on small synthetic demo data.

Usage:
  cd GDRNet
  python demo/run_demo.py
"""

import sys
import io
import numpy as np
import pandas as pd
import torch
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

DEMO = Path(__file__).parent / "data"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from models.gdr_v11 import GDRNetV11, V11Dataset, train_v11


def run_demo():
    print("=" * 60)
    print("  GDRNet Demo — Training on Synthetic Data")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load demo data
    meta = pd.read_csv(DEMO / "gdsc_metadata.csv")
    gene_expr = np.load(DEMO / "gdsc_gene_expr.npy")
    scf_emb = np.load(DEMO / "scfoundation_cell_emb.npy")
    cell_ids = np.load(DEMO / "scfoundation_cell_ids.npy", allow_pickle=True)
    drug_features = np.load(DEMO / "gdsc_drug_features.npy")

    cell_to_idx = {c: i for i, c in enumerate(cell_ids)}
    drug_list = sorted(meta["drug_name"].unique())
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}

    n_cells = len(cell_ids)
    n_drugs = len(drug_list)
    fp = drug_features[:, :2048].astype(np.float32)
    desc_raw = drug_features[:, 2048:2048+188].astype(np.float32)

    # Build input arrays
    cell_idx_arr = np.array([cell_to_idx.get(c, 0) for c in meta["ModelID"]], dtype=np.int64)
    drug_idx_arr = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]], dtype=np.int64)
    scf_arr = np.array([scf_emb[cell_to_idx[c]] for c in meta["ModelID"]], dtype=np.float32)
    gene_arr = np.array([gene_expr[cell_to_idx[c]] for c in meta["ModelID"]], dtype=np.float32)

    y = meta["LN_IC50"].values.astype(np.float32)

    # Split
    n = len(y)
    tr_idx = np.arange(int(n * 0.8))
    te_idx = np.arange(int(n * 0.8), n)

    tr_ds = V11Dataset(gene_arr[tr_idx], scf_arr[tr_idx], fp[tr_idx], desc_raw[tr_idx],
                        cell_idx_arr[tr_idx], drug_idx_arr[tr_idx], y[tr_idx])
    te_ds = V11Dataset(gene_arr[te_idx], scf_arr[te_idx], fp[te_idx], desc_raw[te_idx],
                        cell_idx_arr[te_idx], drug_idx_arr[te_idx], y[te_idx])

    print(f"  Train: {len(tr_ds)}, Test: {len(te_ds)}")
    print(f"  Cells: {n_cells}, Drugs: {n_drugs}")

    # Build model
    model = GDRNetV11(
        n_genes=2000, scf_dim=3072, fp_bits=2048, n_desc=188,
        n_cells=n_cells, n_drugs=n_drugs,
        d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
        n_deep=3, dropout=0.15,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params/1e6:.2f}M")

    # Train
    model, metrics, history, preds = train_v11(
        model, tr_ds, te_ds,
        model_name="demo_model",
        n_epochs=20, batch_size=32, lr=1e-3, patience=10,
        device=device,
    )

    print("\n" + "=" * 60)
    print("  Demo Results")
    print("=" * 60)
    print(f"  Pearson: {metrics['Pearson']:.4f}")
    print(f"  R2:      {metrics['R2']:.4f}")
    print(f"  RMSE:    {metrics['RMSE']:.4f}")
    print(f"  AUROC:   {metrics['AUROC']:.4f}")
    print("=" * 60)
    print("  Note: Results are on synthetic data for demonstration only.")
    print("  Real performance requires actual GDSC/organoid data.")


if __name__ == "__main__":
    run_demo()
