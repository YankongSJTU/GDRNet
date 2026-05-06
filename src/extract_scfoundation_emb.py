# -*- coding: utf-8 -*-
"""
Extract scFoundation Cell Embeddings for GDSC Cell Lines
=========================================================
Loads the scFoundation pre-trained model and generates 768-dim cell embeddings
for all 700 GDSC cell lines using their 2000-gene expression profiles.

Input : data/processed/gdsc_cell_features.parquet  (N_samples × 2000 genes, z-scored)
Output: data/processed/scfoundation_cell_emb.npy   (700 × 768 float32)
        data/processed/scfoundation_cell_ids.npy   (700,)  — ModelID strings in order

Usage:
  cd /export/home/kongyan/project/Organoid
  python src/extract_scfoundation_emb.py
"""

import sys, os, warnings, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT      = Path("/export/home/kongyan/project/Organoid")
PROC_DIR  = ROOT / "data/processed"
SCF_DIR   = ROOT / "external/scFoundation"
MODEL_DIR = SCF_DIR / "model"

# Add scFoundation model dir to path
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "pretrainmodels"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH   = 32   # cell lines per batch (each is 19264-dim → manageable)


def load_scfoundation():
    """Load pre-trained scFoundation model (cell embedding version)."""
    from load import load_model_frommmf
    ckpt = MODEL_DIR / "models" / "models.ckpt"
    assert ckpt.exists(), f"Model weights not found at {ckpt}"
    print(f"  Loading scFoundation from {ckpt} ...")
    model, config = load_model_frommmf(str(ckpt), key="cell")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  scFoundation loaded  |  params={n_params/1e6:.1f}M  |  device={DEVICE}")
    return model, config


def prepare_expression(cell_features_df, gene_list):
    """
    Map z-scored 2000-gene expression to scFoundation 19264-gene space.

    Steps:
      1. Shift z-scores to all-positive (add |min| + 1)  — simulates raw-count-like values
      2. Zero-pad missing genes
      3. Align to scFoundation gene order
    Returns: (N_cells, 19264) float32 numpy array, already shifted but NOT yet CPM-normalised
             (normalisation is done per-cell inside the inference loop)
    """
    print("  Preparing expression matrix ...")
    # One row per unique cell line (average replicates if any)
    # (gdsc_cell_features already has unique cell lines stacked across drugs;
    #  we just need unique rows, since the same cell has same expression)
    df = cell_features_df.copy()

    # Shift z-scores → all positive (simulate library-size-like counts)
    shift = float(-df.values.min()) + 1.0    # ≥ 1 for every gene
    df = df + shift

    # Add missing genes as zero columns
    missing = list(set(gene_list) - set(df.columns))
    if missing:
        zero_df = pd.DataFrame(
            np.zeros((len(df), len(missing)), dtype=np.float32),
            columns=missing,
            index=df.index,
        )
        df = pd.concat([df, zero_df], axis=1)
    df = df[gene_list].astype(np.float32)
    print(f"  Expression matrix: {df.shape}  (shift applied: +{shift:.4f})")
    return df.values   # (N_cells, 19264)


@torch.no_grad()
def extract_embeddings(model, config, expr_arr):
    """
    Run scFoundation cell-embedding inference in batches.
    Returns (N_cells, 768) float32 array.
    """
    from load import gatherData
    import scanpy as sc
    import anndata

    N = expr_arr.shape[0]
    all_emb = []

    pad_id = config.get("pad_token_id", 0)

    for start in tqdm(range(0, N, BATCH), desc="  scFoundation inference"):
        end   = min(start + BATCH, N)
        batch = expr_arr[start:end]               # (B, 19264)

        # CPM normalise + log1p  (bulk pre_normalized='F' style)
        ad = anndata.AnnData(batch.copy())
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        log_expr = torch.tensor(ad.X, dtype=torch.float32).to(DEVICE)  # (B, 19264)

        # Append two total-count tokens (log10 of original sum, bulk-style)
        raw_sum   = torch.tensor(batch, dtype=torch.float32).sum(dim=1, keepdim=True)  # (B,1)
        total_tok = torch.log10(raw_sum.clamp(min=1.0)).to(DEVICE)         # (B,1)
        pretrain_gene_x = torch.cat(
            [log_expr, total_tok, total_tok], dim=1
        )                                                                   # (B, 19266)

        data_gene_ids = torch.arange(19266, device=DEVICE).unsqueeze(0).expand(end - start, -1)

        # Gather non-zero tokens
        value_labels = pretrain_gene_x > 0
        x, x_padding = gatherData(pretrain_gene_x, value_labels, pad_id)

        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pad_id)

        # Forward through encoder
        x_emb  = model.token_emb(x.unsqueeze(2).float(), output_weight=0)
        pos_emb = model.pos_emb(position_gene_ids)
        x_emb  += pos_emb
        enc_out = model.encoder(x_emb, x_padding)           # (B, seq_len, 192)

        # 4-way pooling → 768
        e1 = enc_out[:, -1, :]                              # last token
        e2 = enc_out[:, -2, :]                              # 2nd-to-last
        e3, _ = torch.max(enc_out[:, :-2, :], dim=1)       # max over gene tokens
        e4 = torch.mean(enc_out[:, :-2, :], dim=1)         # mean over gene tokens
        emb = torch.cat([e1, e2, e3, e4], dim=1)           # (B, 768)

        all_emb.append(emb.cpu().float().numpy())

        del x_emb, pos_emb, enc_out, pretrain_gene_x, log_expr
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return np.concatenate(all_emb, axis=0)   # (N, 768)


def main():
    print("=" * 65)
    print("  scFoundation Cell Embedding Extraction")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # ── Gene list ─────────────────────────────────────────────────────
    gene_list_df = pd.read_csv(MODEL_DIR / "OS_scRNA_gene_index.19264.tsv",
                                sep="\t", header=0)
    gene_list = list(gene_list_df["gene_name"])
    print(f"  scFoundation gene space: {len(gene_list)}")

    # ── Load GDSC cell features ───────────────────────────────────────
    X_full = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    meta   = pd.read_parquet(PROC_DIR / "gdsc_metadata.parquet")

    # Unique cell lines (keep first occurrence per ModelID)
    cell_ids = meta["ModelID"].values
    unique_ids, first_idx = np.unique(cell_ids, return_index=True)
    X_unique = X_full.iloc[first_idx].copy()
    X_unique.index = unique_ids
    print(f"  Unique GDSC cell lines: {len(unique_ids)}")

    # ── Gene coverage ─────────────────────────────────────────────────
    our_genes = set(X_unique.columns)
    scf_genes = set(gene_list)
    overlap   = our_genes & scf_genes
    print(f"  Gene overlap: {len(overlap)}/{len(our_genes)} "
          f"({100*len(overlap)/len(our_genes):.1f}%)")

    # ── Prepare expression ────────────────────────────────────────────
    expr_arr = prepare_expression(X_unique, gene_list)

    # ── Load model ────────────────────────────────────────────────────
    model, config = load_scfoundation()

    # ── Extract embeddings ────────────────────────────────────────────
    print(f"\n  Extracting embeddings for {len(unique_ids)} cell lines ...")
    emb = extract_embeddings(model, config, expr_arr)
    print(f"  Embeddings shape: {emb.shape}  (expected: {len(unique_ids)} × 768)")

    # ── Save ──────────────────────────────────────────────────────────
    out_emb = PROC_DIR / "scfoundation_cell_emb.npy"
    out_ids = PROC_DIR / "scfoundation_cell_ids.npy"
    np.save(out_emb, emb)
    np.save(out_ids, unique_ids)
    print(f"\n  Saved embeddings → {out_emb}")
    print(f"  Saved cell IDs   → {out_ids}")
    print("=" * 65)


if __name__ == "__main__":
    main()
