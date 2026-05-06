# -*- coding: utf-8 -*-
"""
GDRNetV11 Organoid Inference
=============================
Run trained V11 ensemble on organoid drug response data.

Handles:
  - No raw gene expression for organoids → use zeros, rely on scF embeddings
  - Unknown cell IDs → use padding index 0
  - Drug desc dimension mismatch (166 vs 188) → zero-pad
  - Drug name → GDSC index mapping
"""

import sys
import io
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

ROOT = Path("/export/home/kongyan/project/Organoid")
PROC = ROOT / "data/processed"
EXT = ROOT / "data/external"
MODELS = ROOT / "models"
TABLES = ROOT / "results/tables"
TABLES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass


def compute_metrics(y_true, y_pred, name=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else float("nan")
    thr = np.percentile(y_true, 30)
    try:
        auroc = float(roc_auc_score((y_true <= thr).astype(int), -y_pred))
    except Exception:
        auroc = float("nan")
    m = dict(Pearson=round(pearson, 4), R2=round(r2, 4),
             RMSE=round(rmse, 4), AUROC=round(auroc, 4))
    if name:
        print(f"  {name:35s}  Pearson={pearson:.4f}  R2={r2:.4f}  "
              f"RMSE={rmse:.4f}  AUROC={auroc:.4f}", flush=True)
    return m


def load_organoid_data():
    """Load organoid data and prepare inputs for V11 model."""
    print("=" * 65, flush=True)
    print("  Loading organoid data", flush=True)
    print("=" * 65, flush=True)

    # ── Organoid features ──
    meta = pd.read_csv(PROC / "organoid_pair_meta.csv")
    cell_emb_all = np.load(PROC / "organoid_cell_emb.npy")         # (64, 3072)
    cell_ids_all = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
    drug_feat = np.load(PROC / "organoid_drug_features.npy")        # (544, 2214)
    response = np.load(PROC / "organoid_response.npy")              # (544,)
    n_samples = len(response)
    print(f"  Organoid pairs: {n_samples}", flush=True)
    print(f"  Organoids: {meta['organoid_id'].nunique()}, Drugs: {meta['drug_name'].nunique()}", flush=True)
    print(f"  IC50 range: [{response.min():.2f}, {response.max():.2f}]", flush=True)

    # ── Build cell_id → scF embedding mapping ──
    cell_emb_map = {cid: cell_emb_all[i] for i, cid in enumerate(cell_ids_all)}

    # ── Map organoid_id → scF embedding for each pair ──
    scf_emb = np.array([cell_emb_map[oid] for oid in meta["organoid_id"]],
                       dtype=np.float32)
    print(f"  scF embeddings: {scf_emb.shape}", flush=True)

    # ── Gene features: not available for organoids, use zeros ──
    n_genes = 2000
    gene_zeros = np.zeros((n_samples, n_genes), dtype=np.float32)
    print(f"  Gene features: zeros ({n_genes})", flush=True)

    # ── Drug FP + desc ──
    fp = drug_feat[:, :2048].astype(np.float32)
    desc_raw = drug_feat[:, 2048:].astype(np.float32)
    print(f"  Drug FP: {fp.shape[1]}, Desc raw: {desc_raw.shape[1]}", flush=True)

    # Pad desc to match GDSC training (188 dims)
    n_desc_target = 188
    if desc_raw.shape[1] < n_desc_target:
        pad = np.zeros((n_samples, n_desc_target - desc_raw.shape[1]), dtype=np.float32)
        desc = np.concatenate([desc_raw, pad], axis=1)
    else:
        desc = desc_raw[:, :n_desc_target]
    print(f"  Drug desc (padded): {desc.shape[1]}", flush=True)

    # ── Build GDSC drug name → index mapping ──
    # Must match training exactly
    gdsc_meta = pd.read_parquet(PROC / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")

    # Filter to valid drugs (same as training)
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    fps_valid = set()
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol:
                fps_valid.add(row["drug_name"])
        except Exception:
            pass

    valid_mask = (gdsc_meta["drug_name"].isin(fps_valid)) & \
                 (gdsc_meta["ModelID"].isin(
                     {cid: True for cid in np.load(PROC / "scfoundation_cell_ids.npy",
                                                    allow_pickle=True)}))
    meta_gdsc = gdsc_meta[valid_mask].reset_index(drop=True)

    drug_list = sorted(meta_gdsc["drug_name"].unique())
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}
    n_drugs = len(drug_list)

    # Map organoid drugs → GDSC indices
    drug_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]],
                        dtype=np.int64)
    n_mapped = (drug_idx > 0).sum()
    print(f"  Drug mapping: {n_mapped}/{n_samples} mapped to GDSC indices", flush=True)

    # Cell IDs: organoids not in GDSC → use padding index 0
    cell_idx = np.zeros(n_samples, dtype=np.int64)
    print(f"  Cell index: all 0 (organoids not in GDSC)", flush=True)

    n_cells = meta_gdsc["ModelID"].nunique()
    scf_dim = 3072

    print(f"  Data ready: {n_samples} samples", flush=True)
    return (gene_zeros, scf_emb, fp, desc, cell_idx, drug_idx, response,
            meta, n_genes, scf_dim, 2048, n_desc_target, n_cells, n_drugs)


def infer_ensemble(device="cuda:0"):
    """Run V11 ensemble inference on organoid data."""
    (x_gene, scf_emb, x_fp, x_desc, cell_idx, drug_idx, y_true,
     meta, n_genes, scf_dim, fp_bits, n_desc, n_cells, n_drugs) = load_organoid_data()

    from models.gdr_v11 import GDRNetV11

    # ── Load each model ──
    seeds = [42, 123, 456]
    all_preds = []

    for seed in seeds:
        ckpt_path = MODELS / f"gdr_v11_s{seed}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping", flush=True)
            continue

        print(f"\n  Loading model seed={seed} ...", flush=True)
        model = GDRNetV11(
            n_genes=n_genes, scf_dim=scf_dim, fp_bits=fp_bits, n_desc=n_desc,
            n_cells=n_cells, n_drugs=n_drugs,
            d_hidden=256, id_emb_dim=64,
            n_cross=3, cross_rank=64, n_deep=3, dropout=0.15,
        )

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Handle DataParallel "module." prefix
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned)
        model = model.to(device).eval()
        print(f"  Loaded {ckpt_path.name}", flush=True)

        # ── Inference ──
        with torch.no_grad():
            batch = 512
            preds = []
            for i in range(0, len(y_true), batch):
                sl = slice(i, min(i + batch, len(y_true)))
                out = model(
                    torch.FloatTensor(x_gene[sl]).to(device),
                    torch.FloatTensor(scf_emb[sl]).to(device),
                    torch.FloatTensor(x_fp[sl]).to(device),
                    torch.FloatTensor(x_desc[sl]).to(device),
                    torch.LongTensor(cell_idx[sl]).to(device),
                    torch.LongTensor(drug_idx[sl]).to(device),
                )
                preds.extend(out.cpu().numpy())
            preds = np.array(preds, dtype=np.float32)
            all_preds.append(preds)
            compute_metrics(y_true, preds, f"V11-s{seed}")

    if not all_preds:
        print("  ERROR: No models loaded!", flush=True)
        return

    # ── Ensemble average ──
    ens_preds = np.mean(all_preds, axis=0)
    print(f"\n  {'='*60}", flush=True)
    print(f"  Ensemble ({len(all_preds)} models)", flush=True)
    print(f"  {'='*60}", flush=True)
    ens_metrics = compute_metrics(y_true, ens_preds, "V11-Ensemble")

    # ── Per-organoid metrics ──
    print(f"\n  Per-organoid performance:", flush=True)
    per_org = []
    for oid in sorted(meta["organoid_id"].unique()):
        mask = meta["organoid_id"] == oid
        if mask.sum() < 3:
            continue
        yt = y_true[mask]
        yp = ens_preds[mask]
        p = float(np.corrcoef(yt, yp)[0, 1]) if mask.sum() > 2 else float("nan")
        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        per_org.append(dict(organoid=oid, n_drugs=int(mask.sum()),
                            Pearson=round(p, 4), R2=round(r2, 4), RMSE=round(rmse, 4)))
        print(f"    {oid:8s}  n={mask.sum():2d}  P={p:.4f}  R2={r2:.4f}  RMSE={rmse:.4f}", flush=True)

    per_org_df = pd.DataFrame(per_org)
    per_org_df.to_csv(TABLES / "organoid_per_organoid.csv", index=False)

    # ── Per-drug metrics ──
    print(f"\n  Per-drug performance:", flush=True)
    per_drug = []
    for dname in sorted(meta["drug_name"].unique()):
        mask = meta["drug_name"] == dname
        if mask.sum() < 3:
            continue
        yt = y_true[mask]
        yp = ens_preds[mask]
        p = float(np.corrcoef(yt, yp)[0, 1]) if mask.sum() > 2 else float("nan")
        r2 = float(r2_score(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        per_drug.append(dict(drug=dname, n_organoids=int(mask.sum()),
                             Pearson=round(p, 4), R2=round(r2, 4), RMSE=round(rmse, 4)))

    per_drug_df = pd.DataFrame(per_drug)
    per_drug_df.to_csv(TABLES / "organoid_per_drug.csv", index=False)

    # ── Save predictions ──
    result_df = meta.copy()
    result_df["predicted_ic50"] = ens_preds
    result_df["error"] = ens_preds - y_true
    result_df.to_csv(TABLES / "organoid_predictions.csv", index=False)
    np.save(TABLES / "organoid_v11_ens_preds.npy", ens_preds)

    # ── Summary ──
    avg_per_org = np.mean([r["Pearson"] for r in per_org if np.isfinite(r["Pearson"])])
    avg_per_drug = np.mean([r["Pearson"] for r in per_drug if np.isfinite(r["Pearson"])])

    print(f"\n{'='*65}", flush=True)
    print(f"  Organoid Test Summary", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  Overall:  Pearson={ens_metrics['Pearson']:.4f}  "
          f"R2={ens_metrics['R2']:.4f}  RMSE={ens_metrics['RMSE']:.4f}", flush=True)
    print(f"  Avg per-organoid Pearson: {avg_per_org:.4f}", flush=True)
    print(f"  Avg per-drug Pearson:     {avg_per_drug:.4f}", flush=True)
    print(f"  Saved predictions -> {TABLES}/organoid_predictions.csv", flush=True)
    print(f"{'='*65}", flush=True)

    return ens_metrics


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    infer_ensemble(device)
