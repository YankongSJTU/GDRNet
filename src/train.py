# -*- coding: utf-8 -*-
"""
GDRNet Training on GDSC (964 Cell Lines)
=========================================
Usage:
  python src/train.py
  python src/train.py --gpus 0,2,4 --epochs 300
"""

import argparse
import io
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path("/export/home/kongyan/project/Organoid")
PROC_DIR = ROOT / "data/processed"
EXT_DIR = ROOT / "data/external"
MODELS_DIR = ROOT / "models"
TABLES = ROOT / "results/tables"
for d in [MODELS_DIR, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--gpus", type=str, default="0,2,4")
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--d_hidden", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0.01)
args = parser.parse_args()


def compute_metrics(y_true, y_pred, name=""):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
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


def load_dataset():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from models.gdrnet import GDRNet, GDRNetDataset, train_model

    print("=" * 65, flush=True)
    print("  Loading dataset for GDRNet", flush=True)
    print("=" * 65)
    t0 = time.time()

    # scFoundation embeddings
    emb_path = PROC_DIR / "scfoundation_cell_emb.npy"
    ids_path = PROC_DIR / "scfoundation_cell_ids.npy"
    scf_emb = np.load(emb_path)
    scf_ids = np.load(ids_path, allow_pickle=True)
    id_to_emb = {cid: scf_emb[i] for i, cid in enumerate(scf_ids)}
    print(f"  [1/5] scFoundation embeddings: {scf_emb.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # Core data
    X_cell_full = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    y_full = pd.read_parquet(PROC_DIR / "gdsc_response_lnic50.parquet").iloc[:, 0]
    meta_full = pd.read_parquet(PROC_DIR / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT_DIR / "gdsc_drug_smiles.csv")
    print(f"  [2/5] Core data: {len(y_full):,} samples  ({time.time()-t0:.1f}s)", flush=True)

    # Morgan FP
    fps = {}
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            try:
                gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
                fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            except Exception:
                fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=2048), dtype=np.float32)
            fps[row["drug_name"]] = fp
        except Exception:
            pass
    print(f"  [3/5] Morgan FP: {len(fps)} drugs  ({time.time()-t0:.1f}s)", flush=True)

    # Filter to valid samples
    valid_mask = (meta_full["drug_name"].isin(fps)) & \
                 (meta_full["ModelID"].isin(id_to_emb))
    orig_idx = meta_full[valid_mask].index
    meta = meta_full[valid_mask].reset_index(drop=True)
    y = y_full[valid_mask].values.astype(np.float32)

    X_gene = X_cell_full.loc[orig_idx].values.astype(np.float32)
    X_scf = np.array([id_to_emb[c] for c in meta["ModelID"]], dtype=np.float32)
    X_fp = np.array([fps[d] for d in meta["drug_name"]], dtype=np.float32)
    X_desc = np.zeros((len(y), 1), dtype=np.float32)

    print(f"  [4/5] Filtered: {len(y):,} samples  "
          f"({meta['ModelID'].nunique()} cells, "
          f"{meta['drug_name'].nunique()} drugs)  ({time.time()-t0:.1f}s)", flush=True)

    # Cell-line based split
    cell_ids_str = meta["ModelID"].values
    unique_cells = np.unique(cell_ids_str)
    tr_cells, val_cells = train_test_split(
        unique_cells, test_size=0.2, random_state=42)
    tr_mask = np.isin(cell_ids_str, tr_cells)
    val_mask = np.isin(cell_ids_str, val_cells)
    print(f"  [5/5] Split: train={tr_mask.sum():,}  val={val_mask.sum():,}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Build cell/drug index
    cell_list = sorted(meta["ModelID"].unique())
    drug_list = sorted(meta["drug_name"].unique())
    cell_to_idx = {c: i + 1 for i, c in enumerate(cell_list)}
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}
    cell_idx_all = np.array([cell_to_idx[c] for c in meta["ModelID"]], dtype=np.int64)
    drug_idx_all = np.array([drug_to_idx[d] for d in meta["drug_name"]], dtype=np.int64)

    n_cells = len(cell_list)
    n_drugs = len(drug_list)
    print(f"  Cell IDs: {n_cells}  Drug IDs: {n_drugs}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    data = dict(
        x_gene_tr=X_gene[tr_mask], x_scf_tr=X_scf[tr_mask],
        x_fp_tr=X_fp[tr_mask], x_desc_tr=X_desc[tr_mask],
        cell_idx_tr=cell_idx_all[tr_mask], drug_idx_tr=drug_idx_all[tr_mask],
        y_tr=y[tr_mask],
        x_gene_val=X_gene[val_mask], x_scf_val=X_scf[val_mask],
        x_fp_val=X_fp[val_mask], x_desc_val=X_desc[val_mask],
        cell_idx_val=cell_idx_all[val_mask], drug_idx_val=drug_idx_all[val_mask],
        y_val=y[val_mask],
        n_cells=n_cells, n_drugs=n_drugs,
    )

    print(f"  Data loading complete in {time.time()-t0:.1f}s", flush=True)
    return data


def run_lightgbm(cell_tr, drug_tr, y_tr, cell_val, drug_val, y_val, label="LightGBM"):
    import lightgbm as lgb
    X_tr = np.concatenate([cell_tr, drug_tr], axis=1)
    X_val = np.concatenate([cell_val, drug_val], axis=1)
    lgbm = lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05,
        num_leaves=63, max_depth=10,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=10, reg_alpha=0.5, reg_lambda=0.5,
        n_jobs=8, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = lgbm.predict(X_val).astype(np.float32)
    m = compute_metrics(y_val, preds, label)
    return preds, m


def main():
    t0 = time.time()
    data = load_dataset()
    results = {}

    # LightGBM baseline
    print("\n  [LightGBM] Training baseline ...", flush=True)
    lgbm_preds, results["LightGBM"] = run_lightgbm(
        data["x_scf_tr"], data["x_fp_tr"], data["y_tr"],
        data["x_scf_val"], data["x_fp_val"], data["y_val"])
    np.save(TABLES / "lgbm_val_preds.npy", lgbm_preds)

    # GDRNet training (3 seeds)
    from models.gdrnet import GDRNet, GDRNetDataset, train_model

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    seeds = [42, 123, 456]
    all_preds = {}

    for seed in seeds:
        print(f"\n  {'='*60}", flush=True)
        print(f"  Training GDRNet seed={seed}", flush=True)
        print(f"  {'='*60}", flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GDRNet(
            n_genes=2000, scf_dim=3072, fp_bits=2048,
            n_cells=data["n_cells"], n_drugs=data["n_drugs"],
            d_hidden=args.d_hidden, id_emb_dim=64,
            n_cross=3, cross_rank=64, n_deep=3, dropout=args.dropout,
        )

        primary_gpu = f"cuda:{gpu_ids[0]}"
        if len(gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        tr_ds = GDRNetDataset(
            data["x_gene_tr"], data["x_scf_tr"], data["x_fp_tr"], data["x_desc_tr"],
            data["cell_idx_tr"], data["drug_idx_tr"], data["y_tr"])
        val_ds = GDRNetDataset(
            data["x_gene_val"], data["x_scf_val"], data["x_fp_val"], data["x_desc_val"],
            data["cell_idx_val"], data["drug_idx_val"], data["y_val"])

        model, metrics, history, val_preds = train_model(
            model, tr_ds, val_ds,
            model_name=f"gdrnet_s{seed}",
            n_epochs=args.epochs, batch_size=args.batch,
            lr=args.lr, patience=args.patience,
            weight_decay=args.weight_decay, device=primary_gpu,
        )

        name = f"GDRNet-s{seed}"
        results[name] = metrics
        all_preds[seed] = val_preds
        history.to_csv(TABLES / f"gdrnet_s{seed}_history.csv", index=False)
        np.save(TABLES / f"gdrnet_s{seed}_val_preds.npy", val_preds)

    # Ensemble
    ens_preds = np.mean([all_preds[s] for s in seeds], axis=0)
    np.save(TABLES / "gdrnet_ensemble_val_preds.npy", ens_preds)
    results["GDRNet-Ensemble"] = compute_metrics(
        data["y_val"], ens_preds, "GDRNet-Ensemble")

    # Summary
    cmp = pd.DataFrame(results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / "model_comparison.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*65}", flush=True)
    print(f"  GDRNet Training Summary", flush=True)
    print(f"{'='*65}", flush=True)
    print(cmp.to_string(), flush=True)
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"  Saved -> {TABLES}/model_comparison.csv", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
