# -*- coding: utf-8 -*-
"""
GDRNetV11 Training Script — CrossNet-Deep Ensemble
====================================================
Fundamental departure from V3-V10:
  - DCN v2 explicit feature crossing instead of attention
  - Cell/Drug ID embeddings for matrix factorization effect
  - Simple MLP encoders instead of Transformer/GNN
  - 3-model ensemble for variance reduction
  - LightGBM with gene+scF features as strong baseline

Multi-GPU on A800 (2-3 GPUs via DataParallel).

Usage:
  cd /export/home/kongyan/project/Organoid
  python src/train.py
  python src/train.py --gpus 0,2,4 --epochs 300 --n_models 5
"""

import argparse
import io
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from scipy.optimize import minimize_scalar

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
parser.add_argument("--n_models", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--d_hidden", type=int, default=256)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--no_lgbm", action="store_true")
args = parser.parse_args()


# ── Metrics ────────────────────────────────────────────────────────────────────

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
              f"RMSE={rmse:.4f}  AUROC={auroc:.4f}")
    return m


# ── RDKit Descriptor Utilities ────────────────────────────────────────────────

def compute_rdkit_descriptors(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    desc_fns = [(name, fn) for name, fn in Descriptors._descList]
    n_desc = len(desc_fns)
    print(f"  Computing {n_desc} RDKit descriptors for {len(smiles_list)} molecules...", flush=True)

    results, valid = [], []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                results.append(None); valid.append(False); continue
            row = []
            for _, fn in desc_fns:
                try:
                    v = float(fn(mol))
                    row.append(v if np.isfinite(v) else np.nan)
                except Exception:
                    row.append(np.nan)
            results.append(row); valid.append(True)
        except Exception:
            results.append(None); valid.append(False)

    arr = np.full((len(smiles_list), n_desc), np.nan, dtype=np.float32)
    for i, (r, v) in enumerate(zip(results, valid)):
        if v:
            arr[i] = r
    return arr, [n for n, _ in desc_fns], np.array(valid)


def normalize_descriptors(arr_tr, arr_val):
    col_median = np.nanmedian(arr_tr, axis=0)
    col_std = np.nanstd(arr_tr, axis=0)

    def _fill(arr):
        out = arr.copy()
        for j in range(arr.shape[1]):
            mask = ~np.isfinite(out[:, j])
            if mask.any():
                out[mask, j] = col_median[j]
        return out

    arr_tr, arr_val = _fill(arr_tr), _fill(arr_val)
    keep = col_std > 1e-6
    arr_tr, arr_val = arr_tr[:, keep], arr_val[:, keep]
    col_std, col_median = col_std[keep], col_median[keep]
    arr_tr = np.clip((arr_tr - col_median) / col_std, -5, 5).astype(np.float32)
    arr_val = np.clip((arr_val - col_median) / col_std, -5, 5).astype(np.float32)
    print(f"  Descriptor dims after filtering: {keep.sum()} / {len(keep)}", flush=True)
    return arr_tr, arr_val, int(keep.sum())


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    print("=" * 65)
    print("  Loading dataset for GDRNetV11 (CrossNet-Deep Ensemble)")
    print("=" * 65)
    t0 = time.time()

    # ── scFoundation embeddings ──
    emb_path = PROC_DIR / "scfoundation_cell_emb.npy"
    ids_path = PROC_DIR / "scfoundation_cell_ids.npy"
    assert emb_path.exists(), "Run extract_scfoundation_emb.py first."
    scf_emb = np.load(emb_path)
    scf_ids = np.load(ids_path, allow_pickle=True)
    id_to_emb = {cid: scf_emb[i] for i, cid in enumerate(scf_ids)}
    print(f"  [1/6] scFoundation embeddings: {scf_emb.shape}  ({time.time()-t0:.1f}s)")

    # ── Core data ──
    X_cell_full = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    y_full = pd.read_parquet(PROC_DIR / "gdsc_response_lnic50.parquet").iloc[:, 0]
    meta_full = pd.read_parquet(PROC_DIR / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT_DIR / "gdsc_drug_smiles.csv")
    print(f"  [2/6] Core data: {len(y_full):,} samples  ({time.time()-t0:.1f}s)")

    # ── Morgan FP ──
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
    print(f"  [3/6] Morgan FP: {len(fps)} drugs  ({time.time()-t0:.1f}s)")

    # ── Filter to valid samples ──
    valid_mask = (meta_full["drug_name"].isin(fps)) & \
                 (meta_full["ModelID"].isin(id_to_emb))
    orig_idx = meta_full[valid_mask].index
    meta = meta_full[valid_mask].reset_index(drop=True)
    y = y_full[valid_mask].values.astype(np.float32)

    X_gene = X_cell_full.loc[orig_idx].values.astype(np.float32)
    X_scf = np.array([id_to_emb[c] for c in meta["ModelID"]], dtype=np.float32)
    X_fp = np.array([fps[d] for d in meta["drug_name"]], dtype=np.float32)

    print(f"  [4/6] Filtered: {len(y):,} samples  "
          f"({meta['ModelID'].nunique()} cells, "
          f"{meta['drug_name'].nunique()} drugs)  ({time.time()-t0:.1f}s)")

    # ── Cell-line based train/val split ──
    cell_ids_str = meta["ModelID"].values
    unique_cells = np.unique(cell_ids_str)
    tr_cells, val_cells = train_test_split(
        unique_cells, test_size=0.2, random_state=42)
    tr_mask = np.isin(cell_ids_str, tr_cells)
    val_mask = np.isin(cell_ids_str, val_cells)
    print(f"  [5/6] Split: train={tr_mask.sum():,}  val={val_mask.sum():,}  "
          f"({time.time()-t0:.1f}s)")

    # ── RDKit descriptors ──
    unique_drugs = smiles_df[smiles_df["drug_name"].isin(fps)] \
        [["drug_name", "smiles"]].drop_duplicates("drug_name").reset_index(drop=True)
    desc_arr, _, _ = compute_rdkit_descriptors(unique_drugs["smiles"].tolist())
    drug_desc_map = {row["drug_name"]: desc_arr[i]
                     for i, row in unique_drugs.iterrows()}
    X_desc_all = np.array([drug_desc_map.get(d, np.zeros(desc_arr.shape[1]))
                           for d in meta["drug_name"]], dtype=np.float32)
    X_desc_tr, X_desc_val, n_desc = normalize_descriptors(
        X_desc_all[tr_mask], X_desc_all[val_mask])

    # ── Build cell_id and drug_id integer mappings ──
    cell_list = sorted(meta["ModelID"].unique())
    drug_list = sorted(meta["drug_name"].unique())
    cell_to_idx = {c: i + 1 for i, c in enumerate(cell_list)}  # 0 = padding
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}
    cell_idx_all = np.array([cell_to_idx[c] for c in meta["ModelID"]], dtype=np.int64)
    drug_idx_all = np.array([drug_to_idx[d] for d in meta["drug_name"]], dtype=np.int64)

    n_cells = len(cell_list)
    n_drugs = len(drug_list)
    print(f"  [6/6] Cell IDs: {n_cells}  Drug IDs: {n_drugs}  "
          f"n_desc: {n_desc}  ({time.time()-t0:.1f}s)")

    # Assemble
    data = dict(
        # Training
        x_gene_tr=X_gene[tr_mask], x_scf_tr=X_scf[tr_mask],
        x_fp_tr=X_fp[tr_mask], x_desc_tr=X_desc_tr,
        cell_idx_tr=cell_idx_all[tr_mask], drug_idx_tr=drug_idx_all[tr_mask],
        y_tr=y[tr_mask],
        # Validation
        x_gene_val=X_gene[val_mask], x_scf_val=X_scf[val_mask],
        x_fp_val=X_fp[val_mask], x_desc_val=X_desc_val,
        cell_idx_val=cell_idx_all[val_mask], drug_idx_val=drug_idx_all[val_mask],
        y_val=y[val_mask],
        # Meta
        n_cells=n_cells, n_drugs=n_drugs, n_desc=n_desc,
        scf_dim=X_scf.shape[1],
    )

    print(f"  Data loading complete in {time.time()-t0:.1f}s")
    return data


# ── LightGBM ──────────────────────────────────────────────────────────────────

def run_lightgbm(cell_tr, drug_tr, y_tr, cell_val, drug_val, y_val, label="LightGBM"):
    import lightgbm as lgb
    print(f"  [LightGBM(scF+FP+desc)] Training ...", flush=True)
    X_tr = np.concatenate([cell_tr, drug_tr], axis=1)
    X_val = np.concatenate([cell_val, drug_val], axis=1)
    lgbm = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=1000, learning_rate=0.05,
        num_leaves=63, max_depth=10,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=8, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
             callbacks=[lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(500)])
    preds = lgbm.predict(X_val).astype(np.float32)
    print(f"  [LightGBM] Done.", flush=True)
    return preds


# ── Blend ─────────────────────────────────────────────────────────────────────

def blend(y_val, p1, p2):
    res = minimize_scalar(
        lambda a: -float(np.corrcoef(y_val, a * p1 + (1 - a) * p2)[0, 1]),
        bounds=(0, 1), method="bounded")
    return res.x * p1 + (1 - res.x) * p2, res.x


# ── Seed helper ───────────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("  GDRNetV11 Training  (CrossNet-Deep Ensemble)")
    print("  DCN v2 + ID Embeddings + Simple MLP + 3-Model Ensemble")
    print("=" * 65)

    # GPU setup
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    primary_gpu = f"cuda:{gpu_ids[0]}"
    n_gpus = len(gpu_ids)
    print(f"  GPUs: {gpu_ids} ({n_gpus} devices)")

    # Load data
    data = load_dataset()
    results = {}

    # ── LightGBM baselines ──
    if not args.no_lgbm:
        # LightGBM with scF + FP + desc (same as V10)
        lgbm_preds = run_lightgbm(
            data["x_scf_tr"],
            np.concatenate([data["x_fp_tr"], data["x_desc_tr"]], axis=1),
            data["y_tr"],
            data["x_scf_val"],
            np.concatenate([data["x_fp_val"], data["x_desc_val"]], axis=1),
            data["y_val"],
            label="LightGBM(scF+FP+desc)")
        np.save(TABLES / "lgbm_v11_val_preds.npy", lgbm_preds)
        results["LightGBM"] = compute_metrics(data["y_val"], lgbm_preds, "LightGBM")

        # LightGBM with gene + scF + FP + desc (richer)
        lgbm2_preds = run_lightgbm(
            np.concatenate([data["x_gene_tr"], data["x_scf_tr"]], axis=1),
            np.concatenate([data["x_fp_tr"], data["x_desc_tr"]], axis=1),
            data["y_tr"],
            np.concatenate([data["x_gene_val"], data["x_scf_val"]], axis=1),
            np.concatenate([data["x_fp_val"], data["x_desc_val"]], axis=1),
            data["y_val"],
            label="LightGBM(gene+scF+FP+desc)")
        np.save(TABLES / "lgbm_v11_full_val_preds.npy", lgbm2_preds)
        results["LightGBM-full"] = compute_metrics(
            data["y_val"], lgbm2_preds, "LightGBM-full")

    # ── Load cached baselines ──
    for name, cache_name in [("LightGBM", "lgbm_v4_val_preds.npy"),
                              ("GDRNetV3", "gdr_v3_val_preds.npy"),
                              ("GDRNetV10", "gdr_v10_val_preds.npy"),
                              ("GDRNetV8", "gdr_v8_val_preds.npy")]:
        cache = TABLES / cache_name
        if name not in results and cache.exists():
            p = np.load(cache)
            results[name] = compute_metrics(data["y_val"], p, f"{name} (cached)")

    # ── GDRNetV11 Ensemble ──
    from models.gdr_v11 import GDRNetV11, V11Dataset, train_v11

    seeds = [42, 123, 456, 789, 2024][:args.n_models]
    ensemble_preds = []
    ensemble_metrics = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*65}")
        print(f"  Model {i+1}/{len(seeds)}  seed={seed}")
        print(f"{'='*65}")

        set_seed(seed)

        model = GDRNetV11(
            n_genes=data["x_gene_tr"].shape[1],
            scf_dim=data["scf_dim"],
            fp_bits=data["x_fp_tr"].shape[1],
            n_desc=data["n_desc"],
            n_cells=data["n_cells"],
            n_drugs=data["n_drugs"],
            d_hidden=args.d_hidden,
            id_emb_dim=64,
            n_cross=3,
            cross_rank=64,
            n_deep=3,
            dropout=args.dropout,
        )

        if n_gpus > 1:
            model = nn.DataParallel(model, device_ids=gpu_ids)

        tr_ds = V11Dataset(
            data["x_gene_tr"], data["x_scf_tr"],
            data["x_fp_tr"], data["x_desc_tr"],
            data["cell_idx_tr"], data["drug_idx_tr"],
            data["y_tr"])
        val_ds = V11Dataset(
            data["x_gene_val"], data["x_scf_val"],
            data["x_fp_val"], data["x_desc_val"],
            data["cell_idx_val"], data["drug_idx_val"],
            data["y_val"])

        tag = f"gdr_v11_s{seed}"
        _, metrics, hist, val_preds = train_v11(
            model, tr_ds, val_ds,
            model_name=tag,
            n_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            patience=args.patience,
            weight_decay=args.weight_decay,
            warmup_frac=0.05,
            device=primary_gpu,
        )
        hist.to_csv(TABLES / f"gdr_v11_s{seed}_history.csv", index=False)
        np.save(TABLES / f"gdr_v11_s{seed}_val_preds.npy", val_preds)

        ensemble_preds.append(val_preds)
        ensemble_metrics.append(metrics)
        results[f"V11-s{seed}"] = metrics

        elapsed = time.time() - t0
        print(f"  Model {i+1} done. Elapsed: {elapsed/60:.1f} min")

    # ── Ensemble: average predictions ──
    ens_preds = np.mean(ensemble_preds, axis=0)
    np.save(TABLES / "gdr_v11_ensemble_val_preds.npy", ens_preds)
    results["V11-Ensemble"] = compute_metrics(
        data["y_val"], ens_preds, "V11-Ensemble")

    # ── Blends ──
    for label, pred_arr in [("LightGBM", "lgbm_v11_val_preds.npy"),
                            ("LightGBM-full", "lgbm_v11_full_val_preds.npy"),
                            ("V3", "gdr_v3_val_preds.npy"),
                            ("V10", "gdr_v10_val_preds.npy")]:
        cache = TABLES / pred_arr
        if cache.exists():
            p = np.load(cache)
            mixed, alpha = blend(data["y_val"], ens_preds, p)
            key = f"V11-Ens+{label}(a={alpha:.2f})"
            results[key] = compute_metrics(data["y_val"], mixed, key)

    # ── Summary ──
    cmp = pd.DataFrame(results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / "model_comparison_v11.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print("  Final Comparison:")
    print(cmp.to_string())
    print(f"\n  Saved -> {TABLES}/model_comparison_v11.csv")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"{'='*65}")

    # V11 vs baselines
    v11_p = results["V11-Ensemble"]["Pearson"]
    for bl in ["LightGBM", "LightGBM-full", "GDRNetV3", "GDRNetV10"]:
        if bl in results:
            bp = results[bl]["Pearson"]
            diff = v11_p - bp
            symbol = "+" if diff > 0 else ""
            status = "PASS" if diff > 0 else "NEEDS WORK"
            print(f"  V11-Ensemble vs {bl}: {v11_p:.4f} vs {bp:.4f}  "
                  f"({symbol}{diff:.4f})  {status}")


if __name__ == "__main__":
    main()
