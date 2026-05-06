# -*- coding: utf-8 -*-
"""
GDRNet Organoid Fine-Tuning (LOOCV)
========================================
Leave-One-Organoid-Out Cross-Validation fine-tuning on organoid data.

Strategy:
  - Freeze feature encoders (gene, scF, FP, desc encoders + embeddings)
  - Fine-tune only interaction layers (cross + deep + output head)
  - Very small LR, strong regularization
  - 16-fold LOOCV: train on 15 organoids, test on 1
  - Compare with LightGBM baseline on same folds
"""

import sys
import io
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from pathlib import Path

ROOT = Path("./Organoid")
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


# ── Dataset ────────────────────────────────────────────────────────────────────

class OrgDataset(Dataset):
    def __init__(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.scf = torch.FloatTensor(scf)
        self.x_fp = torch.FloatTensor(x_fp)
        self.x_desc = torch.FloatTensor(x_desc)
        self.cell_idx = torch.LongTensor(cell_idx)
        self.drug_idx = torch.LongTensor(drug_idx)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.x_gene[i], self.scf[i], self.x_fp[i], self.x_desc[i],
                self.cell_idx[i], self.drug_idx[i], self.y[i])


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_organoid_data():
    print("  Loading organoid data ...", flush=True)
    meta = pd.read_csv(PROC / "organoid_pair_meta.csv")
    cell_emb_all = np.load(PROC / "organoid_cell_emb.npy")
    cell_ids_all = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
    drug_feat = np.load(PROC / "organoid_drug_features.npy")
    response = np.load(PROC / "organoid_response.npy")

    cell_emb_map = {cid: cell_emb_all[i] for i, cid in enumerate(cell_ids_all)}
    scf_emb = np.array([cell_emb_map[oid] for oid in meta["organoid_id"]],
                       dtype=np.float32)
    gene_zeros = np.zeros((len(response), 2000), dtype=np.float32)
    fp = drug_feat[:, :2048].astype(np.float32)
    desc_raw = drug_feat[:, 2048:].astype(np.float32)
    n_desc_target = 188
    if desc_raw.shape[1] < n_desc_target:
        pad = np.zeros((len(response), n_desc_target - desc_raw.shape[1]), dtype=np.float32)
        desc = np.concatenate([desc_raw, pad], axis=1)
    else:
        desc = desc_raw[:, :n_desc_target]

    # Drug name → GDSC index
    from rdkit import Chem
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")
    fps_valid = set()
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol:
                fps_valid.add(row["drug_name"])
        except Exception:
            pass
    gdsc_meta = pd.read_parquet(PROC / "gdsc_metadata.parquet")
    scf_ids = set(np.load(PROC / "scfoundation_cell_ids.npy", allow_pickle=True))
    valid_mask = (gdsc_meta["drug_name"].isin(fps_valid)) & \
                 (gdsc_meta["ModelID"].isin(scf_ids))
    meta_gdsc = gdsc_meta[valid_mask].reset_index(drop=True)
    drug_list = sorted(meta_gdsc["drug_name"].unique())
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}
    n_drugs = len(drug_list)
    n_cells = meta_gdsc["ModelID"].nunique()

    drug_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]],
                        dtype=np.int64)
    cell_idx = np.zeros(len(response), dtype=np.int64)

    print(f"  {len(response)} samples, {meta['organoid_id'].nunique()} organoids, "
          f"{meta['drug_name'].nunique()} drugs", flush=True)

    return (gene_zeros, scf_emb, fp, desc, cell_idx, drug_idx, response,
            meta, n_cells, n_drugs)


# ── Build model with frozen encoders ──────────────────────────────────────────

def build_model(n_cells, n_drugs, seed, device):
    from models.gdr import GDRNet

    model = GDRNet(
        n_genes=2000, scf_dim=3072, fp_bits=2048, n_desc=188,
        n_cells=n_cells, n_drugs=n_drugs,
        d_hidden=256, id_emb_dim=64,
        n_cross=3, cross_rank=64, n_deep=3, dropout=0.15,
    )

    ckpt = MODELS / f"gdr_s{seed}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)

    # Freeze encoders + embeddings (keep learned GDSC representations)
    for name, param in model.named_parameters():
        if any(k in name for k in ["gene_enc", "scf_enc", "fp_enc",
                                     "desc_enc", "cell_emb", "drug_emb",
                                     "input_proj"]):
            param.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model s{seed}: {n_total/1e6:.2f}M total, "
          f"{n_trainable/1e6:.2f}M trainable (cross+deep+head)", flush=True)

    return model.to(device)


# ── LOOCV Fine-tune one fold ──────────────────────────────────────────────────

def finetune_fold(model, tr_ds, val_ds, device,
                  n_epochs=100, lr=5e-4, patience=20, batch_size=64):
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    best_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        # Keep frozen layers in eval mode for BatchNorm
        model.gene_enc.eval()
        model.scf_enc.eval()
        model.fp_enc.eval()
        model.desc_enc.eval()

        for x_gene, scf, x_fp, x_desc, ci, di, y_b in tr_loader:
            batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            y_b = y_b.to(device)
            optimizer.zero_grad()
            out = model(*batch)
            loss = F.huber_loss(out, y_b, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
                batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
                out = model(*batch)
                preds.extend(out.cpu().numpy())
                targets.extend(y_b.numpy())
        preds = np.array(preds, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        rmse = float(np.sqrt(mean_squared_error(targets, preds)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final prediction
    model.eval()
    preds = []
    with torch.no_grad():
        for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
            batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            out = model(*batch)
            preds.extend(out.cpu().numpy())
    return np.array(preds, dtype=np.float32)


# ── LOOCV LightGBM baseline ──────────────────────────────────────────────────

def lightgbm_loocv(scf_emb, fp, desc, y_true, meta):
    import lightgbm as lgb
    print("\n  [LightGBM LOOCV]", flush=True)
    organoids = sorted(meta["organoid_id"].unique())
    all_preds = np.zeros_like(y_true)
    X = np.concatenate([scf_emb, fp, desc], axis=1)

    for oid in organoids:
        tr_mask = meta["organoid_id"] != oid
        te_mask = meta["organoid_id"] == oid
        lgbm = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            num_leaves=15, max_depth=5,
            subsample=0.8, colsample_bytree=0.6,
            min_child_samples=5, reg_alpha=0.5, reg_lambda=0.5,
            n_jobs=4, random_state=42, verbose=-1,
        )
        lgbm.fit(X[tr_mask], y_true[tr_mask])
        all_preds[te_mask] = lgbm.predict(X[te_mask]).astype(np.float32)

    return all_preds


# ── Metrics ────────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 65, flush=True)
    print("  GDRNet Organoid Fine-Tuning (LOOCV)", flush=True)
    print("  Freeze encoders, fine-tune interaction+head", flush=True)
    print("  16-fold Leave-One-Organoid-Out", flush=True)
    print("=" * 65, flush=True)

    (x_gene, scf_emb, x_fp, x_desc, cell_idx, drug_idx,
     y_true, meta, n_cells, n_drugs) = load_organoid_data()

    organoids = sorted(meta["organoid_id"].unique())
    results = {}

    # ── 1. Pre-trained direct inference (baseline) ──
    print("\n  [1/3] Pre-trained direct inference ...", flush=True)
    from models.gdr import GDRNet
    all_preds_direct = np.zeros_like(y_true)
    for seed in [42, 123, 456]:
        model = GDRNet(n_genes=2000, scf_dim=3072, fp_bits=2048, n_desc=188,
                           n_cells=n_cells, n_drugs=n_drugs,
                           d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
                           n_deep=3, dropout=0.15)
        ckpt = MODELS / f"gdr_s{seed}.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned)
        model = model.to(device).eval()
        with torch.no_grad():
            out = model(
                torch.FloatTensor(x_gene).to(device),
                torch.FloatTensor(scf_emb).to(device),
                torch.FloatTensor(x_fp).to(device),
                torch.FloatTensor(x_desc).to(device),
                torch.LongTensor(cell_idx).to(device),
                torch.LongTensor(drug_idx).to(device),
            )
            all_preds_direct += out.cpu().numpy() / 3
    results["Pretrained-Direct"] = compute_metrics(
        y_true, all_preds_direct, "Pretrained-Direct")

    # ── 2. LOOCV Fine-tuning (3 seeds, ensemble) ──
    print("\n  [2/3] LOOCV Fine-tuning (16 folds) ...", flush=True)
    seeds = [42, 123, 456]
    all_preds_ft = {s: np.zeros_like(y_true) for s in seeds}

    for fold_i, test_oid in enumerate(organoids):
        tr_mask = meta["organoid_id"] != test_oid
        te_mask = meta["organoid_id"] == test_oid
        n_tr, n_te = tr_mask.sum(), te_mask.sum()
        print(f"\n  Fold {fold_i+1}/16: test={test_oid} "
              f"train={n_tr} test={n_te}", flush=True)

        for seed in seeds:
            model = build_model(n_cells, n_drugs, seed, device)

            tr_ds = OrgDataset(x_gene[tr_mask], scf_emb[tr_mask],
                               x_fp[tr_mask], x_desc[tr_mask],
                               cell_idx[tr_mask], drug_idx[tr_mask],
                               y_true[tr_mask])
            te_ds = OrgDataset(x_gene[te_mask], scf_emb[te_mask],
                               x_fp[te_mask], x_desc[te_mask],
                               cell_idx[te_mask], drug_idx[te_mask],
                               y_true[te_mask])

            preds = finetune_fold(model, tr_ds, te_ds, device,
                                  n_epochs=100, lr=5e-4, patience=20,
                                  batch_size=64)
            all_preds_ft[seed][te_mask] = preds

        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1} done. Elapsed: {elapsed/60:.1f} min", flush=True)

    # Ensemble average
    ens_preds_ft = np.mean([all_preds_ft[s] for s in seeds], axis=0)
    results["FineTune-LOOCV"] = compute_metrics(
        y_true, ens_preds_ft, "FineTune-LOOCV (3-model ens)")

    for seed in seeds:
        results[f"FineTune-LOOCV-s{seed}"] = compute_metrics(
            y_true, all_preds_ft[seed], f"FineTune-s{seed}")

    # ── 3. LightGBM LOOCV baseline ──
    print("\n  [3/3] LightGBM LOOCV ...", flush=True)
    lgbm_preds = lightgbm_loocv(scf_emb, x_fp, x_desc, y_true, meta)
    results["LightGBM-LOOCV"] = compute_metrics(y_true, lgbm_preds, "LightGBM-LOOCV")

    # ── Per-organoid comparison ──
    print(f"\n  Per-organoid comparison:", flush=True)
    per_org = []
    for oid in organoids:
        mask = meta["organoid_id"] == oid
        yt = y_true[mask]
        if mask.sum() < 3:
            continue
        row = dict(organoid=oid, n=int(mask.sum()))
        for label, preds in [("Pretrain", all_preds_direct),
                              ("FineTune", ens_preds_ft),
                              ("LightGBM", lgbm_preds)]:
            p = float(np.corrcoef(yt, preds[mask])[0, 1])
            row[f"P_{label}"] = round(p, 4)
        per_org.append(row)
        print(f"    {oid:8s}  n={mask.sum():2d}  "
              f"Pre={row['P_Pretrain']:.4f}  "
              f"FT={row['P_FineTune']:.4f}  "
              f"LGB={row['P_LightGBM']:.4f}", flush=True)

    pd.DataFrame(per_org).to_csv(TABLES / "organoid_loocv_per_organoid.csv", index=False)

    # ── Save predictions ──
    result_df = meta.copy()
    result_df["ic50"] = y_true
    result_df["pred_pretrained"] = all_preds_direct
    result_df["pred_finetuned"] = ens_preds_ft
    result_df["pred_lightgbm"] = lgbm_preds
    result_df.to_csv(TABLES / "organoid_loocv_predictions.csv", index=False)

    # ── Summary ──
    cmp = pd.DataFrame(results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / "organoid_comparison.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*65}", flush=True)
    print(f"  Organoid Fine-Tuning Summary", flush=True)
    print(f"{'='*65}", flush=True)
    print(cmp.to_string(), flush=True)
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"  Saved -> {TABLES}/organoid_comparison.csv", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
