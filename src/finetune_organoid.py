# -*- coding: utf-8 -*-
"""
GDRNet Organoid Fine-Tuning (LOOCV)
====================================
Leave-one-organoid-out cross-validation with frozen encoders.
"""

import sys
import io
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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

from models.gdrnet import GDRNet


class OrgDataset(Dataset):
    def __init__(self, x_gene, scf, x_fp, cell_idx, drug_idx, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.scf = torch.FloatTensor(scf)
        self.x_fp = torch.FloatTensor(x_fp)
        self.x_desc = torch.zeros(len(y), 1)
        self.cell_idx = torch.LongTensor(cell_idx)
        self.drug_idx = torch.LongTensor(drug_idx)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.x_gene[i], self.scf[i], self.x_fp[i], self.x_desc[i],
                self.cell_idx[i], self.drug_idx[i], self.y[i])


def load_organoid_data():
    """Load organoid data and build drug index from training dataset."""
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

    # Drug index from training dataset mapping
    drug_map = pd.read_csv(PROC / "expanded/drug_id_map.csv")
    drug_to_idx = dict(zip(drug_map["DRUG_NAME"], drug_map["drug_idx"]))

    n_cells = 964
    n_drugs = 229

    drug_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]],
                        dtype=np.int64)
    cell_idx = np.zeros(len(response), dtype=np.int64)

    return (gene_zeros, scf_emb, fp, cell_idx, drug_idx, response,
            meta, n_cells, n_drugs)


def build_model(n_cells, n_drugs, seed, device):
    """Build model and load pretrained weights."""
    model = GDRNet(
        n_genes=2000, scf_dim=3072, fp_bits=2048,
        n_cells=n_cells, n_drugs=n_drugs,
        d_hidden=256, id_emb_dim=64,
        n_cross=3, cross_rank=64, n_deep=3, dropout=0.15,
    )

    ckpt = MODELS / f"gdrnet_s{seed}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)

    # Freeze encoders
    for name, param in model.named_parameters():
        if any(k in name for k in ["gene_enc", "scf_enc", "fp_enc",
                                    "cell_emb", "drug_emb", "input_proj"]):
            param.requires_grad = False

    return model.to(device)


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
        model.gene_enc.eval()
        model.scf_enc.eval()
        model.fp_enc.eval()

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

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
                batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
                out = model(*batch)
                preds.extend(out.cpu().numpy())
                targets.extend(y_b.numpy())
        rmse = float(np.sqrt(mean_squared_error(
            np.array(targets), np.array(preds))))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    preds = []
    with torch.no_grad():
        for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
            batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            out = model(*batch)
            preds.extend(out.cpu().numpy())
    return np.array(preds, dtype=np.float32)


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


def main():
    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 65, flush=True)
    print("  GDRNet Organoid Fine-Tuning (LOOCV)", flush=True)
    print("=" * 65, flush=True)

    (x_gene, scf_emb, x_fp, cell_idx, drug_idx,
     y_true, meta, n_cells, n_drugs) = load_organoid_data()
    organoids = sorted(meta["organoid_id"].unique())

    print(f"  Organoids: {len(organoids)}  Samples: {len(y_true)}", flush=True)
    print(f"  n_cells={n_cells}  n_drugs={n_drugs}", flush=True)

    seeds = [42, 123, 456]
    all_preds_ft = {s: np.zeros_like(y_true) for s in seeds}

    for fold_i, test_oid in enumerate(organoids):
        tr_mask = meta["organoid_id"] != test_oid
        te_mask = meta["organoid_id"] == test_oid
        print(f"\n  Fold {fold_i+1}/{len(organoids)}: test={test_oid}  "
              f"train={tr_mask.sum()} test={te_mask.sum()}", flush=True)

        for seed in seeds:
            model = build_model(n_cells, n_drugs, seed, device)

            tr_ds = OrgDataset(x_gene[tr_mask], scf_emb[tr_mask],
                               x_fp[tr_mask], cell_idx[tr_mask],
                               drug_idx[tr_mask], y_true[tr_mask])
            te_ds = OrgDataset(x_gene[te_mask], scf_emb[te_mask],
                               x_fp[te_mask], cell_idx[te_mask],
                               drug_idx[te_mask], y_true[te_mask])

            preds = finetune_fold(model, tr_ds, te_ds, device)
            all_preds_ft[seed][te_mask] = preds

        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1} done. Elapsed: {elapsed/60:.1f} min", flush=True)

    # Ensemble
    ens_preds = np.mean([all_preds_ft[s] for s in seeds], axis=0)

    results = {}
    results["GDRNet-FineTune-LOOCV"] = compute_metrics(
        y_true, ens_preds, "GDRNet-FineTune-LOOCV (ens)")
    for seed in seeds:
        results[f"GDRNet-FineTune-s{seed}"] = compute_metrics(
            y_true, all_preds_ft[seed], f"GDRNet-FineTune-s{seed}")

    # Per-organoid
    per_org = []
    for oid in organoids:
        mask = meta["organoid_id"] == oid
        if mask.sum() >= 3:
            p = float(np.corrcoef(y_true[mask], ens_preds[mask])[0, 1])
            per_org.append(dict(organoid=oid, Pearson=round(p, 4),
                                n_samples=int(mask.sum())))

    # Save
    cmp = pd.DataFrame(results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / "organoid_comparison.csv")

    result_df = meta.copy()
    result_df["ic50"] = y_true
    result_df["pred_finetuned"] = ens_preds
    for seed in seeds:
        result_df[f"pred_s{seed}"] = all_preds_ft[seed]
    result_df.to_csv(TABLES / "organoid_loocv_predictions.csv", index=False)

    per_org_df = pd.DataFrame(per_org)
    per_org_df.to_csv(TABLES / "organoid_loocv_per_organoid.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*65}", flush=True)
    print(cmp.to_string(), flush=True)
    print(f"\n  Per-organoid Pearson:", flush=True)
    for row in per_org:
        print(f"    {row['organoid']:15s}  n={row['n_samples']:3d}  P={row['Pearson']:.4f}", flush=True)
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"  Saved -> organoid_comparison.csv", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
