# -*- coding: utf-8 -*-
"""
GDRNet Multi-Cancer Organoid Fine-Tuning (LOOCV)
=================================================
Leave-one-organoid-out cross-validation with frozen GDSC encoders.
Supports multiple cancer types: CRC (existing), PDAC, BLCA.
"""

import sys, io, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from pathlib import Path

ROOT = Path("/export/home/kongyan/project/Organoid")
PROC = ROOT / "data/processed"
GEO_PROC = PROC / "geo"
TABLES = ROOT / "results/tables"
TABLES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "GDRNet" / "src"))

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


def load_cancer_data(cancer_type):
    """Load organoid data for a specific cancer type."""
    if cancer_type == "crc":
        meta = pd.read_csv(PROC / "organoid_pair_meta.csv")
        cell_emb_all = np.load(PROC / "organoid_cell_emb.npy")
        cell_ids_all = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
        drug_feat = np.load(PROC / "organoid_drug_features.npy")
        response = np.load(PROC / "organoid_response.npy")
        response_col = "ic50"
    elif cancer_type == "pdac":
        meta = pd.read_csv(GEO_PROC / "pdac_pair_meta.csv")
        cell_emb_all = np.load(GEO_PROC / "pdac_cell_emb.npy")
        cell_ids_all = np.load(GEO_PROC / "pdac_cell_ids.npy", allow_pickle=True)
        drug_feat = np.load(GEO_PROC / "pdac_drug_features.npy")
        response = np.load(GEO_PROC / "pdac_response.npy")
        response_col = "sensitivity"
    elif cancer_type == "blca":
        meta = pd.read_csv(GEO_PROC / "blca_pair_meta.csv")
        cell_emb_all = np.load(GEO_PROC / "blca_cell_emb.npy")
        cell_ids_all = np.load(GEO_PROC / "blca_cell_ids.npy", allow_pickle=True)
        drug_feat = np.load(GEO_PROC / "blca_drug_features.npy")
        response = np.load(GEO_PROC / "blca_response.npy")
        response_col = "ic50"
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")

    cell_emb_map = {cid: cell_emb_all[i] for i, cid in enumerate(cell_ids_all)}
    scf_emb = np.array([cell_emb_map[oid] for oid in meta["organoid_id"]],
                       dtype=np.float32)
    gene_zeros = np.zeros((len(response), 2000), dtype=np.float32)
    fp = drug_feat[:, :2048].astype(np.float32)

    # Drug index from GDSC mapping
    drug_map = pd.read_csv(PROC / "expanded/drug_id_map.csv")
    drug_to_idx = dict(zip(drug_map["DRUG_NAME"], drug_map["drug_idx"]))

    n_cells = 964
    n_drugs = 229

    drug_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]],
                        dtype=np.int64)
    cell_idx = np.zeros(len(response), dtype=np.int64)

    return (gene_zeros, scf_emb, fp, cell_idx, drug_idx, response,
            meta, n_cells, n_drugs, response_col)


def build_model(n_cells, n_drugs, seed, device):
    model = GDRNet(
        n_genes=2000, scf_dim=3072, fp_bits=2048,
        n_cells=n_cells, n_drugs=n_drugs,
        d_hidden=256, id_emb_dim=64,
        n_cross=3, cross_rank=64, n_deep=3, dropout=0.15,
    )
    ckpt = ROOT / "models" / f"gdrnet_s{seed}.pt"
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
    # AUROC: lower response = more sensitive (works for both IC50 and 1-AUC)
    thr = np.percentile(y_true, 30)
    try:
        auroc = float(roc_auc_score((y_true <= thr).astype(int), -y_pred))
    except Exception:
        auroc = float("nan")
    m = dict(Pearson=round(pearson, 4), R2=round(r2, 4),
             RMSE=round(rmse, 4), AUROC=round(auroc, 4))
    if name:
        print(f"  {name:40s}  Pearson={pearson:.4f}  R2={r2:.4f}  "
              f"RMSE={rmse:.4f}  AUROC={auroc:.4f}", flush=True)
    return m


def run_loocv(cancer_type, device):
    """Run LOOCV for a specific cancer type."""
    t0 = time.time()
    cancer_names = {"crc": "CRC (Colorectal)", "pdac": "PDAC (Pancreatic)",
                    "blca": "BLCA (Bladder)"}
    print(f"\n{'='*65}")
    print(f"  GDRNet LOOCV Fine-Tuning: {cancer_names.get(cancer_type, cancer_type)}")
    print(f"{'='*65}", flush=True)

    (x_gene, scf_emb, x_fp, cell_idx, drug_idx,
     y_true, meta, n_cells, n_drugs, response_col) = load_cancer_data(cancer_type)
    organoids = sorted(meta["organoid_id"].unique())

    print(f"  Organoids: {len(organoids)}  Samples: {len(y_true)}", flush=True)
    print(f"  Drugs: {meta['drug_name'].nunique()}  Response: {response_col}", flush=True)
    print(f"  Response range: {y_true.min():.4f} to {y_true.max():.4f}", flush=True)

    seeds = [42, 123, 456]
    all_preds = {s: np.zeros_like(y_true) for s in seeds}

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
            all_preds[seed][te_mask] = preds

        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1} done. Elapsed: {elapsed/60:.1f} min", flush=True)

    # Ensemble
    ens_preds = np.mean([all_preds[s] for s in seeds], axis=0)

    # Overall metrics
    results = {}
    results["GDRNet-Ensemble"] = compute_metrics(y_true, ens_preds,
                                                  f"{cancer_type.upper()} GDRNet-Ensemble")
    for seed in seeds:
        results[f"GDRNet-s{seed}"] = compute_metrics(y_true, all_preds[seed],
                                                       f"{cancer_type.upper()} GDRNet-s{seed}")

    # Per-organoid
    per_org = []
    for oid in organoids:
        mask = meta["organoid_id"] == oid
        if mask.sum() >= 3:
            p = float(np.corrcoef(y_true[mask], ens_preds[mask])[0, 1])
            r2 = float(r2_score(y_true[mask], ens_preds[mask]))
            per_org.append(dict(organoid=oid, Pearson=round(p, 4),
                                R2=round(r2, 4), n_samples=int(mask.sum()),
                                cancer_type=cancer_type))

    # Per-drug analysis
    per_drug = []
    for drug in meta["drug_name"].unique():
        mask = meta["drug_name"] == drug
        if mask.sum() >= 5:
            p = float(np.corrcoef(y_true[mask], ens_preds[mask])[0, 1])
            per_drug.append(dict(drug=drug, Pearson=round(p, 4),
                                 n_samples=int(mask.sum()), cancer_type=cancer_type))

    # Save results
    prefix = cancer_type
    cmp = pd.DataFrame(results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / f"{prefix}_loocv_comparison.csv")

    result_df = meta.copy()
    result_df["response"] = y_true
    result_df["pred_ensemble"] = ens_preds
    for seed in seeds:
        result_df[f"pred_s{seed}"] = all_preds[seed]
    result_df.to_csv(TABLES / f"{prefix}_loocv_predictions.csv", index=False)

    per_org_df = pd.DataFrame(per_org)
    per_org_df.to_csv(TABLES / f"{prefix}_loocv_per_organoid.csv", index=False)

    if per_drug:
        per_drug_df = pd.DataFrame(per_drug)
        per_drug_df.to_csv(TABLES / f"{prefix}_loocv_per_drug.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*65}", flush=True)
    print(cmp.to_string(), flush=True)
    print(f"\n  Per-organoid Pearson:", flush=True)
    for row in per_org:
        print(f"    {row['organoid']:15s}  n={row['n_samples']:3d}  "
              f"P={row['Pearson']:.4f}  R2={row['R2']:.4f}", flush=True)
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"  Saved → {prefix}_loocv_*.csv", flush=True)
    print(f"{'='*65}", flush=True)

    return results, per_org, per_drug


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    all_results = {}
    all_per_org = []
    all_per_drug = []

    # Run for each cancer type
    for ctype in ["crc", "pdac", "blca"]:
        results, per_org, per_drug = run_loocv(ctype, device)
        all_results[ctype] = results
        all_per_org.extend(per_org)
        all_per_drug.extend(per_drug)

    # Save combined results
    combined_rows = []
    for ctype, results in all_results.items():
        ens = results.get("GDRNet-Ensemble", {})
        cancer_names = {"crc": "CRC", "pdac": "PDAC", "blca": "BLCA"}
        combined_rows.append({
            "Dataset": ctype.upper(),
            "Cancer": cancer_names[ctype],
            **ens,
        })
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(TABLES / "multicancer_loocv_comparison.csv", index=False)

    # Combined per-organoid
    all_per_org_df = pd.DataFrame(all_per_org)
    all_per_org_df.to_csv(TABLES / "multicancer_per_organoid.csv", index=False)

    # Combined per-drug
    if all_per_drug:
        all_per_drug_df = pd.DataFrame(all_per_drug)
        all_per_drug_df.to_csv(TABLES / "multicancer_per_drug.csv", index=False)

    # Print summary
    print(f"\n{'='*65}")
    print("  MULTI-CANCER LOOCV SUMMARY")
    print(f"{'='*65}")
    print(combined_df.to_string(index=False))
    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
