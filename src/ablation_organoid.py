# -*- coding: utf-8 -*-
"""
GDRNet Ablation Study (Organoid LOOCV)
==========================================
Test contribution of each component by zeroing it out:

  1. Full model        — baseline
  2. w/o ID Embeddings — remove cell/drug ID embeddings
  3. w/o Cross Network — remove DCN v2 cross layers
  4. w/o Deep Network  — remove deep MLP layers
  5. w/o scF Emb       — remove scFoundation embeddings
  6. w/o Drug Desc     — remove RDKit descriptors

Uses same LOOCV + fine-tuning protocol as finetune_organoid.py.
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

from models.gdr import GDRNet


# ── Ablation Model Variants ───────────────────────────────────────────────────

class AblationNoID(GDRNet):
    """w/o Cell/Drug ID Embeddings."""
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)

        h_fp = self.fp_enc(x_fp)
        h_desc = self.desc_enc(x_desc)
        h_did = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        drug_repr = torch.cat([h_fp, h_desc, h_did], dim=-1)

        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


class AblationNoCross(GDRNet):
    """w/o DCN v2 Cross Network."""
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)

        h_fp = self.fp_enc(x_fp)
        h_desc = self.desc_enc(x_desc)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_desc, h_did], dim=-1)

        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        # Skip cross network: use x0 directly
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        # Use x0 as "cross" to keep head input dim correct
        return self.head(torch.cat([x0, xd], dim=-1)).squeeze(-1)


class AblationNoDeep(GDRNet):
    """w/o Deep MLP Network."""
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)

        h_fp = self.fp_enc(x_fp)
        h_desc = self.desc_enc(x_desc)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_desc, h_did], dim=-1)

        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        # Skip deep: use zeros for deep dim
        xd = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


class AblationNoScF(GDRNet):
    """w/o scFoundation Embeddings."""
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = torch.zeros(x_gene.size(0), 256, device=x_gene.device)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)

        h_fp = self.fp_enc(x_fp)
        h_desc = self.desc_enc(x_desc)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_desc, h_did], dim=-1)

        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


class AblationNoDesc(GDRNet):
    """w/o RDKit Descriptors."""
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)

        h_fp = self.fp_enc(x_fp)
        h_desc = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_desc, h_did], dim=-1)

        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


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
    if desc_raw.shape[1] < 188:
        pad = np.zeros((len(response), 188 - desc_raw.shape[1]), dtype=np.float32)
        desc = np.concatenate([desc_raw, pad], axis=1)
    else:
        desc = desc_raw[:, :188]

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

    return (gene_zeros, scf_emb, fp, desc, cell_idx, drug_idx, response,
            meta, n_cells, n_drugs)


# ── Build + Freeze model ──────────────────────────────────────────────────────

def build_model(model_cls, n_cells, n_drugs, seed, device):
    model = model_cls(
        n_genes=2000, scf_dim=3072, fp_bits=2048, n_desc=188,
        n_cells=n_cells, n_drugs=n_drugs,
        d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
        n_deep=3, dropout=0.15,
    )
    ckpt = MODELS / f"gdr_s{seed}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)

    # Freeze encoders + embeddings + input_proj
    for name, param in model.named_parameters():
        if any(k in name for k in ["gene_enc", "scf_enc", "fp_enc",
                                     "desc_enc", "cell_emb", "drug_emb",
                                     "input_proj"]):
            param.requires_grad = False

    return model.to(device)


# ── LOOCV fine-tune one fold ──────────────────────────────────────────────────

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
    print("  GDRNet Ablation Study (Organoid LOOCV)", flush=True)
    print("  6 variants × 16 folds × 1 seed (s42)", flush=True)
    print("=" * 65, flush=True)

    (x_gene, scf_emb, x_fp, x_desc, cell_idx, drug_idx,
     y_true, meta, n_cells, n_drugs) = load_organoid_data()
    organoids = sorted(meta["organoid_id"].unique())

    # Ablation variants
    ABLATIONS = [
        ("Full Model", GDRNet),
        ("w/o ID Embeddings", AblationNoID),
        ("w/o Cross Network", AblationNoCross),
        ("w/o Deep Network",  AblationNoDeep),
        ("w/o scF Emb",       AblationNoScF),
        ("w/o Drug Desc",     AblationNoDesc),
    ]

    all_results = {}

    for abl_name, model_cls in ABLATIONS:
        print(f"\n{'='*65}", flush=True)
        print(f"  Ablation: {abl_name}", flush=True)
        print(f"{'='*65}", flush=True)

        all_preds = np.zeros_like(y_true)

        for fold_i, test_oid in enumerate(organoids):
            tr_mask = meta["organoid_id"] != test_oid
            te_mask = meta["organoid_id"] == test_oid

            model = build_model(model_cls, n_cells, n_drugs, seed=42, device=device)

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
            all_preds[te_mask] = preds

            elapsed = time.time() - t0
            print(f"    Fold {fold_i+1:2d}/16  test={test_oid}  "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

        all_results[abl_name] = compute_metrics(y_true, all_preds, abl_name)

        # Per-organoid
        per_org = []
        for oid in organoids:
            mask = meta["organoid_id"] == oid
            if mask.sum() >= 3:
                p = float(np.corrcoef(y_true[mask], all_preds[mask])[0, 1])
                per_org.append(dict(organoid=oid, Pearson=round(p, 4)))
        avg_p = np.mean([r["Pearson"] for r in per_org])
        print(f"    Avg per-organoid Pearson: {avg_p:.4f}", flush=True)

        # Save this variant's predictions
        np.save(TABLES / f"ablation_{abl_name.replace(' ', '_').replace('/', '')}_preds.npy",
                all_preds)

    # ── Summary table ──
    cmp = pd.DataFrame(all_results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / "ablation_comparison.csv")

    # Compute drops vs Full
    full_p = all_results.get("Full Model", {}).get("Pearson", 0)
    print(f"\n{'='*65}", flush=True)
    print(f"  Ablation Results (sorted by Pearson)", flush=True)
    print(f"{'='*65}", flush=True)
    print(cmp.to_string(), flush=True)
    print(f"\n  Component contribution (drop from Full):", flush=True)
    for name, row in cmp.iterrows():
        drop = full_p - row["Pearson"]
        pct = drop / full_p * 100 if full_p > 0 else 0
        print(f"    {name:25s}  P={row['Pearson']:.4f}  "
              f"drop={drop:+.4f} ({pct:+.1f}%)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"  Saved -> {TABLES}/ablation_comparison.csv", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
