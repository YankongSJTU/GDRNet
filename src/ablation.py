# -*- coding: utf-8 -*-
"""
GDRNet Ablation Study — GDSC + Organoid
============================================
Test contribution of each component on both datasets.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from pathlib import Path

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

from models.gdrnet import GDRNet, GDRNetDataset


# ── Ablation Variants ─────────────────────────────────────────────────────────

class AblationNoID(GDRNet):
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)
        h_fp = self.fp_enc(x_fp)
        h_did = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        drug_repr = torch.cat([h_fp, h_did], dim=-1)
        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


class AblationNoCross(GDRNet):
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)
        h_fp = self.fp_enc(x_fp)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_did], dim=-1)
        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([x0, xd], dim=-1)).squeeze(-1)


class AblationNoDeep(GDRNet):
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)
        h_fp = self.fp_enc(x_fp)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_did], dim=-1)
        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = torch.zeros(x_gene.size(0), 64, device=x_gene.device)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


class AblationNoScF(GDRNet):
    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = torch.zeros(x_gene.size(0), 256, device=x_gene.device)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)
        h_fp = self.fp_enc(x_fp)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_did], dim=-1)
        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)
        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


# ── GDSC Training ─────────────────────────────────────────────────────────────

def train_variant(model_cls, tr_ds, val_ds, n_cells, n_drugs, device, n_epochs=150, patience=30):
    model = model_cls(
        n_genes=2000, scf_dim=3072, fp_bits=2048,
        n_cells=n_cells, n_drugs=n_drugs,
        d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
        n_deep=3, dropout=0.15,
    ).to(device)

    tr_loader = DataLoader(tr_ds, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    total_steps = n_epochs * len(tr_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse, best_state, wait = float("inf"), None, 0

    for epoch in range(n_epochs):
        model.train()
        for x_gene, scf, x_fp, x_desc, ci, di, y_b in tr_loader:
            batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            y_b = y_b.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(*batch)
                loss = F.huber_loss(out, y_b, delta=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
                batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
                out = model(*batch)
                preds.extend(out.cpu().numpy())
                targets.extend(y_b.numpy())
        rmse = float(np.sqrt(mean_squared_error(np.array(targets), np.array(preds))))

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
    preds, targets = [], []
    with torch.no_grad():
        for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
            batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            out = model(*batch)
            preds.extend(out.cpu().numpy())
            targets.extend(y_b.numpy())
    preds, targets = np.array(preds), np.array(targets)
    pearson = float(np.corrcoef(targets, preds)[0, 1])
    r2 = float(r2_score(targets, preds))
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    thr = np.percentile(targets, 30)
    try:
        auroc = float(roc_auc_score((targets <= thr).astype(int), -preds))
    except:
        auroc = float("nan")
    return dict(Pearson=round(pearson, 4), R2=round(r2, 4), RMSE=round(rmse, 4), AUROC=round(auroc, 4))


def load_gdsc_data():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    scf_emb = np.load(PROC / "scfoundation_cell_emb.npy")
    scf_ids = np.load(PROC / "scfoundation_cell_ids.npy", allow_pickle=True)
    id_to_emb = {cid: scf_emb[i] for i, cid in enumerate(scf_ids)}

    X_cell = pd.read_parquet(PROC / "gdsc_cell_features.parquet")
    y_full = pd.read_parquet(PROC / "gdsc_response_lnic50.parquet").iloc[:, 0].values.astype(np.float32)
    meta_full = pd.read_parquet(PROC / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")

    fps = {}
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol:
                try:
                    gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
                    fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
                except:
                    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
                fps[row["drug_name"]] = fp
        except:
            pass

    valid_mask = (meta_full["drug_name"].isin(fps)) & (meta_full["ModelID"].isin(id_to_emb))
    meta = meta_full[valid_mask].reset_index(drop=True)
    y = y_full[valid_mask]
    X_gene = X_cell.loc[meta_full[valid_mask].index].values.astype(np.float32)
    X_scf = np.array([id_to_emb[c] for c in meta["ModelID"]], dtype=np.float32)
    X_fp = np.array([fps[d] for d in meta["drug_name"]], dtype=np.float32)
    X_desc_dummy = np.zeros((len(y), 1), dtype=np.float32)

    cell_ids = meta["ModelID"].values
    tr_cells, val_cells = train_test_split(np.unique(cell_ids), test_size=0.2, random_state=42)
    tr_mask = np.isin(cell_ids, tr_cells)
    val_mask = np.isin(cell_ids, val_cells)

    cell_list = sorted(meta["ModelID"].unique())
    drug_list = sorted(meta["drug_name"].unique())
    cell_to_idx = {c: i+1 for i, c in enumerate(cell_list)}
    drug_to_idx = {d: i+1 for i, d in enumerate(drug_list)}
    cell_idx = np.array([cell_to_idx[c] for c in meta["ModelID"]], dtype=np.int64)
    drug_idx = np.array([drug_to_idx[d] for d in meta["drug_name"]], dtype=np.int64)

    tr_ds = GDRNetDataset(X_gene[tr_mask], X_scf[tr_mask], X_fp[tr_mask], X_desc_dummy[tr_mask],
                        cell_idx[tr_mask], drug_idx[tr_mask], y[tr_mask])
    val_ds = GDRNetDataset(X_gene[val_mask], X_scf[val_mask], X_fp[val_mask], X_desc_dummy[val_mask],
                         cell_idx[val_mask], drug_idx[val_mask], y[val_mask])

    return tr_ds, val_ds, len(cell_list), len(drug_list)


# ── Organoid LOOCV ────────────────────────────────────────────────────────────

def organoid_ablation(model_cls, n_cells, n_drugs, device):
    meta = pd.read_csv(PROC / "organoid_pair_meta.csv")
    cell_emb_all = np.load(PROC / "organoid_cell_emb.npy")
    cell_ids_all = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
    drug_feat = np.load(PROC / "organoid_drug_features.npy")
    response = np.load(PROC / "organoid_response.npy")

    cell_emb_map = {cid: cell_emb_all[i] for i, cid in enumerate(cell_ids_all)}
    scf_emb = np.array([cell_emb_map[oid] for oid in meta["organoid_id"]], dtype=np.float32)
    gene_zeros = np.zeros((len(response), 2000), dtype=np.float32)
    fp = drug_feat[:, :2048].astype(np.float32)
    desc_dummy = np.zeros((len(response), 1), dtype=np.float32)

    from rdkit import Chem
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")
    fps_valid = set()
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol: fps_valid.add(row["drug_name"])
        except: pass
    gdsc_meta = pd.read_parquet(PROC / "gdsc_metadata.parquet")
    scf_ids = set(np.load(PROC / "scfoundation_cell_ids.npy", allow_pickle=True))
    valid_mask = (gdsc_meta["drug_name"].isin(fps_valid)) & (gdsc_meta["ModelID"].isin(scf_ids))
    meta_gdsc = gdsc_meta[valid_mask].reset_index(drop=True)
    drug_list = sorted(meta_gdsc["drug_name"].unique())
    drug_to_idx = {d: i+1 for i, d in enumerate(drug_list)}
    drug_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]], dtype=np.int64)
    cell_idx = np.zeros(len(response), dtype=np.int64)

    organoids = sorted(meta["organoid_id"].unique())
    all_preds = np.zeros_like(response)

    for test_oid in organoids:
        tr_mask = meta["organoid_id"] != test_oid
        te_mask = meta["organoid_id"] == test_oid

        model = model_cls(
            n_genes=2000, scf_dim=3072, fp_bits=2048,
            n_cells=n_cells, n_drugs=n_drugs,
            d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
            n_deep=3, dropout=0.15,
        )
        ckpt = MODELS / "gdrnet_s42.pt"
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)

        for name, param in model.named_parameters():
            if any(k in name for k in ["gene_enc", "scf_enc", "fp_enc",
                                         "cell_emb", "drug_emb", "input_proj"]):
                param.requires_grad = False
        model = model.to(device)

        tr_ds = GDRNetDataset(gene_zeros[tr_mask], scf_emb[tr_mask], fp[tr_mask], desc_dummy[tr_mask],
                            cell_idx[tr_mask], drug_idx[tr_mask], response[tr_mask])
        te_ds = GDRNetDataset(gene_zeros[te_mask], scf_emb[te_mask], fp[te_mask], desc_dummy[te_mask],
                            cell_idx[te_mask], drug_idx[te_mask], response[te_mask])

        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(te_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=5e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-6)

        best_rmse, best_state, wait = float("inf"), None, 0
        for epoch in range(100):
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
            preds_t, targets_t = [], []
            with torch.no_grad():
                for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
                    batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
                    out = model(*batch)
                    preds_t.extend(out.cpu().numpy())
                    targets_t.extend(y_b.numpy())
            rmse = float(np.sqrt(mean_squared_error(np.array(targets_t), np.array(preds_t))))
            if rmse < best_rmse:
                best_rmse = rmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 20:
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        model.eval()
        preds_f = []
        with torch.no_grad():
            for x_gene, scf, x_fp, x_desc, ci, di, y_b in val_loader:
                batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
                out = model(*batch)
                preds_f.extend(out.cpu().numpy())
        all_preds[te_mask] = np.array(preds_f, dtype=np.float32)

    pearson = float(np.corrcoef(response, all_preds)[0, 1])
    r2 = float(r2_score(response, all_preds))
    rmse = float(np.sqrt(mean_squared_error(response, all_preds)))
    thr = np.percentile(response, 30)
    try:
        auroc = float(roc_auc_score((response <= thr).astype(int), -all_preds))
    except:
        auroc = float("nan")
    return dict(Pearson=round(pearson, 4), R2=round(r2, 4), RMSE=round(rmse, 4), AUROC=round(auroc, 4))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ABLATIONS = [
        ("Full Model",        GDRNet),
        ("w/o ID Embeddings", AblationNoID),
        ("w/o Cross Network", AblationNoCross),
        ("w/o Deep Network",  AblationNoDeep),
        ("w/o scF Emb",       AblationNoScF),
    ]

    # ── GDSC Ablation ──
    print("=" * 65, flush=True)
    print("  GDRNet Ablation — GDSC (training from scratch)", flush=True)
    print("=" * 65, flush=True)

    tr_ds, val_ds, n_cells, n_drugs = load_gdsc_data()
    gdsc_results = {}
    for name, cls in ABLATIONS:
        print(f"\n  GDSC: {name}", flush=True)
        m = train_variant(cls, tr_ds, val_ds, n_cells, n_drugs, device)
        gdsc_results[name] = m
        print(f"    Pearson={m['Pearson']:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

    gdsc_df = pd.DataFrame(gdsc_results).T.sort_values("Pearson", ascending=False)
    gdsc_df.to_csv(TABLES / "ablation_gdsc.csv")
    print(f"\n  GDSC Ablation Results:", flush=True)
    print(gdsc_df.to_string(), flush=True)

    # ── Organoid Ablation ──
    print(f"\n{'='*65}", flush=True)
    print(f"  GDRNet Ablation — Organoid LOOCV", flush=True)
    print(f"{'='*65}", flush=True)

    # Need n_cells, n_drugs for organoid
    gdsc_meta = pd.read_parquet(PROC / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")
    from rdkit import Chem
    fps_valid = set()
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol: fps_valid.add(row["drug_name"])
        except: pass
    scf_ids = set(np.load(PROC / "scfoundation_cell_ids.npy", allow_pickle=True))
    valid_mask = (gdsc_meta["drug_name"].isin(fps_valid)) & (gdsc_meta["ModelID"].isin(scf_ids))
    meta_gdsc = gdsc_meta[valid_mask].reset_index(drop=True)
    n_cells_o = meta_gdsc["ModelID"].nunique()
    n_drugs_o = meta_gdsc["drug_name"].nunique()

    org_results = {}
    for name, cls in ABLATIONS:
        print(f"\n  Organoid: {name}", flush=True)
        m = organoid_ablation(cls, n_cells_o, n_drugs_o, device)
        org_results[name] = m
        print(f"    Pearson={m['Pearson']:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)

    org_df = pd.DataFrame(org_results).T.sort_values("Pearson", ascending=False)
    org_df.to_csv(TABLES / "ablation_organoid.csv")
    print(f"\n  Organoid Ablation Results:", flush=True)
    print(org_df.to_string(), flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
