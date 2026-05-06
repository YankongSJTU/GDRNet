# -*- coding: utf-8 -*-
"""
GDRNetV11 Ablation Study (GDSC)
================================
Test contribution of each component by training from scratch with each variant:

  1. Full model        — baseline
  2. w/o ID Embeddings — remove cell/drug ID embeddings
  3. w/o Cross Network — remove DCN v2 cross layers
  4. w/o Deep Network  — remove deep MLP layers
  5. w/o scF Emb       — remove scFoundation embeddings
  6. w/o Drug Desc     — remove RDKit descriptors

Each variant is trained independently on GDSC with the same hyperparameters.
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

from models.gdr_v11 import GDRNetV11, V11Dataset, evaluate_v11


# ── Ablation Model Variants ───────────────────────────────────────────────────

class AblationNoID(GDRNetV11):
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


class AblationNoCross(GDRNetV11):
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


class AblationNoDeep(GDRNetV11):
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


class AblationNoScF(GDRNetV11):
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


class AblationNoDesc(GDRNetV11):
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


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_gdsc_data():
    """Load GDSC data for ablation study."""
    print("Loading GDSC data ...", flush=True)
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    # scFoundation embeddings
    scf_emb = np.load(PROC / "scfoundation_cell_emb.npy")
    scf_ids = np.load(PROC / "scfoundation_cell_ids.npy", allow_pickle=True)
    id_to_emb = {cid: scf_emb[i] for i, cid in enumerate(scf_ids)}
    print(f"  scFoundation: {scf_emb.shape}", flush=True)

    # Core data
    X_cell_full = pd.read_parquet(PROC / "gdsc_cell_features.parquet")
    y_full = pd.read_parquet(PROC / "gdsc_response_lnic50.parquet").iloc[:, 0].values.astype(np.float32)
    meta_full = pd.read_parquet(PROC / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT / "gdsc_drug_smiles.csv")

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
                fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
            fps[row["drug_name"]] = fp
        except Exception:
            pass
    print(f"  Morgan FP: {len(fps)} drugs", flush=True)

    # Filter to valid samples
    valid_mask = (meta_full["drug_name"].isin(fps)) & (meta_full["ModelID"].isin(id_to_emb))
    orig_idx = meta_full[valid_mask].index
    meta = meta_full[valid_mask].reset_index(drop=True)
    y = y_full[valid_mask]

    X_gene = X_cell_full.loc[orig_idx].values.astype(np.float32)
    X_scf = np.array([id_to_emb[c] for c in meta["ModelID"]], dtype=np.float32)
    X_fp = np.array([fps[d] for d in meta["drug_name"]], dtype=np.float32)

    print(f"  Filtered: {len(y):,} samples ({meta['ModelID'].nunique()} cells, {meta['drug_name'].nunique()} drugs)", flush=True)

    # RDKit descriptors
    unique_drugs = smiles_df[smiles_df["drug_name"].isin(fps)][["drug_name", "smiles"]].drop_duplicates("drug_name").reset_index(drop=True)
    desc_arr = compute_rdkit_descriptors(unique_drugs["smiles"].tolist())
    drug_desc_map = {row["drug_name"]: desc_arr[i] for i, row in unique_drugs.iterrows()}
    X_desc_all = np.array([drug_desc_map.get(d, np.zeros(desc_arr.shape[1])) for d in meta["drug_name"]], dtype=np.float32)

    # Cell-line based split
    cell_ids_str = meta["ModelID"].values
    unique_cells = np.unique(cell_ids_str)
    tr_cells, val_cells = train_test_split(unique_cells, test_size=0.2, random_state=42)
    tr_mask = np.isin(cell_ids_str, tr_cells)
    val_mask = np.isin(cell_ids_str, val_cells)
    print(f"  Split: train={tr_mask.sum():,}  val={val_mask.sum():,}", flush=True)

    # Normalize descriptors
    X_desc_tr, X_desc_val, n_desc = normalize_descriptors(X_desc_all[tr_mask], X_desc_all[val_mask])

    # Build cell/drug index
    cell_list = sorted(meta["ModelID"].unique())
    drug_list = sorted(meta["drug_name"].unique())
    cell_to_idx = {c: i + 1 for i, c in enumerate(cell_list)}
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}
    cell_idx = np.array([cell_to_idx[c] for c in meta["ModelID"]], dtype=np.int64)
    drug_idx = np.array([drug_to_idx[d] for d in meta["drug_name"]], dtype=np.int64)

    n_cells = len(cell_list)
    n_drugs = len(drug_list)

    # Assemble
    data = dict(
        x_gene_tr=X_gene[tr_mask], x_scf_tr=X_scf[tr_mask],
        x_fp_tr=X_fp[tr_mask], x_desc_tr=X_desc_tr,
        cell_idx_tr=cell_idx[tr_mask], drug_idx_tr=drug_idx[tr_mask],
        y_tr=y[tr_mask],
        x_gene_val=X_gene[val_mask], x_scf_val=X_scf[val_mask],
        x_fp_val=X_fp[val_mask], x_desc_val=X_desc_val,
        cell_idx_val=cell_idx[val_mask], drug_idx_val=drug_idx[val_mask],
        y_val=y[val_mask],
        n_cells=n_cells, n_drugs=n_drugs, n_desc=n_desc,
    )
    return data


def compute_rdkit_descriptors(smiles_list):
    """Compute RDKit molecular descriptors."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    desc_fns = [
        ("MolWt", rdMolDescriptors.CalcExactMolWt),
        ("LogP", rdMolDescriptors.CalcCrippenDescriptors),
        ("TPSA", rdMolDescriptors.CalcTPSA),
        ("NumHDonors", rdMolDescriptors.CalcNumHBD),
        ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
        ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
        ("NumHeavyAtoms", lambda m: m.GetNumHeavyAtoms()),
        ("NumRings", rdMolDescriptors.CalcNumRings),
        ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ]
    # Add more descriptors (simplified)
    for name in ["MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
                 "LabuteASA", "BalabanJ", "BertzCT", "Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n",
                 "Chi4n", "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "HallKierAlpha", "Ipc", "Kappa1",
                 "Kappa2", "Kappa3", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5",
                 "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "PEOE_VSA10", "PEOE_VSA11",
                 "SMR_VSA1", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7",
                 "SlogP_VSA1", "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6",
                 "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
                 "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5",
                 "MQN1", "MQN2", "MQN3", "MQN4", "MQN5", "MQN6", "MQN7", "MQN8", "MQN9", "MQN10",
                 "MQN11", "MQN12", "MQN13", "MQN14", "MQN15", "MQN16", "MQN17", "MQN18", "MQN19",
                 "MQN20", "MQN21", "MQN22", "MQN23", "MQN24", "MQN25", "MQN26", "MQN27", "MQN28",
                 "MQN29", "MQN30", "MQN31", "MQN32", "MQN33", "MQN34", "MQN35", "MQN36", "MQN37",
                 "MQN38", "MQN39", "MQN40", "MQN41", "MQN42", "NHOHCount", "NOCount", "NumAliphaticRings",
                 "NumAromaticRings", "NumAliphaticHeterocycles", "NumAromaticHeterocycles",
                 "NumHeterocycles", "NumSaturatedRings", "NumSaturatedHeterocycles"]:
        try:
            desc_fns.append((name, getattr(rdMolDescriptors, f"Calc{name}")))
        except AttributeError:
            pass

    n_desc = len(desc_fns)
    arr = np.zeros((len(smiles_list), n_desc), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for j, (name, fn) in enumerate(desc_fns):
                    try:
                        val = float(fn(mol))
                        arr[i, j] = val if np.isfinite(val) else 0.0
                    except Exception:
                        arr[i, j] = 0.0
        except Exception:
            pass
    return arr


def normalize_descriptors(arr_tr, arr_val):
    """Normalize descriptors using training statistics."""
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
    print(f"  Descriptor dims after filtering: {keep.sum()}", flush=True)
    return arr_tr, arr_val, int(keep.sum())


# ── Training ──────────────────────────────────────────────────────────────────

def train_ablation_variant(model, tr_ds, val_ds, device, n_epochs=150, lr=1e-3, patience=30, batch_size=1024):
    """Train one ablation variant."""
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = n_epochs * len(tr_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        for x_gene, scf, x_fp, x_desc, ci, di, y_b in tr_loader:
            batch = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            y_b = y_b.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(*batch)
                loss = F.huber_loss(out, y_b, delta=1.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Evaluate
        rmse, r2, pearson, auroc, _ = evaluate_v11(model, val_loader, device)

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

    # Final metrics
    rmse, r2, pearson, auroc, _ = evaluate_v11(model, val_loader, device)
    return dict(Pearson=round(pearson, 4), R2=round(r2, 4), RMSE=round(rmse, 4), AUROC=round(auroc, 4))


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
    m = dict(Pearson=round(pearson, 4), R2=round(r2, 4), RMSE=round(rmse, 4), AUROC=round(auroc, 4))
    if name:
        print(f"  {name:35s}  Pearson={pearson:.4f}  R2={r2:.4f}  RMSE={rmse:.4f}  AUROC={auroc:.4f}", flush=True)
    return m


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("=" * 65, flush=True)
    print("  GDRNetV11 Ablation Study (GDSC)", flush=True)
    print("  6 variants × 1 seed (s42) × ~150 epochs", flush=True)
    print("=" * 65, flush=True)

    data = load_gdsc_data()

    # Build datasets
    tr_ds = V11Dataset(
        data["x_gene_tr"], data["x_scf_tr"], data["x_fp_tr"], data["x_desc_tr"],
        data["cell_idx_tr"], data["drug_idx_tr"], data["y_tr"]
    )
    val_ds = V11Dataset(
        data["x_gene_val"], data["x_scf_val"], data["x_fp_val"], data["x_desc_val"],
        data["cell_idx_val"], data["drug_idx_val"], data["y_val"]
    )

    # Ablation variants
    ABLATIONS = [
        ("Full Model",        GDRNetV11),
        ("w/o ID Embeddings", AblationNoID),
        ("w/o Cross Network", AblationNoCross),
        ("w/o Deep Network",  AblationNoDeep),
        ("w/o scF Emb",       AblationNoScF),
        ("w/o Drug Desc",     AblationNoDesc),
    ]

    all_results = {}

    for abl_name, model_cls in ABLATIONS:
        print(f"\n{'='*65}", flush=True)
        print(f"  Training: {abl_name}", flush=True)
        print(f"{'='*65}", flush=True)

        model = model_cls(
            n_genes=2000, scf_dim=3072, fp_bits=2048, n_desc=data["n_desc"],
            n_cells=data["n_cells"], n_drugs=data["n_drugs"],
            d_hidden=256, id_emb_dim=64, n_cross=3, cross_rank=64,
            n_deep=3, dropout=0.15,
        )
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params/1e6:.2f}M", flush=True)

        metrics = train_ablation_variant(model, tr_ds, val_ds, device)
        all_results[abl_name] = metrics

        elapsed = time.time() - t0
        print(f"  Result: Pearson={metrics['Pearson']:.4f}  elapsed={elapsed/60:.1f}min", flush=True)

    # ── Summary ──
    cmp = pd.DataFrame(all_results).T.sort_values("Pearson", ascending=False)
    cmp.to_csv(TABLES / "ablation_gdsc_comparison.csv")

    print(f"\n{'='*65}", flush=True)
    print(f"  GDSC Ablation Results (sorted by Pearson)", flush=True)
    print(f"{'='*65}", flush=True)
    print(cmp.to_string(), flush=True)

    # Compute drops vs Full
    full_p = all_results.get("Full Model", {}).get("Pearson", 0)
    print(f"\n  Component contribution (drop from Full):", flush=True)
    for name, row in cmp.iterrows():
        drop = full_p - row["Pearson"]
        pct = drop / full_p * 100 if full_p > 0 else 0
        print(f"    {name:25s}  P={row['Pearson']:.4f}  drop={drop:+.4f} ({pct:+.1f}%)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"  Saved -> {TABLES}/ablation_gdsc_comparison.csv", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()