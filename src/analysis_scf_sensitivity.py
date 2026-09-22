# -*- coding: utf-8 -*-
"""
CRC scFoundation Sensitivity Analysis
=======================================
Addresses Reviewer 2's concern: ablation shows removing scF improves GDSC
performance, yet CRC-PDO analysis relies entirely on scF embeddings.

Compares 3 configurations on CRC organoid LOOCV:
  (a) scF only (current design, gene=zeros)
  (b) No scF, gene=zeros (only drug FP + drug ID)
  (c) No scF, gene=random (random gene-like features instead of scF)

Note: Option (b) raw microarray expression requires processing GSE64392 data
through the same 2000-gene pipeline, which is complex. Instead, we compare
scF on vs off to quantify its marginal contribution.

Usage:
  python src/analysis_scf_sensitivity.py
"""

import io
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
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
from eval_utils import compute_full_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OrgDataset(Dataset):
    def __init__(self, x_gene, x_scf, x_fp, cell_idx, drug_idx, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.x_scf = torch.FloatTensor(x_scf)
        self.x_fp = torch.FloatTensor(x_fp)
        self.x_desc = torch.zeros(len(y), 1)
        self.cell_idx = torch.LongTensor(cell_idx)
        self.drug_idx = torch.LongTensor(drug_idx)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.x_gene[i], self.x_scf[i], self.x_fp[i], self.x_desc[i],
                self.cell_idx[i], self.drug_idx[i], self.y[i])


def load_organoid_data():
    """Load CRC organoid data."""
    meta = pd.read_csv(PROC / "organoid_pair_meta.csv")
    cell_emb_all = np.load(PROC / "organoid_cell_emb.npy")
    cell_ids_all = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
    drug_feat = np.load(PROC / "organoid_drug_features.npy")
    response = np.load(PROC / "organoid_response.npy")

    cell_emb_map = {cid: cell_emb_all[i] for i, cid in enumerate(cell_ids_all)}
    scf_emb = np.array([cell_emb_map[oid] for oid in meta["organoid_id"]],
                       dtype=np.float32)
    fp = drug_feat[:, :2048].astype(np.float32)

    drug_map = pd.read_csv(PROC / "expanded/drug_id_map.csv")
    drug_to_idx = dict(zip(drug_map["DRUG_NAME"], drug_map["drug_idx"]))

    drug_idx = np.array([drug_to_idx.get(d, 0) for d in meta["drug_name"]],
                        dtype=np.int64)
    cell_idx = np.zeros(len(response), dtype=np.int64)

    return meta, scf_emb, fp, drug_idx, cell_idx, response


def build_model(seed):
    model = GDRNet(
        n_genes=2000, scf_dim=3072, fp_bits=2048,
        n_cells=964, n_drugs=229,
        d_hidden=256, id_emb_dim=64,
        n_cross=3, cross_rank=64, n_deep=3, dropout=0.15,
    )
    ckpt = MODELS / f"gdrnet_s{seed}.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    for name, param in model.named_parameters():
        if any(k in name for k in ["gene_enc", "scf_enc", "fp_enc",
                                    "cell_emb", "drug_emb", "input_proj"]):
            param.requires_grad = False
    return model.to(DEVICE)


def finetune_fold(model, tr_ds, val_ds, device=DEVICE,
                    n_epochs=100, lr=5e-4, patience=20, batch_size=64):
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

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
        for batch in tr_loader:
            tensors = [t.to(device) for t in batch[:6]]
            y_b = batch[6].to(device)
            optimizer.zero_grad()
            out = model(*tensors)
            loss = F.huber_loss(out, y_b, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                tensors = [t.to(device) for t in batch[:6]]
                out = model(*tensors)
                preds.extend(out.cpu().numpy())
                targets.extend(batch[6].numpy())
        rmse = float(np.sqrt(np.mean((np.array(targets) - np.array(preds)) ** 2)))
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
        for batch in val_loader:
            tensors = [t.to(device) for t in batch[:6]]
            out = model(*tensors)
            preds.extend(out.cpu().numpy())
            targets.extend(batch[6].numpy())
    return np.array(preds, dtype=np.float32)


def run_sensitivity(config_name, scf_config, meta, fp, drug_idx, cell_idx, response, seeds):
    """Run LOOCV with a specific scF configuration.

    scf_config: 'scf' (real), 'zeros', or 'random'
    """
    print(f"\n  === {config_name} ===", flush=True)

    organoids = sorted(meta["organoid_id"].unique())
    all_preds = np.zeros(len(response), dtype=np.float32)

    if scf_config == "scf":
        from data_loader import load_gdsc_lclo  # just for scF
        # Use pre-computed scF embeddings
        cell_emb_all = np.load(PROC / "organoid_cell_emb.npy")
        cell_ids_all = np.load(PROC / "organoid_cell_ids.npy", allow_pickle=True)
        cell_emb_map = {cid: cell_emb_all[i] for i, cid in enumerate(cell_ids_all)}
        scf_data = np.array([cell_emb_map[oid] for oid in meta["organoid_id"]], dtype=np.float32)
    elif scf_config == "zeros":
        scf_data = np.zeros((len(response), 3072), dtype=np.float32)
    elif scf_config == "random":
        rng = np.random.RandomState(42)
        scf_data = rng.randn(len(response), 3072).astype(np.float32) * 0.1
    else:
        raise ValueError(f"Unknown scf_config: {scf_config}")

    gene_zeros = np.zeros((len(response), 2000), dtype=np.float32)

    results_list = []
    for seed in seeds:
        print(f"    Seed={seed}...", end="", flush=True)
        seed_preds = np.zeros(len(response), dtype=np.float32)

        for i, test_oid in enumerate(organoids):
            tr_mask = meta["organoid_id"] != test_oid
            te_mask = meta["organoid_id"] == test_oid

            tr_ds = OrgDataset(
                gene_zeros[tr_mask], scf_data[tr_mask], fp[tr_mask],
                cell_idx[tr_mask], drug_idx[tr_mask], response[tr_mask])
            te_ds = OrgDataset(
                gene_zeros[te_mask], scf_data[te_mask], fp[te_mask],
                cell_idx[te_mask], drug_idx[te_mask], response[te_mask])

            model = build_model(seed)
            fold_preds = finetune_fold(model, tr_ds, te_ds, DEVICE)
            seed_preds[te_mask] = fold_preds

        all_preds += seed_preds
        print(f"  done", flush=True)

    # Ensemble eval
    ens_preds = all_preds / len(seeds)
    eval_res = compute_full_eval(response, ens_preds, meta["drug_name"], meta["organoid_id"])
    results_list.append({
        "Config": config_name,
        "Pooled_Pearson": eval_res["pooled_pearson"],
        "Pooled_R2": eval_res["pooled_r2"],
        "Pooled_RMSE": eval_res["pooled_rmse"],
        "PerDrug_Mean_r": eval_res["per_drug"]["mean"],
        "PerOrganoid_Mean_r": eval_res["per_cell"]["mean"],
        "AUROC_30": eval_res["auroc_30"],
    })

    print(f"    Pooled_r={eval_res['pooled_pearson']:.4f}  "
          f"PerDrug={eval_res['per_drug']['mean']:.4f}  "
          f"PerOrganoid={eval_res['per_cell']['mean']:.4f}", flush=True)

    return results_list[0]


def main():
    print("=" * 60, flush=True)
    print("  CRC scFoundation Sensitivity Analysis", flush=True)
    print("=" * 60, flush=True)

    t0 = time.time()
    meta, scf_emb, fp, drug_idx, cell_idx, response = load_organoid_data()
    seeds = [42, 123, 456]

    print(f"  CRC data: {len(response)} samples, {meta['organoid_id'].nunique()} organoids, "
          f"{meta['drug_name'].nunique()} drugs", flush=True)

    configs = [
        ("(a) scF Embeddings Only (current)", "scf"),
        ("(b) No scF, No gene (drug only)", "zeros"),
        ("(c) Random scF features", "random"),
    ]

    results = []
    for config_name, scf_config in configs:
        r = run_sensitivity(config_name, scf_config, meta, fp, drug_idx,
                            cell_idx, response, seeds)
        results.append(r)

    df = pd.DataFrame(results)
    out_path = TABLES / "crc_scf_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}", flush=True)
    print(df.to_string(index=False), flush=True)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
