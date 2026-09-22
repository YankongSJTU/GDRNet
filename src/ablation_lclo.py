# -*- coding: utf-8 -*-
"""
GDRNet Ablation Study — LCLO 5-Fold Protocol
===================================================
Re-runs ablation variants under strict LCLO 5-fold CV.
Headline metric: per-drug mean Pearson r (with bootstrapped CI).
Includes Wilcoxon signed-rank test vs Full Model.

Fine-grained (reuse from ablation.py):
  - w/o ID Embeddings (cell_id + drug_id)
  - w/o Cross Network (DCN v2)
  - w/o Deep Network (ResMLP)
  - w/o scF Emb (scFoundation)

Coarse-grained (new, to amplify differences):
  - No Interaction (concat + MLP only, no DCN cross)
  - w/o Cell Path (zero gene + scF + cell_emb, keep drug path only)

Usage:
  python src/ablation_lclo.py
  python src/ablation_lclo.py --epochs 300 --seeds 42 --gpus 0,2,4
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results/tables"
TABLES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass
# Force line buffering even when redirected to a file
if not sys.stdout.isatty():
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace', closefd=False)
if not sys.stderr.isatty():
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1, encoding='utf-8', errors='replace', closefd=False)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--gpus", type=str, default="0,2,4")
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--seeds", type=str, default="42")
parser.add_argument("--skip-coarse", action="store_true", help="skip coarse ablations")
args = parser.parse_args()


# ─── Fine-grained Ablation Variants (from ablation.py) ────────────────────

class AblationNoID(nn.Module):
    """Remove both cell_id and drug_id embeddings."""
    def __init__(self, base_model):
        super().__init__()
        self.gene_enc = base_model.gene_enc
        self.scf_enc = base_model.scf_enc
        self.fp_enc = base_model.fp_enc
        self.cross_layers = base_model.cross_layers
        self.deep_layers = base_model.deep_layers
        self.head = base_model.head
        self.input_proj = base_model.input_proj

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


class AblationNoCross(nn.Module):
    """Remove DCN v2 cross network (pass x0 directly)."""
    def __init__(self, base_model):
        super().__init__()
        self.gene_enc = base_model.gene_enc
        self.scf_enc = base_model.scf_enc
        self.cell_emb = base_model.cell_emb
        self.fp_enc = base_model.fp_enc
        self.drug_emb = base_model.drug_emb
        self.deep_layers = base_model.deep_layers
        self.head = base_model.head
        self.input_proj = base_model.input_proj

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


class AblationNoDeep(nn.Module):
    """Remove deep residual MLP (zeros instead)."""
    def __init__(self, base_model):
        super().__init__()
        self.gene_enc = base_model.gene_enc
        self.scf_enc = base_model.scf_enc
        self.cell_emb = base_model.cell_emb
        self.fp_enc = base_model.fp_enc
        self.drug_emb = base_model.drug_emb
        self.cross_layers = base_model.cross_layers
        self.head = base_model.head
        self.input_proj = base_model.input_proj

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


class AblationNoScF(nn.Module):
    """Remove scFoundation embeddings."""
    def __init__(self, base_model):
        super().__init__()
        self.gene_enc = base_model.gene_enc
        self.fp_enc = base_model.fp_enc
        self.cell_emb = base_model.cell_emb
        self.drug_emb = base_model.drug_emb
        self.cross_layers = base_model.cross_layers
        self.deep_layers = base_model.deep_layers
        self.head = base_model.head
        self.input_proj = base_model.input_proj

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


# ─── Coarse-grained Ablation Variants ──────────────────────────────────────

class AblationNoInteraction(nn.Module):
    """Remove DCN v2 cross AND deep network. Just concat + MLP head.
    Input: (896) -> proj(512) -> head(512+64=576 -> 128 -> 1)
    """
    def __init__(self, base_model):
        super().__init__()
        self.gene_enc = base_model.gene_enc
        self.scf_enc = base_model.scf_enc
        self.cell_emb = base_model.cell_emb
        self.fp_enc = base_model.fp_enc
        self.drug_emb = base_model.drug_emb
        self.head = base_model.head
        self.input_proj = base_model.input_proj

    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)
        h_fp = self.fp_enc(x_fp)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_did], dim=-1)
        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))
        # No interaction modules: cross=x0, deep=zeros(64) to match head input dim 576
        deep_dummy = torch.zeros(x0.size(0), 64, device=x0.device)
        return self.head(torch.cat([x0, deep_dummy], dim=-1)).squeeze(-1)


class AblationNoCellPath(nn.Module):
    """Remove entire cell feature path: zero gene + scF + cell_emb.
    Only drug FP + drug_id -> input_proj -> interaction -> predict.
    NOTE: input_proj expects 896 input, we pad cell_repr with zeros.
    """
    def __init__(self, base_model):
        super().__init__()
        self.fp_enc = base_model.fp_enc
        self.drug_emb = base_model.drug_emb
        self.cross_layers = base_model.cross_layers
        self.deep_layers = base_model.deep_layers
        self.head = base_model.head
        self.input_proj = base_model.input_proj
        self._cell_dim = 256 * 2 + 64  # 576

    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        cell_repr = torch.zeros(x_gene.size(0), self._cell_dim, device=x_gene.device)
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


# ─── Training Function ─────────────────────────────────────────────────────

# ─── Unseen Cell Embedding Fix ─────────────────────────────────────────

def _fix_unseen_cell_emb(model, tr_cell_idx, device):
    """Replace unseen cell embeddings with mean of trained ones for LCLO."""
    raw = model.module if hasattr(model, 'module') else model
    if not hasattr(raw, 'cell_emb'):
        return
    tr_cells = np.unique(tr_cell_idx)
    all_cells = np.arange(raw.cell_emb.num_embeddings)
    unseen = np.setdiff1d(all_cells, tr_cells)
    if len(unseen) == 0:
        return
    with torch.no_grad():
        trained_emb = raw.cell_emb.weight[tr_cells].mean(dim=0)
        for idx in unseen:
            raw.cell_emb.weight[idx] = trained_emb.clone()


def train_variant_fold(model, tr_ds, te_ds, device, epochs, batch_size,
                         lr, patience, weight_decay, tr_cell_idx=None):
    """Train an ablation variant on one LCLO fold."""
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(tr_loader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for batch in tr_loader:
            x_gene, scf, x_fp, x_desc, ci, di, y_b = batch
            batch_tensors = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            y_b = y_b.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(*batch_tensors)
                loss = F.huber_loss(out, y_b, delta=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in te_loader:
                x_gene, scf, x_fp, x_desc, ci, di, y_b = batch
                batch_tensors = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
                out = model(*batch_tensors)
                preds.extend(out.cpu().numpy())
                targets.extend(y_b.numpy())

        rmse = float(np.sqrt(np.mean((np.array(targets) - np.array(preds)) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Fix unseen cell embeddings before inference
    if tr_cell_idx is not None:
        _fix_unseen_cell_emb(model, tr_cell_idx, device)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in te_loader:
            x_gene, scf, x_fp, x_desc, ci, di, y_b = batch
            batch_tensors = [t.to(device) for t in [x_gene, scf, x_fp, x_desc, ci, di]]
            out = model(*batch_tensors)
            preds.extend(out.cpu().numpy())
            targets.extend(y_b.numpy())

    return np.array(preds, dtype=np.float32)


def make_variant(model_name, base_gdrnet):
    """Create an ablation variant from a trained GDRNet."""
    if model_name == "Full Model":
        return base_gdrnet
    elif model_name == "w/o ID Embeddings":
        return AblationNoID(base_gdrnet)
    elif model_name == "w/o Cross Network":
        return AblationNoCross(base_gdrnet)
    elif model_name == "w/o Deep Network":
        return AblationNoDeep(base_gdrnet)
    elif model_name == "w/o scF Emb":
        return AblationNoScF(base_gdrnet)
    elif model_name == "No Interaction":
        return AblationNoInteraction(base_gdrnet)
    elif model_name == "w/o Cell Path":
        return AblationNoCellPath(base_gdrnet)
    else:
        raise ValueError(f"Unknown variant: {model_name}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    t0_global = time.time()
    from data_loader import load_gdsc_lclo
    from eval_utils import (compute_full_eval, aggregate_fold_results,
                            format_results_table, paired_wilcoxon_test)
    from models.gdrnet import GDRNet, GDRNetDataset

    data = load_gdsc_lclo(n_splits=5, seed=42)
    device = f"cuda:{args.gpus.split(',')[0]}" if torch.cuda.is_available() else "cpu"
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Variant definitions
    fine_variants = ["Full Model", "w/o ID Embeddings", "w/o Cross Network",
                     "w/o Deep Network", "w/o scF Emb"]
    coarse_variants = ["No Interaction", "w/o Cell Path"]
    if args.skip_coarse:
        all_variants = fine_variants
    else:
        all_variants = fine_variants + coarse_variants

    # Storage
    all_results = {}   # {variant: [fold_eval, ...]}
    all_per_drug = {}  # {variant: {fold: per_drug_r_array}}

    for variant_name in all_variants:
        all_results[variant_name] = []
        all_per_drug[variant_name] = {}

    for fold_idx, (tr_mask, te_mask) in enumerate(data.folds):
        print(f"\n{'='*70}", flush=True)
        print(f"  ABLATION FOLD {fold_idx+1}/5  train={tr_mask.sum():,}  test={te_mask.sum():,}", flush=True)
        print(f"{'='*70}", flush=True)
        fold_t0 = time.time()

        y_te = data.y[te_mask]
        drug_te = data.drug_names[te_mask]
        cell_te = data.cell_ids[te_mask]

        X_desc_dummy = lambda n: np.zeros((n, 1), dtype=np.float32)
        tr_ds = GDRNetDataset(
            data.X_gene[tr_mask], data.X_scf[tr_mask], data.X_fp[tr_mask], X_desc_dummy(tr_mask.sum()),
            data.cell_idx[tr_mask], data.drug_idx[tr_mask], data.y[tr_mask])
        te_ds = GDRNetDataset(
            data.X_gene[te_mask], data.X_scf[te_mask], data.X_fp[te_mask], X_desc_dummy(te_mask.sum()),
            data.cell_idx[te_mask], data.drug_idx[te_mask], data.y[te_mask])

        for seed in seeds:
            print(f"  Seed={seed}:", flush=True)
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Train Full Model
            base_gdrnet = GDRNet(
                n_genes=2000, scf_dim=3072, fp_bits=2048,
                n_cells=data.n_cells, n_drugs=data.n_drugs,
                d_hidden=256, id_emb_dim=64,
                n_cross=3, cross_rank=64, n_deep=3, dropout=args.dropout,
            )

            # Train all variants from same initialization
            for vname in all_variants:
                print(f"    {vname:25s}...", end="", flush=True)
                torch.manual_seed(seed)
                np.random.seed(seed)

                if vname == "Full Model":
                    variant = GDRNet(
                        n_genes=2000, scf_dim=3072, fp_bits=2048,
                        n_cells=data.n_cells, n_drugs=data.n_drugs,
                        d_hidden=256, id_emb_dim=64,
                        n_cross=3, cross_rank=64, n_deep=3, dropout=args.dropout,
                    )
                else:
                    # Create a fresh base, then wrap it
                    fresh_base = GDRNet(
                        n_genes=2000, scf_dim=3072, fp_bits=2048,
                        n_cells=data.n_cells, n_drugs=data.n_drugs,
                        d_hidden=256, id_emb_dim=64,
                        n_cross=3, cross_rank=64, n_deep=3, dropout=args.dropout,
                    )
                    variant = make_variant(vname, fresh_base)

                vpreds = train_variant_fold(
                    variant, tr_ds, te_ds, device,
                    epochs=args.epochs, batch_size=args.batch,
                    lr=args.lr, patience=args.patience,
                    weight_decay=args.weight_decay,
                    tr_cell_idx=data.cell_idx[tr_mask],
                )

                eval_res = compute_full_eval(y_te, vpreds, drug_te, cell_te)
                all_results[vname].append(eval_res)
                all_per_drug[vname][fold_idx] = eval_res["per_drug"]["values"]

                print(f"  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                      f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)

        print(f"  Fold {fold_idx+1} done in {time.time()-fold_t0:.0f}s", flush=True)

    # ── Aggregate Results ───────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  ABLATION RESULTS (LCLO 5-Fold)", flush=True)
    print(f"{'='*70}", flush=True)

    aggregated = {}
    for vname in all_variants:
        aggregated[vname] = aggregate_fold_results(all_results[vname])

    # Save table
    table = format_results_table(aggregated)
    table.to_csv(TABLES / "table4_ablation_lclo.csv", index=False)
    print(f"\n  Table 4 saved to {TABLES / 'table4_ablation_lclo.csv'}", flush=True)
    print(table.to_string(index=False), flush=True)

    # Wilcoxon test (Full Model vs each variant)
    print(f"\n  Wilcoxon Signed-Rank Tests (per-drug r, Full Model vs variant):", flush=True)
    full_dr = np.concatenate([all_per_drug["Full Model"][f] for f in range(5) if len(all_per_drug["Full Model"][f]) > 0])
    for vname in all_variants[1:]:
        v_dr = np.concatenate([all_per_drug[vname][f] for f in range(5) if len(all_per_drug[vname][f]) > 0])
        # Paired: align by drug name across folds
        # Use fold-level per-drug means for paired comparison
        full_fold_means = []
        v_fold_means = []
        for f in range(5):
            fd = all_per_drug["Full Model"][f]
            vd = all_per_drug[vname][f]
            n = min(len(fd), len(vd))
            if n > 5:
                full_fold_means.extend(fd[:n].tolist())
                v_fold_means.extend(vd[:n].tolist())
        if len(full_fold_means) >= 10:
            stat, p = paired_wilcoxon_test(
                np.array(full_fold_means), np.array(v_fold_means))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"    {vname:25s}  p={p:.4e}  {sig}", flush=True)
        else:
            print(f"    {vname:25s}  (insufficient paired samples)", flush=True)

    elapsed = time.time() - t0_global
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
