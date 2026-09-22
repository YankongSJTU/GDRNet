# -*- coding: utf-8 -*-
"""
GDRNet LCLO 5-Fold Training Orchestrator
=============================================
Trains all models (additive baselines, tree models, GDRNet, GraphDRP,
DrugCell, DeepCDR) under identical LCLO 5-fold cross-validation.

Usage:
  python src/train_lclo.py
  python src/train_lclo.py --gpus 0,2,4 --epochs 300 --skip-dl
  python src/train_lclo.py --skip-trees --skip-baselines  # only DL models
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
parser.add_argument("--seeds", type=str, default="42,123,456")
parser.add_argument("--skip-trees", action="store_true")
parser.add_argument("--skip-dl", action="store_true")
parser.add_argument("--skip-baselines", action="store_true")
parser.add_argument("--skip-gdrnet", action="store_true")
args = parser.parse_args()


# ─── Unseen Cell Embedding Fix ─────────────────────────────────────────

def _fix_unseen_embeddings(model, tr_cell_idx, device):
    """Replace unseen cell/drug embeddings with mean of trained ones.

    In LCLO, test cell lines never appear in training, so their embeddings
    remain at random initialization. Replace them with the trained mean.
    """
    # Get the underlying model (unwrap DataParallel if needed)
    raw = model.module if hasattr(model, 'module') else model
    tr_cells = np.unique(tr_cell_idx)
    all_cells = np.arange(raw.cell_emb.num_embeddings)
    unseen = np.setdiff1d(all_cells, tr_cells)
    if len(unseen) == 0:
        return
    with torch.no_grad():
        trained_emb = raw.cell_emb.weight[tr_cells].mean(dim=0)
        for idx in unseen:
            raw.cell_emb.weight[idx] = trained_emb.clone()


# ─── GDRNet Fold Training (reuses gdrnet.py training loop) ──────────────────

def train_gdrnet_fold(data, tr_mask, te_mask, seed, device, epochs, batch_size,
                      lr, patience, weight_decay, dropout):
    """Train GDRNet on one LCLO fold, returning test predictions."""
    from models.gdrnet import GDRNet, GDRNetDataset
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = GDRNet(
        n_genes=2000, scf_dim=3072, fp_bits=2048,
        n_cells=data.n_cells, n_drugs=data.n_drugs,
        d_hidden=256, id_emb_dim=64,
        n_cross=3, cross_rank=64, n_deep=3, dropout=dropout,
    )

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    if len(gpu_ids) > 1 and device.startswith("cuda"):
        model = nn.DataParallel(model, device_ids=gpu_ids)

    X_desc_dummy = lambda n: np.zeros((n, 1), dtype=np.float32)

    tr_ds = GDRNetDataset(
        data.X_gene[tr_mask], data.X_scf[tr_mask], data.X_fp[tr_mask], X_desc_dummy(tr_mask.sum()),
        data.cell_idx[tr_mask], data.drug_idx[tr_mask], data.y[tr_mask])
    te_ds = GDRNetDataset(
        data.X_gene[te_mask], data.X_scf[te_mask], data.X_fp[te_mask], X_desc_dummy(te_mask.sum()),
        data.cell_idx[te_mask], data.drug_idx[te_mask], data.y[te_mask])

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

        # Evaluate
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

    # Fix unseen cell embeddings: replace with mean of trained cells
    # In LCLO, test cell embeddings were never updated (stayed at random init)
    _fix_unseen_embeddings(model, data.cell_idx[tr_mask], device)

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


# ─── Tree Model Training ────────────────────────────────────────────────────

def get_tree_models():
    """Return dict of {name: (model_instance, fit_params)}."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet

    models = {
        "ElasticNet": (
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42),
            {},
        ),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=100, max_depth=20, min_samples_leaf=5,
                n_jobs=-1, random_state=42),
            {},
        ),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = (
            xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                reg_alpha=0.5, reg_lambda=0.5,
                n_jobs=-1, random_state=42, verbosity=0),
            {},
        )
    except ImportError:
        print("  [WARN] XGBoost not available, skipping")
    try:
        import lightgbm as lgb
        models["LightGBM"] = (
            lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05,
                num_leaves=63, max_depth=10,
                subsample=0.8, colsample_bytree=0.6,
                min_child_samples=10, reg_alpha=0.5, reg_lambda=0.5,
                n_jobs=-1, random_state=42, verbose=-1),
            {},
        )
    except ImportError:
        print("  [WARN] LightGBM not available, skipping")
    return models


# ─── DL Baseline Registry ──────────────────────────────────────────────────

def get_dl_configs():
    """Return list of (name, model_class, train_fn, dataset_class)."""
    configs = []
    try:
        from models.graphdrp import GraphDRP, train_graphdrp, GraphDRPDataset
        configs.append(("GraphDRP", GraphDRP, train_graphdrp, GraphDRPDataset))
    except ImportError as e:
        print(f"  [WARN] GraphDRP import failed: {e}")
    try:
        from models.drugcell import DrugCellSimplified, train_drugcell, DrugCellDataset
        configs.append(("DrugCell", DrugCellSimplified, train_drugcell, DrugCellDataset))
    except ImportError as e:
        print(f"  [WARN] DrugCell import failed: {e}")
    try:
        from models.deepcdr import DeepCDR, train_deepcdr, DeepCDRDataset
        configs.append(("DeepCDR", DeepCDR, train_deepcdr, DeepCDRDataset))
    except ImportError as e:
        print(f"  [WARN] DeepCDR import failed: {e}")
    return configs


def train_dl_baseline_fold(model_cls, train_fn, dataset_cls,
                            X_gene_tr, X_fp_tr, y_tr,
                            X_gene_te, X_fp_te, y_te_len,
                            device, epochs, batch_size, lr, patience,
                            weight_decay, dropout, model_name):
    """Train a DL baseline on one fold."""
    tr_ds = dataset_cls(X_gene_tr, X_fp_tr, y_tr)
    te_ds = dataset_cls(X_gene_te, X_fp_te, np.zeros(y_te_len, dtype=np.float32))
    model = model_cls(n_genes=2000, fp_bits=2048, dropout=dropout)
    _, preds = train_fn(
        model, tr_ds, te_ds,
        model_name=model_name, n_epochs=epochs, batch_size=batch_size,
        lr=lr, patience=patience, weight_decay=weight_decay, device=device,
    )
    return preds


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    t0_global = time.time()
    from data_loader import load_gdsc_lclo
    from eval_utils import (compute_full_eval, additive_baselines,
                            aggregate_fold_results, format_results_table)

    # Load data
    data = load_gdsc_lclo(n_splits=5, seed=42)
    device = f"cuda:{args.gpus.split(',')[0]}" if torch.cuda.is_available() else "cpu"
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    dl_configs = get_dl_configs()

    # Storage
    all_results = {}     # {model_name: [fold_eval_dict, ...]}
    all_predictions = {} # {model_name: {fold_idx: preds_array}}

    n_folds = len(data.folds)
    for fold_idx, (tr_mask, te_mask) in enumerate(data.folds):
        print(f"\n{'='*70}", flush=True)
        print(f"  FOLD {fold_idx+1}/{n_folds}  "
              f"train={tr_mask.sum():,}  test={te_mask.sum():,}", flush=True)
        print(f"{'='*70}", flush=True)
        fold_t0 = time.time()

        y_te = data.y[te_mask]
        drug_te = data.drug_names[te_mask]
        cell_te = data.cell_ids[te_mask]
        fold_results = {}

        # ── 1. Additive baselines (instant) ──
        if not args.skip_baselines:
            print("  [1/4] Additive baselines...", flush=True)
            baselines = additive_baselines(data.y, data.drug_names, data.cell_ids, tr_mask, te_mask)
            for bname, bpreds in baselines.items():
                eval_res = compute_full_eval(y_te, bpreds, drug_te, cell_te)
                fold_results[bname] = eval_res
                all_predictions.setdefault(bname, {})[fold_idx] = bpreds
                print(f"    {bname:25s}  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                      f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)

        # ── 2. Tree baselines ──
        if not args.skip_trees:
            print("  [2/4] Tree baselines...", flush=True)
            X_tr_flat = np.concatenate([data.X_gene[tr_mask], data.X_fp[tr_mask]], axis=1)
            X_te_flat = np.concatenate([data.X_gene[te_mask], data.X_fp[te_mask]], axis=1)
            y_tr = data.y[tr_mask]

            for tname, (tmodel, _) in get_tree_models().items():
                tmodel.fit(X_tr_flat, y_tr)
                tpreds = tmodel.predict(X_te_flat).astype(np.float32)
                eval_res = compute_full_eval(y_te, tpreds, drug_te, cell_te)
                fold_results[tname] = eval_res
                all_predictions.setdefault(tname, {})[fold_idx] = tpreds
                print(f"    {tname:25s}  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                      f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)

        # ── 3. GDRNet (multiple seeds) ──
        if not args.skip_gdrnet:
            print(f"  [3/4] GDRNet ({len(seeds)} seeds)...", flush=True)
            gdrnet_fold_preds = []
            for seed in seeds:
                print(f"    GDRNet seed={seed}...", end="", flush=True)
                gpreds = train_gdrnet_fold(
                    data, tr_mask, te_mask, seed, device,
                    epochs=args.epochs, batch_size=args.batch,
                    lr=args.lr, patience=args.patience,
                    weight_decay=args.weight_decay, dropout=args.dropout,
                )
                eval_res = compute_full_eval(y_te, gpreds, drug_te, cell_te)
                fold_results[f"GDRNet-s{seed}"] = eval_res
                gdrnet_fold_preds.append(gpreds)
                all_predictions.setdefault(f"GDRNet-s{seed}", {})[fold_idx] = gpreds
                print(f"  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                      f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)

            # Ensemble
            ens_preds = np.mean(gdrnet_fold_preds, axis=0).astype(np.float32)
            eval_res = compute_full_eval(y_te, ens_preds, drug_te, cell_te)
            fold_results["GDRNet-Ensemble"] = eval_res
            all_predictions.setdefault("GDRNet-Ensemble", {})[fold_idx] = ens_preds
            print(f"    {'GDRNet-Ensemble':25s}  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                  f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)

        # ── 4. DL baselines ──
        if not args.skip_dl and dl_configs:
            print("  [4/4] DL baselines...", flush=True)
            for dl_name, Cls, tr_fn, Ds in dl_configs:
                print(f"    {dl_name}...", end="", flush=True)
                dpreds = train_dl_baseline_fold(
                    Cls, tr_fn, Ds,
                    data.X_gene[tr_mask], data.X_fp[tr_mask], data.y[tr_mask],
                    data.X_gene[te_mask], data.X_fp[te_mask], te_mask.sum(),
                    device, args.epochs, args.batch, args.lr, args.patience,
                    args.weight_decay, args.dropout, dl_name,
                )
                eval_res = compute_full_eval(y_te, dpreds, drug_te, cell_te)
                fold_results[dl_name] = eval_res
                all_predictions.setdefault(dl_name, {})[fold_idx] = dpreds
                print(f"  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                      f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)

        # Collect fold results
        for mname, mres in fold_results.items():
            all_results.setdefault(mname, []).append(mres)

        print(f"  Fold {fold_idx+1} done in {time.time()-fold_t0:.0f}s", flush=True)

    # ── Aggregate and Save ──────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print(f"  AGGREGATING RESULTS ACROSS {n_folds} FOLDS", flush=True)
    print(f"{'='*70}", flush=True)

    aggregated = {}
    for mname, fold_evals in all_results.items():
        aggregated[mname] = aggregate_fold_results(fold_evals)

    # Format and save Table 2
    table = format_results_table(aggregated)
    table.to_csv(TABLES / "table2_gdsc_lclo.csv", index=False)
    print(f"\n  Table 2 saved to {TABLES / 'table2_gdsc_lclo.csv'}", flush=True)
    print(table.to_string(index=False), flush=True)

    # Save predictions (fold 0 only for CSV)
    pred_rows = {"cell_id": data.cell_ids[data.folds[0][1]],
                 "drug_name": data.drug_names[data.folds[0][1]],
                 "y_true": data.y[data.folds[0][1]]}
    for mname, fold_preds in all_predictions.items():
        if 0 in fold_preds:
            pred_rows[mname] = fold_preds[0]
    pd.DataFrame(pred_rows).to_csv(TABLES / "lclo_predictions_fold0.csv", index=False)

    elapsed = time.time() - t0_global
    print(f"\n  Total time: {elapsed/60:.1f} min", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
