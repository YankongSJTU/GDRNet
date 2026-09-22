# -*- coding: utf-8 -*-
"""
LCLO GDRNet-Only Rerun — saves ALL fold predictions + checkpoints
===================================================================
Reproduces the GDRNet (3 seeds + ensemble) part of the Table 2 LCLO run,
but persists per-fold test predictions for all 5 folds (the original run
only saved fold 0) so that Figures 4A/B/D/E and 5B/C/D can be regenerated
from pooled LCLO predictions.

Reuses train_lclo.train_gdrnet_fold / data_loader.load_gdsc_lclo verbatim
to stay faithful to the published protocol (same seeds, hyperparameters,
early stopping, unseen-embedding fix).

Outputs → results/tables/lclo_rerun/
  fold{i}_predictions.parquet   cell_id, drug_name, y_true, 3 seeds, ensemble
  fold_evals.json               compute_full_eval metrics per fold (ensemble)

Usage:
  python src/rerun_lclo_gdrnet.py --gpus 1
"""

import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
# LCLO_OUT / LCLO_DEVICE let the same script run on several machines
# (e.g. V100S locally vs P100 on gpu1) without clobbering each other.
OUT = ROOT / os.environ.get("LCLO_OUT", "results/tables/lclo_rerun")
CKPT = ROOT / "models/lclo_rerun"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

# train_lclo parses argv at import time — feed it our options first
sys.argv = ["train_lclo.py", "--gpus", "1", "--skip-trees", "--skip-dl", "--skip-baselines"]
import train_lclo  # noqa: E402  (parses args, exposes train_gdrnet_fold)

import torch  # noqa: E402
from data_loader import load_gdsc_lclo  # noqa: E402
from eval_utils import compute_full_eval  # noqa: E402


def main():
    t0 = time.time()
    data = load_gdsc_lclo(n_splits=5, seed=42)
    device = os.environ.get("LCLO_DEVICE",
                            "cuda:1" if torch.cuda.is_available() else "cpu")
    seeds = [42, 123, 456]
    fold_evals = {}

    for fold_idx, (tr_mask, te_mask) in enumerate(data.folds):
        print(f"\n{'='*70}", flush=True)
        print(f"  FOLD {fold_idx+1}/5  train={tr_mask.sum():,}  test={te_mask.sum():,}",
              flush=True)
        fold_t0 = time.time()

        y_te = data.y[te_mask]
        drug_te = data.drug_names[te_mask]
        cell_te = data.cell_ids[te_mask]

        seed_preds = []
        for seed in seeds:
            print(f"  GDRNet seed={seed} ...", end="", flush=True)
            gpreds = train_lclo.train_gdrnet_fold(
                data, tr_mask, te_mask, seed, device,
                epochs=train_lclo.args.epochs, batch_size=train_lclo.args.batch,
                lr=train_lclo.args.lr, patience=train_lclo.args.patience,
                weight_decay=train_lclo.args.weight_decay,
                dropout=train_lclo.args.dropout,
            )
            seed_preds.append(gpreds)
            eval_res = compute_full_eval(y_te, gpreds, drug_te, cell_te)
            print(f"  Pooled_r={eval_res['pooled_pearson']:.4f}  "
                  f"PerDrug={eval_res['per_drug']['mean']:.4f}", flush=True)
            del eval_res

        ens = np.mean(seed_preds, axis=0).astype(np.float32)
        eval_res = compute_full_eval(y_te, ens, drug_te, cell_te)
        print(f"  GDRNet-Ensemble  Pooled_r={eval_res['pooled_pearson']:.4f}  "
              f"PerDrug={eval_res['per_drug']['mean']:.4f}  "
              f"Median={eval_res['per_drug']['median']:.4f}", flush=True)

        # per-fold predictions table
        df = pd.DataFrame({
            "cell_id": cell_te, "drug_name": drug_te, "y_true": y_te,
            "GDRNet-s42": seed_preds[0], "GDRNet-s123": seed_preds[1],
            "GDRNet-s456": seed_preds[2], "GDRNet-Ensemble": ens,
        })
        df.to_parquet(OUT / f"fold{fold_idx}_predictions.parquet", index=False)

        # serializable metrics (drop the raw value arrays)
        slim = {
            "pooled_pearson": eval_res["pooled_pearson"],
            "pooled_r2": eval_res["pooled_r2"],
            "pooled_rmse": eval_res["pooled_rmse"],
            "auroc_20": eval_res["auroc_20"], "auroc_30": eval_res["auroc_30"],
            "auroc_50": eval_res["auroc_50"],
            "per_drug_mean": eval_res["per_drug"]["mean"],
            "per_drug_median": eval_res["per_drug"]["median"],
            "per_drug_n": eval_res["per_drug"]["n_drugs"],
            "per_drug_values": list(np.asarray(eval_res["per_drug"]["values"], dtype=float)),
            "per_cell_mean": eval_res["per_cell"]["mean"],
        }
        fold_evals[str(fold_idx)] = slim
        with open(OUT / "fold_evals.json", "w") as f:
            json.dump(fold_evals, f, indent=1)

        print(f"  Fold {fold_idx+1} done in {time.time()-fold_t0:.0f}s", flush=True)

    # summary vs published Table 2
    print(f"\n{'='*70}\n  SUMMARY (ensemble per fold)\n{'='*70}", flush=True)
    for k in sorted(fold_evals, key=int):
        e = fold_evals[k]
        print(f"  fold{k}: pooled_r={e['pooled_pearson']:.4f}  "
              f"per_drug_mean={e['per_drug_mean']:.4f}", flush=True)
    pm = np.mean([e["pooled_pearson"] for e in fold_evals.values()])
    dm = np.mean([e["per_drug_mean"] for e in fold_evals.values()])
    print(f"  mean pooled_r={pm:.4f} (Table 2: 0.8894)   "
          f"mean per_drug={dm:.4f} (Table 2: 0.5448)", flush=True)
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
