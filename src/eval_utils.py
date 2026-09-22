# -*- coding: utf-8 -*-
"""
Unified Evaluation Utilities for GDRNet LCLO Benchmarking
=============================================================
All models and scripts use this single source of truth for metrics.

Key functions:
  - make_lclo_folds(): GroupKFold LCLO split, saved for reproducibility
  - compute_per_drug_metrics(): Per-drug Pearson r distribution
  - compute_per_cell_metrics(): Per-cell Pearson r distribution
  - bootstrap_ci(): Bootstrapped 95% CI for any statistic
  - compute_full_eval(): Master evaluation returning all metrics
  - additive_baselines(): Drug-mean, Cell-mean, Two-way additive baselines
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

TABLES_DIR = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ─── LCLO Fold Generation ─────────────────────────────────────────────────

_fold_cache = {}


def make_lclo_folds(meta, n_splits=5, seed=42):
    """Generate Leave-Cell-Line-Out folds using GroupKFold.

    Parameters
    ----------
    meta : pd.DataFrame with 'ModelID' column
    n_splits : int
    seed : int  (used as random_state for GroupKFold shuffle)

    Returns
    -------
    list of (tr_mask, te_mask) boolean np.ndarray pairs
    """
    cache_key = (len(meta), n_splits, seed, meta["ModelID"].nunique())
    if cache_key in _fold_cache:
        return _fold_cache[cache_key]

    gkf = GroupKFold(n_splits=n_splits)
    groups = meta["ModelID"].values

    # GroupKFold doesn't shuffle by itself; we shuffle group labels
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)
    group_order = {g: i for i, g in enumerate(unique_groups)}
    ordered_groups = np.array([group_order[g] for g in groups])

    folds = []
    for tr_idx, te_idx in gkf.split(X=meta, y=meta["ModelID"].values, groups=ordered_groups):
        tr_mask = np.zeros(len(meta), dtype=bool)
        te_mask = np.zeros(len(meta), dtype=bool)
        tr_mask[tr_idx] = True
        te_mask[te_idx] = True
        folds.append((tr_mask, te_mask))

    # Verify no cell line overlap between train and test in any fold
    for i, (tr_m, te_m) in enumerate(folds):
        tr_cells = set(meta.loc[tr_m, "ModelID"].unique())
        te_cells = set(meta.loc[te_m, "ModelID"].unique())
        overlap = tr_cells & te_cells
        assert len(overlap) == 0, f"Fold {i}: {len(overlap)} cell lines overlap!"

    # Save for reproducibility
    save_path = TABLES_DIR / "lclo_fold_assignments.npy"
    np.save(save_path, np.array([(tr_m, te_m) for tr_m, te_m in folds], dtype=object))
    print(f"  LCLO folds saved to {save_path}")

    _fold_cache[cache_key] = folds
    return folds


# ─── Per-Entity Metrics ────────────────────────────────────────────────────

def compute_per_drug_metrics(y_true, y_pred, drug_names, min_samples=5):
    """Compute Pearson r for each drug across cell lines.

    Returns dict with:
      mean, median, std, q25, q75, iqr, n_drugs,
      per_entity: list of {name, r, n} dicts,
      values: np.ndarray of r values
    """
    results = []
    for drug in np.unique(drug_names):
        mask = drug_names == drug
        if mask.sum() < min_samples:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.std(yt) < 1e-10 or np.std(yp) < 1e-10:
            continue
        with np.errstate(invalid='ignore'):
            r = float(np.corrcoef(yt, yp)[0, 1])
        if np.isnan(r):
            continue
        results.append({"name": drug, "r": r, "n": int(mask.sum())})

    values = np.array([d["r"] for d in results])
    if len(values) == 0:
        return {"mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0,
                    "iqr": 0, "n_drugs": 0, "per_entity": [], "values": np.array([])}

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "n_drugs": len(values),
        "per_entity": results,
        "values": values,
    }


def compute_per_cell_metrics(y_true, y_pred, cell_ids, min_samples=5):
    """Compute Pearson r for each cell line across drugs.

    Same return format as compute_per_drug_metrics.
    """
    results = []
    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        if mask.sum() < min_samples:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.std(yt) < 1e-10 or np.std(yp) < 1e-10:
            continue
        with np.errstate(invalid='ignore'):
            r = float(np.corrcoef(yt, yp)[0, 1])
        if np.isnan(r):
            continue
        results.append({"name": cell, "r": r, "n": int(mask.sum())})

    values = np.array([d["r"] for d in results])
    if len(values) == 0:
        return {"mean": 0, "median": 0, "std": 0, "q25": 0, "q75": 0,
                    "iqr": 0, "n_cells": 0, "per_entity": [], "values": np.array([])}

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "n_cells": len(values),
        "per_entity": results,
        "values": values,
    }


# ─── Bootstrapped Confidence Intervals ─────────────────────────────────────

def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, alpha=0.05, seed=42):
    """Compute bootstrapped confidence interval for a statistic.

    Parameters
    ----------
    values : array-like
    stat_fn : callable, default np.mean
    n_boot : int
    alpha : float (default 0.05 for 95% CI)
    seed : int

    Returns
    -------
    (point_estimate, ci_lo, ci_hi)
    """
    values = np.asarray(values)
    if len(values) < 5:
        est = float(stat_fn(values))
        return (est, est, est)

    rng = np.random.RandomState(seed)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(float(stat_fn(sample)))

    boot_stats = np.array(boot_stats)
    point_est = float(stat_fn(values))
    ci_lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return (point_est, ci_lo, ci_hi)


def bootstrap_ci_per_drug(y_true, y_pred, drug_names, min_samples=5, n_boot=2000, seed=42):
    """Bootstrap CI for the mean of per-drug Pearson r values.

    Returns (mean, ci_lo, ci_hi)
    """
    per_drug = compute_per_drug_metrics(y_true, y_pred, drug_names, min_samples)
    if len(per_drug["values"]) == 0:
        return (0, 0, 0)
    return bootstrap_ci(per_drug["values"], stat_fn=np.mean, n_boot=n_boot, seed=seed)


def bootstrap_ci_per_cell(y_true, y_pred, cell_ids, min_samples=5, n_boot=2000, seed=42):
    """Bootstrap CI for the mean of per-cell Pearson r values.

    Returns (mean, ci_lo, ci_hi)
    """
    per_cell = compute_per_cell_metrics(y_true, y_pred, cell_ids, min_samples)
    if len(per_cell["values"]) == 0:
        return (0, 0, 0)
    return bootstrap_ci(per_cell["values"], stat_fn=np.mean, n_boot=n_boot, seed=seed)


# ─── AUROC at Multiple Thresholds ──────────────────────────────────────────

def auroc_at_threshold(y_true, y_pred, percentile):
    """Compute AUROC with sensitivity defined as y_true <= percentile threshold.
    """
    thr = np.percentile(y_true, percentile)
    labels = (y_true <= thr).astype(int)
    scores = -y_pred  # lower IC50 = more sensitive
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


# ─── Additive Main-Effect Baselines ────────────────────────────────────────

def additive_baselines(y, drug_names, cell_ids, tr_mask, te_mask):
    """Compute fold-specific additive baselines using only training statistics.

    NOTE: Under LCLO (leave-cell-line-out), test-fold cells are unseen in
    training, so Cell-Mean falls back to grand_mean for all test samples.
    This produces constant predictions and thus Pearson=0 and undefined R2.
    This is expected and demonstrates that Cell-Mean cannot generalize.

    Parameters
    ----------
    y : np.ndarray, full response values
    drug_names : np.ndarray, drug name per sample
    cell_ids : np.ndarray, cell ID per sample
    tr_mask, te_mask : boolean arrays

    Returns
    -------
    dict of {baseline_name: predictions_on_test_fold}
    """
    y_tr = y[tr_mask]
    drugs_tr = drug_names[tr_mask]
    cells_tr = cell_ids[tr_mask]

    grand_mean = np.mean(y_tr)

    # Drug-mean baseline
    drug_means = {}
    for d in np.unique(drug_names):
        mask = drugs_tr == d
        if mask.sum() > 0:
            drug_means[d] = np.mean(y_tr[mask])
        else:
            drug_means[d] = grand_mean

    # Cell-mean baseline
    cell_means = {}
    for c in np.unique(cells_tr):
        mask = cells_tr == c
        if mask.sum() > 0:
            cell_means[c] = np.mean(y_tr[mask])
        else:
            cell_means[c] = grand_mean

    drugs_te = drug_names[te_mask]
    cells_te = cell_ids[te_mask]

    pred_drug_mean = np.array([drug_means.get(d, grand_mean) for d in drugs_te])
    pred_cell_mean = np.array([cell_means.get(c, grand_mean) for c in cells_te])
    pred_two_way = np.array([
        grand_mean + (drug_means.get(d, grand_mean) - grand_mean) + (cell_means.get(c, grand_mean) - grand_mean)
        for d, c in zip(drugs_te, cells_te)
    ])

    return {
        "Drug-Mean": pred_drug_mean,
        "Cell-Mean": pred_cell_mean,
        "Two-Way-Additive": pred_two_way,
    }


# ─── Master Evaluation Function ────────────────────────────────────────────

def compute_full_eval(y_true, y_pred, drug_names, cell_ids,
                      min_samples=5, n_boot=2000, boot_seed=42):
    """Compute all evaluation metrics for a single fold's predictions.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    drug_names : np.ndarray of str
    cell_ids : np.ndarray of str
    min_samples : int, minimum samples for per-entity metrics
    n_boot : int, bootstrap iterations
    boot_seed : int

    Returns
    -------
    dict with keys:
      pooled_pearson, pooled_r2, pooled_rmse,
      auroc_20, auroc_30, auroc_50,
      per_drug: {mean, median, std, q25, q75, iqr, n_drugs, values, ci_lo, ci_hi},
      per_cell: {mean, median, std, q25, q75, iqr, n_cells, values, ci_lo, ci_hi},
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    # Pooled metrics (handle constant predictions gracefully)
    if np.std(y_pred) < 1e-10 or np.std(y_true) < 1e-10:
        pooled_pearson = 0.0
    else:
        with np.errstate(invalid='ignore'):
            r = np.corrcoef(y_true, y_pred)[0, 1]
            pooled_pearson = 0.0 if np.isnan(r) else float(r)
    pooled_r2 = float(r2_score(y_true, y_pred))
    pooled_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Multi-threshold AUROC
    auroc_20 = auroc_at_threshold(y_true, y_pred, 20)
    auroc_30 = auroc_at_threshold(y_true, y_pred, 30)
    auroc_50 = auroc_at_threshold(y_true, y_pred, 50)

    # Per-drug
    per_drug = compute_per_drug_metrics(y_true, y_pred, drug_names, min_samples)
    pd_mean, pd_ci_lo, pd_ci_hi = bootstrap_ci(
        per_drug["values"], stat_fn=np.mean, n_boot=n_boot, seed=boot_seed
    ) if len(per_drug["values"]) > 0 else (0, 0, 0)
    per_drug["ci_lo"] = pd_ci_lo
    per_drug["ci_hi"] = pd_ci_hi

    # Per-cell
    per_cell = compute_per_cell_metrics(y_true, y_pred, cell_ids, min_samples)
    pc_mean, pc_ci_lo, pc_ci_hi = bootstrap_ci(
        per_cell["values"], stat_fn=np.mean, n_boot=n_boot, seed=boot_seed
    ) if len(per_cell["values"]) > 0 else (0, 0, 0)
    per_cell["ci_lo"] = pc_ci_lo
    per_cell["ci_hi"] = pc_ci_hi

    return {
        "pooled_pearson": round(pooled_pearson, 4),
        "pooled_r2": round(pooled_r2, 4),
        "pooled_rmse": round(pooled_rmse, 4),
        "auroc_20": round(auroc_20, 4) if not np.isnan(auroc_20) else "nan",
        "auroc_30": round(auroc_30, 4) if not np.isnan(auroc_30) else "nan",
        "auroc_50": round(auroc_50, 4) if not np.isnan(auroc_50) else "nan",
        "per_drug": per_drug,
        "per_cell": per_cell,
    }


# ─── Aggregate Across Folds ────────────────────────────────────────────────

def aggregate_fold_results(fold_results_list):
    """Aggregate evaluation results across 5 folds.

    Parameters
    ----------
    fold_results_list : list of dicts from compute_full_eval()

    Returns
    -------
    dict with mean ± std (or CI) for each metric
    """
    n_folds = len(fold_results_list)

    def _mean_std(key):
        vals = [f[key] for f in fold_results_list if not isinstance(f[key], str)]
        if not vals:
            return ("nan", "nan")
        return (round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4) if n_folds > 1 else 0.0)

    # Pooled metrics: mean across folds
    p_pearson, p_pearson_std = _mean_std("pooled_pearson")
    p_r2, p_r2_std = _mean_std("pooled_r2")
    p_rmse, p_rmse_std = _mean_std("pooled_rmse")
    a20, a20_std = _mean_std("auroc_20")
    a30, a30_std = _mean_std("auroc_30")
    a50, a50_std = _mean_std("auroc_50")

    # Per-drug: collect ALL per-drug r values across folds, then compute distribution
    _pd_vals = [f["per_drug"]["values"] for f in fold_results_list
                if len(f["per_drug"]["values"]) > 0]
    all_pd_r = np.concatenate(_pd_vals) if _pd_vals else np.array([0.0])
    if len(all_pd_r) > 1:
        pd_mean, pd_ci_lo, pd_ci_hi = bootstrap_ci(all_pd_r, stat_fn=np.mean)
        pd_median = float(np.median(all_pd_r))
        pd_std = float(np.std(all_pd_r, ddof=1))
        pd_q25 = float(np.percentile(all_pd_r, 25))
        pd_q75 = float(np.percentile(all_pd_r, 75))
    else:
        pd_mean = pd_ci_lo = pd_ci_hi = pd_median = pd_std = pd_q25 = pd_q75 = 0.0

    # Per-cell: same
    _pc_vals = [f["per_cell"]["values"] for f in fold_results_list
                if len(f["per_cell"]["values"]) > 0]
    all_pc_r = np.concatenate(_pc_vals) if _pc_vals else np.array([0.0])
    if len(all_pc_r) > 1:
        pc_mean, pc_ci_lo, pc_ci_hi = bootstrap_ci(all_pc_r, stat_fn=np.mean)
        pc_median = float(np.median(all_pc_r))
        pc_std = float(np.std(all_pc_r, ddof=1))
        pc_q25 = float(np.percentile(all_pc_r, 25))
        pc_q75 = float(np.percentile(all_pc_r, 75))
    else:
        pc_mean = pc_ci_lo = pc_ci_hi = pc_median = pc_std = pc_q25 = pc_q75 = 0.0

    return {
        "pooled_pearson": p_pearson, "pooled_pearson_std": p_pearson_std,
        "pooled_r2": p_r2, "pooled_r2_std": p_r2_std,
        "pooled_rmse": p_rmse, "pooled_rmse_std": p_rmse_std,
        "auroc_20": a20, "auroc_20_std": a20_std,
        "auroc_30": a30, "auroc_30_std": a30_std,
        "auroc_50": a50, "auroc_50_std": a50_std,
        "per_drug_mean": round(pd_mean, 4), "per_drug_ci_lo": round(pd_ci_lo, 4),
        "per_drug_ci_hi": round(pd_ci_hi, 4),
        "per_drug_median": round(pd_median, 4), "per_drug_std": round(pd_std, 4),
        "per_drug_q25": round(pd_q25, 4), "per_drug_q75": round(pd_q75, 4),
        "per_drug_n": len(all_pd_r),
        "per_cell_mean": round(pc_mean, 4), "per_cell_ci_lo": round(pc_ci_lo, 4),
        "per_cell_ci_hi": round(pc_ci_hi, 4),
        "per_cell_median": round(pc_median, 4), "per_cell_std": round(pc_std, 4),
        "per_cell_q25": round(pc_q25, 4), "per_cell_q75": round(pc_q75, 4),
        "per_cell_n": len(all_pc_r),
    }


def format_results_table(all_model_results):
    """Format aggregated results into a publication-ready DataFrame.

    Parameters
    ----------
    all_model_results : dict of {model_name: aggregated_result_dict}

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, res in all_model_results.items():
        rows.append({
            "Model": name,
            "Pooled_Pearson": f"{res['pooled_pearson']:.4f} ± {res['pooled_pearson_std']:.4f}",
            "Pooled_R2": f"{res['pooled_r2']:.4f} ± {res['pooled_r2_std']:.4f}",
            "Pooled_RMSE": f"{res['pooled_rmse']:.4f} ± {res['pooled_rmse_std']:.4f}",
            "PerDrug_Mean_r": f"{res['per_drug_mean']:.4f} [{res['per_drug_ci_lo']:.4f}, {res['per_drug_ci_hi']:.4f}]",
            "PerDrug_Median_r": f"{res['per_drug_median']:.4f}",
            "PerDrug_IQR": f"[{res['per_drug_q25']:.4f}, {res['per_drug_q75']:.4f}]",
            "PerDrug_n": res["per_drug_n"],
            "PerCell_Mean_r": f"{res['per_cell_mean']:.4f} [{res['per_cell_ci_lo']:.4f}, {res['per_cell_ci_hi']:.4f}]",
            "PerCell_Median_r": f"{res['per_cell_median']:.4f}",
            "PerCell_n": res["per_cell_n"],
            "AUROC_20": f"{res['auroc_20']} ± {res['auroc_20_std']}",
            "AUROC_30": f"{res['auroc_30']} ± {res['auroc_30_std']}",
            "AUROC_50": f"{res['auroc_50']} ± {res['auroc_50_std']}",
        })
    return pd.DataFrame(rows)


# ─── Statistical Tests ──────────────────────────────────────────────────────

def paired_wilcoxon_test(r_values_a, r_values_b):
    """Wilcoxon signed-rank test for paired per-drug r values.

    Parameters
    ----------
    r_values_a, r_values_b : np.ndarray (same length, paired by drug)

    Returns
    -------
    (statistic, p_value)
    """
    if len(r_values_a) < 5:
        return (float("nan"), float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = stats.wilcoxon(r_values_a, r_values_b)
    return (float(stat), float(p))


# ─── Quick Self-Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing eval_utils...")
    np.random.seed(42)
    n = 1000
    y_true = np.random.randn(n).astype(np.float32)
    y_pred = y_true * 0.5 + np.random.randn(n).astype(np.float32) * 0.5
    drug_names = np.array([f"drug_{i % 50}" for i in range(n)])
    cell_ids = np.array([f"cell_{i // 10}" for i in range(n)])

    # Test full eval
    result = compute_full_eval(y_true, y_pred, drug_names, cell_ids)
    print(f"  Pooled Pearson: {result['pooled_pearson']:.4f}")
    print(f"  Per-drug mean r: {result['per_drug']['mean']:.4f} "
          f"CI=[{result['per_drug']['ci_lo']:.4f}, {result['per_drug']['ci_hi']:.4f}]")
    print(f"  Per-cell mean r: {result['per_cell']['mean']:.4f} "
          f"CI=[{result['per_cell']['ci_lo']:.4f}, {result['per_cell']['ci_hi']:.4f}]")
    print(f"  AUROC (20/30/50): {result['auroc_20']:.4f} / {result['auroc_30']:.4f} / {result['auroc_50']:.4f}")

    # Test additive baselines
    tr_mask = np.random.rand(n) < 0.8
    te_mask = ~tr_mask
    baselines = additive_baselines(y_true, drug_names, cell_ids, tr_mask, te_mask)
    for name, preds in baselines.items():
        r = np.corrcoef(y_true[te_mask], preds)[0, 1]
        print(f"  Baseline {name}: Pearson={r:.4f}")

    # Test bootstrap CI
    vals = np.random.randn(100)
    est, lo, hi = bootstrap_ci(vals, n_boot=1000)
    print(f"  Bootstrap CI: {est:.4f} [{lo:.4f}, {hi:.4f}]")

    print("  All tests passed!")
