"""
Baseline models for drug efficacy prediction:
- Random Forest
- XGBoost / LightGBM
- ElasticNet (linear baseline)

All models predict ln_IC50 (regression) with optional binary classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

RESULTS_DIR = Path("/export/home/kongyan/project/Organoid/results")
MODELS_DIR = Path("/export/home/kongyan/project/Organoid/models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def regression_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Pearson": np.corrcoef(y_true, y_pred)[0, 1],
    }


def classification_metrics(y_true, y_score, threshold_pct=30):
    """
    Convert continuous IC50 to binary (sensitive/resistant) and compute AUROC.
    Sensitive = bottom `threshold_pct`% IC50 values.
    """
    threshold = np.percentile(y_true, threshold_pct)
    y_bin = (y_true <= threshold).astype(int)
    # Invert score: lower IC50 = more sensitive = higher score
    y_score_inv = -y_score
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"AUROC": np.nan, "AUPR": np.nan}
    return {
        "AUROC": roc_auc_score(y_bin, y_score_inv),
        "AUPR": average_precision_score(y_bin, y_score_inv),
    }


def cross_validate_model(model, X, y, groups=None, n_splits=5, model_name="model"):
    """
    Run cross-validation. Uses GroupKFold if groups provided (e.g., by cell line).
    Returns per-fold metrics and out-of-fold predictions.
    """
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        split_args = (X, y, groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_args = (X, y)

    y_pred_oof = np.zeros(len(y))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(*split_args)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_pred_oof[val_idx] = y_pred

        metrics = regression_metrics(y_val, y_pred)
        cls_metrics = classification_metrics(y_val.values, y_pred)
        metrics.update(cls_metrics)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)
        print(f"  Fold {fold+1}: RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}, "
              f"Pearson={metrics['Pearson']:.3f}, AUROC={metrics['AUROC']:.3f}")

    df_metrics = pd.DataFrame(fold_metrics)
    mean_metrics = df_metrics.drop(columns="fold").mean()
    std_metrics = df_metrics.drop(columns="fold").std()

    print(f"\n  {model_name} CV Summary:")
    for col in ["RMSE", "R2", "Pearson", "AUROC", "AUPR"]:
        print(f"    {col}: {mean_metrics[col]:.3f} ± {std_metrics[col]:.3f}")

    return y_pred_oof, df_metrics, mean_metrics


def train_elasticnet(X, y, groups=None):
    """ElasticNet linear baseline."""
    print("\n[ElasticNet Baseline]")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    return cross_validate_model(model, X_scaled, y, groups, model_name="ElasticNet")


def train_random_forest(X, y, groups=None, n_estimators=200):
    """Random Forest regression."""
    print("\n[Random Forest]")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    return cross_validate_model(model, X, y, groups, model_name="RandomForest")


def train_xgboost(X, y, groups=None):
    """XGBoost regression."""
    if not HAS_XGB:
        print("[XGBoost] Not installed, skipping.")
        return None, None, None
    print("\n[XGBoost]")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    return cross_validate_model(model, X, y, groups, model_name="XGBoost")


def train_lightgbm(X, y, groups=None):
    """LightGBM regression."""
    if not HAS_LGB:
        print("[LightGBM] Not installed, skipping.")
        return None, None, None
    print("\n[LightGBM]")
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    return cross_validate_model(model, X, y, groups, model_name="LightGBM")


def compute_shap_importance(model, X, model_type="xgb", top_n=30, save_prefix="model"):
    """Compute SHAP feature importance."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] SHAP not installed.")
        return None

    print(f"\nComputing SHAP values for {model_type}...")
    if model_type in ("xgb", "lgb", "rf"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
        columns=["mean_abs_shap"]
    ).sort_values("mean_abs_shap", ascending=False)

    # Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "figures" / f"{save_prefix}_shap_summary.png", dpi=150)
    plt.close()

    shap_df.to_csv(RESULTS_DIR / "tables" / f"{save_prefix}_shap_importance.csv")
    print(f"  Top 10 features:\n{shap_df.head(10).to_string()}")
    return shap_df


def run_baseline_pipeline(X, y, meta, save_prefix="gdsc"):
    """
    Run all baseline models and save results.
    Uses drug_name as group for GroupKFold to prevent data leakage.
    """
    print("\n" + "=" * 60)
    print("BASELINE MODEL PIPELINE")
    print("=" * 60)
    print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")

    # Group by cell line to simulate real prediction scenario
    groups = meta["cosmic_id"].values if "cosmic_id" in meta.columns else None

    all_results = {}

    # 1. ElasticNet
    _, _, en_metrics = train_elasticnet(X, y, groups)
    if en_metrics is not None:
        all_results["ElasticNet"] = en_metrics

    # 2. Random Forest
    _, _, rf_metrics = train_random_forest(X, y, groups, n_estimators=100)
    if rf_metrics is not None:
        all_results["RandomForest"] = rf_metrics

    # 3. XGBoost
    xgb_preds, _, xgb_metrics = train_xgboost(X, y, groups)
    if xgb_metrics is not None:
        all_results["XGBoost"] = xgb_metrics

    # 4. LightGBM
    _, _, lgb_metrics = train_lightgbm(X, y, groups)
    if lgb_metrics is not None:
        all_results["LightGBM"] = lgb_metrics

    # Save comparison table
    if all_results:
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv(RESULTS_DIR / "tables" / f"{save_prefix}_baseline_comparison.csv")
        print(f"\n{'='*60}")
        print("Model Comparison Summary:")
        print(results_df[["RMSE", "R2", "Pearson", "AUROC"]].round(3).to_string())
        print(f"{'='*60}")

    return all_results
