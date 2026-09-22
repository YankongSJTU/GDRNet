# -*- coding: utf-8 -*-
"""
GDRNet-guided drug prioritization for organoid-based preclinical screening
==========================================================================
Application-oriented validation: for each organoid, rank all candidate drugs
by predicted lnIC50 / sensitivity (lower = more sensitive), then evaluate how
well the top-ranked candidates match the experimentally validated responses.

Baselines compared:
  * GDRNet-Ensemble : per-organoid predictions from the LOOCV runs
  * Drug-Mean (LOO): rank drugs by the mean response across the OTHER
        organoids of the same cancer type (ignore individual variation)
  * Random          : uniform random ranking (fixed seed, 100 repeats)

Metrics (per organoid, lower response = more sensitive):
  * Top-1 / Top-3 / Top-5 hit rate : is the true most-sensitive drug among
        the first k ranked candidates
  * Spearman rho                    : ranking correlation pred vs. response
  * NDCG@5                          : ranking quality weighted by sensitivity
  * Regret@1                        : |response(top-1 predicted) - response(true best)|
        (drop in achieved sensitivity when only the single top candidate is tested)

Outputs:
  results/tables/prioritization_summary.csv      (per dataset x method)
  results/tables/prioritization_per_organoid.csv (per organoid x method)
  results/tables/prioritization_hitcurve.csv     (cumulative hit rate vs k)
  results/tables/prioritization_examples.csv     (representative organoids)

Usage:
  python src/compute_drug_prioritization.py
"""

import argparse
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

DATASETS = {
    "CRC":  dict(file="crc_loocv_predictions.csv",  id="organoid_id", response="response"),
    "PDAC": dict(file="pdac_loocv_predictions.csv", id="organoid_id", response="response"),
    "BLCA": dict(file="blca_loocv_predictions.csv", id="organoid_id", response="response"),
}
SEED = 42
N_RANDOM = 100


def ndcg_at_k(response, order, k=5):
    """NDCG for a response where lower values indicate greater sensitivity."""
    k = min(k, len(order))
    # Shift gains to non-negative relevance, preserving sensitivity ordering.
    gains = response.max() - response
    ideal = np.argsort(gains)[::-1]
    dcg = sum(gains[order[i]] / np.log2(i + 2) for i in range(k))
    idcg = sum(gains[ideal[i]] / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def compute_per_organoid(dataset, file, idcol, respcol):
    d = pd.read_csv(TABLES / file)
    rows = []
    hitcurve_rows = []
    for oid, g in d.groupby(idcol):
        if len(g) < 5:
            continue
        y = g[respcol].to_numpy(float)
        pred = g["pred_ensemble"].to_numpy(float)
        drugs = g["drug_name"].to_numpy()

        # --- GDRNet ranking (lower pred = more sensitive) ---
        order_g = np.argsort(pred).tolist()
        top1_g = drugs[order_g[0]]
        true_best = drugs[np.argmin(y)]
        rho_g, _ = stats.spearmanr(pred, y)

        # --- Drug-Mean (LOO): rank by mean response of drug across other orgs ---
        # mean over all other organoids of the SAME dataset
        others = d[d[idcol] != oid]
        m = others.groupby("drug_name")[respcol].mean().reindex(drugs).to_numpy(float)
        order_dm = np.argsort(m).tolist()
        rho_dm, _ = stats.spearmanr(m, y)

        # --- Random baselines ---
        rng = np.random.default_rng(SEED)
        rho_r = np.zeros(N_RANDOM)
        for r in range(N_RANDOM):
            perm = rng.permutation(len(y))
            rho_r[r], _ = stats.spearmanr(pred[perm], y)

        # --- cumulative hit curve: is true best drug within first k preds ---
        cur = {"dataset": dataset, idcol: oid, "n_drugs": len(g)}
        for k in range(1, len(g) + 1):
            cur[f"G_hit_k{k}"] = int(true_best in drugs[order_g[:k]])
            cur[f"D_hit_k{k}"] = int(true_best in drugs[order_dm[:k]])
            cur[f"R_hit_k{k}"] = float(np.mean([
                int(true_best in drugs[rng.permutation(len(drugs))[:k]])
                for _ in range(N_RANDOM)]))
        hitcurve_rows.append(cur)

        rows.append({
            "dataset": dataset, idcol: oid, "n_drugs": len(g),
            "true_best": true_best, "true_best_response": float(y[np.argmin(y)]),
            # GDRNet
            "G_pred_top1": top1_g, "G_top1_hit": top1_g == true_best,
            "G_top3_hit": true_best in drugs[order_g[:3]],
            "G_top5_hit": true_best in drugs[order_g[:5]],
            "G_spearman": rho_g,
            "G_ndcg5": ndcg_at_k(y, order_g, 5),
            "G_regret1": float(abs(y[order_g[0]] - y[np.argmin(y)])),
            "G_pred_top1_response": float(y[order_g[0]]),
            # Drug-Mean (LOO)
            "D_top1_hit": drugs[order_dm[0]] == true_best,
            "D_top3_hit": true_best in drugs[order_dm[:3]],
            "D_top5_hit": true_best in drugs[order_dm[:5]],
            "D_spearman": rho_dm,
            "D_ndcg5": ndcg_at_k(y, order_dm, 5),
            "D_regret1": float(abs(y[order_dm[0]] - y[np.argmin(y)])),
            "D_pred_top1_response": float(y[order_dm[0]]),
            # Random
            "R_spearman_mean": float(rho_r.mean()),
        })
    return pd.DataFrame(rows), pd.DataFrame(hitcurve_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(TABLES))
    args = parser.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    hitcols = ["top1_hit", "top3_hit", "top5_hit"]
    for dataset, cfg in DATASETS.items():
        p, hc = compute_per_organoid(dataset, cfg["file"], cfg["id"], cfg["response"])
        p.to_csv(out / f"prioritization_per_organoid_{dataset}.csv", index=False)
        hc.to_csv(out / f"prioritization_hitcurve_{dataset}.csv", index=False)

        # --- 100-repeat random Top-1/3/5 hit (conditional: best drug index) ---
        rng = np.random.default_rng(SEED)
        rand_top = {k: [] for k in hitcols}
        d = pd.read_csv(TABLES / cfg["file"]); idcol = cfg["id"]; respcol = cfg["response"]
        n_org = p.shape[0]
        for oid, g in d.groupby(idcol):
            if len(g) < 5: continue
            y = g[respcol].to_numpy(float)
            true_best_idx = int(np.argmin(y)); n = len(y)
            for r in range(N_RANDOM):
                perm = rng.permutation(n)
                pos = np.where(perm == true_best_idx)[0][0]
                rand_top["top1_hit"].append(pos == 0)
                rand_top["top3_hit"].append(pos in (0, 1, 2))
                rand_top["top5_hit"].append(pos in range(5))
        rec = {"dataset": dataset, "n_organoids": n_org,
               "G_top1": p["G_top1_hit"].mean(), "G_top3": p["G_top3_hit"].mean(), "G_top5": p["G_top5_hit"].mean(),
               "D_top1": p["D_top1_hit"].mean(), "D_top3": p["D_top3_hit"].mean(), "D_top5": p["D_top5_hit"].mean(),
               "R_top1": np.mean(rand_top["top1_hit"]), "R_top3": np.mean(rand_top["top3_hit"]), "R_top5": np.mean(rand_top["top5_hit"]),
               "G_spearman": p["G_spearman"].mean(), "D_spearman": p["D_spearman"].mean(),
               "G_ndcg5": p["G_ndcg5"].mean(), "D_ndcg5": p["D_ndcg5"].mean(),
               "G_regret1": p["G_regret1"].mean(), "D_regret1": p["D_regret1"].mean(),
               "GvsD_spearman_p": stats.wilcoxon(p["G_spearman"], p["D_spearman"]).pvalue
               if p.shape[0] > 3 else float("nan"),
               "GvsD_ndcg5_p": stats.wilcoxon(p["G_ndcg5"], p["D_ndcg5"]).pvalue
               if p.shape[0] > 3 else float("nan"),
               "GvsD_regret_p": stats.wilcoxon(p["G_regret1"], p["D_regret1"]).pvalue
               if p.shape[0] > 3 else float("nan")}
        parts_summary = [rec]
        df_sum = pd.DataFrame(parts_summary)
        df_sum.to_csv(out / f"prioritization_summary_{dataset}.csv", index=False)
        print(f"\n=== {dataset} (n={n_org} organoids) ===")
        print(df_sum.T.to_string())

    # combined summary
    all_sum = []
    for dataset in DATASETS:
        all_sum.append(pd.read_csv(out / f"prioritization_summary_{dataset}.csv"))
    comb = pd.concat(all_sum, ignore_index=True)
    comb.to_csv(out / "prioritization_summary.csv", index=False)
    print("\nSaved ->", comb.shape)


if __name__ == "__main__":
    main()