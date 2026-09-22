# -*- coding: utf-8 -*-
"""
Mutation Subgroup Analysis
=============================
Analyzes whether GDRNet prediction errors correlate with TP53, KRAS, BRAF
mutation status in GDSC cell lines.

Input:  lclo_predictions_fold0.csv (from train_lclo.py)
Output: results/tables/mutation_subgroup_analysis.csv

Usage:
  python src/analysis_mutation.py
  python src/analysis_mutation.py --predictions results/tables/lclo_predictions_fold0.csv
"""

import argparse
import io
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/ccle"
TABLES = ROOT / "results/tables"

sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except (AttributeError, io.UnsupportedOperation):
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", type=str,
                    default=str(TABLES / "lclo_predictions_fold0.csv"))
parser.add_argument("--mutation-file", type=str,
                    default=str(RAW / "OmicsSomaticMutations.csv"))
parser.add_argument("--genes", type=str, default="TP53,KRAS,BRAF")
parser.add_argument("--model", type=str, default="GDRNet-Ensemble")
args = parser.parse_args()


def load_mutation_matrix(mutation_file, genes_of_interest):
    """Build binary mutation matrix (cell x gene) from CCLE somatic mutations.

    Filters to HIGH/MODERATE impact variants only.
    Returns DataFrame with ModelID as index, gene columns as bool.
    """
    print(f"  Loading mutations from {mutation_file}...", flush=True)
    cols = ["ModelID", "HugoSymbol", "VepImpact"]
    mut = pd.read_csv(mutation_file, usecols=cols, low_memory=False)

    # Filter to damaging mutations
    damaging = mut[mut["VepImpact"].isin(["HIGH", "MODERATE"])]
    print(f"    {len(mut):,} total variants, {len(damaging):,} HIGH/MODERATE", flush=True)

    # Binary pivot: 1 if any damaging variant in gene for cell line
    mut_binary = damaging.groupby(["ModelID", "HugoSymbol"]).size().unstack(fill_value=0)
    mut_binary = (mut_binary > 0).astype(int)

    # Filter to genes of interest
    available = [g for g in genes_of_interest if g in mut_binary.columns]
    missing = [g for g in genes_of_interest if g not in mut_binary.columns]
    if missing:
        print(f"    [WARN] Genes not found: {missing}", flush=True)

    print(f"    Cell lines with mutations: " + ", ".join(
        f"{g}: {mut_binary[g].sum()}" for g in available), flush=True)

    return mut_binary[available]


def mutation_subgroup_analysis(predictions_file, mutation_matrix, model_name):
    """For each gene, compare prediction errors between mutant vs wild-type.
    """
    preds = pd.read_csv(predictions_file)

    # Merge with mutation data
    preds = preds.merge(mutation_matrix, left_on="cell_id", right_index=True, how="inner")
    preds["error"] = np.abs(preds["y_true"] - preds[model_name])
    preds["sq_error"] = preds["error"] ** 2

    print(f"  Matched {len(preds):,} samples with mutation data", flush=True)

    results = []
    gene_cols = [c for c in preds.columns if c in mutation_matrix.columns]

    for gene in gene_cols:
        mutant = preds[preds[gene] == 1]
        wildtype = preds[preds[gene] == 0]

        if len(mutant) < 10 or len(wildtype) < 10:
            results.append({
                "Gene": gene,
                "N_mutant": len(mutant),
                "N_wildtype": len(wildtype),
                "MeanAE_mutant": np.nan, "MeanAE_wildtype": np.nan,
                "MedianAE_mutant": np.nan, "MedianAE_wildtype": np.nan,
                "RMSE_mutant": np.nan, "RMSE_wildtype": np.nan,
                "Pearson_mutant": np.nan, "Pearson_wildtype": np.nan,
                "MannWhitney_U": np.nan, "p_value": np.nan,
                "significant": "N/A (too few)",
            })
            continue

        # Metrics per group
        mae_mut = np.mean(mutant["error"])
        mae_wt = np.mean(wildtype["error"])
        rmse_mut = np.sqrt(np.mean(mutant["sq_error"]))
        rmse_wt = np.sqrt(np.mean(wildtype["sq_error"]))

        # Per-group Pearson (across all samples in group)
        r_mut = np.corrcoef(mutant["y_true"], mutant[model_name])[0, 1] if np.std(mutant[model_name]) > 1e-10 else 0
        r_wt = np.corrcoef(wildtype["y_true"], wildtype[model_name])[0, 1] if np.std(wildtype[model_name]) > 1e-10 else 0

        # Statistical test
        u_stat, p_val = stats.mannwhitneyu(mutant["error"], wildtype["error"], alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        results.append({
            "Gene": gene,
            "N_mutant": len(mutant),
            "N_wildtype": len(wildtype),
            "MeanAE_mutant": round(mae_mut, 4),
            "MeanAE_wildtype": round(mae_wt, 4),
            "MedianAE_mutant": round(np.median(mutant["error"]), 4),
            "MedianAE_wildtype": round(np.median(wildtype["error"]), 4),
            "RMSE_mutant": round(rmse_mut, 4),
            "RMSE_wildtype": round(rmse_wt, 4),
            "Pearson_mutant": round(float(r_mut), 4),
            "Pearson_wildtype": round(float(r_wt), 4),
            "MannWhitney_U": round(float(u_stat), 1),
            "p_value": f"{p_val:.2e}",
            "significant": sig,
        })

        print(f"    {gene}: mutant={len(mutant):,} wt={len(wildtype):,}  "
              f"MAE={mae_mut:.4f} vs {mae_wt:.4f}  p={p_val:.2e} {sig}", flush=True)

    return pd.DataFrame(results)


def main():
    print("=" * 60, flush=True)
    print("  Mutation Subgroup Analysis", flush=True)
    print("=" * 60, flush=True)

    genes = args.genes.split(",")
    mut_matrix = load_mutation_matrix(args.mutation_file, genes)

    result_df = mutation_subgroup_analysis(args.predictions, mut_matrix, args.model)

    out_path = TABLES / "mutation_subgroup_analysis.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}", flush=True)
    print(result_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
