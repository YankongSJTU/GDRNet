"""
Data preprocessing pipeline for organoid drug efficacy prediction.
Handles GDSC, CCLE/DepMap, and GEO organoid datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = Path("/export/home/kongyan/project/Organoid/data/raw")
PROC_DIR = Path("/export/home/kongyan/project/Organoid/data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─── GDSC ────────────────────────────────────────────────────────────────────

def load_gdsc_response(version=2):
    """Load GDSC drug response data. Returns DataFrame with IC50 (ln) and AUC."""
    fname = RAW_DIR / "gdsc" / f"GDSC{version}_fitted_dose_response.xlsx"
    print(f"Loading GDSC{version} drug response from {fname}...")
    df = pd.read_excel(fname)

    # Standardize column names
    col_map = {
        "CELL_LINE_NAME": "cell_line",
        "COSMIC_ID": "cosmic_id",
        "DRUG_NAME": "drug_name",
        "DRUG_ID": "drug_id",
        "LN_IC50": "ln_ic50",
        "AUC": "auc",
        "RMSE": "rmse",
        "Z_SCORE": "z_score",
        "TISSUE_NAME": "tissue",
        "TCGA_DESC": "cancer_type",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Remove low-quality fits
    if "rmse" in df.columns:
        df = df[df["rmse"] < 0.3]

    print(f"  GDSC{version}: {len(df):,} drug-cell pairs, "
          f"{df['cell_line'].nunique()} cell lines, "
          f"{df['drug_name'].nunique()} drugs")
    return df


def load_gdsc_expression():
    """Load GDSC RMA-normalized gene expression matrix."""
    expr_file = RAW_DIR / "gdsc" / "Cell_line_RMA_proc_basalExp.txt"
    if not expr_file.exists():
        # Try zip
        import zipfile
        zf = RAW_DIR / "gdsc" / "Cell_line_RMA_proc_basalExp.txt.zip"
        if zf.exists():
            with zipfile.ZipFile(zf) as z:
                z.extractall(RAW_DIR / "gdsc")

    print(f"Loading GDSC gene expression...")
    df = pd.read_csv(expr_file, sep="\t", index_col=0)
    # First row is gene name, first col is probe ID; second column onward are cell lines
    # Format: rows=genes, cols=cell lines with "DATA.COSMICID" header
    # Extract COSMIC ID from column names
    cols = df.columns.tolist()
    # Columns like "DATA.1240135"
    cosmic_ids = [c.replace("DATA.", "") for c in cols]
    df.columns = cosmic_ids
    df.index.name = "gene"
    print(f"  Expression: {df.shape[0]} genes x {df.shape[1]} cell lines")
    return df


def load_gdsc_drug_info():
    """Load GDSC drug annotations."""
    f = RAW_DIR / "gdsc" / "drug_annotations.csv"
    if f.exists():
        df = pd.read_csv(f)
        return df
    # Fallback: create minimal from response data
    return None


# ─── DepMap/CCLE ─────────────────────────────────────────────────────────────

def load_depmap_expression():
    """Load DepMap RNA expression (TPM log1p, protein coding genes)."""
    f = RAW_DIR / "ccle" / "OmicsExpression_TPM_log1p.csv"
    print(f"Loading DepMap gene expression...")
    df = pd.read_csv(f, index_col=0)
    # Rows = cell lines (ModelID), cols = genes (SYMBOL (ENTREZ))
    # Clean column names: "TSPAN6 (7105)" -> "TSPAN6"
    df.columns = [c.split(" (")[0] for c in df.columns]
    print(f"  DepMap expression: {df.shape[0]} cell lines x {df.shape[1]} genes")
    return df


def load_depmap_mutations():
    """Load DepMap somatic mutations as binary gene x cell line matrix."""
    f = RAW_DIR / "ccle" / "OmicsSomaticMutations.csv"
    print(f"Loading DepMap mutations...")
    df = pd.read_csv(f)
    # Pivot to binary matrix: cell line x gene
    if "ModelID" in df.columns and "HugoSymbol" in df.columns:
        mut_mat = df.groupby(["ModelID", "HugoSymbol"]).size().unstack(fill_value=0)
        mut_mat = (mut_mat > 0).astype(int)
        print(f"  Mutations: {mut_mat.shape[0]} cell lines x {mut_mat.shape[1]} genes")
        return mut_mat
    return df


def load_depmap_metadata():
    """Load DepMap cell line metadata."""
    f = RAW_DIR / "ccle" / "Model.csv"
    df = pd.read_csv(f)
    print(f"DepMap metadata: {len(df)} cell lines")
    return df


def load_prism_drug_response():
    """Load PRISM drug repurposing screen (log2 viability)."""
    # Try secondary screen first (more doses, more reliable)
    f = RAW_DIR / "ccle" / "PRISM_secondary_screen.csv"
    if not f.exists():
        f = RAW_DIR / "ccle" / "PRISM_primary_screen.csv"
    print(f"Loading PRISM drug response...")
    df = pd.read_csv(f, index_col=0)
    print(f"  PRISM: {df.shape[0]} cell lines x {df.shape[1]} compounds")
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def select_top_variance_genes(expr_df, n_genes=2000):
    """Select top N genes by variance across cell lines."""
    var = expr_df.var(axis=1)
    top_genes = var.nlargest(n_genes).index
    return expr_df.loc[top_genes]


def normalize_expression(expr_df, method="standard"):
    """Normalize gene expression matrix (genes x samples)."""
    expr_t = expr_df.T  # samples x genes
    if method == "standard":
        scaler = StandardScaler()
    elif method == "quantile":
        scaler = QuantileTransformer(output_distribution="normal", n_quantiles=500)
    else:
        return expr_df

    expr_norm = pd.DataFrame(
        scaler.fit_transform(expr_t),
        index=expr_t.index,
        columns=expr_t.columns
    )
    return expr_norm.T  # back to genes x samples


def remove_low_variance_features(df, threshold=0.01):
    """Remove near-zero variance features."""
    selector = VarianceThreshold(threshold=threshold)
    mask = selector.fit(df).get_support()
    return df.loc[:, mask]


# ─── Drug Features ────────────────────────────────────────────────────────────

def compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan (circular) fingerprints from SMILES strings."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import numpy as np
    except ImportError:
        print("[WARN] RDKit not available. Drug fingerprints skipped.")
        return None

    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(np.array(fp))
                valid_idx.append(i)
        except Exception:
            pass

    if not fps:
        return None, []

    return np.array(fps), valid_idx


def compute_drug_descriptors(smiles_list):
    """Compute physicochemical descriptors from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return None

    desc_names = [
        "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
        "TPSA", "NumRotatableBonds", "NumAromaticRings",
        "NumAliphaticRings", "FractionCSP3", "NumHeavyAtoms"
    ]
    descriptor_fns = {name: getattr(Descriptors, name) for name in desc_names}

    results = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                row = {name: fn(mol) for name, fn in descriptor_fns.items()}
            else:
                row = {name: np.nan for name in desc_names}
        except Exception:
            row = {name: np.nan for name in desc_names}
        results.append(row)

    return pd.DataFrame(results)


# ─── Dataset Integration ──────────────────────────────────────────────────────

def build_gdsc_dataset(n_genes=2000, response_col="ln_ic50", version=2):
    """
    Build integrated GDSC dataset for drug response prediction.

    Returns:
        X_cell: DataFrame (n_pairs x n_genes) - cell line expression features
        X_drug: DataFrame (n_pairs x n_drug_features) - drug features
        y: Series (n_pairs,) - drug response (ln_IC50 or AUC)
        meta: DataFrame - metadata (cell_line, drug_name, tissue, etc.)
    """
    print("\n" + "=" * 50)
    print("Building GDSC dataset...")
    print("=" * 50)

    # Load response
    resp = load_gdsc_response(version)

    # Load expression
    try:
        expr = load_gdsc_expression()  # genes x cell lines (COSMIC ID as columns)
        expr = select_top_variance_genes(expr, n_genes)
        # Normalize
        expr_norm = normalize_expression(expr, method="standard")
        print(f"  Selected {n_genes} high-variance genes")
    except Exception as e:
        print(f"  [WARN] Expression load failed: {e}")
        return None

    # Map COSMIC ID to cell line names
    # In response: cosmic_id; in expression: column = COSMIC ID
    resp["cosmic_str"] = resp["cosmic_id"].astype(str)
    expr_cols = expr_norm.columns.astype(str).tolist()
    valid_cosmic = set(expr_cols) & set(resp["cosmic_str"].unique())
    print(f"  Cell lines with both expression and response: {len(valid_cosmic)}")

    resp_filtered = resp[resp["cosmic_str"].isin(valid_cosmic)].copy()

    # Build feature matrix
    records = []
    for _, row in resp_filtered.iterrows():
        cid = row["cosmic_str"]
        if cid in expr_norm.columns:
            cell_features = expr_norm[cid].values
            records.append({
                "cosmic_id": cid,
                "drug_name": row.get("drug_name", ""),
                "drug_id": row.get("drug_id", ""),
                "ln_ic50": row.get("ln_ic50", np.nan),
                "auc": row.get("auc", np.nan),
                "tissue": row.get("tissue", ""),
                "cancer_type": row.get("cancer_type", ""),
                "cell_features": cell_features,
            })

    df_pairs = pd.DataFrame(records)
    df_pairs = df_pairs.dropna(subset=[response_col])

    gene_names = expr_norm.index.tolist()
    X_cell = pd.DataFrame(
        np.stack(df_pairs["cell_features"].values),
        columns=gene_names
    )

    meta = df_pairs[["cosmic_id", "drug_name", "drug_id", "tissue", "cancer_type"]].reset_index(drop=True)
    y = df_pairs[response_col].reset_index(drop=True)

    print(f"  Final dataset: {len(y):,} samples, {X_cell.shape[1]} cell features")
    return X_cell, y, meta, gene_names


def build_depmap_dataset(n_genes=2000):
    """Build DepMap/CCLE + PRISM dataset."""
    print("\n" + "=" * 50)
    print("Building DepMap/PRISM dataset...")
    print("=" * 50)

    try:
        expr = load_depmap_expression()  # cell lines x genes (index=ModelID)
        drug = load_prism_drug_response()  # cell lines x drugs
        meta = load_depmap_metadata()

        # Align cell lines
        common_cells = expr.index.intersection(drug.index)
        print(f"  Common cell lines: {len(common_cells)}")

        expr_aligned = expr.loc[common_cells]
        drug_aligned = drug.loc[common_cells]

        # Select top variance genes
        gene_var = expr_aligned.var(axis=0)
        top_genes = gene_var.nlargest(n_genes).index
        expr_sub = expr_aligned[top_genes]

        # Normalize
        scaler = StandardScaler()
        expr_norm = pd.DataFrame(
            scaler.fit_transform(expr_sub),
            index=expr_sub.index,
            columns=expr_sub.columns
        )

        print(f"  Expression: {expr_norm.shape}, Drug screen: {drug_aligned.shape}")
        return expr_norm, drug_aligned, meta

    except Exception as e:
        print(f"  [FAIL] {e}")
        return None


# ─── Save processed data ──────────────────────────────────────────────────────

def save_processed(obj, name):
    path = PROC_DIR / f"{name}.parquet"
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(path)
        print(f"  Saved: {path} ({obj.shape})")
    elif isinstance(obj, pd.Series):
        obj.to_frame(name).to_parquet(path)
        print(f"  Saved: {path} (len={len(obj)})")


def run_preprocessing():
    """Run full preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # GDSC pipeline
    result = build_gdsc_dataset(n_genes=2000, response_col="ln_ic50")
    if result:
        X_cell, y, meta, genes = result
        save_processed(X_cell, "gdsc_cell_features")
        save_processed(y, "gdsc_response_lnic50")
        save_processed(meta, "gdsc_metadata")
        print(f"\n[OK] GDSC dataset ready: {X_cell.shape[0]:,} pairs")

    # DepMap pipeline
    dep_result = build_depmap_dataset(n_genes=2000)
    if dep_result:
        expr_norm, drug_screen, meta_dep = dep_result
        save_processed(expr_norm, "depmap_expression_norm")
        save_processed(drug_screen, "depmap_drug_screen")
        save_processed(meta_dep, "depmap_metadata")
        print(f"\n[OK] DepMap dataset ready")

    print("\n[DONE] Preprocessing complete.")


if __name__ == "__main__":
    run_preprocessing()
