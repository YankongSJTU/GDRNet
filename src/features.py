"""
Feature engineering for drug efficacy prediction.
Generates drug molecular features and cell line genomic features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW_DIR = Path("/export/home/kongyan/project/Organoid/data/raw")
PROC_DIR = Path("/export/home/kongyan/project/Organoid/data/processed")
EXT_DIR = Path("/export/home/kongyan/project/Organoid/data/external")
EXT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Drug Features ────────────────────────────────────────────────────────────

def get_drug_smiles_from_pubchem(drug_names):
    """Fetch SMILES from PubChem for a list of drug names."""
    import requests, time
    smiles_dict = {}
    for name in drug_names:
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                smi = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
                smiles_dict[name] = smi
            time.sleep(0.2)  # rate limit
        except Exception:
            pass
    return smiles_dict


def build_drug_fingerprint_matrix(drug_smiles_df, radius=2, n_bits=2048):
    """
    Build Morgan fingerprint matrix from drug SMILES.
    Input: DataFrame with columns ['drug_name', 'smiles']
    Returns: DataFrame (drugs x fingerprint_bits)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("[WARN] RDKit not installed. Using dummy drug features.")
        # Return one-hot encoded drug names as fallback
        return None

    fps = {}
    for _, row in drug_smiles_df.iterrows():
        name = row["drug_name"]
        smi = row.get("smiles", row.get("SMILES", ""))
        if not smi or pd.isna(smi):
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps[name] = np.array(fp)
        except Exception:
            continue

    if not fps:
        return None

    fp_df = pd.DataFrame(fps).T
    fp_df.columns = [f"fp_{i}" for i in range(n_bits)]
    print(f"Drug fingerprints: {fp_df.shape[0]} drugs x {fp_df.shape[1]} bits")
    return fp_df


def build_drug_onehot(drug_names):
    """Fallback: one-hot encode drug names."""
    unique = sorted(set(drug_names))
    drug_to_idx = {d: i for i, d in enumerate(unique)}
    mat = np.zeros((len(drug_names), len(unique)), dtype=np.float32)
    for i, name in enumerate(drug_names):
        if name in drug_to_idx:
            mat[i, drug_to_idx[name]] = 1.0
    df = pd.DataFrame(mat, columns=[f"drug_{d}" for d in unique])
    return df


# ─── Cell Line Features ───────────────────────────────────────────────────────

def reduce_expression_pca(expr_df, n_components=128):
    """
    Reduce gene expression dimensionality via PCA.
    Input: DataFrame (n_samples x n_genes)
    Returns: PCA-reduced DataFrame (n_samples x n_components)
    """
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(expr_df.values)
    df_pca = pd.DataFrame(
        reduced,
        index=expr_df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA: {n_components} components explain {explained:.1%} variance")
    return df_pca, pca


def build_combined_cell_features(expr_df, mut_df=None, cn_df=None):
    """
    Combine expression + mutation + copy number features.
    All DataFrames: index = cell line ID.
    """
    common = expr_df.index
    if mut_df is not None:
        common = common.intersection(mut_df.index)
    if cn_df is not None:
        common = common.intersection(cn_df.index)

    parts = [expr_df.loc[common]]

    if mut_df is not None:
        # Use cancer driver genes for mutations
        cancer_genes = [
            "TP53", "KRAS", "PIK3CA", "APC", "PTEN", "BRAF", "EGFR",
            "MYC", "RB1", "CDKN2A", "CDH1", "VHL", "BRCA1", "BRCA2",
            "MLH1", "SMAD4", "STK11", "ALK", "MET", "ERBB2", "NRAS",
            "IDH1", "IDH2", "FLT3", "NPM1", "DNMT3A", "TET2"
        ]
        avail_genes = [g for g in cancer_genes if g in mut_df.columns]
        if avail_genes:
            mut_sub = mut_df.loc[common, avail_genes].fillna(0)
            mut_sub.columns = [f"mut_{g}" for g in avail_genes]
            parts.append(mut_sub)

    if cn_df is not None:
        # Use top variable copy number features
        cn_aligned = cn_df.loc[common]
        cn_var = cn_aligned.var(axis=0)
        top_cn_genes = cn_var.nlargest(500).index
        cn_sub = cn_aligned[top_cn_genes].fillna(0)
        cn_sub.columns = [f"cn_{g}" for g in top_cn_genes]
        parts.append(cn_sub)

    combined = pd.concat(parts, axis=1)
    print(f"Combined cell features: {combined.shape[0]} cell lines x {combined.shape[1]} features")
    return combined


# ─── Build Final ML Dataset ───────────────────────────────────────────────────

def build_ml_dataset_gdsc(n_genes=2000, use_fingerprints=True):
    """
    Build final (X, y) dataset for ML from GDSC data.
    X = [cell_expression_features | drug_features]
    y = ln_IC50
    """
    # Load preprocessed data
    X_cell = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    y = pd.read_parquet(PROC_DIR / "gdsc_response_lnic50.parquet").iloc[:, 0]
    meta = pd.read_parquet(PROC_DIR / "gdsc_metadata.parquet")

    if use_fingerprints:
        # Load drug SMILES
        smiles_file = RAW_DIR / "drug_info" / "gdsc_drugs_smiles.csv"
        if smiles_file.exists():
            drug_info = pd.read_csv(smiles_file)
            # Normalize column names
            drug_info.columns = drug_info.columns.str.lower()
            smiles_col = [c for c in drug_info.columns if "smiles" in c]
            name_col = [c for c in drug_info.columns if "name" in c or "drug" in c]
            if smiles_col and name_col:
                drug_info = drug_info.rename(columns={
                    name_col[0]: "drug_name",
                    smiles_col[0]: "smiles"
                })
                fp_mat = build_drug_fingerprint_matrix(drug_info[["drug_name", "smiles"]])
                if fp_mat is not None:
                    # Merge fingerprints
                    drug_fps = meta["drug_name"].map(lambda d: fp_mat.loc[d].values if d in fp_mat.index else None)
                    valid = drug_fps.notna()
                    X_cell_v = X_cell[valid].reset_index(drop=True)
                    X_drug_v = pd.DataFrame(
                        np.stack(drug_fps[valid].values),
                        columns=fp_mat.columns
                    )
                    y_v = y[valid].reset_index(drop=True)
                    meta_v = meta[valid].reset_index(drop=True)
                    X = pd.concat([X_cell_v, X_drug_v], axis=1)
                    print(f"Dataset with fingerprints: {X.shape[0]:,} x {X.shape[1]}")
                    return X, y_v, meta_v

    # Fallback: one-hot encode drugs
    print("Using drug one-hot encoding (no fingerprints available)")
    X_drug_oh = build_drug_onehot(meta["drug_name"].values)
    X = pd.concat([X_cell.reset_index(drop=True), X_drug_oh], axis=1)
    return X, y, meta
