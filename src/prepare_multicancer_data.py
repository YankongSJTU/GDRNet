# -*- coding: utf-8 -*-
"""
Prepare Multi-Cancer Organoid Data for GDRNet Fine-tuning
=========================================================
Sources:
  1. PDAC organoids (GSE194249, Chen et al. 2022, N=39, 64 drugs)
     Expression: FPKM RNA-seq
     Drug response: Normalized AUC → 1-AUC (sensitivity)

  2. Bladder cancer organoids (GSE103990, Lee et al. 2018, N=11, 50 drugs)
     Expression: Normalized counts RNA-seq
     Drug response: LogIC50 from Supplementary Table S3

Outputs (saved to data/processed/geo/):
  pdac_cell_emb.npy, pdac_pair_meta.csv, pdac_response.npy, ...
  blca_cell_emb.npy, blca_pair_meta.csv, blca_response.npy, ...
"""

import sys, warnings, gc, gzip
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT     = Path("/export/home/kongyan/project/Organoid")
PROC_DIR = ROOT / "data/processed"
GEO_PROC = PROC_DIR / "geo"
RAW_DIR  = ROOT / "data/raw/geo"
EXT_DIR  = ROOT / "data/external"
SCF_DIR  = ROOT / "external/scFoundation/model"

sys.path.insert(0, str(ROOT / "GDRNet" / "src"))
sys.path.insert(0, str(SCF_DIR))
sys.path.insert(0, str(SCF_DIR / "pretrainmodels"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

GEO_PROC.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 32


# ── Gene lists ────────────────────────────────────────────────────────────────

def load_gene_list():
    df = pd.read_csv(SCF_DIR / "OS_scRNA_gene_index.19264.tsv", sep="\t", header=0)
    return list(df["gene_name"])

def load_gdsc_gene_list():
    df = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    return list(df.columns)


# ── Expression normalization for scFoundation ────────────────────────────────

def prep_for_scfoundation(expr_df, gene_list, gdsc_genes, source="rna"):
    """Map expression to scFoundation gene space (19264 genes).
    Restrict to GDSC 2000-gene subset to keep sequence length ~2000."""
    import scanpy as sc
    import anndata

    df = expr_df.T.copy()  # samples × genes

    # Keep only GDSC-overlapping genes
    common = [g for g in gdsc_genes if g in df.columns]
    df = df[common].astype(np.float32)
    print(f"    {source}: genes kept (GDSC overlap): {len(common)}/{len(gdsc_genes)}")

    if source == "microarray":
        mean = df.mean(axis=0)
        std  = df.std(axis=0).replace(0, 1.0)
        df   = (df - mean) / std
        shift = float(-df.values.min()) + 1.0
        df    = df + shift
    else:  # RNA-seq
        row_sums = df.sum(axis=1).replace(0, 1.0)
        df = df.div(row_sums, axis=0) * 1e4  # CPM
        df = np.log1p(df)
        mean = df.mean(axis=0)
        std  = df.std(axis=0).replace(0, 1.0)
        df   = (df - mean) / std
        shift = float(-df.values.min()) + 1.0
        df    = df + shift

    # Add missing scFoundation genes as zero
    missing = list(set(gene_list) - set(df.columns))
    zero_df = pd.DataFrame(
        np.zeros((len(df), len(missing)), dtype=np.float32),
        columns=missing, index=df.index,
    )
    df = pd.concat([df, zero_df], axis=1)
    df = df[gene_list].astype(np.float32)
    print(f"    → aligned to {df.shape[1]} scFoundation genes")
    return df.values  # (N_samples, 19264)


# ── scFoundation inference ────────────────────────────────────────────────────

@torch.no_grad()
def extract_scf_embeddings(model, config, expr_arr):
    """4-way pooling → (N, 3072)."""
    from load import gatherData
    import scanpy as sc
    import anndata

    N = expr_arr.shape[0]
    all_emb = []
    pad_id  = config.get("pad_token_id", 0)

    for start in range(0, N, BATCH):
        end   = min(start + BATCH, N)
        batch = expr_arr[start:end]

        ad = anndata.AnnData(batch.copy())
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        log_expr = torch.tensor(ad.X, dtype=torch.float32).to(DEVICE)

        raw_sum   = torch.tensor(batch, dtype=torch.float32).sum(dim=1, keepdim=True)
        total_tok = torch.log10(raw_sum.clamp(min=1.0)).to(DEVICE)
        pretrain_x = torch.cat([log_expr, total_tok, total_tok], dim=1)

        data_ids = torch.arange(19266, device=DEVICE).unsqueeze(0).expand(end - start, -1)
        value_labels = pretrain_x > 0
        x, x_pad = gatherData(pretrain_x, value_labels, pad_id)
        pos_ids, _ = gatherData(data_ids, value_labels, pad_id)

        x_emb  = model.token_emb(x.unsqueeze(2).float(), output_weight=0)
        x_emb += model.pos_emb(pos_ids)
        enc    = model.encoder(x_emb, x_pad)

        e1 = enc[:, -1, :]
        e2 = enc[:, -2, :]
        e3, _ = torch.max(enc[:, :-2, :], dim=1)
        e4 = torch.mean(enc[:, :-2, :], dim=1)
        emb = torch.cat([e1, e2, e3, e4], dim=1)

        all_emb.append(emb.cpu().float().numpy())
        del x_emb, enc, pretrain_x, log_expr
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        print(f"    batch {start//BATCH+1}/{(N+BATCH-1)//BATCH} done")

    return np.concatenate(all_emb, axis=0)


# ── Drug features ─────────────────────────────────────────────────────────────

def build_drug_features_from_smiles(drug_names, smiles_df):
    """Build Morgan FP + RDKit descriptors from a SMILES DataFrame."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    smiles_map = dict(zip(smiles_df["drug_name"], smiles_df["smiles"]))
    fps = {}
    for d in drug_names:
        smi = smiles_map.get(d)
        if not smi or pd.isna(smi):
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            try:
                gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
                fp  = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            except Exception:
                fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                              dtype=np.float32)
            fps[d] = fp
        except Exception:
            pass

    valid_drugs = [d for d in drug_names if d in fps]
    print(f"  Drugs with valid fingerprints: {len(valid_drugs)}/{len(drug_names)}")
    return fps, valid_drugs


def normalize_descriptors(desc_arr):
    """Normalize descriptor array, remove near-constant features."""
    desc_mean = desc_arr.mean(axis=0)
    desc_std  = desc_arr.std(axis=0) + 1e-8
    desc_norm = (desc_arr - desc_mean) / desc_std
    valid_cols = desc_std > 1e-6
    return desc_norm[:, valid_cols]


# ── PDAC Data Preparation ────────────────────────────────────────────────────

def prepare_pdac(gene_list, gdsc_genes, scf_model, scf_config):
    """Prepare PDAC organoid data from GSE194249."""
    print("\n" + "=" * 60)
    print("  [1] PDAC Organoids (GSE194249, Chen et al. 2022)")
    print("=" * 60)

    # Load FPKM expression
    with gzip.open(RAW_DIR / "GSE194249/GSE194249_PDPCOs_FPKM.txt.gz", "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    print(f"  Expression: {expr.shape}  (genes × samples)")
    print(f"  Sample IDs: {expr.columns[:5].tolist()} ...")

    # Load AUC data
    auc = pd.read_excel(RAW_DIR / "GSE194249/GSE194249_drug_AUC_MOESM11.xlsx", header=1)
    auc.columns = [c.strip() for c in auc.columns]
    auc_samples = [c for c in auc.columns if c.startswith("CAS-DAC")]
    print(f"  AUC samples: {len(auc_samples)}")
    print(f"  AUC drugs: {len(auc)}")

    # Filter expression to AUC samples
    common_samples = [s for s in auc_samples if s in expr.columns]
    expr_filtered = expr[common_samples]
    print(f"  Expression filtered to {len(common_samples)} AUC samples")

    # Prepare for scFoundation
    pdac_arr = prep_for_scfoundation(expr_filtered, gene_list, gdsc_genes, source="rna")

    # Extract embeddings
    print(f"  Extracting scFoundation embeddings for {len(common_samples)} PDAC organoids...")
    pdac_emb = extract_scf_embeddings(scf_model, scf_config, pdac_arr)
    print(f"  PDAC embeddings: {pdac_emb.shape}")

    # Load drug SMILES
    pdac_smiles = pd.read_csv(EXT_DIR / "pdac_drug_smiles.csv")

    # Build AUC drug name mapping: catalog → drug_name
    auc_drug_ids = auc.iloc[:, 0].tolist()  # S* or chemo abbreviation
    cat_to_name = dict(zip(pdac_smiles["catalog"], pdac_smiles["drug_name"]))

    # Melt AUC matrix to long format
    pairs = []
    for _, row in auc.iterrows():
        catalog = row.iloc[0]
        drug_name = cat_to_name.get(catalog, catalog)
        for sample in auc_samples:
            val = row[sample]
            if pd.isna(val):
                continue
            pairs.append({
                "organoid_id": sample,
                "drug_name": drug_name,
                "drug_catalog": catalog,
                "auc": float(val),
                "sensitivity": 1.0 - float(val),  # 1-AUC: higher = more sensitive
            })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Total pairs: {len(pairs_df)}")
    print(f"  Organoids: {pairs_df['organoid_id'].nunique()}")
    print(f"  Drugs: {pairs_df['drug_name'].nunique()}")
    print(f"  Sensitivity (1-AUC) range: {pairs_df['sensitivity'].min():.4f} to {pairs_df['sensitivity'].max():.4f}")

    # Build drug features
    fps, valid_drugs = build_drug_features_from_smiles(
        pairs_df["drug_name"].unique().tolist(), pdac_smiles)
    pairs_df = pairs_df[pairs_df["drug_name"].isin(valid_drugs)].reset_index(drop=True)
    print(f"  Valid pairs after SMILES filter: {len(pairs_df)}")

    # Build arrays
    org_to_idx = {oid: i for i, oid in enumerate(common_samples)}
    X_cell = np.array([pdac_emb[org_to_idx[r["organoid_id"]]]
                        for _, r in pairs_df.iterrows()], dtype=np.float32)
    X_fp   = np.array([fps[r["drug_name"]]
                        for _, r in pairs_df.iterrows()], dtype=np.float32)
    y      = pairs_df["sensitivity"].values.astype(np.float32)

    # Save
    cell_ids = np.array(common_samples)
    cell_emb = pdac_emb[:len(common_samples)]

    np.save(GEO_PROC / "pdac_cell_emb.npy", cell_emb.astype(np.float32))
    np.save(GEO_PROC / "pdac_cell_ids.npy", cell_ids)
    np.save(GEO_PROC / "pdac_drug_features.npy", X_fp)
    np.save(GEO_PROC / "pdac_response.npy", y)
    pairs_df.to_csv(GEO_PROC / "pdac_pair_meta.csv", index=False)

    print(f"\n  Saved PDAC data:")
    print(f"    pdac_cell_emb.npy:       {cell_emb.shape}")
    print(f"    pdac_cell_ids.npy:       {cell_ids.shape}")
    print(f"    pdac_drug_features.npy:  {X_fp.shape}")
    print(f"    pdac_response.npy:       {y.shape}")
    print(f"    pdac_pair_meta.csv:      {len(pairs_df)} rows")

    return pairs_df


# ── Bladder Cancer Data Preparation ──────────────────────────────────────────

def prepare_blca(gene_list, gdsc_genes, scf_model, scf_config):
    """Prepare bladder cancer organoid data from GSE103990."""
    print("\n" + "=" * 60)
    print("  [2] Bladder Cancer Organoids (GSE103990, Lee et al. 2018)")
    print("=" * 60)

    # Load expression (normalized counts)
    with gzip.open(RAW_DIR / "GSE103990/GSE103990_Normalized_counts.txt.gz", "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    # Fix gene IDs: ENSG_SYMBOL → SYMBOL
    expr.index = expr.index.str.split("_", n=1).str[-1]
    expr = expr[~expr.index.duplicated()]
    print(f"  Expression: {expr.shape}  (genes × samples)")

    # Load drug response (Table S3)
    dr = pd.read_excel(RAW_DIR / "GSE103990/drug_response_S3.xlsx", header=1)
    dr.columns = ["Drug", "Line", "HillSlope", "LogIC50", "AUC"]
    dr["LogIC50"] = pd.to_numeric(dr["LogIC50"], errors="coerce")
    dr["AUC"] = pd.to_numeric(dr["AUC"], errors="coerce")

    # Average replicates
    dr_avg = dr.groupby(["Drug", "Line"]).agg({"LogIC50": "mean", "AUC": "mean"}).reset_index()
    print(f"  Drug response (averaged): {len(dr_avg)} pairs")
    print(f"  Organoids: {sorted(dr_avg['Line'].unique())}")
    print(f"  Drugs: {dr_avg['Drug'].nunique()}")
    print(f"  LogIC50 range: {dr_avg['LogIC50'].min():.3f} to {dr_avg['LogIC50'].max():.3f}")

    # Match expression samples to drug response lines
    # Drug response uses SCBO-* IDs, expression may use different IDs
    # Check overlap
    expr_samples = expr.columns.tolist()
    dr_lines = dr_avg["Line"].unique().tolist()
    print(f"\n  Expression sample IDs: {expr_samples[:5]} ...")
    print(f"  Drug response line IDs: {dr_lines}")

    # Expression samples might have different naming - try to match
    # GSE103990 expression uses GSM* or descriptive names
    # Read metadata for mapping
    meta = pd.read_csv(PROC_DIR / "geo/GSE103990_metadata.csv")
    print(f"  Metadata columns: {meta.columns.tolist()}")
    print(f"  Metadata sample rows:")
    print(meta.head(5).to_string())

    # Build mapping from expression columns to SCBO IDs
    # The metadata should have both sample accessions and SCBO IDs
    # Try to find the mapping
    possible_id_cols = [c for c in meta.columns if any(
        x in c.lower() for x in ['title', 'sample', 'name', 'id', 'line', 'scbo'])]
    print(f"  Possible ID columns in metadata: {possible_id_cols}")

    # Check if expression columns match any metadata field
    for col in meta.columns:
        vals = set(meta[col].astype(str))
        overlap = vals & set(expr_samples)
        if overlap:
            print(f"  Column '{col}' overlaps with expression: {len(overlap)} samples")

    # Try to match via sample titles containing SCBO
    title_col = None
    for c in meta.columns:
        if 'title' in c.lower():
            title_col = c
            break

    if title_col:
        for _, row in meta.iterrows():
            title = str(row[title_col])
            if 'SCBO' in title:
                # Try to extract SCBO ID
                import re
                match = re.search(r'SCBO-\d+', title)
                if match:
                    print(f"  Found SCBO in title: {title} -> {match.group()}")

    return None  # Will complete after checking metadata


def prepare_blca_v2(gene_list, gdsc_genes, scf_model, scf_config):
    """Prepare bladder cancer organoid data from GSE103990 (v2 with MS→SCBO mapping)."""
    import re as re_mod

    print("\n" + "=" * 60)
    print("  [2] Bladder Cancer Organoids (GSE103990, Lee et al. 2018)")
    print("=" * 60)

    # Build MS* → SCBO* mapping from SOFT file
    with gzip.open(RAW_DIR / "GSE103990/GSE103990_family.soft.gz", "rt") as f:
        soft_content = f.read()
    soft_lines = soft_content.split("\n")
    current_gsm = None
    current_ms = None
    current_scbo = None
    mapping = []
    for line in soft_lines:
        line = line.strip()
        if line.startswith("^SAMPLE"):
            if current_gsm and current_ms and current_scbo:
                mapping.append((current_gsm, current_ms, current_scbo))
            current_gsm = line.split("= ")[1] if "= " in line else None
            current_ms = None
            current_scbo = None
        elif line.startswith("!Sample_title"):
            title = line.split("= ")[1] if "= " in line else ""
            m = re_mod.search(r"(SCBO[\-\.][\d\.]+)", title)
            if m:
                current_scbo = m.group(1)
        elif line.startswith("!Sample_description"):
            desc = line.split("= ")[1].strip() if "= " in line else ""
            if desc.startswith("MS"):
                current_ms = desc
    if current_gsm and current_ms and current_scbo:
        mapping.append((current_gsm, current_ms, current_scbo))

    ms_to_scbo = {ms: scbo for _, ms, scbo in mapping}
    print(f"  MS→SCBO mapping: {len(ms_to_scbo)} entries")

    # Load expression
    with gzip.open(RAW_DIR / "GSE103990/GSE103990_Normalized_counts.txt.gz", "rt") as f:
        expr = pd.read_csv(f, sep="\t", index_col=0)
    expr.index = expr.index.str.split("_", n=1).str[-1]
    expr = expr[~expr.index.duplicated()]
    print(f"  Expression: {expr.shape}  (genes × samples)")

    # Load drug response
    dr = pd.read_excel(RAW_DIR / "GSE103990/drug_response_S3.xlsx", header=1)
    dr.columns = ["Drug", "Line", "HillSlope", "LogIC50", "AUC"]
    dr["LogIC50"] = pd.to_numeric(dr["LogIC50"], errors="coerce")
    dr["AUC"] = pd.to_numeric(dr["AUC"], errors="coerce")
    dr_avg = dr.groupby(["Drug", "Line"]).agg({"LogIC50": "mean", "AUC": "mean"}).reset_index()
    dr_lines = set(dr_avg["Line"].unique())
    print(f"  Drug response lines: {sorted(dr_lines)}")
    print(f"  Drugs: {dr_avg['Drug'].nunique()}")

    # Find expression samples matching drug response lines
    matched_ms = {ms: scbo for ms, scbo in ms_to_scbo.items()
                  if scbo in dr_lines and ms in expr.columns}
    print(f"  Matched expression samples: {len(matched_ms)}")
    for ms, scbo in sorted(matched_ms.items()):
        print(f"    {ms} -> {scbo}")

    # For each SCBO line, pick the first MS sample (lowest passage)
    scbo_to_ms = {}
    for ms, scbo in sorted(matched_ms.items()):
        if scbo not in scbo_to_ms:
            scbo_to_ms[scbo] = ms
    print(f"  Selected samples (1 per line): {len(scbo_to_ms)}")

    # Filter expression
    selected_ms = list(scbo_to_ms.values())
    expr_filtered = expr[selected_ms]
    print(f"  Filtered expression: {expr_filtered.shape}")

    # Prepare for scFoundation
    blca_arr = prep_for_scfoundation(expr_filtered, gene_list, gdsc_genes, source="rna")

    # Extract embeddings
    print(f"  Extracting scFoundation embeddings for {len(selected_ms)} bladder organoids...")
    blca_emb = extract_scf_embeddings(scf_model, scf_config, blca_arr)
    print(f"  Bladder embeddings: {blca_emb.shape}")

    # Build pairs using LogIC50 (consistent with GDSC lnIC50)
    pairs = []
    for _, row in dr_avg.iterrows():
        scbo = row["Line"]
        if scbo not in scbo_to_ms:
            continue
        ms_id = scbo_to_ms[scbo]
        log_ic50 = row["LogIC50"]
        if pd.isna(log_ic50):
            continue
        pairs.append({
            "organoid_id": ms_id,
            "scbo_id": scbo,
            "drug_name": row["Drug"],
            "ic50": float(log_ic50),
        })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Total pairs: {len(pairs_df)}")
    print(f"  Organoids: {pairs_df['organoid_id'].nunique()}")
    print(f"  Drugs: {pairs_df['drug_name'].nunique()}")
    print(f"  LogIC50 range: {pairs_df['ic50'].min():.3f} to {pairs_df['ic50'].max():.3f}")

    # Build drug features using GDSC SMILES
    gdsc_smiles = pd.read_csv(EXT_DIR / "gdsc_drug_smiles.csv")
    fps, valid_drugs = build_drug_features_from_smiles(
        pairs_df["drug_name"].unique().tolist(), gdsc_smiles)
    pairs_df = pairs_df[pairs_df["drug_name"].isin(valid_drugs)].reset_index(drop=True)
    print(f"  Valid pairs after SMILES filter: {len(pairs_df)}")

    # Build arrays
    org_to_idx = {ms: i for i, ms in enumerate(selected_ms)}
    X_cell = np.array([blca_emb[org_to_idx[r["organoid_id"]]]
                        for _, r in pairs_df.iterrows()], dtype=np.float32)
    X_fp   = np.array([fps[r["drug_name"]]
                        for _, r in pairs_df.iterrows()], dtype=np.float32)
    y      = pairs_df["ic50"].values.astype(np.float32)

    # Save
    cell_ids = np.array(selected_ms)
    cell_emb_final = blca_emb[:len(selected_ms)]

    np.save(GEO_PROC / "blca_cell_emb.npy", cell_emb_final.astype(np.float32))
    np.save(GEO_PROC / "blca_cell_ids.npy", cell_ids)
    np.save(GEO_PROC / "blca_drug_features.npy", X_fp)
    np.save(GEO_PROC / "blca_response.npy", y)
    pairs_df.to_csv(GEO_PROC / "blca_pair_meta.csv", index=False)

    print(f"\n  Saved BLCA data:")
    print(f"    blca_cell_emb.npy:       {cell_emb_final.shape}")
    print(f"    blca_cell_ids.npy:       {cell_ids.shape}")
    print(f"    blca_drug_features.npy:  {X_fp.shape}")
    print(f"    blca_response.npy:       {y.shape}")
    print(f"    blca_pair_meta.csv:      {len(pairs_df)} rows")

    return pairs_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Preparing Multi-Cancer Organoid Data")
    print("=" * 65)

    gene_list  = load_gene_list()
    gdsc_genes = load_gdsc_gene_list()
    print(f"  scFoundation gene space: {len(gene_list)}")
    print(f"  GDSC gene subset: {len(gdsc_genes)}")

    # Load scFoundation model
    from load import load_model_frommmf
    ckpt = SCF_DIR / "models/models.ckpt"
    model, config = load_model_frommmf(str(ckpt), key="cell")
    model.eval()
    print(f"  scFoundation loaded  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # [1] PDAC
    prepare_pdac(gene_list, gdsc_genes, model, config)

    # [2] Bladder Cancer
    prepare_blca_v2(gene_list, gdsc_genes, model, config)

    print("\n" + "=" * 65)
    print("  Done!")
    print("=" * 65)


if __name__ == "__main__":
    main()
