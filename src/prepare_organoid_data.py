# -*- coding: utf-8 -*-
"""
Prepare Organoid Data for GDRNetV3 Fine-tuning
===============================================
Sources:
  1. CRC organoids (van de Wetering 2015, N=16, 83 drugs)
     Expression: GSE64392 microarray (12037 genes)
     Drug response: median IC50 (log-scale, ~comparable to GDSC LnIC50)

  2. Bladder cancer organoids (Lee 2018, GSE103990, N=42)
     Expression only (RNA-seq, ~54k genes after ENSG_SYMBOL extraction)
     No drug labels → used only for scFoundation embedding pre-computation

Outputs (saved to data/processed/):
  organoid_cell_emb.npy       (N_org, 3072) scFoundation embeddings
  organoid_cell_ids.npy       (N_org,)      organoid IDs
  organoid_drug_features.npy  (N_pairs, fp+desc) drug features
  organoid_response.npy       (N_pairs,)    LnIC50-scale response
  organoid_pair_meta.csv      organoid_id, drug_name, ic50, split
"""

import sys, warnings, gc
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import anndata

warnings.filterwarnings("ignore")

ROOT     = Path("/export/home/kongyan/project/Organoid")
PROC_DIR = ROOT / "data/processed"
RAW_DIR  = ROOT / "data/raw/geo"
EXT_DIR  = ROOT / "data/external"
SCF_DIR  = ROOT / "external/scFoundation/model"

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCF_DIR))
sys.path.insert(0, str(SCF_DIR / "pretrainmodels"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 32   # 2000-gene space keeps seq_len ~2000, safe at batch 32


# ── Gene list ──────────────────────────────────────────────────────────────────

def load_gene_list():
    df = pd.read_csv(SCF_DIR / "OS_scRNA_gene_index.19264.tsv", sep="\t", header=0)
    return list(df["gene_name"])


# ── Load CRC organoid expression ──────────────────────────────────────────────

def load_crc_expression():
    """Microarray expression (gene symbol × 22 organoids), log2 intensity."""
    expr = pd.read_csv(RAW_DIR / "organoid_COAD/expression_median.txt",
                       sep="\t", index_col=0)
    expr = expr.drop(columns=["uniprotID"], errors="ignore")
    # Already gene symbols as index, samples as columns
    print(f"  CRC expression: {expr.shape}  (genes × organoids)")
    return expr   # float, log2-scale microarray intensities


# ── Load bladder organoid expression ──────────────────────────────────────────

def load_bladder_expression():
    """RNA-seq raw counts (ENSG_SYMBOL × 42 samples)."""
    import gzip
    with gzip.open(RAW_DIR / "GSE103990/GSE103990_Normalized_counts.txt.gz", "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    # Fix gene IDs: ENSG_SYMBOL → SYMBOL
    df.index = df.index.str.split("_", n=1).str[-1]
    df = df[~df.index.duplicated()]
    print(f"  Bladder expression: {df.shape}  (genes × organoids)")
    return df   # raw counts


# ── Prepare expression for scFoundation ───────────────────────────────────────

def load_gdsc_gene_list():
    """Load the 2000 GDSC genes used for cell-line embeddings."""
    df = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    return list(df.columns)   # 2000 gene symbols


def prep_for_scfoundation(expr_df, gene_list, gdsc_genes, source="rna"):
    """
    Map expression dataframe to scFoundation gene space.

    To avoid OOM (microarray has ~12k non-zero genes → O(L²) attention = 99 GB),
    we restrict to the same 2000-gene GDSC subset used for cell-line embeddings.
    This keeps sequence length ≈ 2000 (matching cell-line extraction) and ensures
    the organoid embeddings live in the same representation space as cell lines,
    which is critical for transfer learning.

    expr_df: genes × samples, already in appropriate scale
    Returns (N_samples × 19264) numpy array ready for scFoundation.
    """
    df = expr_df.T.copy()   # samples × genes

    if source == "microarray":
        # Keep only GDSC-overlapping genes, then z-score + shift (same as GDSC pipeline)
        common = [g for g in gdsc_genes if g in df.columns]
        df = df[common].astype(np.float32)
        print(f"    microarray genes kept (GDSC overlap): {len(common)}/{len(gdsc_genes)}")
        # Z-score per gene (column) across samples
        mean = df.mean(axis=0)
        std  = df.std(axis=0).replace(0, 1.0)
        df   = (df - mean) / std
        # Shift to all-positive (same as extract_scfoundation_emb.py)
        shift = float(-df.values.min()) + 1.0
        df    = df + shift
    else:  # RNA-seq raw counts → same 2000-gene space
        common = [g for g in gdsc_genes if g in df.columns]
        df = df[common].astype(np.float32)
        print(f"    RNA-seq genes kept (GDSC overlap): {len(common)}/{len(gdsc_genes)}")
        # Library-size normalise → log1p → z-score + shift
        row_sums = df.sum(axis=1).replace(0, 1.0)
        df = df.div(row_sums, axis=0) * 1e4   # CPM
        df = np.log1p(df)
        mean = df.mean(axis=0)
        std  = df.std(axis=0).replace(0, 1.0)
        df   = (df - mean) / std
        shift = float(-df.values.min()) + 1.0
        df    = df + shift

    # Add missing scFoundation genes as zero columns
    missing = list(set(gene_list) - set(df.columns))
    zero_df = pd.DataFrame(
        np.zeros((len(df), len(missing)), dtype=np.float32),
        columns=missing, index=df.index,
    )
    df = pd.concat([df, zero_df], axis=1)
    df = df[gene_list].astype(np.float32)
    print(f"    → aligned to {df.shape[1]} scFoundation genes  "
          f"(non-zero: {len(common)}  zero-padded: {len(missing)})")
    return df.values  # (N_samples, 19264)


# ── scFoundation inference ────────────────────────────────────────────────────

@torch.no_grad()
def extract_scf_embeddings(model, config, expr_arr):
    """Same 4-way pooling as in extract_scfoundation_emb.py → (N, 3072)."""
    from load import gatherData
    N = expr_arr.shape[0]
    all_emb = []
    pad_id  = config.get("pad_token_id", 0)

    for start in range(0, N, BATCH):
        end   = min(start + BATCH, N)
        batch = expr_arr[start:end]

        # CPM normalise + log1p
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

def build_drug_features(drug_names, smiles_df):
    """Morgan FP + RDKit desc for a list of drug names, using GDSC SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    from train_v3 import compute_rdkit_descriptors, normalize_descriptors

    smiles_map = dict(zip(smiles_df["drug_name"], smiles_df["smiles"]))
    fps = {}
    for d in drug_names:
        smi = smiles_map.get(d)
        if smi is None:
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
    print(f"  Drugs with SMILES: {len(valid_drugs)}/{len(drug_names)}")

    # RDKit descriptors
    valid_smiles_df = smiles_df[smiles_df["drug_name"].isin(fps)].drop_duplicates("drug_name")
    desc_arr, _, _  = compute_rdkit_descriptors(valid_smiles_df["smiles"].tolist())
    desc_map = {row["drug_name"]: desc_arr[i]
                for i, row in valid_smiles_df.reset_index(drop=True).iterrows()}

    return fps, desc_map, valid_drugs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Preparing Organoid Data for Fine-tuning")
    print("=" * 65)

    gene_list  = load_gene_list()
    gdsc_genes = load_gdsc_gene_list()
    print(f"  scFoundation gene space: {len(gene_list)}")
    print(f"  GDSC gene subset:        {len(gdsc_genes)} genes (used for seq-len control)")

    # ── Load scFoundation model ──────────────────────────────────────
    from load import load_model_frommmf
    ckpt = SCF_DIR / "models/models.ckpt"
    model, config = load_model_frommmf(str(ckpt), key="cell")
    model.eval()
    print(f"  scFoundation loaded  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

    # ── CRC organoids ────────────────────────────────────────────────
    print("\n  [1] CRC organoids (van de Wetering 2015)")
    crc_expr = load_crc_expression()
    crc_arr  = prep_for_scfoundation(crc_expr, gene_list, gdsc_genes, source="microarray")

    print(f"  Extracting scFoundation embeddings for {len(crc_expr.columns)} CRC organoids...")
    crc_emb = extract_scf_embeddings(model, config, crc_arr)
    print(f"  CRC embeddings: {crc_emb.shape}")

    # ── Bladder cancer organoids ─────────────────────────────────────
    print("\n  [2] Bladder cancer organoids (Lee 2018, GSE103990)")
    blca_expr = load_bladder_expression()
    blca_arr  = prep_for_scfoundation(blca_expr, gene_list, gdsc_genes, source="rna")

    print(f"  Extracting scFoundation embeddings for {len(blca_expr.columns)} bladder organoids...")
    blca_emb = extract_scf_embeddings(model, config, blca_arr)
    print(f"  Bladder embeddings: {blca_emb.shape}")

    # ── Combine embeddings ───────────────────────────────────────────
    all_ids = list(crc_expr.columns) + [f"BLCA_{s}" for s in blca_expr.columns]
    all_emb = np.vstack([crc_emb, blca_emb])
    print(f"\n  Total organoid embeddings: {all_emb.shape}")

    np.save(PROC_DIR / "organoid_cell_emb.npy",  all_emb.astype(np.float32))
    np.save(PROC_DIR / "organoid_cell_ids.npy",  np.array(all_ids))
    print(f"  Saved → organoid_cell_emb.npy, organoid_cell_ids.npy")

    # ── Build training pairs (CRC only — has drug labels) ────────────
    print("\n  [3] Building CRC organoid-drug pairs")

    dr = pd.read_csv(RAW_DIR / "organoid_COAD/drug_response_median.txt",
                     sep="\t", index_col=0).reset_index()
    dr.columns = ["organoid_id", "drug_name", "ic50"]

    smiles_df = pd.read_csv(EXT_DIR / "gdsc_drug_smiles.csv")
    fps, desc_map, valid_drugs = build_drug_features(
        dr["drug_name"].unique().tolist(), smiles_df)

    # Get paired samples
    crc_id_to_emb = {oid: crc_emb[i] for i, oid in enumerate(crc_expr.columns)}
    pairs = []
    for _, row in dr.iterrows():
        oid  = row["organoid_id"]
        drug = row["drug_name"]
        ic50 = row["ic50"]
        if oid not in crc_id_to_emb or drug not in fps:
            continue
        if np.isnan(ic50):
            continue
        pairs.append({
            "organoid_id": oid,
            "drug_name":   drug,
            "ic50":        float(ic50),
        })

    pairs_df = pd.DataFrame(pairs)
    print(f"  Valid pairs: {len(pairs_df)}")
    print(f"  Organoids: {pairs_df['organoid_id'].nunique()}")
    print(f"  Drugs:     {pairs_df['drug_name'].nunique()}")
    print(f"  IC50 range: {pairs_df['ic50'].min():.3f} to {pairs_df['ic50'].max():.3f}")

    # Build arrays
    X_cell = np.array([crc_id_to_emb[r["organoid_id"]] for _, r in pairs_df.iterrows()],
                      dtype=np.float32)
    X_fp   = np.array([fps[r["drug_name"]] for _, r in pairs_df.iterrows()],
                      dtype=np.float32)

    # Normalise descriptors
    n_desc_dim = next(iter(desc_map.values())).shape[0]
    desc_raw = np.array([desc_map.get(r["drug_name"],
                         np.zeros(n_desc_dim))
                         for _, r in pairs_df.iterrows()], dtype=np.float32)

    # Simple std-normalisation of descriptors
    desc_mean = desc_raw.mean(axis=0)
    desc_std  = desc_raw.std(axis=0) + 1e-8
    desc_norm = (desc_raw - desc_mean) / desc_std
    # Remove near-constant features
    valid_cols = desc_std > 1e-6
    desc_norm  = desc_norm[:, valid_cols]
    print(f"  Descriptor dims after filtering: {desc_norm.shape[1]}")

    X_drug = np.concatenate([X_fp, desc_norm], axis=1)
    y      = pairs_df["ic50"].values.astype(np.float32)

    # Save
    np.save(PROC_DIR / "organoid_cell_features.npy",  X_cell)
    np.save(PROC_DIR / "organoid_drug_features.npy",  X_drug)
    np.save(PROC_DIR / "organoid_response.npy",        y)
    pairs_df.to_csv(PROC_DIR / "organoid_pair_meta.csv", index=False)

    print(f"\n  Saved arrays:")
    print(f"    organoid_cell_features.npy:  {X_cell.shape}")
    print(f"    organoid_drug_features.npy:  {X_drug.shape}")
    print(f"    organoid_response.npy:       {y.shape}")
    print(f"    organoid_pair_meta.csv:      {len(pairs_df)} rows")
    print("=" * 65)


if __name__ == "__main__":
    main()
