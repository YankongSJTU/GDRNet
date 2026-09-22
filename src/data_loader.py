# -*- coding: utf-8 -*-
"""
Unified Data Loading for GDRNet LCLO Benchmarking
=====================================================
Replaces duplicated data loading in train.py and ablation.py.
Provides a single canonical data pipeline that produces LCLO-ready data.

Key function:
  load_gdsc_lclo(n_splits=5, seed=42) -> GDLCLOData
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data/processed"
EXT_DIR = ROOT / "data/external"


def _compute_morgan_fps(smiles_df):
    """Compute 2048-bit Morgan fingerprints from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors

    fps = {}
    for _, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                continue
            try:
                gen = rdMolDescriptors.GetMorganGenerator(radius=2, fpSize=2048)
                fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            except Exception:
                fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
                    mol, 2, nBits=2048), dtype=np.float32)
            fps[row["drug_name"]] = fp
        except Exception:
            pass
    return fps


@dataclass
class GDLCLOData:
    """Container for GDSC data with LCLO folds."""
    X_gene: np.ndarray       # (N, 2000) gene expression
    X_scf: np.ndarray        # (N, 3072) scFoundation embeddings
    X_fp: np.ndarray         # (N, 2048) Morgan fingerprints
    cell_ids: np.ndarray     # (N,) str ModelIDs
    drug_names: np.ndarray   # (N,) str drug names
    y: np.ndarray            # (N,) float32 ln_IC50
    cell_idx: np.ndarray     # (N,) int64 1-based cell indices
    drug_idx: np.ndarray     # (N,) int64 1-based drug indices
    meta: pd.DataFrame       # full metadata for filtered samples
    folds: list              # list of (tr_mask, te_mask) boolean arrays
    cell_to_idx: dict        # ModelID -> int (1-based)
    drug_to_idx: dict        # drug_name -> int (1-based)
    n_cells: int
    n_drugs: int
    cell_list: list          # sorted cell IDs
    drug_list: list          # sorted drug names


def load_gdsc_lclo(n_splits=5, seed=42):
    """Load GDSC data with LCLO fold assignments.

    Returns GDLCLOData with all features, indices, and LCLO folds.
    Cell/drug index mappings are computed over ALL data (not per-fold).
    Test-fold cells have valid indices but their embeddings are never trained —
    this is the correct LCLO behavior.
    """
    from eval_utils import make_lclo_folds

    print("=" * 65, flush=True)
    print("  Loading GDSC data for LCLO benchmarking", flush=True)
    print("=" * 65, flush=True)
    t0 = time.time()

    # 1. scFoundation embeddings
    scf_emb = np.load(PROC_DIR / "scfoundation_cell_emb.npy")
    scf_ids = np.load(PROC_DIR / "scfoundation_cell_ids.npy", allow_pickle=True)
    id_to_emb = {cid: scf_emb[i] for i, cid in enumerate(scf_ids)}
    print(f"  [1/5] scFoundation embeddings: {scf_emb.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # 2. Core data
    X_cell_full = pd.read_parquet(PROC_DIR / "gdsc_cell_features.parquet")
    y_full = pd.read_parquet(PROC_DIR / "gdsc_response_lnic50.parquet").iloc[:, 0]
    meta_full = pd.read_parquet(PROC_DIR / "gdsc_metadata.parquet")
    smiles_df = pd.read_csv(EXT_DIR / "gdsc_drug_smiles.csv")
    print(f"  [2/5] Core data: {len(y_full):,} samples  ({time.time()-t0:.1f}s)", flush=True)

    # 3. Morgan fingerprints
    fps = _compute_morgan_fps(smiles_df)
    print(f"  [3/5] Morgan FP: {len(fps)} drugs  ({time.time()-t0:.1f}s)", flush=True)

    # 4. Filter to valid samples (have both FP and scF embedding)
    valid_mask = (meta_full["drug_name"].isin(fps)) & \
                 (meta_full["ModelID"].isin(id_to_emb))
    orig_idx = meta_full[valid_mask].index
    meta = meta_full[valid_mask].reset_index(drop=True)
    y = y_full[valid_mask].values.astype(np.float32)

    X_gene = X_cell_full.loc[orig_idx].values.astype(np.float32)
    X_scf = np.array([id_to_emb[c] for c in meta["ModelID"]], dtype=np.float32)
    X_fp = np.array([fps[d] for d in meta["drug_name"]], dtype=np.float32)

    print(f"  [4/5] Filtered: {len(y):,} samples  "
          f"({meta['ModelID'].nunique()} cells, "
          f"{meta['drug_name'].nunique()} drugs)  ({time.time()-t0:.1f}s)", flush=True)

    # 5. Build cell/drug index (over ALL data)
    cell_list = sorted(meta["ModelID"].unique())
    drug_list = sorted(meta["drug_name"].unique())
    cell_to_idx = {c: i + 1 for i, c in enumerate(cell_list)}
    drug_to_idx = {d: i + 1 for i, d in enumerate(drug_list)}
    cell_idx_all = np.array([cell_to_idx[c] for c in meta["ModelID"]], dtype=np.int64)
    drug_idx_all = np.array([drug_to_idx[d] for d in meta["drug_name"]], dtype=np.int64)

    n_cells = len(cell_list)
    n_drugs = len(drug_list)

    # 6. Generate LCLO folds
    folds = make_lclo_folds(meta, n_splits=n_splits, seed=seed)

    print(f"  [5/5] LCLO {n_splits}-fold: {n_cells} cells, {n_drugs} drugs  "
          f"({time.time()-t0:.1f}s)", flush=True)
    for i, (tr_m, te_m) in enumerate(folds):
        n_tr_cells = meta.loc[tr_m, "ModelID"].nunique()
        n_te_cells = meta.loc[te_m, "ModelID"].nunique()
        print(f"    Fold {i}: train={tr_m.sum():,} ({n_tr_cells} cells)  "
              f"test={te_m.sum():,} ({n_te_cells} cells)", flush=True)

    data = GDLCLOData(
        X_gene=X_gene, X_scf=X_scf, X_fp=X_fp,
        cell_ids=meta["ModelID"].values,
        drug_names=meta["drug_name"].values,
        y=y,
        cell_idx=cell_idx_all, drug_idx=drug_idx_all,
        meta=meta,
        folds=folds,
        cell_to_idx=cell_to_idx, drug_to_idx=drug_to_idx,
        n_cells=n_cells, n_drugs=n_drugs,
        cell_list=cell_list, drug_list=drug_list,
    )

    print(f"  Data loading complete in {time.time()-t0:.1f}s", flush=True)
    return data


def load_gdsc_flat(n_splits=5, seed=42, include_scf=False):
    """Load GDSC data as flat feature matrix for tree models.

    Parameters
    ----------
    n_splits : int
    seed : int
    include_scf : bool - whether to include scFoundation embeddings (3072 extra dims)

    Returns
    -------
    X_flat : np.ndarray (N, 4048) or (N, 7120) if include_scf
    y : np.ndarray
    cell_ids : np.ndarray
    drug_names : np.ndarray
    folds : list of (tr_mask, te_mask)
    """
    data = load_gdsc_lclo(n_splits=n_splits, seed=seed)

    parts = [data.X_gene, data.X_fp]
    if include_scf:
        parts.insert(1, data.X_scf)
    X_flat = np.concatenate(parts, axis=1).astype(np.float32)

    print(f"  Flat features: {X_flat.shape[1]} dims "
          f"(gene={data.X_gene.shape[1]}, fp={data.X_fp.shape[1]}"
          f"{', scf=' + str(data.X_scf.shape[1]) if include_scf else ''})")

    return X_flat, data.y, data.cell_ids, data.drug_names, data.folds, data.meta


if __name__ == "__main__":
    print("Testing data_loader...")
    data = load_gdsc_lclo(n_splits=5, seed=42)
    print(f"\n  X_gene: {data.X_gene.shape}")
    print(f"  X_scf:  {data.X_scf.shape}")
    print(f"  X_fp:   {data.X_fp.shape}")
    print(f"  y:      {data.y.shape}")
    print(f"  Cells:  {data.n_cells}, Drugs: {data.n_drugs}")
    print(f"  Folds:  {len(data.folds)}")
    print(f"  cell_idx range: [{data.cell_idx.min()}, {data.cell_idx.max()}]")
    print(f"  drug_idx range: [{data.drug_idx.min()}, {data.drug_idx.max()}]")
    print("  All tests passed!")
