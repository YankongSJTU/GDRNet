# -*- coding: utf-8 -*-
"""
Generate Demo Data for GDRNet
==============================
Create small synthetic datasets to test the pipeline without real data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DEMO = Path(__file__).parent / "data"
DEMO.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ── GDSC Demo Data ──
n_cells = 20
n_drugs = 10
n_genes = 2000
n_pairs = 200

cell_ids = [f"CellLine_{i:03d}" for i in range(n_cells)]
drug_names = [f"Drug_{i:02d}" for i in range(n_drugs)]

# Generate pairs
pairs = []
for c in cell_ids:
    for d in drug_names:
        if np.random.rand() < 0.5:
            pairs.append((c, d))

n_pairs = len(pairs)
print(f"Demo pairs: {n_pairs}")

# Features
gene_expr = np.random.randn(n_cells, n_genes).astype(np.float32) * 0.5 + 5.0
scf_emb = np.random.randn(n_cells, 3072).astype(np.float32) * 0.1
drug_features = np.random.rand(n_drugs, 2236).astype(np.float32)  # 2048 FP + 188 desc

# Response
cell_to_idx = {c: i for i, c in enumerate(cell_ids)}
drug_to_idx = {d: i for i, d in enumerate(drug_names)}

response = np.zeros(n_pairs, dtype=np.float32)
for k, (c, d) in enumerate(pairs):
    ci, di = cell_to_idx[c], drug_to_idx[d]
    response[k] = 2.0 + np.dot(gene_expr[ci, :10], drug_features[di, :10]) * 0.01 \
                  + np.random.randn() * 0.5

# Save
meta = pd.DataFrame(pairs, columns=["ModelID", "drug_name"])
meta["LN_IC50"] = response

np.save(DEMO / "gdsc_gene_expr.npy", gene_expr)
np.save(DEMO / "scfoundation_cell_emb.npy", scf_emb)
np.save(DEMO / "scfoundation_cell_ids.npy", np.array(cell_ids))
np.save(DEMO / "gdsc_drug_features.npy", drug_features)
np.save(DEMO / "gdsc_drug_names.npy", np.array(drug_names))
meta.to_csv(DEMO / "gdsc_metadata.csv", index=False)

print(f"GDSC demo: {n_cells} cells, {n_drugs} drugs, {n_pairs} pairs")

# ── Organoid Demo Data ──
n_org = 5
org_ids = [f"Organoid_{i:02d}" for i in range(n_org)]
org_drugs = drug_names[:5]

org_pairs = [(o, d) for o in org_ids for d in org_drugs]
n_org_pairs = len(org_pairs)

org_scf = np.random.randn(n_org, 3072).astype(np.float32) * 0.1
org_response = np.random.randn(n_org_pairs).astype(np.float32) * 0.5 + 2.0

org_meta = pd.DataFrame(org_pairs, columns=["organoid_id", "drug_name"])
org_meta["ic50"] = org_response

np.save(DEMO / "organoid_cell_emb.npy", org_scf)
np.save(DEMO / "organoid_cell_ids.npy", np.array(org_ids))
np.save(DEMO / "organoid_drug_features.npy", drug_features[:5])
np.save(DEMO / "organoid_response.npy", org_response)
org_meta.to_csv(DEMO / "organoid_pair_meta.csv", index=False)

print(f"Organoid demo: {n_org} organoids, {len(org_drugs)} drugs, {n_org_pairs} pairs")
print(f"Demo data saved to {DEMO}")
