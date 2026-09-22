# GDRNet: Deep Cross-Network for Drug Sensitivity Prediction

GDRNet is a deep learning framework for predicting drug sensitivity (IC50) in cancer cell lines and patient-derived organoids (PDOs). It uses a DCN v2 cross network with multi-modal feature integration to achieve state-of-the-art performance. The model is pre-trained on the GDSC cell line benchmark and transferred to patient-derived organoids via a frozen-encoder fine-tuning strategy, demonstrating robust cross-cancer generalizability.
<div>
 <img src="data/workflow.jpg"   width="100%">
</div>
## Key Features

- **DCN v2 Cross Network**: Explicit feature crossing via low-rank factorization for cell-drug interaction modeling
- **Multi-modal Input**: Integrates gene expression (2,000 landmark genes), scFoundation single-cell embeddings (3,072-d), and Morgan fingerprints (2,048-bit)
- **Cell/Drug ID Embeddings**: Learnable entity embeddings capturing systematic biases (matrix factorization effect)
- **Transfer Learning**: Pre-trained on GDSC, fine-tuned on patient-derived organoid data with frozen encoders
- **Cross-Cancer Validation**: Validated on organoids from three cancer types—colorectal (CRC), pancreatic (PDAC), and bladder (BLCA)
- **Drug Generalization**: Demonstrates accurate prediction for structurally novel compounds not seen during pretraining
- **LCLO 5-Fold Benchmark**: Replica-consistent LCLO evaluation with 14 baselines including DrugCell, GraphDRP, DeepCDR, ElasticNet, and tree ensembles

## Performance

### GDSC LCLO 5-Fold (PerDrug Pearson r)

| Method | PerDrug Mean | Pooled r |
|--------|-----------:|:-------:|
| **GDRNet-Ensemble** | **0.5453** | **0.8892** |
| XGBoost | 0.4989 | 0.8790 |
| LightGBM | 0.4919 | 0.8790 |
| RandomForest | 0.4532 | 0.8494 |
| ElasticNet | 0.4048 | 0.8374 |
| DrugCell | 0.3307 | 0.5935 |
| GraphDRP | 0.2319 | 0.6280 |
| DeepCDR | 0.1272 | 0.6771 |

### Cross-Cancer Organoid LOOCV

| Dataset | Cancer Type | Organoids | Drugs | Pairs | Pearson r |
|---------|------------|-----------|-------|-------|-----------|
| CRC-PDO | Colorectal | 16 | 34 | 544 | **0.890** |
| PDAC | Pancreatic | 38 | 64 | 2,432 | **0.741** |
| BLCA | Bladder | 11 | 30 | 265 | **0.640** |

## Project Structure

```
GDRNet_github/
├── src/                           # Core source code
│   ├── train.py                   # GDSC pretraining (3-seed ensemble)
│   ├── train_lclo.py              # LCLO 5-fold training variant
│   ├── finetune_organoid.py       # CRC organoid LOOCV fine-tuning
│   ├── finetune_multicancer.py    # Multi-cancer LOOCV (CRC + PDAC + BLCA)
│   ├── ablation.py                # GDSC + organoid ablation study
│   ├── ablation_lclo.py           # LCLO ablation study (7 variants + Wilcoxon)
│   ├── preprocess.py              # GDSC data preprocessing
│   ├── features.py                # Drug feature engineering (Morgan FP)
│   ├── extract_scfoundation_emb.py # scFoundation embedding extraction
│   ├── data_loader.py             # Data loading utilities
│   ├── eval_utils.py              # Evaluation utilities
│   ├── prepare_organoid_data.py   # CRC organoid data preparation
│   ├── prepare_multicancer_data.py # Multi-cancer data preparation
│   ├── resolve_pdac_smiles.py     # PDAC SMILES resolution
│   ├── rerun_lclo_gdrnet.py       # LCLO rerun experiment
│   ├── compute_drug_prioritization.py # Drug prioritization
│   ├── analysis_failure.py        # Failure mode analysis
│   ├── analysis_mutation.py       # Mutation subgroup analysis
│   ├── analysis_moa_clustering.py # MOA clustering analysis
│   └──analysis_scf_sensitivity.py # scF sensitivity analysis
├── models/                        # Model definitions (imported by src/)
│   ├── gdrnet.py                  # GDRNet model architecture (DCN v2)
│   ├── baseline.py                # Tree ensemble baselines (LGBM, RF, XGB)
│   ├── deepcdr.py                 # DeepCDR baseline
│   ├── drugcell.py                # DrugCell baseline
│   └── graphdrp.py                # GraphDRP baseline
├── data/                       # Launch scripts (local/remote GPU)
│   └── external/
│   │   ├── pdac_drug_smiles.csv         
│   |   └── gdsc_drug_smiles.csv   
│   └── processed/
│       ├── organoid_pair_meta.csv         
│       └── top_genes.txt   
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Preparation

Raw GDSC data must be downloaded separately. Place raw data in `data/raw/gdsc/` and GEO data in `data/raw/geo/`.

```bash
# GDSC preprocessing
python src/preprocess.py
python src/extract_scfoundation_emb.py
python src/features.py

# Organoid data preparation
python src/prepare_organoid_data.py
python src/prepare_multicancer_data.py
```

### Training

```bash
# GDSC pretraining (3-seed ensemble: seeds 42, 123, 456)
python src/train.py --gpus 0,2,4 --epochs 300

# LCLO 5-fold evaluation
python src/train_lclo.py --epochs 300 --seeds 42

# Fine-tune on CRC organoid (LOOCV)
python src/finetune_organoid.py

# Multi-cancer LOOCV (CRC + PDAC + BLCA)
python src/finetune_multicancer.py
```

### Analysis

```bash
# LCLO ablation study (7 variants + Wilcoxon)
python src/ablation_lclo.py

# Failure mode analysis
python src/analysis_failure.py

# Mutation subgroup analysis (TP53/KRAS/BRAF)
python src/analysis_mutation.py

# scF sensitivity analysis
python src/analysis_scf_sensitivity.py

# Drug prioritization
python src/compute_drug_prioritization.py
```

### Generating Figures

All figures in `results/figures/` and `submission/figures/` are generated by scripts in `src/`:

- `generate_fig1_pipeline.py` → Fig. 1 (multi-modal pipeline, corrected dataset counts)
- `generate_fig2_architecture.py` → Fig. 2 (architecture, Eqs. 1-15 aligned)
- `generate_fig3_lclo.py` → Fig. 3 (LCLO 14-model benchmark + CRC seeds)
- `generate_fig4_composite.py` → Fig. 4 (LCLO scatter/errors/ROC + MOA)
- `generate_fig5_composite.py` → Fig. 5 (gradient x input + per-drug)
- `generate_fig6_hires.py` → Fig. 6 (CRC/PDAC/BLCA cross-cancer, lnIC50)
- `generate_fig7_ablation.py` → Fig. 7 (LCLO + CRC ablation + Δr)

## Model Architecture

```
Cell Branch:
  Gene Expression (2000-d) → MLP Encoder → 256-d
  scFoundation Embedding (3072-d) → MLP Encoder → 256-d
  Cell ID → Embedding → 64-d
  → Concatenate → 576-d cell representation

Drug Branch:
  Morgan Fingerprint (2048-bit) → MLP Encoder → 256-d
  Drug ID → Embedding → 64-d
  → Concatenate → 320-d drug representation

Interaction:
  Concat (896-d) → Project (512-d)
  → DCN v2 Cross Network (3 layers, rank=64)
  → Deep MLP Network (3 ResBlocks, 512→64)
  → Concat → Output Head → lnIC50

Transfer Learning:
  Pretrain on GDSC → Freeze encoders → Fine-tune interaction on organoid LOOCV
```

## Datasets

| Dataset | Type | Samples | Drugs | Pairs | Response | Source |
|---------|------|---------|-------|-------|----------|--------|
| GDSC2 | Cell lines | 700 | 229 | ~146,000 | lnIC50 | GDSC + DepMap |
| CRC-PDO | Organoid | 16 | 34 | 544 | LogIC50 | van de Wetering 2015 |
| PDAC | Organoid | 38 | 64 | 2,432 | 1−AUC | Chen 2022 |
| BLCA | Organoid | 11 | 30 | 265 | LogIC50 | Lee 2018 |

## Citation

If you use this code, please cite:
```
[Citation to be added upon publication]
```

## License

MIT License
