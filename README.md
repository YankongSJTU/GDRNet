# GDRNet: Deep Cross-Network for Drug Sensitivity Prediction

GDRNet is a deep learning framework for predicting drug sensitivity (IC50) in cancer cell lines and patient-derived organoids (PDOs). It uses a DCN v2 cross network with multi-modal feature integration to achieve state-of-the-art performance. The model is pre-trained on the GDSC cell line benchmark and transferred to patient-derived organoids via a frozen-encoder fine-tuning strategy, demonstrating robust cross-cancer generalizability.

## Key Features

- **DCN v2 Cross Network**: Explicit feature crossing via low-rank factorization for cell-drug interaction modeling
- **Multi-modal Input**: Integrates gene expression (2,000 landmark genes), scFoundation single-cell embeddings (3,072-d), and Morgan fingerprints (2,048-bit)
- **Cell/Drug ID Embeddings**: Learnable entity embeddings capturing systematic biases (matrix factorization effect)
- **Transfer Learning**: Pre-trained on GDSC, fine-tuned on patient-derived organoid data with frozen encoders
- **Cross-Cancer Validation**: Validated on organoids from three cancer types—colorectal (CRC), pancreatic (PDAC), and bladder (BLCA)
- **Drug Generalization**: Demonstrates accurate prediction for structurally novel compounds not seen during pretraining
- **Dual-Dataset Ablation**: Ablation studies on both GDSC and organoid datasets reveal dataset-dependent component importance

## Performance

### GDSC Cell Line Benchmark

| Method | Pearson r | R² | RMSE | AUROC |
|--------|-----------|-----|------|-------|
| **GDRNet-Ensemble** | **0.8788** | **0.7711** | **1.3635** | **0.9356** |
| LightGBM | 0.8529 | 0.7265 | 1.4904 | 0.9217 |

### Cross-Cancer Organoid LOOCV

| Dataset | Cancer Type | Organoids | Drugs | Pairs | Pearson r | R² | RMSE | AUROC |
|---------|------------|-----------|-------|-------|-----------|-----|------|-------|
| CRC-PDO | Colorectal | 16 | 34 | 544 | **0.890** | 0.792 | 1.311 | **0.941** |
| PDAC | Pancreatic | 38 | 64 | 2,432 | **0.741** | 0.545 | 0.096 | **0.831** |
| BLCA | Bladder | 11 | 30 | 265 | **0.640** | 0.389 | 1.967 | **0.868** |

> **CRC**: van de Wetering et al. (2015, *Science*) | **PDAC**: Chen et al. (2022, *Nat Commun*, GSE194249) | **BLCA**: Lee et al. (2018, *Cell*, GSE103990)

### Drug Generalization on PDAC

| Drug Category | Count | Mean Per-Drug Pearson r |
|--------------|-------|------------------------|
| GDSC-overlapping (known) | 12 | 0.558 |
| PDAC-specific (novel) | 52 | 0.524 |

The comparable prediction accuracy between known and novel drugs demonstrates that the Morgan fingerprint encoder captures transferable chemical structure representations, enabling zero-shot generalization to compounds not encountered during pretraining.

## Project Structure

```
GDRNet/
├── src/
│   ├── models/
│   │   ├── gdrnet.py                      # GDRNet model architecture (DCN v2)
│   │   └── baseline.py                    # Baseline models (LightGBM, RF, XGBoost)
│   ├── train.py                            # GDSC training (3-seed ensemble)
│   ├── finetune_organoid.py                # CRC organoid LOOCV fine-tuning
│   ├── finetune_multicancer.py             # Multi-cancer LOOCV (CRC + PDAC + BLCA)
│   ├── ablation.py                         # Ablation study (GDSC + Organoid)
│   ├── preprocess.py                       # GDSC data preprocessing
│   ├── features.py                         # Drug feature engineering (Morgan FP, PubChem)
│   ├── extract_scfoundation_emb.py         # scFoundation embedding extraction
│   ├── prepare_organoid_data.py            # CRC organoid data preparation
│   ├── prepare_multicancer_data.py         # Multi-cancer organoid data preparation
│   ├── resolve_pdac_smiles.py              # PDAC drug SMILES resolution (PubChem API)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Data Preparation

### GDSC Data

1. Download GDSC2 data from [cancerrxgene.org](https://www.cancerrxgene.org/)
2. Place raw data in `data/raw/gdsc/`
3. Run preprocessing:

```bash
python src/preprocess.py
python src/extract_scfoundation_emb.py
python src/features.py
```

### CRC Organoid Data

```bash
python src/prepare_organoid_data.py
```

### Multi-Cancer Organoid Data (PDAC + BLCA)

```bash
# Step 1: Resolve SMILES for PDAC drugs (queries PubChem API)
python src/resolve_pdac_smiles.py

# Step 2: Prepare PDAC and bladder cancer organoid data
# (requires scFoundation model in external/scFoundation/)
python src/prepare_multicancer_data.py
```

**Data sources:**
- **PDAC (GSE194249)**: Download FPKM expression and drug AUC data from [GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194249). Place in `data/raw/geo/GSE194249/`.
- **BLCA (GSE103990)**: Download normalized counts from GEO and drug response (Table S3) from [Lee et al. 2018, Cell](https://doi.org/10.1016/j.cell.2018.03.017). Place in `data/raw/geo/GSE103990/`.

## Training & Evaluation

### Train on GDSC

```bash
# Single GPU
python src/train.py --gpus 0 --epochs 300

# Multi-GPU (recommended)
python src/train.py --gpus 0,2,4 --epochs 300
```

An ensemble of 3 models (seeds 42, 123, 456) is trained and averaged.

### Fine-tune on CRC Organoid (LOOCV)

```bash
python src/finetune_organoid.py
```

Uses 16-fold leave-one-organoid-out cross-validation with frozen encoders.

### Multi-Cancer LOOCV (CRC + PDAC + BLCA)

```bash
python src/finetune_multicancer.py
```

Runs LOOCV fine-tuning on all three organoid datasets sequentially and produces:
- Per-dataset comparison tables (`results/tables/{crc,pdac,blca}_loocv_*.csv`)
- Combined cross-cancer summary (`results/tables/multicancer_loocv_comparison.csv`)

### Run Ablation Study

```bash
python src/ablation.py
```

Four ablation variants evaluated on both GDSC and organoid:
1. w/o ID Embeddings
2. w/o Cross Network (DCN v2)
3. w/o Deep Network (MLP)
4. w/o scF Embeddings

### Generate Figures

```bash
# Paper figures (GDSC + CRC organoid)
python src/generate_paper_figures_enhanced.py

# Multi-cancer comparison figures
python src/generate_multicancer_figures.py
```

## Model Architecture

```
Cell Branch:
  Gene Expression (2000-d) -> MLP Encoder -> 256-d
  scFoundation Embedding (3072-d) -> MLP Encoder -> 256-d
  Cell ID -> Embedding -> 64-d
  -> Concatenate -> 576-d cell representation

Drug Branch:
  Morgan Fingerprint (2048-bit) -> MLP Encoder -> 256-d
  Drug ID -> Embedding -> 64-d
  -> Concatenate -> 320-d drug representation

Interaction:
  Concat (896-d) -> Project (512-d)
  -> DCN v2 Cross Network (3 layers, rank=64)    |
  -> Deep MLP Network (3 ResBlocks, 512->64)     |-> Concat -> Output Head -> IC50

Transfer Learning:
  Pretrain on GDSC (964 cell lines × 229 drugs)
  → Freeze encoders → Fine-tune interaction layers on organoid LOOCV
```

## Datasets

| Dataset | Type | Samples | Drugs | Pairs | Response | Source |
|---------|------|---------|-------|-------|----------|--------|
| GDSC2 | Cell lines | 964 | 229 | ~199,000 | lnIC50 | GDSC + DepMap |
| CRC-PDO | Organoid | 16 | 34 | 544 | LogIC50 | van de Wetering 2015 |
| PDAC | Organoid | 38 | 64 | 2,432 | 1−AUC | Chen 2022, GSE194249 |
| BLCA | Organoid | 11 | 30 | 265 | LogIC50 | Lee 2018, GSE103990 |

## Citation

If you use this code, please cite:

```
[Citation to be added upon publication]
```

## License

MIT License
