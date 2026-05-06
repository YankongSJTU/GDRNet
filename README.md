# GDRNet: Graph-Drug Response Network for Drug Sensitivity Prediction

GDRNet is a deep learning model for predicting drug sensitivity (IC50) in cancer cell lines and patient-derived organoids.

## Key Features

- **DCN v2 Cross Network**: Explicit feature crossing via low-rank factorization for cell-drug interaction modeling
- **Multi-modal Input**: Integrates gene expression, scFoundation single-cell embeddings, Morgan fingerprints, and RDKit molecular descriptors
- **Cell/Drug ID Embeddings**: Learnable entity embeddings capturing systematic biases
- **Transfer Learning**: Pre-trained on GDSC, fine-tuned on patient-derived organoid data with frozen encoders

## Architecture

```
Cell Branch: Gene Expr (2000-d) + scF Emb (3072-d) + Cell ID (64-d) → 384-d
Drug Branch: Morgan FP (2048-bit) + RDKit Desc (188-d) + Drug ID (64-d) → 384-d
Interaction: Input Projection (768→128) → DCN v2 Cross (3 layers) + Deep MLP (3 layers) → Output (IC50)
```

## Performance

| Dataset | Method | Pearson | R2 | RMSE | AUROC |
|---------|--------|---------|-----|------|-------|
| GDSC | Ensemble | 0.8848 | 0.7806 | 1.3165 | 0.9392 |
| GDSC | LightGBM | 0.8795 | 0.7731 | 1.3387 | 0.9360 |
| Organoid LOOCV | FineTune | 0.8754 | 0.7658 | 1.3891 | 0.9297 |
| Organoid LOOCV | LightGBM | 0.8331 | 0.6902 | 1.5977 | 0.9091 |

## Project Structure

```
GDRNet/
├── src/
│   ├── models/
│   │   └── gdr.py          # V11 model architecture
│   ├── train.py            # GDSC training with multi-GPU
│   ├── finetune_organoid.py    # LOOCV fine-tuning on organoid data
│   ├── ablation_organoid.py    # Ablation study
│   ├── infer_organoid.py       # Direct inference on organoid data
│   └── generate_paper_figures.py  # Paper figure generation
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- scikit-learn
- pandas, numpy
- rdkit
- matplotlib, seaborn
- lightgbm

See `requirements.txt` for full list.

## Data

The model uses data from:
- **GDSC** (Genomics of Drug Sensitivity in Cancer): ~177K cell line-drug pairs
- **Organoid**: 16 patient-derived organoids × 34 drugs = 544 pairs

Due to data sharing agreements, raw data is not included in this repository. Please download GDSC data from [https://www.cancerrxgene.org/](https://www.cancerrxgene.org/).

## Usage

### Training on GDSC

```bash
python src/train.py --gpus 0,1,2 --epochs 300 --n_models 3
```

### Fine-tuning on Organoid Data (LOOCV)

```bash
python src/finetune_organoid.py
```

### Ablation Study

```bash
python src/ablation_organoid.py
```

### Generate Paper Figures

```bash
python src/generate_paper_figures.py
```

## Citation

If you use this code, please cite:

```
[Paper citation to be added upon publication]
```

## License

MIT License
