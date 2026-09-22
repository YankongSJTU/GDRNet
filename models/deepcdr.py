# -*- coding: utf-8 -*-
"""
DeepCDR Baseline — PyTorch Adaptation
=========================================
Adapted from external/scFoundation/DeepCDR/prog/model.py (Keras).
Preserves the distinctive 1D-conv interaction head from DeepCDR.

Key changes from original:
- Drug: Morgan FP MLP replaces molecular graph GCN
- Cell: gene expression MLP (no mutation/methylation branches)
- Interaction head: 1D-conv layers preserved (signature DeepCDR feature)

Key difference from GDRNet:
- 1D-conv interaction head (vs. DCN v2 + Deep MLP)
- No ID embeddings, no scFoundation
- tanh activation in encoders (DeepCDR convention)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DrugEncoder(nn.Module):
    """Drug encoder: FP → Dense(tanh) → BN → Drop → Dense(relu) → BN → Drop."""
    def __init__(self, fp_bits=2048, hidden=512, out_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(fp_bits),
            nn.Linear(fp_bits, hidden),
            nn.Tanh(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CellEncoder(nn.Module):
    """Cell encoder: gene → Dense(tanh) → BN → Drop → Dense(relu)."""
    def __init__(self, n_genes=2000, hidden=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_genes),
            nn.Linear(n_genes, hidden),
            nn.Tanh(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvInteractionHead(nn.Module):
    """1D-conv interaction head — the distinctive DeepCDR component.

    Takes the concatenated drug+cell representation, reshapes to (B, 1, dim),
    then applies 3 Conv1d layers with decreasing channels and pooling.
    This captures local interaction patterns between cell and drug features.
    """
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        # in_dim = drug_out_dim + cell_out_dim = 256 + 128 = 384
        self.pre_fc = nn.Sequential(
            nn.Linear(in_dim, 300),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.conv = nn.Sequential(
            # Conv block 1: 1 → 30 channels
            nn.Conv1d(1, 30, kernel_size=150, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            # Conv block 2: 30 → 10 channels
            nn.Conv1d(30, 10, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            # Conv block 3: 10 → 5 channels
            nn.Conv1d(10, 5, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
        )
        # After conv: depends on input length 300
        # L1: 300-150+1=151 → pool/2 → 75
        # L2: 75-5+1=71 → pool/3 → 23
        # L3: 23-5+1=19 → pool/3 → 6
        # Output: 5 * 6 = 30
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(30, 1),
        )

    def forward(self, x):
        h = self.pre_fc(x)                          # (B, 300)
        h = h.unsqueeze(1)                           # (B, 1, 300)
        h = self.conv(h)                             # (B, 5, ~6)
        h = h.reshape(h.size(0), -1)                 # (B, 30)
        return self.head(h).squeeze(-1)              # (B,)


class DeepCDR(nn.Module):
    """
    DeepCDR (PyTorch adaptation) with our feature set.

    Drug: FP(2048) → tanh+BN encoder → 256
    Cell: gene(2000) → tanh+BN encoder → 128
    Combined: concat(384) → tanh → 1D-conv interaction → 1
    """
    def __init__(
        self,
        n_genes=2000,
        fp_bits=2048,
        drug_out=256,
        cell_out=128,
        dropout=0.1,
    ):
        super().__init__()
        self.drug_enc = DrugEncoder(fp_bits, 512, drug_out, dropout)
        self.cell_enc = CellEncoder(n_genes, 256, cell_out, dropout)
        self.interaction = ConvInteractionHead(drug_out + cell_out, dropout)

    def forward(self, x_gene, x_fp):
        h_drug = self.drug_enc(x_fp)     # (B, 256)
        h_cell = self.cell_enc(x_gene)   # (B, 128)
        return self.interaction(torch.cat([h_drug, h_cell], dim=-1))


class DeepCDRDataset(Dataset):
    def __init__(self, x_gene, x_fp, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.x_fp = torch.FloatTensor(x_fp)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_gene[i], self.x_fp[i], self.y[i]


def train_deepcdr(
    model,
    tr_ds, te_ds,
    model_name="deepcdr",
    n_epochs=300,
    batch_size=1024,
    lr=1e-3,
    patience=40,
    weight_decay=0.01,
    warmup_frac=0.05,
    device=DEVICE,
):
    """Training loop matching GDRNet's recipe."""
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = n_epochs * len(tr_loader)
    warmup_steps = int(warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_rmse = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        for x_gene, x_fp, y_b in tr_loader:
            x_gene, x_fp, y_b = x_gene.to(device), x_fp.to(device), y_b.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x_gene, x_fp)
                loss = F.huber_loss(out, y_b, delta=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_gene, x_fp, _ in te_loader:
                x_gene, x_fp = x_gene.to(device), x_fp.to(device)
                out = model(x_gene, x_fp)
                preds.extend(out.cpu().numpy())
                targets.extend(_.numpy())

        rmse = float(np.sqrt(((np.array(targets) - np.array(preds)) ** 2).mean()))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final predictions
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_gene, x_fp, _ in te_loader:
            x_gene, x_fp = x_gene.to(device), x_fp.to(device)
            out = model(x_gene, x_fp)
            preds.extend(out.cpu().numpy())
            targets.extend(_.numpy())

    return model, np.array(preds, dtype=np.float32)
