# -*- coding: utf-8 -*-
"""
DrugCell Baseline — Simplified Hierarchical Adaptation
=======================================================
Original DrugCell uses a 'visible neural network' over the Gene Ontology
DAG where each GO term is a neuron. We implement a simplified version that
preserves the core concept:
  - Hierarchical cell encoder with skip connections (mimics GO levels)
  - Drug FP encoder
  - Element-wise multiply interaction (DrugCell's signature mechanism)
  - MLP prediction head

Key difference from GDRNet:
- Hierarchical cell encoding (3 levels with skip connections)
- Multiplicative interaction (element-wise multiply, not DCN v2 cross)
- No ID embeddings, no scFoundation, no explicit feature crossing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class HierarchicalCellEncoder(nn.Module):
    """3-level hierarchical cell encoder mimicking GO graph levels.

    Level 1 (broad): 2000 → 512
    Level 2 (mid): 512 → 256
    Level 3 (specific): 256 → 128
    Skip connections from level 1 and 2 to final output.
    """
    def __init__(self, n_genes=2000, dropout=0.15):
        super().__init__()
        self.level1 = nn.Sequential(
            nn.BatchNorm1d(n_genes),
            nn.Linear(n_genes, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.level2 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.level3 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Project skip connections to match level 3 dim (128)
        self.skip1_proj = nn.Linear(512, 128)
        self.skip2_proj = nn.Linear(256, 128)
        # Final fusion: 128*3 = 384 → 128
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(384),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(self.level1[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.level2[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.level3[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.skip1_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.skip2_proj.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.fusion[1].weight, nonlinearity='relu')

    def forward(self, x):
        h1 = self.level1(x)      # (B, 512)
        h2 = self.level2(h1)     # (B, 256)
        h3 = self.level3(h2)     # (B, 128)
        s1 = self.skip1_proj(h1) # (B, 128)
        s2 = self.skip2_proj(h2) # (B, 128)
        return self.fusion(torch.cat([h3, s2, s1], dim=-1))  # (B, 128)


class DrugEncoder(nn.Module):
    """MLP encoder for drug Morgan fingerprints."""
    def __init__(self, fp_bits=2048, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(fp_bits),
            nn.Linear(fp_bits, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[4].weight, nonlinearity='relu')

    def forward(self, x):
        return self.net(x)


class DrugCellSimplified(nn.Module):
    """
    Simplified DrugCell adapted to our feature set.

    Cell: gene(2000) → hierarchical encoder → 128
    Drug: FP(2048) → MLP → 128
    Interaction: element-wise multiply (cell * drug) → 128
    Prediction: MLP(128 → 64 → 1)
    """
    def __init__(self, n_genes=2000, fp_bits=2048, dropout=0.15):
        super().__init__()
        self.cell_enc = HierarchicalCellEncoder(n_genes, dropout)
        self.drug_enc = DrugEncoder(fp_bits, dropout)
        self.head = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        nn.init.kaiming_normal_(self.head[1].weight, nonlinearity='relu')

    def forward(self, x_gene, x_fp):
        h_cell = self.cell_enc(x_gene)   # (B, 128)
        h_drug = self.drug_enc(x_fp)     # (B, 128)
        interaction = h_cell * h_drug      # element-wise multiply
        return self.head(interaction).squeeze(-1)


class DrugCellDataset(Dataset):
    def __init__(self, x_gene, x_fp, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.x_fp = torch.FloatTensor(x_fp)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_gene[i], self.x_fp[i], self.y[i]


def train_drugcell(
    model,
    tr_ds, te_ds,
    model_name="drugcell",
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
