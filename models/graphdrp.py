# -*- coding: utf-8 -*-
"""
GraphDRP Baseline — Simplified MLP Adaptation
================================================
Original GraphDRP uses GNN for drugs + MLP for cell lines.
Since we use Morgan FPs (not molecular graphs), we replace the GNN drug
encoder with an MLP, preserving the core design: separate cell/drug
encoding via MLP → concat → interaction MLP → predict.

Key difference from GDRNet:
- No DCN v2 feature crossing (simple concat + MLP)
- No ID embeddings (no cell_id_emb or drug_id_emb)
- No scFoundation embeddings (gene expression only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DrugEncoder(nn.Module):
    """MLP encoder for drug Morgan fingerprints."""
    def __init__(self, fp_bits=2048, hidden=512, out_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(fp_bits),
            nn.Linear(fp_bits, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[4].weight, nonlinearity='linear')

    def forward(self, x):
        return self.net(x)


class CellEncoder(nn.Module):
    """MLP encoder for cell line gene expression."""
    def __init__(self, n_genes=2000, hidden=512, out_dim=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_genes),
            nn.Linear(n_genes, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[4].weight, nonlinearity='linear')

    def forward(self, x):
        return self.net(x)


class GraphDRP(nn.Module):
    """
    GraphDRP (simplified MLP variant).
    Drug: FP(2048) → MLP → 256
    Cell: gene(2000) → MLP → 256
    Combined: concat(512) → MLP(512→256→128) → 1
    """
    def __init__(
        self,
        n_genes=2000,
        fp_bits=2048,
        hidden=512,
        repr_dim=256,
        dropout=0.15,
    ):
        super().__init__()
        self.drug_enc = DrugEncoder(fp_bits, hidden, repr_dim, dropout)
        self.cell_enc = CellEncoder(n_genes, hidden, repr_dim, dropout)

        combined = repr_dim * 2  # 512
        self.head = nn.Sequential(
            nn.BatchNorm1d(combined),
            nn.Linear(combined, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_gene, x_fp):
        h_drug = self.drug_enc(x_fp)
        h_cell = self.cell_enc(x_gene)
        return self.head(torch.cat([h_cell, h_drug], dim=-1)).squeeze(-1)


class GraphDRPDataset(Dataset):
    def __init__(self, x_gene, x_fp, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.x_fp = torch.FloatTensor(x_fp)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_gene[i], self.x_fp[i], self.y[i]


def train_graphdrp(
    model,
    tr_ds, te_ds,
    model_name="graphdrp",
    n_epochs=300,
    batch_size=1024,
    lr=1e-3,
    patience=40,
    weight_decay=0.01,
    warmup_frac=0.05,
    device=DEVICE,
):
    """Training loop matching GDRNet's recipe for fair comparison."""
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
