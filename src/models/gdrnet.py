# -*- coding: utf-8 -*-
"""
GDRNet — CrossNet-Deep Drug Response Network
=================================================

Root cause: drug response prediction is tabular data. Tree-based methods win
because they directly capture feature interactions via axis-aligned splits.
Deep models need EXPLICIT feature crossing, not attention.

innovations:
  1. Cell/Drug ID Embeddings — matrix factorization effect,
     captures systematic biases that feature-based models miss.
  2. DCN v2 Cross Network — explicit feature crossing that mimics
     tree split behavior (proven to match GBDT on tabular benchmarks).
  3. Simple MLP encoders — no attention, no GNN, no MoE.
     Less complexity = less overfitting risk on ~170K tabular samples.
  4. BatchNorm + ReLU — proven better for tabular than LayerNorm + SELU.
  5. Ensemble of 3 diverse models — variance reduction.

References:
  - DCN v2: Wang et al., "DCN V2: Improved Deep & Cross Network" (WWW 2021)
  - Tabular DL: Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path("/export/home/kongyan/project/Organoid/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── DCN v2 Low-Rank Cross Layer ──────────────────────────────────────────────

class CrossLayer(nn.Module):
    """
    x_{l+1} = x_0 ⊙ (U_l(V_l(x_l)) + b_l) + x_l

    Low-rank factorization of the weight matrix reduces parameters from
    O(d^2) to O(d*r) while maintaining expressiveness.
    """
    def __init__(self, dim, rank=64):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=False)
        self.V = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)

    def forward(self, x0, xl):
        return x0 * self.V(F.relu(self.U(xl))) + self.bias + xl


# ─── Residual MLP Block ──────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)
        self.drop = nn.Dropout(dropout)
        self.shortcut = (nn.Linear(d_in, d_out) if d_in != d_out
                         else nn.Identity())
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        if isinstance(self.shortcut, nn.Linear):
            nn.init.kaiming_normal_(self.shortcut.weight, nonlinearity='linear')

    def forward(self, x):
        return self.drop(F.relu(self.bn(self.fc(x)))) + self.shortcut(x)


# ─── Simple MLP Encoder ──────────────────────────────────────────────────────

class MLPEncoder(nn.Module):
    """2-layer MLP with BatchNorm: in_dim → hidden → out_dim."""
    def __init__(self, in_dim, hidden, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        nn.init.kaiming_normal_(self.net[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.net[4].weight, nonlinearity='linear')

    def forward(self, x):
        return self.net(x)


# ─── GDRNet ────────────────────────────────────────────────────────────────

class GDRNet(nn.Module):
    """
    Architecture:
      Cell path: gene MLP + scF MLP + cell_id_emb → cell_repr
      Drug path: FP MLP + desc MLP + drug_id_emb → drug_repr
      Interaction: concat → project → DCN v2 cross + Deep MLP → predict
    """
    def __init__(
        self,
        n_genes=2000,
        scf_dim=3072,
        fp_bits=2048,
        n_desc=195,
        n_cells=700,
        n_drugs=286,
        d_hidden=256,
        id_emb_dim=64,
        n_cross=3,
        cross_rank=64,
        n_deep=3,
        dropout=0.15,
    ):
        super().__init__()

        # ── Cell encoders ──
        self.gene_enc = MLPEncoder(n_genes, 512, d_hidden, dropout)
        self.scf_enc = MLPEncoder(scf_dim, 512, d_hidden, dropout)
        self.cell_emb = nn.Embedding(n_cells + 1, id_emb_dim, padding_idx=0)
        nn.init.normal_(self.cell_emb.weight, std=0.02)

        # ── Drug encoders ──
        self.fp_enc = MLPEncoder(fp_bits, 512, d_hidden, dropout)
        self.desc_enc = MLPEncoder(n_desc, 128, id_emb_dim, dropout)
        self.drug_emb = nn.Embedding(n_drugs + 1, id_emb_dim, padding_idx=0)
        nn.init.normal_(self.drug_emb.weight, std=0.02)

        # ── Combined dimensions ──
        cell_dim = d_hidden * 2 + id_emb_dim   # 256*2 + 64 = 576
        drug_dim = d_hidden + id_emb_dim * 2    # 256 + 64*2 = 384
        combined = cell_dim + drug_dim           # 960

        # Project to working dim
        self.input_proj = nn.Sequential(
            nn.BatchNorm1d(combined),
            nn.Linear(combined, 512),
            nn.ReLU(inplace=True),
        )

        # ── DCN v2 Cross Network ──
        self.cross_layers = nn.ModuleList([
            CrossLayer(512, rank=cross_rank) for _ in range(n_cross)
        ])

        # ── Deep Network ──
        deep_cfg = [(512, 256), (256, 128), (128, 64)]
        self.deep_layers = nn.ModuleList([
            ResBlock(di, do, dropout) for di, do in deep_cfg
        ])

        # ── Output head ──
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx):
        # Cell
        h_gene = self.gene_enc(x_gene)
        h_scf = self.scf_enc(scf)
        h_cid = self.cell_emb(cell_idx)
        cell_repr = torch.cat([h_gene, h_scf, h_cid], dim=-1)

        # Drug
        h_fp = self.fp_enc(x_fp)
        h_desc = self.desc_enc(x_desc)
        h_did = self.drug_emb(drug_idx)
        drug_repr = torch.cat([h_fp, h_desc, h_did], dim=-1)

        # Combine
        x0 = self.input_proj(torch.cat([cell_repr, drug_repr], dim=-1))

        # Cross network
        xc = x0
        for layer in self.cross_layers:
            xc = layer(x0, xc)

        # Deep network
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)

        return self.head(torch.cat([xc, xd], dim=-1)).squeeze(-1)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class Dataset(Dataset):
    def __init__(self, x_gene, scf, x_fp, x_desc, cell_idx, drug_idx, y):
        self.x_gene = torch.FloatTensor(x_gene)
        self.scf = torch.FloatTensor(scf)
        self.x_fp = torch.FloatTensor(x_fp)
        self.x_desc = torch.FloatTensor(x_desc)
        self.cell_idx = torch.LongTensor(cell_idx)
        self.drug_idx = torch.LongTensor(drug_idx)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.x_gene[i], self.scf[i], self.x_fp[i], self.x_desc[i],
                self.cell_idx[i], self.drug_idx[i], self.y[i])


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x_gene, scf, x_fp, x_desc, ci, di, y in loader:
        batch = [x_gene, scf, x_fp, x_desc, ci, di]
        out = model(*[t.to(device) for t in batch])
        preds.extend(out.cpu().numpy())
        targets.extend(y.numpy())
    preds = np.array(preds, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    r2 = float(r2_score(targets, preds))
    pearson = float(np.corrcoef(targets, preds)[0, 1])
    thr = np.percentile(targets, 30)
    try:
        auroc = float(roc_auc_score((targets <= thr).astype(int), -preds))
    except Exception:
        auroc = float("nan")
    return rmse, r2, pearson, auroc, preds


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(
    model,
    tr_ds, val_ds,
    model_name="gdr",
    n_epochs=300,
    batch_size=1024,
    lr=1e-3,
    patience=40,
    weight_decay=0.01,
    warmup_frac=0.05,
    device=DEVICE,
):
    import pandas as pd

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    total_steps = n_epochs * len(tr_loader)
    warmup_steps = int(warmup_frac * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rmse = float("inf")
    best_state = None
    wait = 0
    history = []

    print(f"\n  {'='*60}")
    print(f"  GDRNet  params={n_params/1e6:.2f}M  device={device}")
    print(f"  Epochs={n_epochs}  BS={batch_size}  LR={lr:.0e}  Patience={patience}")
    print(f"  Train={len(tr_ds):,}  Val={len(val_ds):,}")
    print(f"  {'='*60}")

    t0 = time.time()
    last_print = t0

    for epoch in range(n_epochs):
        model.train()
        tr_loss = 0.0
        n_batch = 0

        for x_gene, scf, x_fp, x_desc, ci, di, y_b in tr_loader:
            batch = [x_gene, scf, x_fp, x_desc, ci, di]
            batch = [t.to(device) for t in batch]
            y_b = y_b.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(*batch)
                loss = F.huber_loss(out, y_b, delta=1.0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tr_loss += loss.item()
            n_batch += 1

            # Heartbeat: print dot every 90s to prevent PuTTY disconnect
            now = time.time()
            if now - last_print > 90:
                print(f"    ... epoch {epoch+1} batch {n_batch}/{len(tr_loader)} "
                      f"loss={loss.item():.4f} elapsed={now-t0:.0f}s")
                last_print = now

        # Epoch eval
        rmse, r2, pearson, auroc, _ = evaluate(model, val_loader, device)
        avg_loss = tr_loss / max(1, n_batch)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history.append(dict(
            epoch=epoch + 1, train_loss=round(avg_loss, 4),
            val_rmse=round(rmse, 4), val_r2=round(r2, 4),
            val_pearson=round(pearson, 4), val_auroc=round(auroc, 4),
        ))

        print(f"  Ep {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  "
              f"RMSE={rmse:.4f}  R2={r2:.4f}  P={pearson:.4f}  "
              f"AUROC={auroc:.4f}  lr={cur_lr:.2e}  [{elapsed:.0f}s]")
        last_print = time.time()

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop @ epoch {epoch+1}  best RMSE={best_rmse:.4f}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final eval
    rmse, r2, pearson, auroc, val_preds = evaluate(model, val_loader, device)
    print(f"\n  --- Best weights ---")
    print(f"  RMSE={rmse:.4f}  R2={r2:.4f}  Pearson={pearson:.4f}  AUROC={auroc:.4f}")

    save_path = MODELS_DIR / f"{model_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved -> {save_path}")

    metrics = dict(RMSE=round(rmse, 4), R2=round(r2, 4),
                   Pearson=round(pearson, 4), AUROC=round(auroc, 4))
    return model, metrics, pd.DataFrame(history), val_preds
