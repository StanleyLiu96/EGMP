#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_train.py
Train a global 4-class offline classifier on full 19s trials:
Inputs: mixture audio features, EEG features, mixture metadata (gains/panning summaries)
Output: attended class in {Gt, Vx, Dr, Bs}

Model:
- Audio encoder (mixture-only): small TCN over [log-Mel L/R, ILD, IPD] -> sequence A_enc (B,T,D)
- EEG encoder: small TCN with short receptive field (captures 0-250ms lag) -> sequence E_enc -> temporal attention -> EEG token e (B,D)
- 4 instrument queries q_k built from learned instrument embeddings plus metadata projection -> cross-attend over A_enc -> audio token a_k
- Match scores s_k = MLP([e, a_k, e⊙a_k, |e-a_k|]); softmax over 4 -> CE loss

Includes:
- Optional presence-prior head from A_enc to encourage down-weighting absent classes (not a hard mask)
- Metadata dropout (set in config)
"""

import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from EGMP_util import EGMPDataset, collate_fulltrials

# -------------------------
# Config
# -------------------------
SEED = 1337
BATCH_SIZE = 4
EPOCHS = 40
LR = 2e-4
WD = 1e-4
D_MODEL = 192
N_HEADS = 4
D_FF = 384
D_QUERY = 64
META_DROPOUT_P = 0.4     # probability to zero metadata during training
PRESENCE_PRIOR_W = 0.2   # weight to add log p_k to class scores (soft prior). Set 0 to disable.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Utils
# -------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class TemporalConvBlock(nn.Module):
    def __init__(self, d_in, d_h, k=5, d_out=None, p=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.net = nn.Sequential(
            nn.Conv1d(d_in, d_h, kernel_size=k, padding=k//2),
            nn.GELU(),
            nn.Conv1d(d_h, d_out, kernel_size=k, padding=k//2),
            nn.GELU(),
            nn.Dropout(p)
        )
        self.proj = nn.Conv1d(d_in, d_out, kernel_size=1) if d_in != d_out else nn.Identity()

    def forward(self, x):  # (B, T, C)
        x = x.transpose(1,2)  # (B, C, T)
        y = self.net(x) + self.proj(x)
        return y.transpose(1,2)  # (B, T, C)

class TemporalAttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x):  # (B, T, D)
        a = self.w(x).squeeze(-1)              # (B, T)
        a = torch.softmax(a, dim=1)            # weights sum to 1
        pooled = torch.bmm(a.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled, a

class CrossAttentionQuery(nn.Module):
    """
    One-step cross-attention: query vector attends over audio sequence.
    """
    def __init__(self, d_model, d_query):
        super().__init__()
        self.q_proj = nn.Linear(d_query, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, q_vec, seq):  # q_vec: (B, d_query); seq: (B, T, D)
        Q = self.q_proj(q_vec).unsqueeze(1)  # (B, 1, D)
        K = self.k(seq)                      # (B, T, D)
        V = self.v(seq)                      # (B, T, D)
        attn = torch.softmax((Q @ K.transpose(1,2))/self.scale, dim=-1)  # (B,1,T)
        tok = (attn @ V).squeeze(1)  # (B, D)
        return tok, attn.squeeze(1)  # (B,T)

class EGMPModel(nn.Module):
    def __init__(self, fa, fe, meta_dim, d_model=D_MODEL, n_classes=4):
        super().__init__()
        self.n_classes = n_classes

        # Audio encoder
        self.a_in = nn.Linear(fa, d_model)
        self.a_tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
        self.a_tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)

        # Presence-prior head (from audio sequence)
        self.pres_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

        # EEG encoder
        self.e_in = nn.Linear(fe, d_model)
        self.e_tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
        self.e_tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
        self.pool = TemporalAttentionPool(d_model)

        # Instrument embeddings (Gt, Vx, Dr, Bs)
        self.inst_embed = nn.Embedding(n_classes, D_QUERY//2)

        # Metadata projection (pan histogram + gains summary + present mask + missing mask)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, D_QUERY//2),
            nn.GELU(),
            nn.Linear(D_QUERY//2, D_QUERY//2)
        )

        # Cross-attention per class
        self.cross = CrossAttentionQuery(d_model, D_QUERY)

        # Matcher MLP
        self.match_mlp = nn.Sequential(
            nn.Linear(4*d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, audio_seq, eeg_seq, meta_vec, meta_drop=0.0):
        """
        audio_seq: (B, T, Fa)
        eeg_seq:   (B, T, Fe)
        meta_vec:  (B, M)
        """
        B, T, _ = audio_seq.shape
        # Audio encoder
        A = self.a_in(audio_seq)      # (B,T,D)
        A = self.a_tcn1(A)
        A = self.a_tcn2(A)

        # Presence prior (soft)
        pres_logits_seq = self.pres_head(A)       # (B,T,4)
        pres_logits = pres_logits_seq.mean(dim=1) # (B,4)
        pres_logprob = torch.log_softmax(pres_logits, dim=-1)  # (B,4)

        # EEG encoder
        E = self.e_in(eeg_seq)        # (B,T,D)
        E = self.e_tcn1(E)
        E = self.e_tcn2(E)
        e_tok, attn_eeg = self.pool(E)  # (B,D), (B,T)

        # Metadata
        if self.training and meta_drop > 0.0:
            mask = (torch.rand_like(meta_vec[:, :1]) < meta_drop).float()
            meta_in = meta_vec * (1.0 - mask)
        else:
            meta_in = meta_vec
        m_tok = self.meta_proj(meta_in)  # (B, D_QUERY/2)

        # For each class, build query, cross-attend, and match
        scores = []
        attns = []
        for k in range(self.n_classes):
            inst_tok = self.inst_embed.weight[k].unsqueeze(0).expand(B, -1)  # (B, D_QUERY/2)
            qk = torch.cat([inst_tok, m_tok], dim=-1)  # (B, D_QUERY)
            a_k, attn = self.cross(qk, A)             # (B, D), (B,T)
            # Matching features
            feat = torch.cat([e_tok, a_k, e_tok*a_k, torch.abs(e_tok - a_k)], dim=-1)  # (B, 4D)
            s_k = self.match_mlp(feat).squeeze(-1)  # (B,)
            # Add soft presence prior
            if PRESENCE_PRIOR_W > 0:
                s_k = s_k + PRESENCE_PRIOR_W * pres_logprob[:, k]
            scores.append(s_k)
            attns.append(attn.unsqueeze(1))
        S = torch.stack(scores, dim=-1)  # (B, 4)
        A_attns = torch.cat(attns, dim=1)  # (B, 4, T) attention per class (for analysis)
        return S, pres_logits, attn_eeg, A_attns

# -------------------------
# Training
# -------------------------
def train_loop(args):
    set_seed(SEED)
    ds_tr = EGMPDataset(os.path.join(args.data_root, "manifest.csv"), split="train", split_seed=0, val_ratio=0.2)
    ds_va = EGMPDataset(os.path.join(args.data_root, "manifest.csv"), split="val",   split_seed=0, val_ratio=0.2)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fulltrials, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fulltrials, pin_memory=True)

    # infer feature dims from one sample
    samp = ds_tr[0]
    Fa = samp["audio"].shape[1]
    Fe = samp["eeg"].shape[1]
    M  = samp["meta"].shape[0]

    model = EGMPModel(fa=Fa, fe=Fe, meta_dim=M).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    bce = nn.BCEWithLogitsLoss()

    best_va = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(1, EPOCHS+1):
        model.train()
        t0 = time.time()
        tr_loss = tr_acc = n_tr = 0
        for batch in dl_tr:
            audio = batch["audio"].to(DEVICE, non_blocking=True)
            eeg   = batch["eeg"].to(DEVICE, non_blocking=True)
            meta  = batch["meta"].to(DEVICE, non_blocking=True)
            y     = batch["label"].to(DEVICE, non_blocking=True)
            present = batch["present"].to(DEVICE, non_blocking=True)

            S, pres_logits, _, _ = model(audio, eeg, meta, meta_drop=META_DROPOUT_P)
            loss_cls = ce(S, y)
            loss_pres = bce(pres_logits, present)
            loss = loss_cls + 0.2*loss_pres

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            with torch.no_grad():
                pred = S.argmax(dim=-1)
                acc = (pred == y).float().mean().item()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += acc * bs
            n_tr    += bs

        tr_loss /= max(1, n_tr)
        tr_acc  /= max(1, n_tr)

        # Validation
        model.eval()
        va_loss = va_acc = n_va = 0
        with torch.no_grad():
            for batch in dl_va:
                audio = batch["audio"].to(DEVICE, non_blocking=True)
                eeg   = batch["eeg"].to(DEVICE, non_blocking=True)
                meta  = batch["meta"].to(DEVICE, non_blocking=True)
                y     = batch["label"].to(DEVICE, non_blocking=True)
                present = batch["present"].to(DEVICE, non_blocking=True)

                S, pres_logits, _, _ = model(audio, eeg, meta, meta_drop=0.0)
                loss_cls = ce(S, y)
                loss_pres = bce(pres_logits, present)
                loss = loss_cls + 0.2*loss_pres

                pred = S.argmax(dim=-1)
                acc = (pred == y).float().mean().item()

                bs = y.size(0)
                va_loss += loss.item() * bs
                va_acc  += acc * bs
                n_va    += bs

        va_loss /= max(1, n_va)
        va_acc  /= max(1, n_va)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s")

        # Save best
        if va_acc > best_va:
            best_va = va_acc
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": va_acc},
                       os.path.join(args.ckpt_dir, "best.pt"))

    print(f"Best val acc: {best_va:.3f}")

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./EGMP_preprocessed", help="Folder containing manifest.csv")
    p.add_argument("--ckpt_dir", type=str, default="./EGMP_ckpt")
    args = p.parse_args()
    train_loop(args)
