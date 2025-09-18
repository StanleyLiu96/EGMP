#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_train.py

Train a global 4-class offline classifier using full 19s trials:
Inputs: mixture audio features, EEG features
Output: attended class in {Gt, Vx, Dr, Bs}

Pipeline:
---------
1. Audio encoder
   - Encode mixture audio (log-Mel L/R, ILD, IPD) with small TCN
   - Output: audio sequence A_enc (B,T,D)

2. Presence-prior head
   - Predict which instruments are present in the mixture
   - Used as a soft prior (boost true classes, penalize absent ones)

3. EEG encoder
   - Two modes, controlled by USE_EEG_CNN:
       * False → flatten spectrogram (F*C) per frame, encode with Linear+TCN
       * True  → keep full 2D (F,C) spectrogram, encode with CNN
   - Apply temporal attention pooling → one EEG token e_tok (B,D)

4. Instrument cross-attention
   - Each instrument (Gt, Vx, Dr, Bs) has a learnable embedding vector
   - Embedding acts as a query into audio sequence → instrument-specific audio token a_k
   - Combine (e_tok, a_k) into joint features and score alignment

5. Classification
   - Collect 4 scores → logits (B,4)
   - Softmax during training → CrossEntropyLoss

Training:
---------
- Logs to TensorBoard
- Saves both "Latest" and "Best" checkpoints
- Supports resume
"""

import argparse
import glob
import math
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from EGMP_util import EGMPDataset, collate_fulltrials
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 10
D_MODEL = 192
D_QUERY = 64
EPOCHS = 200
LR = 2e-4
PRESENCE_PRIOR_W = 0.2  # weight for presence prior, 0 = disable
SEED = 1337
USE_EEG_CNN = False  # True = use 2D CNN encoder, False = use Linear+TCN
WD = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

# -------------------------
# Utils
# -------------------------
def set_seed(s):
    """Set random seeds for reproducibility."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

# -------------------------
# Model components
# -------------------------
class TemporalConvBlock(nn.Module):
    """Small 1D temporal convolution block with residual connection."""
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

    def forward(self, x):  # (B,T,C)
        x = x.transpose(1,2)              # (B,C,T)
        y = self.net(x) + self.proj(x)    # residual
        return y.transpose(1,2)           # (B,T,C)

class TemporalAttentionPool(nn.Module):
    """Learn attention weights over time → weighted sum of sequence."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x):  # (B,T,D)
        a = self.w(x).squeeze(-1)           # (B,T)
        a = torch.softmax(a, dim=1)         # normalize weights
        pooled = torch.bmm(a.unsqueeze(1), x).squeeze(1)  # (B,D)
        return pooled, a

class CrossAttentionQuery(nn.Module):
    """
    One-step cross-attention:
    - Query: instrument embedding
    - Keys/Values: audio sequence
    - Output: instrument-specific audio token
    """
    def __init__(self, d_model, d_query):
        super().__init__()
        self.q_proj = nn.Linear(d_query, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, q_vec, seq):  # q_vec: (B,d_query), seq: (B,T,D)
        Q = self.q_proj(q_vec).unsqueeze(1)    # (B,1,D)
        K = self.k(seq)                        # (B,T,D)
        V = self.v(seq)                        # (B,T,D)
        attn = torch.softmax((Q @ K.transpose(1,2))/self.scale, dim=-1)  # (B,1,T)
        tok = (attn @ V).squeeze(1)            # (B,D)
        return tok, attn.squeeze(1)            # (B,T)

class EEG2DCNNEncoder(nn.Module):
    """
    Encode EEG time × freq × channel features using 2D convolutions.
    Assumes input shape: (B, T, F, C) where F = #freq bins, C = #channels.
    Internally reshaped into (B, 1, T, F*C) for Conv2d.
    """
    def __init__(self, fe, d_model):
        super().__init__()
        self.fe = fe
        # Simple 2D conv stack → reduce F*C into d_model
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5,5), stride=(2,2), padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((None, 1))  # collapse freq*chan axis → keep time
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        # Input: (B,T,F,C)
        B, T, F, C = x.shape
        x = x.view(B, 1, T, F*C)      # (B,1,T,F*C)
        y = self.conv(x)              # (B,64,T',1)
        y = y.squeeze(-1).transpose(1,2)  # (B,T',64)
        return self.proj(y)           # (B,T',D)

# -------------------------
# Main model
# -------------------------
class EGMPModel(nn.Module):
    def __init__(self, fa, fe, d_model=D_MODEL, n_classes=4):
        """
        fa = audio feature dimension per frame
        fe = EEG feature dimension per frame
            - If USE_EEG_CNN=False → fe = flattened size (F*C, e.g. 2580 for 129×20)
            - If USE_EEG_CNN=True  → fe = tuple (F,C), passed as (129,20)
        """
        super().__init__()
        self.n_classes = n_classes

        # ---- Audio encoder ----
        self.a_in = nn.Linear(fa, d_model)
        self.a_tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
        self.a_tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)

        # Presence-prior head (predicts which instruments exist in mixture)
        self.pres_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

        # ---- EEG encoder ----
        if USE_EEG_CNN:
            self.eeg_encoder = EEG2DCNNEncoder(fe, d_model)
            self.pool = TemporalAttentionPool(d_model)
        else:
            self.e_in = nn.Linear(fe, d_model)
            self.e_tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
            self.e_tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
            self.pool = TemporalAttentionPool(d_model)

        # ---- Instrument embeddings (Gt,Vx,Dr,Bs) ----
        self.inst_embed = nn.Embedding(n_classes, D_QUERY//2)

        # ---- Cross-attention + Matching ----
        self.cross = CrossAttentionQuery(d_model, D_QUERY//2)
        self.match_mlp = nn.Sequential(
            nn.Linear(4*d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, audio_seq, eeg_seq):
        B, T, _ = audio_seq.shape

        # ===== AUDIO ENCODER =====
        A = self.a_in(audio_seq)  # (B,T,D)
        A = self.a_tcn1(A)
        A = self.a_tcn2(A)

        # ===== PRESENCE PRIOR =====
        pres_logits_seq = self.pres_head(A)          # (B,T,4)
        pres_logits = pres_logits_seq.mean(dim=1)    # (B,4)
        pres_logprob = torch.log_softmax(pres_logits, dim=-1)

        # ===== EEG ENCODER =====
        if USE_EEG_CNN:
            E = self.eeg_encoder(eeg_seq)     # (B,T',D)
        else:
            E = self.e_in(eeg_seq)            # (B,T,D)
            E = self.e_tcn1(E)
            E = self.e_tcn2(E)

        e_tok, attn_eeg = self.pool(E)        # (B,D), (B,T or T')

        # ===== MATCHING LOOP =====
        scores, attns = [], []
        for k in range(self.n_classes):
            inst_tok = self.inst_embed.weight[k].unsqueeze(0).expand(B, -1)  # (B,Dq/2)
            a_k, attn = self.cross(inst_tok, A)                              # (B,D), (B,T)
            feat = torch.cat([e_tok, a_k, e_tok*a_k, torch.abs(e_tok-a_k)], dim=-1)
            s_k = self.match_mlp(feat).squeeze(-1)
            if PRESENCE_PRIOR_W > 0:
                s_k = s_k + PRESENCE_PRIOR_W * pres_logprob[:, k]
            scores.append(s_k)
            attns.append(attn.unsqueeze(1))   # (B,1,T)

        # ===== FINAL STACK =====
        S = torch.stack(scores, dim=-1)       # (B,4)
        A_attns = torch.cat(attns, dim=1)     # (B,4,T)

        return S, pres_logits, attn_eeg, A_attns

# -------------------------
# Training loop
# -------------------------
def train_loop(args):
    set_seed(SEED)

    # ---- Dataset & DataLoaders ----
    val_txt = os.path.join(args.data_root, "val_split_filenames.txt")
    ds_tr = EGMPDataset(os.path.join(args.data_root, "manifest.csv"), split="train",
                        split_seed=0, val_ratio=0.2, val_trial_ids_path=val_txt)
    ds_va = EGMPDataset(os.path.join(args.data_root, "manifest.csv"), split="val",
                        split_seed=0, val_ratio=0.2, val_trial_ids_path=val_txt)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=1, collate_fn=collate_fulltrials, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=1, collate_fn=collate_fulltrials, pin_memory=True)

    # Infer feature dims from a sample
    # - Audio: always (T_audio, Fa)
    # - EEG:
    #     * If USE_EEG_CNN=False → flatten (T_eeg, F*C)
    #     * If USE_EEG_CNN=True  → keep structured (T_eeg, F, C)
    samp = ds_tr[0]
    Fa = samp["audio"].shape[1]
    if USE_EEG_CNN:
        Fe = samp["eeg"].shape[1:]   # (129,20)
    else:
        Fe = samp["eeg"].shape[1] * samp["eeg"].shape[2]  # 129*20=2580

    # ---- Model & Optimizer ----
    model = EGMPModel(fa=Fa, fe=Fe).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    bce = nn.BCEWithLogitsLoss()

    # ---- Logging dirs ----
    tb_dir = os.path.join(args.ckpt_dir, "tensorboard")
    best_dir = os.path.join(args.ckpt_dir, "Best")
    latest_dir = os.path.join(args.ckpt_dir, "Latest")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    # ---- Resume checkpoints ----
    # Load best model (for comparison only)
    best_va = best_tr = 0.0
    best_ckpts = sorted([f for f in os.listdir(best_dir) if f.endswith(".pt")],
                        key=lambda x: int(x.replace(".pt",""))) 
    if best_ckpts:
        ckpt = torch.load(os.path.join(best_dir, best_ckpts[-1]), map_location="cpu")
        best_va, best_tr = ckpt["val_acc"], ckpt["train_acc"]
        print(f"Best Validation Acc: {best_va:.3f}")
        print(f"Best Training Acc: {best_tr:.3f}")

    # Load latest model (to resume training)
    start_epoch = 1
    latest_ckpts = sorted([f for f in os.listdir(latest_dir) if f.endswith(".pt")],
                          key=lambda x: int(x.replace(".pt","")))
    if latest_ckpts:
        ckpt = torch.load(os.path.join(latest_dir, latest_ckpts[-1]), map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"Resuming from {latest_ckpts[-1]} at epoch {start_epoch}")
    else:
        print("Starting from scratch.")

    # ---- Resume TensorBoard ----
    old_tb_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))

    if len(old_tb_files) == 0:
        # Case 1: No old TensorBoard logs, start fresh
        writer = SummaryWriter(log_dir=tb_dir)
        print("No old TensorBoard logs found, starting fresh.")

    elif len(old_tb_files) == 1:
        # Case 2: Found exactly one old TensorBoard log
        old_tb_file = old_tb_files[0]
        print(f"Found old TensorBoard file: {old_tb_file}")

        # Load old events
        ea = event_accumulator.EventAccumulator(old_tb_file)
        ea.Reload()

        # Create a new writer in the same folder (new file)
        writer = SummaryWriter(log_dir=tb_dir)

        # Re-log all scalars up to the last finished epoch
        # NOTE: training resumes from start_epoch, so we only re-log <= start_epoch-1
        for tag in ea.Tags()['scalars']:
            for event in ea.Scalars(tag):
                if event.step <= start_epoch - 1:
                    writer.add_scalar(tag, event.value, event.step)

        writer.flush()
        print(f"Re-logged TensorBoard scalars up to epoch {start_epoch-1}")

        # Delete old tfevents file so we only keep one
        os.remove(old_tb_file)
        print(f"Deleted old TensorBoard file. Continuing logging from epoch {start_epoch}")

    else:
        # Case 3: Unexpected (multiple tfevent files in tb_dir)
        raise RuntimeError(f"Expected 0 or 1 tfevents file in {tb_dir}, found {len(old_tb_files)}")

    # ---- Training epochs ----
    for epoch in range(start_epoch, EPOCHS+1):
        # ===== TRAIN =====
        model.train()
        t0 = time.time()
        tr_loss = tr_acc = n_tr = 0
        for batch in dl_tr:
            audio = batch["audio"].to(DEVICE, non_blocking=True)
            eeg = batch["eeg"].to(DEVICE, non_blocking=True)

            # Flatten if not using CNN
            if not USE_EEG_CNN:
                B, T, F, C = eeg.shape   # (B,73,129,20)
                eeg = eeg.reshape(B, T, F*C)  # (B,73,2580)
            y     = batch["label"].to(DEVICE, non_blocking=True)
            present = batch["present"].to(DEVICE, non_blocking=True)

            S, pres_logits, _, _ = model(audio, eeg)
            loss_cls = ce(S, y)
            loss_pres = bce(pres_logits, present)
            loss = loss_cls + 0.2*loss_pres

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            # batch accuracy
            with torch.no_grad():
                acc = (S.argmax(dim=-1) == y).float().mean().item()

            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += acc * bs
            n_tr    += bs

        tr_loss /= max(1,n_tr)
        tr_acc  /= max(1,n_tr)

        # ===== VALIDATION =====
        model.eval()
        va_loss = va_acc = n_va = 0
        true_labels_all, pred_labels_all = [], []
        with torch.no_grad():
            for batch in dl_va:
                audio = batch["audio"].to(DEVICE, non_blocking=True)
                eeg   = batch["eeg"].to(DEVICE, non_blocking=True)
                y     = batch["label"].to(DEVICE, non_blocking=True)
                present = batch["present"].to(DEVICE, non_blocking=True)

                S, pres_logits, _, _ = model(audio, eeg)
                loss_cls = ce(S, y)
                loss_pres = bce(pres_logits, present)
                loss = loss_cls + 0.2*loss_pres

                pred = S.argmax(dim=-1)
                true_labels_all.extend(y.cpu().tolist())
                pred_labels_all.extend(pred.cpu().tolist())
                acc = (pred == y).float().mean().item()

                bs = y.size(0)
                va_loss += loss.item() * bs
                va_acc  += acc * bs
                n_va    += bs

        va_loss /= max(1,n_va)
        va_acc  /= max(1,n_va)

        # print epoch report
        idx2label = ["Gt","Vx","Dr","Bs"]
        print("True:", [idx2label[i] for i in true_labels_all])
        print("Pred:", [idx2label[i] for i in pred_labels_all])
        print(f"[Epoch {epoch:03d}] train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f} | {time.time()-t0:.1f}s")

        # log text file
        log_path = os.path.join(args.ckpt_dir, "..", "EGMP_ckpt_log.txt")
        with open(log_path, "a") as f:
            f.write(f"[Epoch {epoch:03d}] train {tr_loss:.4f}/{tr_acc:.3f} | "
                    f"val {va_loss:.4f}/{va_acc:.3f}\n")

        # log TensorBoard
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)
        writer.flush()
        # save latest checkpoint
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "val_acc": va_acc,
            "train_acc": tr_acc},
            os.path.join(latest_dir, f"{epoch}.pt"))

        # keep only last
        latest_ckpts = sorted([f for f in os.listdir(latest_dir) if f.endswith(".pt")],
                            key=lambda x: int(x.replace(".pt","")))
        for fname in latest_ckpts[:-1]:
            os.remove(os.path.join(latest_dir, fname))


        # save best checkpoint
        if va_acc > best_va or (va_acc == best_va and tr_acc > best_tr):
            best_va, best_tr = va_acc, tr_acc
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "train_acc": tr_acc},
                os.path.join(best_dir, f"{epoch}.pt"))

            best_ckpts = sorted([f for f in os.listdir(best_dir) if f.endswith(".pt")],
                                key=lambda x: int(x.replace(".pt","")))
            for fname in best_ckpts[:-1]:
                os.remove(os.path.join(best_dir, fname))

    print(f"Best val acc: {best_va:.3f}")
    writer.close()

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./EGMP_preprocessed",
                   help="Folder containing manifest.csv")
    p.add_argument("--ckpt_dir", type=str, default="./EGMP_ckpt")
    args = p.parse_args()
    train_loop(args)
