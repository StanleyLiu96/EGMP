#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_train.py
Train a global 4-class offline classifier using full 19s trials:
Inputs: mixture audio features, EEG features
Output: attended class in {Gt, Vx, Dr, Bs}

Model:
- Audio encoder (mixture-only): small TCN over [log-Mel L/R, ILD, IPD] -> sequence A_enc (B,T,D)
- EEG encoder: small TCN with a short receptive field (captures 0-250ms lag) -> sequence E_enc -> temporal attention -> EEG token e (B,D)
- 4 instrument queries q_k built from learned instrument embeddings -> cross-attend over A_enc -> audio token a_k
- Match scores s_k = MLP([e, a_k, e⊙a_k, |e-a_k|]); softmax over 4 -> CE loss

Includes:
- Optional presence-prior head from A_enc to encourage down-weighting absent classes (not a hard mask)

Adds:
1. TensorBoard logging to tensorboard/ folder
2. Resume training via epoch-based checkpoint saving
3. Minimal edits, fully commented, existing logic preserved
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
from torch.utils.tensorboard import SummaryWriter

from EGMP_util import EGMPDataset, collate_fulltrials

# -------------------------
# Config
# -------------------------
SEED = 1337
BATCH_SIZE = 4
EPOCHS = 200
LR = 2e-4
WD = 1e-4
D_MODEL = 192
N_HEADS = 4
D_FF = 384
D_QUERY = 64
PRESENCE_PRIOR_W = 0.2   # weight to add log p_k to class scores (soft prior). Set 0 to disable.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

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
    def __init__(self, fa, fe, d_model=D_MODEL, n_classes=4):
        super().__init__()
        self.n_classes = n_classes

        # Audio encoder
        self.a_in = nn.Linear(fa, d_model)
        self.a_tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)
        self.a_tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=7, p=0.2)

        # Presence-prior head (from audio sequence)
        # acts like a soft mask on which instrument embeddings to trust
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

        # Cross-attention per class
        self.cross = CrossAttentionQuery(d_model, D_QUERY//2)

        # Matcher MLP
        self.match_mlp = nn.Sequential(
            nn.Linear(4*d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, audio_seq, eeg_seq):
        """
        Forward pass: combine audio, EEG, and instrument embeddings to predict
        which instrument the listener is attending to.

        Processing pipeline:
        1. Audio encoder:
           - Input: audio sequence (B,T,Fa).
           - Output: encoded audio features (B,T,D).
        
        2. Presence-prior head:
           - Predicts which instruments are present in the mixture.
           - Frame-level predictions: (B,T,4).
           - Averaged to global clip-level predictions: (B,4).
           - Log-softmax applied → (B,4) log-probabilities used to boost/suppress class scores.
        
        3. EEG encoder:
           - Input: EEG sequence (B,T,Fe).
           - Output: encoded EEG sequence (B,T,D).
           - Temporal attention pooling reduces to one pooled EEG token (B,D),
             summarizing the listener's neural evidence of attention.
        
        4. Instrument-by-instrument matching loop:
           - For each instrument (Gt, Vx, Dr, Bs):
             a) Get the learnable embedding vector for that instrument (its "identity").
             b) Use this embedding as a query into the audio sequence via cross-attention:
                "Given the Guitar embedding, where is Guitar in the audio?"
                Output: instrument-specific audio feature (B,D) + attention map over time (B,T).
             c) Fuse EEG token and instrument-specific audio feature into a joint feature
                (concatenating EEG, audio, their product, and their difference).
             d) Pass through match MLP → scalar score (B,), reflecting
                "How much does the listener's EEG match this instrument's audio?"
             e) Add presence-prior log probability (boost if present, penalize if absent).
             f) Save the score and attention map.
        
        5. Final stacking:
           - Collect all per-instrument scores → logits (B,4), one per instrument.
           - Collect per-instrument attention maps → (B,4,T), showing audio time weights.

        Returns:
            S           (B,4)   : classification logits for attended instrument
            pres_logits (B,4)   : raw instrument presence predictions
            attn_eeg    (B,T)   : EEG temporal attention weights
            A_attns     (B,4,T) : audio attention maps per instrument class
        """
        B, T, _ = audio_seq.shape
        # ======================
        # AUDIO ENCODER
        # ======================
        # Pass the raw audio sequence through the input layer and temporal conv blocks.
        # This produces a higher-level representation of the mixture audio over time.
        A = self.a_in(audio_seq)      # (B, T, D) where D = hidden feature dimension
        A = self.a_tcn1(A)            # temporal convolution block 1 (refines temporal patterns)
        A = self.a_tcn2(A)            # temporal convolution block 2 (further refines features)

        # ======================
        # PRESENCE PRIOR HEAD
        # ======================
        # At each time step, predict the likelihood of each instrument being present.
        # pres_logits_seq: raw (unnormalized) scores for 4 instruments, for each frame.
        pres_logits_seq = self.pres_head(A)       # (B,T,4)
        # Average over time to get a single presence prediction for the whole clip.
        # pres_logits: one 4-dimensional vector per sample, saying which instruments exist.
        pres_logits = pres_logits_seq.mean(dim=1) # (B,4)
        # Convert these logits to log-probabilities (normalized over the 4 instruments).
        # pres_logprob[b,k] = log probability that instrument k is present in sample b.
        pres_logprob = torch.log_softmax(pres_logits, dim=-1)  # (B,4)

        # ======================
        # EEG ENCODER
        # ======================
        # Pass the raw EEG sequence through input + temporal conv blocks.
        # Output is a temporally refined EEG sequence, aligned to audio.
        E = self.e_in(eeg_seq)        # (B,T,D)
        E = self.e_tcn1(E)
        E = self.e_tcn2(E)
        # Pool EEG sequence into one fixed-length vector per trial.
        # e_tok: pooled EEG embedding, summarizing attention over the entire clip.
        # attn_eeg: attention weights across time (for analysis/visualization).
        e_tok, attn_eeg = self.pool(E)  # (B,D), (B,T)

        # ======================
        # INSTRUMENT-BY-INSTRUMENT MATCHING
        # For each class, build query, cross-attend, and match
        # ======================
        scores = []  # list to collect attention scores for each instrument
        attns = []   # list to collect per-class attention maps

        # Loop over all instruments (n_classes = 4: Guitar, Voice, Drums, Bass).
        for k in range(self.n_classes):

            # 1) Get the learnable embedding vector for instrument k.
            #    Shape: (D_QUERY/2,) → expand to (B, D_QUERY/2) so every sample in the batch gets one.
            inst_tok = self.inst_embed.weight[k].unsqueeze(0).expand(B, -1)  # (B, D_QUERY/2)

            # 2) Use this instrument embedding as the query for cross-attention.
            #    Think: "Given this instrument identity (e.g. Guitar), where is it in the audio?"
            qk = inst_tok  # (B, D_QUERY//2)

            # 3) Cross-attention:
            # Given the Guitar embedding, where in the audio do I hear Guitar-like patterns?
            # cross-attention between instrument embedding and the audio sequence
            #    Query = instrument embedding
            #    Keys/Values = audio sequence A
            #    Output a_k: instrument-specific audio feature (summarized across time).
            #    Output attn: time weights showing where the instrument is most active.
            a_k, attn = self.cross(qk, A) # (B, D), (B,T)

            # 4) Fuse EEG and audio evidence for this instrument.
            #    Build a "joint feature" with four parts:
            #      - e_tok (EEG vector)
            #      - a_k (audio vector for this instrument)
            #      - e_tok * a_k (interaction term)
            #      - |e_tok - a_k| (disagreement term)
            #    Result is a richer combined representation of brain + sound.
            feat = torch.cat([e_tok, a_k, e_tok*a_k, torch.abs(e_tok - a_k)], dim=-1)  # (B, 4D)

            # 5) Match MLP: turn the joint feature into a single scalar score.
            #    This score reflects how strongly the listener's EEG aligns with this instrument.
            s_k = self.match_mlp(feat).squeeze(-1)  # (B,)

            # 6) Add soft presence prior:
            #    If presence-prior predicts this instrument is absent, penalize the score.
            #    If presence-prior predicts it's present, boost the score.
            if PRESENCE_PRIOR_W > 0:
                s_k = s_k + PRESENCE_PRIOR_W * pres_logprob[:, k]

            # 7) Save score and attention map for this instrument.
            scores.append(s_k) # scalar per batch item
            attns.append(attn.unsqueeze(1)) # expand to (B, 1, T) for stacking later

        # ======================
        # FINAL STACKING
        # ======================
        # Stack per-class scores → final classification logits (before softmax).
        # Shape: (B, 4) one score per instrument.
        S = torch.stack(scores, dim=-1)  # (B, 4)

        # Concatenate attention maps for all classes.
        # Shape: (B, 4, T) → per-class, per-time attention (useful for analysis/visualization).
        A_attns = torch.cat(attns, dim=1)  # (B, 4, T) attention per class (for analysis)

        # ======================
        # RETURN VALUES
        # ======================
        # S:       (B, 4) classification logits (attended instrument)
        # pres_logits: (B, 4) raw presence predictions
        # attn_eeg: (B, T) EEG temporal attention weights
        # A_attns:  (B, 4, T) audio attention per instrument
        return S, pres_logits, attn_eeg, A_attns

# -------------------------
# Training
# -------------------------
def train_loop(args):
    set_seed(SEED)
    val_txt = os.path.join(args.data_root, "val_split_filenames.txt")
    ds_tr = EGMPDataset(os.path.join(args.data_root, "manifest.csv"), split="train", split_seed=0, val_ratio=0.2, val_trial_ids_path=val_txt)
    ds_va = EGMPDataset(os.path.join(args.data_root, "manifest.csv"), split="val", split_seed=0, val_ratio=0.2, val_trial_ids_path=val_txt)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, collate_fn=collate_fulltrials, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, collate_fn=collate_fulltrials, pin_memory=True)

    # infer feature dims from one sample
    samp = ds_tr[0]
    Fa = samp["audio"].shape[1]
    Fe = samp["eeg"].shape[1]

    model = EGMPModel(fa=Fa, fe=Fe).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    bce = nn.BCEWithLogitsLoss()

    tb_dir = os.path.join(args.ckpt_dir, "tensorboard")
    best_dir = os.path.join(args.ckpt_dir, "Best")
    latest_dir = os.path.join(args.ckpt_dir, "Latest")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)  # init TensorBoard

    # # Resume logic
    # latest_ckpts = [f for f in os.listdir(latest_dir) if f.endswith(".pt")]
    # latest_ckpts.sort(key=lambda x: int(x.replace(".pt", "")))  # sort numerically
    # start_epoch = 1
    # best_va = None
    # best_tr = None
    # if latest_ckpts:
    #     latest_ckpt_file = latest_ckpts[-1]
    #     ckpt_path = os.path.join(latest_dir, latest_ckpt_file)
    #     print(f"Resuming from checkpoint: {ckpt_path}")
    #     checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    #     model.load_state_dict(checkpoint["model"])
    #     optim.load_state_dict(checkpoint["optim"])
    #     best_va = checkpoint.get("val_acc")
    #     best_tr = checkpoint.get("train_acc")
    #     start_epoch = checkpoint.get("epoch", 1) + 1
    #     print(f"--> Starting at epoch {start_epoch} (prev best acc: {best_va:.3f})")
    # else:
    #     best_va = 0.0
    #     best_tr = 0.0

    # Resume logic (Latest model for training, Best model for val comparison)

    # Load best validation stats (for comparison)
    best_ckpts = [f for f in os.listdir(best_dir) if f.endswith(".pt")]
    best_ckpts.sort(key=lambda x: int(x.replace(".pt", "")))
    if best_ckpts:
        best_ckpt_file = best_ckpts[-1]
        best_ckpt_path = os.path.join(best_dir, best_ckpt_file)
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=True)
        best_va = best_ckpt["val_acc"]
        print("Best Validation Acc: ", best_va, sep="")
        best_tr = best_ckpt["train_acc"]
        print("Best Training Acc: ", best_tr, sep="")
        del best_ckpt
    else:
        print("Failed to read best model")
        best_va = 0.0
        best_tr = 0.0

    # Load latest model
    latest_ckpts = [f for f in os.listdir(latest_dir) if f.endswith(".pt")]
    latest_ckpts.sort(key=lambda x: int(x.replace(".pt", "")))  # sort numerically
    start_epoch = 1

    if latest_ckpts:
        latest_ckpt_file = latest_ckpts[-1]
        ckpt_path = os.path.join(latest_dir, latest_ckpt_file)
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        start_epoch = checkpoint.get("epoch", 1) + 1
        print(f"--> Starting at epoch {start_epoch}")
    else:
        print("Starting from scratch.")

    # Train
    for epoch in range(start_epoch, EPOCHS+1):
        model.train()
        t0 = time.time()
        tr_loss = tr_acc = n_tr = 0
        for batch in dl_tr:
            audio = batch["audio"].to(DEVICE, non_blocking=True)
            eeg   = batch["eeg"].to(DEVICE, non_blocking=True)
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
            true_labels_all = []
            pred_labels_all = []
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

        va_loss /= max(1, n_va)
        va_acc  /= max(1, n_va)

        idx2label = ["Gt", "Vx", "Dr", "Bs"]
        true_strs = [idx2label[i] for i in true_labels_all]
        pred_strs = [idx2label[i] for i in pred_labels_all]
        print("True: ", true_strs)
        print("Pred: ", pred_strs)

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s")

        # Append to log file
        log_path = os.path.join(args.ckpt_dir, "..", "EGMP_ckpt_log.txt")
        with open(log_path, "a") as f:
            f.write("True:  " + str(true_strs) + "\n")
            f.write("Pred:  " + str(pred_strs) + "\n")
            f.write(f"[Epoch {epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s\n")
            f.write("---\n")

        # TensorBoard log
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val", va_loss, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val", va_acc, epoch)
        writer.flush()

        # Save latest checkpoint
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "val_acc": va_acc,
            "train_acc": tr_acc},
            os.path.join(latest_dir, f"{epoch}.pt"))
        # Delete old latest
        latest_ckpts = [f for f in os.listdir(latest_dir) if f.endswith(".pt")]
        latest_ckpts.sort(key=lambda x: int(x.replace(".pt", "")))
        for fname in latest_ckpts[:-1]:  # keep only the last (highest epoch)
            os.remove(os.path.join(latest_dir, fname))

        # Save best checkpoint
        if va_acc > best_va or (va_acc == best_va and tr_acc > best_tr):
            best_va = va_acc
            best_tr = tr_acc
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "train_acc": tr_acc},
                os.path.join(best_dir, f"{epoch}.pt"))
        # Delete old best
        best_ckpts = [f for f in os.listdir(best_dir) if f.endswith(".pt")]
        best_ckpts.sort(key=lambda x: int(x.replace(".pt", "")))
        for fname in best_ckpts[:-1]:  # keep only the last (highest epoch)
            os.remove(os.path.join(best_dir, fname))

    print(f"Best val acc: {best_va:.3f}")
    writer.close()

# -------------------------
# Entry
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./EGMP_preprocessed", help="Folder containing manifest.csv")
    p.add_argument("--ckpt_dir", type=str, default="./EGMP_ckpt")
    args = p.parse_args()
    train_loop(args)
