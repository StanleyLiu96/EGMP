#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from util import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Small building blocks
# -----------------------------
class TemporalConvBlock(nn.Module):
    """Small residual temporal conv block over (B,T,D)."""
    def __init__(self, d_in: int, d_h: int, k: int = 7, p: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_in)
        self.conv1 = nn.Conv1d(d_in, d_h, kernel_size=k, padding=k//2)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.norm2 = nn.LayerNorm(d_h)
        self.conv2 = nn.Conv1d(d_h, d_in, kernel_size=k, padding=k//2)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D)
        h = self.norm1(x)
        h = h.transpose(1, 2)          # (B,D,T)
        h = self.conv1(h)
        h = self.act1(h)
        h = self.drop1(h)
        h = h.transpose(1, 2)          # (B,T,D_h)
        h = self.norm2(h)
        h = h.transpose(1, 2)          # (B,D_h,T)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.drop2(h)
        h = h.transpose(1, 2)          # (B,T,D)
        return x + h                    # residual

class TemporalAttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B,T,D)
        a = self.w(x).squeeze(-1)          # (B,T)
        a = torch.softmax(a, dim=1)
        z = torch.bmm(a.unsqueeze(1), x).squeeze(1)  # (B,D)
        return z, a

# -----------------------------
# Front-ends & Branches
# -----------------------------
class AudioFrontend(nn.Module):
    """
    Raw mono waveform @256 Hz -> (B,T,Ca) with same T (padding='same').
    Keep it extremely simple: 2 conv layers.
    """
    def __init__(self, c_out: int = 64, k: int = 15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, c_out, kernel_size=k, padding=k//2),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=k//2),
            nn.GELU(),
        )

    def forward(self, x_wave: torch.Tensor) -> torch.Tensor:
        # x_wave: (B,1,T)
        y = self.net(x_wave)               # (B,Ca,T)
        return y.transpose(1, 2)           # (B,T,Ca)

class EEGFrontend(nn.Module):
    """Single-channel EEG @256 Hz -> (B,T,Deeg) with same T."""
    def __init__(self, d_eeg: int = 128, k: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, d_eeg, kernel_size=k, padding=k//2),
            nn.GELU(),
            nn.Conv1d(d_eeg, d_eeg, kernel_size=k, padding=k//2),
            nn.GELU(),
        )

    def forward(self, x_eeg: torch.Tensor) -> torch.Tensor:
        y = self.net(x_eeg)                # (B,Deeg,T)
        return y.transpose(1, 2)           # (B,T,Deeg)

class AudioBranch(nn.Module):
    def __init__(self, fa: int, d_model: int, k: int = 7, p: float = 0.2, n_classes: int = 4):
        super().__init__()
        self.proj = nn.Linear(fa, d_model)
        self.tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=k, p=p)
        self.tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=k, p=p)
        self.pres_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),   # logits per class
        )

    def forward(self, audio_seq: torch.Tensor):
        # audio_seq: (B,T,Fa)
        A = self.proj(audio_seq)            # (B,T,D)
        A = self.tcn1(A)
        A = self.tcn2(A)
        pres_seq = self.pres_head(A)        # (B,T,4)
        audio_presence_logits = pres_seq.mean(dim=1)  # (B,4)
        return A, audio_presence_logits

class EEGBranch(nn.Module):
    def __init__(self, d_in: int, d_model: int, k: int = 7, p: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.tcn1 = TemporalConvBlock(d_model, d_h=d_model, k=k, p=p)
        self.tcn2 = TemporalConvBlock(d_model, d_h=d_model, k=k, p=p)

    def forward(self, eeg_seq: torch.Tensor):
        # eeg_seq: (B,T,Deeg)
        E = self.proj(eeg_seq)              # (B,T,D)
        E = self.tcn1(E)
        E = self.tcn2(E)
        return E

class CrossAttnClassifier(nn.Module):
    """
    EEG queries, Audio keys/values (single-layer MHA), then attention pool and classify.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_classes: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.pool = TemporalAttentionPool(d_model)
        self.cls = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, E: torch.Tensor, A: torch.Tensor):
        # E: (B,T,D) as queries
        # A: (B,T,D) as keys/values (we assume same T; if not, one could allow different T)
        attn_out, _ = self.mha(query=E, key=A, value=A)  # (B,T,D)
        z, _ = self.pool(attn_out)                       # (B,D)
        logits = self.cls(z)                             # (B,4)
        return logits

# -----------------------------
# Full Model wrapper
# -----------------------------
class EGMP_OneChan(nn.Module):
    def __init__(self, d_model: int = 192, n_classes: int = 4, a_feat: int = 64, e_feat: int = 128):
        super().__init__()
        self.af = AudioFrontend(c_out=a_feat)
        self.ef = EEGFrontend(d_eeg=e_feat)
        self.audio_branch = AudioBranch(fa=a_feat, d_model=d_model, n_classes=n_classes)
        self.eeg_branch   = EEGBranch(d_in=e_feat, d_model=d_model)
        self.head = CrossAttnClassifier(d_model=d_model, n_heads=4, n_classes=n_classes)

    def forward(self, wave_mono: torch.Tensor, eeg_mono: torch.Tensor):
        """
        wave_mono: (B,1,T) raw mono @256 Hz
        eeg_mono:  (B,1,T) preprocessed mono @256 Hz
        """
        A0 = self.af(wave_mono)          # (B,T,Fa)
        E0 = self.ef(eeg_mono)           # (B,T,Fe)
        A, pres_logits = self.audio_branch(A0)         # (B,T,D), (B,4)
        E = self.eeg_branch(E0)                          # (B,T,D)
        logits = self.head(E, A)                         # (B,4)
        return logits, pres_logits, A, E

# -----------------------------
# Train / Eval utilities
# -----------------------------
def hard_mask_from_presence(pres_logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """pres_logits: (B,4) -> probs -> hard mask in {0,1} with threshold."""
    probs = torch.sigmoid(pres_logits)
    return (probs >= thr).float()

def apply_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set logits to very negative where mask==0 (hard mask)."""
    neg_inf = torch.finfo(logits.dtype).min / 2  # safe very negative
    masked = logits.clone()
    masked = masked.masked_fill(mask == 0.0, neg_inf)
    return masked

# -----------------------------
# Main train loop
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixed_dir", type=str, default="/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/stimulus_wav")
    p.add_argument("--solo_dir", type=str, default="/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/isolated_wav")
    p.add_argument("--eeg_dir", type=str, default="/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/response_npy")
    p.add_argument("--meta_yaml", type=str, default="/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/madeeg_preprocessed.yaml")

    p.add_argument("--eeg_channel", type=str, default="F3")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr_attn", type=float, default=2e-4)   # classifier branch optimizer
    p.add_argument("--lr_pres", type=float, default=2e-4)   # presence head optimizer
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--presence_thr", type=float, default=0.5)
    p.add_argument("--ckpt_dir", type=str, default="./EGMP1C_ckpt")
    p.add_argument("--val_list", type=str, default="/users/PAS2301/liu215229932/Music_Project/Models/EGMP/part_1_OneEEGChannel/val_split_filenames.txt")
    args = p.parse_args()
    # put checkpoints under a channel-named subfolder
    chan = args.eeg_channel.strip().replace("/", "_")
    args.ckpt_dir = os.path.join(args.ckpt_dir, chan)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    paths = Paths(
        mixed_dir=args.mixed_dir,
        solo_dir=args.solo_dir,
        eeg_dir=args.eeg_dir,
        meta_yaml=args.meta_yaml,
    )
    items = load_meta_trials(paths, eeg_channel_name=args.eeg_channel)
    if len(items) == 0:
        raise RuntimeError("No valid trials found after filtering to {Gt,Vx,Dr,Bs}.")

    tr_items, va_items = split_train_val_from_file(items, args.val_list)

    ds_tr = OneChanDataset(tr_items, eeg_channel_name=args.eeg_channel)
    ds_va = OneChanDataset(va_items, eeg_channel_name=args.eeg_channel)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=1, collate_fn=collate_crop_min, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=1, collate_fn=collate_crop_min, pin_memory=True)

    model = EGMP_OneChan(d_model=192, n_classes=4, a_feat=64, e_feat=128).to(DEVICE)
    # losses
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    # parameter groups
    pres_params = list(model.audio_branch.pres_head.parameters())
    # Everything else goes to the attended optimizer (audio encoders, EEG encoders, cross-attn, classifier)
    attn_params = [p for n, p in model.named_parameters() if not any(p is q for q in pres_params)]

    opt_attn = torch.optim.AdamW(attn_params, lr=args.lr_attn, weight_decay=args.wd)
    opt_pres = torch.optim.AdamW(pres_params, lr=args.lr_pres, weight_decay=args.wd)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # TensorBoard
    tb_dir = os.path.join(args.ckpt_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # Best trackers (use -inf so first epoch always wins)
    best_va = float("-inf")
    best_tr = float("-inf")

    for epoch in range(1, args.epochs + 1):
        # ----------------- Train -----------------
        model.train()
        tr_loss_attn = tr_loss_pres = 0.0
        tr_acc = 0.0
        n_tr = 0

        for batch in dl_tr:
            # tensors
            x_aud = torch.from_numpy(batch["audio"]).to(DEVICE)  # (B,T)
            x_eeg = torch.from_numpy(batch["eeg"]).to(DEVICE)    # (B,T)
            y_att = torch.from_numpy(batch["label"]).long().to(DEVICE)        # (B,)
            y_pres = torch.from_numpy(batch["present"]).to(DEVICE)            # (B,4)

            # reshape to (B,1,T) for front-ends
            x_aud = x_aud.unsqueeze(1)
            x_eeg = x_eeg.unsqueeze(1)

            # Forward
            logits, pres_logits, A, E = model(x_aud, x_eeg)

            # -------- Presence loss (detach audio features into presence head path) --------
            # We already designed presence head to consume A from audio branch internally.
            # To guarantee no gradient reaches encoder from presence loss, we compute presence
            # loss only over pres_head params via a separate backward pass.
            loss_pres = bce(pres_logits, y_pres)

            # ---- Backward presence (only presence head params) ----
            opt_pres.zero_grad(set_to_none=True)
            loss_pres.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.audio_branch.pres_head.parameters(), args.clip)
            opt_pres.step()

            # -------- Attended loss (audio + EEG + cross-attn + classifier) --------
            loss_attn = ce(logits, y_att)

            # ---- Backward attended (all except presence head) ----
            opt_attn.zero_grad(set_to_none=True)
            loss_attn.backward()
            nn.utils.clip_grad_norm_(attn_params, args.clip)
            opt_attn.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == y_att).float().mean().item()
            bs = y_att.size(0)
            tr_loss_attn += loss_attn.item() * bs
            tr_loss_pres += loss_pres.item() * bs
            tr_acc += acc * bs
            n_tr += bs

        tr_loss_attn /= max(1, n_tr)
        tr_loss_pres /= max(1, n_tr)
        tr_acc /= max(1, n_tr)

        # ----------------- Validate (with HARD MASK) -----------------
        model.eval()
        va_acc = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in dl_va:
                x_aud = torch.from_numpy(batch["audio"]).to(DEVICE).unsqueeze(1)
                x_eeg = torch.from_numpy(batch["eeg"]).to(DEVICE).unsqueeze(1)
                y_att = torch.from_numpy(batch["label"]).long().to(DEVICE)
                # y_pres = torch.from_numpy(batch["present"]).to(DEVICE)

                logits, pres_logits, A, E = model(x_aud, x_eeg)
                # hard mask
                mask = hard_mask_from_presence(pres_logits, thr=args.presence_thr)  # (B,4)
                masked_logits = apply_mask_to_logits(logits, mask)
                pred = torch.argmax(masked_logits, dim=-1)
                acc = (pred == y_att).float().mean().item()
                bs = y_att.size(0)
                va_acc += acc * bs
                n_va += bs
        va_acc /= max(1, n_va)

        print(f"[Epoch {epoch:03d}] "
              f"train_attn_loss={tr_loss_attn:.4f} train_pres_loss={tr_loss_pres:.4f} train_acc={tr_acc:.3f} "
              f"| val_acc(hardmask)={va_acc:.3f}")

        # ---- TensorBoard scalars ----
        writer.add_scalar("Loss/train_attn", tr_loss_attn, epoch)
        writer.add_scalar("Loss/train_pres", tr_loss_pres, epoch)
        writer.add_scalar("Acc/train", tr_acc, epoch)
        writer.add_scalar("Acc/val_hardmask", va_acc, epoch)
        writer.flush()

        # save best with tie-break: higher train acc wins if val equal
        if (va_acc > best_va) or (va_acc == best_va and tr_acc > best_tr):
            best_va = va_acc
            best_tr = tr_acc
            save_path = os.path.join(args.ckpt_dir, "best.pt")
            torch.save({
                "model": model.state_dict(),
                "opt_attn": opt_attn.state_dict(),
                "opt_pres": opt_pres.state_dict(),
                "epoch": epoch,
                "val_acc": va_acc,
                "train_acc": tr_acc
            }, save_path)

        # always save latest
        latest_path = os.path.join(args.ckpt_dir, f"latest.pt")
        torch.save({"model": model.state_dict(),
                    "opt_attn": opt_attn.state_dict(),
                    "opt_pres": opt_pres.state_dict(),
                    "epoch": epoch,
                    "val_acc": va_acc}, latest_path)

    writer.close()

if __name__ == "__main__":
    main()


# F3, F1, Fz, F2, F4, C3, C1, Cz, C2, C4, CPz, P3, P1, Pz, P2, P4, POz, O1, Oz, O2

"""

F3, F1, Fz, F2, F4, C3, C1, Cz, C2, C4, CPz, P3, P1, Pz, P2, P4, POz, O1, Oz, O2

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2301 --time 24:00:00
squeue -u $USER
conda activate basen
cd /users/PAS2301/liu215229932/Music_Project/Models/EGMP/part_1_OneEEGChannel

python train.py --eeg_channel F3 37591467
python train.py --eeg_channel F1 37591646
python train.py --eeg_channel Fz 37591650
python train.py --eeg_channel F2 37591648 nan
python train.py --eeg_channel F4 37591701
python train.py --eeg_channel C3 37591648 nan
python train.py --eeg_channel C1 37591648 nan
python train.py --eeg_channel Cz 37591648 nan
python train.py --eeg_channel C2 37591648 nan
python train.py --eeg_channel C4 37591648 nan
python train.py --eeg_channel CPz 37591648 nan
python train.py --eeg_channel P3 37591648 nan
python train.py --eeg_channel P1 37591648 nan
python train.py --eeg_channel Pz 37591648 nan
python train.py --eeg_channel P2 37591648 nan
python train.py --eeg_channel P4 37591648 nan
python train.py --eeg_channel POz 37591648 nan
python train.py --eeg_channel O1 37591648 nan
python train.py --eeg_channel Oz 37591648 nan
python train.py --eeg_channel O2 37591648 nan


"""

