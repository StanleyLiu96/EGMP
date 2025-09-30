#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train.py
--------
Training loop for PAIC (Presence Assisted Instrument Classifier).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import random
import numpy as np

# Our modules
from load_dataset import load_all_datasets, INSTRUMENTS, CHANNELS
from encoders import AudioTimeAlignedEncoder, EEGTimeAlignedEncoder
from paic_model import PAIC


# -------------------------
# Helpers
# -------------------------

def set_seed(seed=42):
    random.seed(seed)                 # Python random
    np.random.seed(seed)              # NumPy
    torch.manual_seed(seed)           # PyTorch CPU
    torch.cuda.manual_seed(seed)      # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_labels(solo_files, device):
    """
    Convert solo audio filenames into presence + attended labels.

    Each filename looks like:
        "0002_pop_mixtape_trio_GtDrVx_theme2_stereo_Vx_soli.wav"

    - Attended instrument = the token before "_soli.wav" (e.g., "Vx")
    - Present instruments = the token before "themeX" (e.g., "GtDrVx")

    Parameters
    ----------
    solo_files : list of str
        Filenames for a batch of solo audio
    device : str or torch.device
        Where to place the tensors

    Returns
    -------
    cls_labels  : [B] long tensor
        Index of attended instrument
    pres_labels : [B, 4] float tensor
        Multi-hot vector for presence of instruments
    """
    B = len(solo_files)

    cls_labels = torch.zeros(B, dtype=torch.long, device=device)     # [B]
    pres_labels = torch.zeros((B, len(INSTRUMENTS)), device=device)  # [B,4]

    for b, fname in enumerate(solo_files):
        parts = fname.split("_")

        # Attended instrument = second-to-last token (before "soli.wav")
        attended_instr = parts[-2]
        cls_labels[b] = INSTRUMENTS.index(attended_instr)

        # Present instruments = token before "themeX"
        present_token = parts[-5]
        for instr in INSTRUMENTS:
            if instr in present_token:
                pres_labels[b, INSTRUMENTS.index(instr)] = 1.0

    return cls_labels, pres_labels


# -------------------------
# Train loop
# -------------------------
def train_loop(EEG_train, EEG_val, mixed_audio_train, mixed_audio_val, solo_audio_train, solo_audio_val, n_epochs, batch_size, lr, device="cuda"):

    # Convert to PyTorch datasets
    train_dataset = list(zip(EEG_train, mixed_audio_train, solo_audio_train))
    val_dataset   = list(zip(EEG_val, mixed_audio_val, solo_audio_val))

    # Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Infer feature dims
    eeg_sample, _ = EEG_train[0]
    audio_sample, _ = mixed_audio_train[0]
    in_ch_eeg = eeg_sample.shape[0] # EEG channel count

    # -------------------------
    # Encoders (convert raw → [B,76,C])
    # -------------------------
    eeg_encoder   = EEGTimeAlignedEncoder(in_ch=in_ch_eeg).to(device)
    audio_encoder = AudioTimeAlignedEncoder().to(device)

    with torch.no_grad():
        eeg_feat = eeg_encoder(eeg_sample.unsqueeze(0).to(device))     # [1,76,C_eeg]
        audio_feat = audio_encoder(audio_sample.unsqueeze(0).to(device)) # [1,76,C_audio]
    C_eeg = eeg_feat.shape[-1]
    C_audio = audio_feat.shape[-1]

    # -------------------------
    # Build PAIC model
    # -------------------------
    model = PAIC(audio_in_ch=C_audio, eeg_in_ch=C_eeg).to(device)

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()       # attended instrument
    bce_loss = nn.BCEWithLogitsLoss()     # presence prior

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Setup checkpoint dirs & TensorBoard
    # -------------------------
    if len(CHANNELS) == 20:
        channels = 'ALL'
    else:
        channels = "_".join(CHANNELS)
    base_dir = f"./ckpt/{channels}"
    best_dir = os.path.join(base_dir, "best")
    latest_dir = os.path.join(base_dir, "latest")
    tb_dir = os.path.join(base_dir, "tensorboard")

    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=tb_dir)

    # Track previous "best" values
    # prev_train_cls_loss = float("inf")
    # prev_val_cls_loss = float("inf")
    prev_train_acc = 0.0
    prev_val_acc = 0.0

    # -------------------------
    # Epoch loop
    # -------------------------
    for epoch in range(1, n_epochs+1):
        model.train()
        epoch_loss_cls = 0.0
        epoch_loss_pres = 0.0
        train_acc = 0.0
        n_train = 0

        # Mini-batch loop
        for (eeg_batch, eeg_files), (audio_batch, audio_files), (solo_batch, solo_files) in train_loader:

            # print("=== DEBUG: train_loader sample ===")
            # for i, batch in enumerate(train_loader):
            #     print(f"\n===== BATCH {i} =====")
            #     (eeg_batch, eeg_files), (audio_batch, audio_files), (solo_batch, solo_files) = batch

            #     print("EEG batch:", type(eeg_batch), eeg_batch.shape if isinstance(eeg_batch, torch.Tensor) else "not tensor")
            #     print("EEG files:", eeg_files)

            #     print("Audio batch:", type(audio_batch), audio_batch.shape if isinstance(audio_batch, torch.Tensor) else "not tensor")
            #     print("Audio files:", audio_files)

            #     print("Solo batch:", type(solo_batch), solo_batch.shape if isinstance(solo_batch, torch.Tensor) else "not tensor")
            #     print("Solo files:", solo_files)

            #     # stop after a few batches to avoid spam
            #     if i >= 2:
            #         break

            # print("\n===== NEW BATCH =====")
            # print("EEG files:  ", eeg_files)
            # print("Audio files:", audio_files)
            # print("Solo files: ", solo_files)
            # print("EEG batch shape:  ", eeg_batch.shape)
            # print("Audio batch shape:", audio_batch.shape)
            # print("Solo batch shape: ", solo_batch.shape)

            # Extract tensors
            eeg_tensors   = eeg_batch.to(device)   # [B, C, 4864]
            audio_tensors = audio_batch.to(device)  # [B, S, 2]

            # Encode into 76-step aligned features
            encoded_eeg   = eeg_encoder(eeg_tensors)     # [B,76,C_eeg]
            encoded_audio = audio_encoder(audio_tensors) # [B,76,C_audio]

            # print("Encoded EEG shape:  ", encoded_eeg.shape)
            # print("Encoded Audio shape:", encoded_audio.shape)

            # Forward pass
            cross_logits, pres_logits = model(encoded_eeg, encoded_audio)

            # Labels
            cls_labels, pres_labels = build_labels(solo_files, device)

            # print("Class labels (attended):", cls_labels)
            # print("Presence labels (GT):   ", pres_labels)

            # Losses
            loss_cls = ce_loss(cross_logits, cls_labels)
            loss_pres = bce_loss(pres_logits, pres_labels)
            loss = loss_cls + 0.2*loss_pres

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy (masked prediction at inference style)
            pres_probs = torch.sigmoid(pres_logits)              # [B,4]
            pres_mask = (pres_probs >= 0.5).long()               # [B,4]

            # print("Presence mask (pred):   ", pres_mask)

            cross_probs = torch.softmax(cross_logits, dim=-1)    # [B,4]
            masked_probs = cross_probs * pres_mask.float()       # apply presence mask
            pred = masked_probs.argmax(dim=-1)                   # [B]
            acc = (pred == cls_labels).float().mean().item()

            # Logging accumulators
            B = eeg_tensors.size(0)
            epoch_loss_cls += loss_cls.item() * B
            epoch_loss_pres += loss_pres.item() * B
            train_acc += acc * B
            n_train += B

        # Normalize
        epoch_loss_cls /= n_train
        epoch_loss_pres /= n_train
        train_acc /= n_train

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss_cls = 0.0
        val_loss_pres = 0.0
        val_acc = 0.0
        n_val = 0

        with torch.no_grad():
            for (eeg_batch, eeg_files), (audio_batch, audio_files), (solo_batch, solo_files) in val_loader:
                eeg_tensors = eeg_batch.to(device)   # [B, C, 4864]
                audio_tensors = audio_batch.to(device)  # [B, S, 2]

                encoded_eeg   = eeg_encoder(eeg_tensors)
                encoded_audio = audio_encoder(audio_tensors)

                cross_logits, pres_logits = model(encoded_eeg, encoded_audio)
                cls_labels, pres_labels = build_labels(solo_files, device)
                # print("!!!!!pres_labels: ", pres_labels)

                loss_cls = ce_loss(cross_logits, cls_labels)
                loss_pres = bce_loss(pres_logits, pres_labels)

                pres_probs = torch.sigmoid(pres_logits)
                pres_mask = (pres_probs >= 0.5).long()
                # print("!!!!!pres_mask: ", pres_mask)

                cross_probs = torch.softmax(cross_logits, dim=-1)
                masked_probs = cross_probs * pres_mask.float()
                pred = masked_probs.argmax(dim=-1)
                acc = (pred == cls_labels).float().mean().item()

                B = eeg_tensors.size(0)
                val_loss_cls += loss_cls.item() * B
                val_loss_pres += loss_pres.item() * B
                val_acc += acc * B
                n_val += B
            
        # Normalize validation metrics
        val_loss_cls /= n_val
        val_loss_pres /= n_val
        val_acc /= n_val

        # -------------------------
        # Epoch summary
        # -------------------------
        print(f"[Epoch {epoch:02d}] "
              f"train_cls={epoch_loss_cls:.4f}, train_pres={epoch_loss_pres:.4f}, train_acc={train_acc:.3f} | "
              f"val_cls={val_loss_cls:.4f}, val_pres={val_loss_pres:.4f}, val_acc={val_acc:.3f}")

        # -------------------------
        # TensorBoard
        # -------------------------
        writer.add_scalar("train/pres_loss",  epoch_loss_pres, epoch)
        writer.add_scalar("train/cross_loss", epoch_loss_cls,  epoch)
        writer.add_scalar("train/acc",        train_acc,       epoch)
        writer.add_scalar("val/pres_loss",    val_loss_pres,   epoch)
        writer.add_scalar("val/cross_loss",   val_loss_cls,    epoch)
        writer.add_scalar("val/acc",          val_acc,         epoch)
        writer.flush()

        # -------------------------
        # Checkpoints
        # -------------------------

        # Always save "latest" (keep only the last one)
        latest_path = os.path.join(latest_dir, f"{epoch}.pt")
        torch.save(model.state_dict(), latest_path)
        latest_ckpts = sorted([f for f in os.listdir(latest_dir) if f.endswith(".pt")], key=lambda x: int(x.replace(".pt", "")) )
        for f in latest_ckpts[:-1]:
            os.remove(os.path.join(latest_dir, f))

        # Save "best" only if all three conditions meet:
        # (cur train_cls <= prev train_cls) and (cur val_cls <= prev val_cls) and (cur val_acc >= prev val_acc)
        if (val_acc > prev_val_acc) or (val_acc == prev_val_acc and train_acc >= prev_train_acc):
            best_path = os.path.join(best_dir, f"{epoch}.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Saved new best model at {best_path}")

            # Keep only the most recent best
            best_ckpts = sorted([f for f in os.listdir(best_dir) if f.endswith(".pt")], key=lambda x: int(x.replace(".pt", "")))
            for f in best_ckpts[:-1]:
                os.remove(os.path.join(best_dir, f))

            # Update thresholds for next comparison
            # prev_train_cls_loss = epoch_loss_cls
            # prev_val_cls_loss   = val_loss_cls
            prev_train_acc      = train_acc
            prev_val_acc        = val_acc

    writer.close()
    return model


if __name__ == "__main__":
    set_seed(42)  

    EEG_train, EEG_val, mixed_audio_train, mixed_audio_val, solo_audio_train, solo_audio_val = load_all_datasets()

    train_loop(
        EEG_train, EEG_val,
        mixed_audio_train, mixed_audio_val,
        solo_audio_train, solo_audio_val,
        n_epochs=200, batch_size=2, lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )