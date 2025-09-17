# EGMP_util.py
# Small utilities shared by EGMP_train.py

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

# Only keep the 4 instruments of interest (consistent with preprocessing filter)
INSTR_LIST = ["Gt", "Vx", "Dr", "Bs"]

def load_manifest(manifest_path):
    """
    Reads the manifest.csv file produced by EGMP_pre-processing_band_EEG.py.
    Each row contains:
        trial_id, subject, npz_path, label_idx, present_mask
    Returns:
        rows: list of dicts (keys from CSV header)
    """
    rows = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

class EGMPDataset(Dataset):
    """
    PyTorch Dataset for loading .npz files from EGMP_pre-processing_band_EEG.py.

    For each trial, returns a dict with:
        'audio':   (T_audio, Fa)   concatenated [melLR, ILD, IPD]
        'eeg':     (T_eeg, Fe)     bandpower features (bands × channels flattened)
        'present': (4,)            instrument presence mask [Gt, Vx, Dr, Bs]
        'label':   scalar long     attended instrument label index (0–3)
        'trial_id': str            trial identifier
    """
    def __init__(self, manifest_path, split="train", split_seed=0,
                 val_ratio=0.2, val_trial_ids_path=None):
        rows = load_manifest(manifest_path)
        self.items = []

        # Option 1: use a manual list of validation trial_ids
        if val_trial_ids_path is not None:
            with open(val_trial_ids_path, "r") as f:
                val_ids = set(line.strip() for line in f if line.strip())
            for r in rows:
                trial_id = r["trial_id"]
                in_val_set = trial_id in val_ids
                if (split == "val" and in_val_set) or (split == "train" and not in_val_set):
                    self.items.append(r)
        else:
            # Option 2: fallback deterministic hash-based split
            for r in rows:
                trial_id = r["trial_id"]
                h = (abs(hash((trial_id, split_seed))) % 1000)/1000.0
                is_val = (h < val_ratio)
                if (split == "train" and not is_val) or (split == "val" and is_val):
                    self.items.append(r)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        r = self.items[idx]
        z = np.load(r["npz_path"])

        # --------------------
        # AUDIO FEATURES
        # --------------------
        mel_lr = z["audio_mel_lr"]     # (4864, 2*NMELS)
        ild    = z["audio_ild"]        # (4864, NMELS)
        ipd    = z["audio_ipd"]        # (4864, NMELS)

        # Default = all features concatenated
        audio = np.concatenate([mel_lr, ild, ipd], axis=1).astype(np.float32)  # (4864, Fa)

        # --------------------
        # EEG FEATURES
        # --------------------
        eeg_band = z["eeg_band"].astype(np.float32)  # (73, 5, 20)
        T, B, C = eeg_band.shape  # T=73 frames, B=5 bands, C=20 channels
        # Flatten bands × channels → 100 features per frame
        eeg = eeg_band.reshape(T, B*C).astype(np.float32)  # (73, 100)

        # --------------------
        # LABELS
        # --------------------
        present_mask = z["present_mask"].astype(np.float32)  # (4,)
        y = int(z["label_idx"])  # scalar
        trial_id = str(z["trial_id"])  # identifier

        # --------------------
        # PACKAGE AS DICT
        # --------------------
        sample = {
            "audio": torch.from_numpy(audio),  # (T_audio=4864, Fa)
            "eeg": torch.from_numpy(eeg),      # (T_eeg=73, Fe=100)
            "present": torch.from_numpy(present_mask),  # (4,)
            "label": torch.tensor(y, dtype=torch.long),
            "trial_id": trial_id
        }
        return sample

def collate_fulltrials(batch):
    """
    Collate function for DataLoader.
    Pads variable-length sequences to the maximum T in the batch.
    Works for both audio (4864 frames) and EEG (73 frames).
    """
    T_max_audio = max(x["audio"].shape[0] for x in batch)
    T_max_eeg   = max(x["eeg"].shape[0]   for x in batch)

    def pad_seq(x, Tm):
        T, F = x.shape
        if T == Tm: return x
        pad = torch.zeros((Tm - T, F), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    audio = torch.stack([pad_seq(b["audio"], T_max_audio) for b in batch], dim=0)  # (B, T_audio, Fa)
    eeg   = torch.stack([pad_seq(b["eeg"],   T_max_eeg)   for b in batch], dim=0)  # (B, T_eeg, Fe)
    present = torch.stack([b["present"] for b in batch], dim=0)                    # (B, 4)
    labels  = torch.stack([b["label"]   for b in batch], dim=0)                    # (B,)
    trial_ids = [b["trial_id"] for b in batch]

    return {
        "audio": audio,
        "eeg": eeg,
        "present": present,
        "label": labels,
        "trial_id": trial_ids
    }
