# EGMP_util.py
# Small utilities shared by EGMP_train.py

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

INSTR_LIST = ["Gt", "Vx", "Dr", "Bs"]

def load_manifest(manifest_path):
    rows = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

class EGMPDataset(Dataset):
    """
    Loads full-trial .npz produced by EGMP_pre-processing.py
    Returns dict with:
      'audio':   (T, Fa)  concatenated [melLR, ild, ipd]
      'eeg':     (T, Fe)
      'meta':    (M,)     metadata vector: [pan_hist(3), gain_sorted(K->pad3), present_mask(4)]
      'present': (4,)
      'label':   scalar long
      'trial_id': str
    """
    def __init__(self, manifest_path, split="train", split_seed=0, val_ratio=0.2):
        rows = load_manifest(manifest_path)
        # simple deterministic split by hashing trial_id
        self.items = []
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
        mel_lr = z["audio_mel_lr"]     # (T, 2*nmels)
        ild    = z["audio_ild"]        # (T, nmels)
        ipd    = z["audio_ipd"]        # (T, nmels)
        audio  = np.concatenate([mel_lr, ild, ipd], axis=1).astype(np.float32)  # (T, Fa)

        eeg    = z["eeg_proc"].astype(np.float32)  # (T, Fe)

        # metadata vector (fixed-length): pan_hist(3), gain_sorted padded to 3, present_mask(4)
        pan_hist = z["pan_hist"].astype(np.float32)      # (3,)
        gain_sorted = z["gain_sorted"].astype(np.float32)  # (<=3,)
        if gain_sorted.shape[0] < 3:
            pad = np.full((3 - gain_sorted.shape[0],), np.nan, dtype=np.float32)
            gain_sorted = np.concatenate([gain_sorted, pad], axis=0)
        present_mask = z["present_mask"].astype(np.float32)  # (4,)
        meta = np.concatenate([pan_hist, gain_sorted[:3], present_mask], axis=0)

        y = int(z["label_idx"])
        trial_id = str(z["trial_id"])

        # Convert NaNs in meta to zeros (and add a companion mask bit)
        nan_mask = np.isnan(meta).astype(np.float32)
        meta = np.nan_to_num(meta, nan=0.0)
        meta = np.concatenate([meta, nan_mask], axis=0)  # append missingness mask

        sample = {
            "audio": torch.from_numpy(audio),  # (T, Fa)
            "eeg": torch.from_numpy(eeg),      # (T, Fe)
            "meta": torch.from_numpy(meta),    # (M,)
            "present": torch.from_numpy(present_mask),  # (4,)
            "label": torch.tensor(y, dtype=torch.long),
            "trial_id": trial_id
        }
        return sample

def collate_fulltrials(batch):
    # pad to max T in batch if necessary (they should be identical T, but keep safe)
    T_max = max(x["audio"].shape[0] for x in batch)
    def pad_seq(x, Tm):
        T, F = x.shape
        if T == Tm: return x
        pad = torch.zeros((Tm - T, F), dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    import torch
    audio = torch.stack([pad_seq(b["audio"], T_max) for b in batch], dim=0)  # (B, T, Fa)
    eeg   = torch.stack([pad_seq(b["eeg"],   T_max) for b in batch], dim=0)  # (B, T, Fe)
    meta  = torch.stack([b["meta"] for b in batch], dim=0)                   # (B, M)
    present = torch.stack([b["present"] for b in batch], dim=0)              # (B, 4)
    labels = torch.stack([b["label"] for b in batch], dim=0)                 # (B,)
    trial_ids = [b["trial_id"] for b in batch]
    return {"audio": audio, "eeg": eeg, "meta": meta, "present": present, "label": labels, "trial_id": trial_ids}
