#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from scipy.io import wavfile
from scipy.signal import iirnotch, butter, filtfilt, resample_poly

# -----------------------------
# Config / Constants
# -----------------------------
ALLOW_INSTR = ["Gt", "Vx", "Dr", "Bs"]
INSTR2IDX = {k: i for i, k in enumerate(ALLOW_INSTR)}

@dataclass
class Paths:
    mixed_dir: str
    solo_dir: str
    eeg_dir: str
    meta_yaml: str

@dataclass
class TrialItem:
    base: str                 # e.g., "0001_pop_falldead_duo_GtVx_theme1_stereo_Gt"
    eeg_ch_names: List[str]
    eeg_sfreq: float
    wav_sfreq: int
    present_vec: np.ndarray   # (4,) float {0,1}
    target_idx: int           # int in [0..3]
    eeg_path: str
    mixed_path: str
    # solo_path optional; we don't need it for training in this part

# -----------------------------
# YAML / Listing
# -----------------------------
def load_meta_trials(paths: Paths, eeg_channel_name: str = "F3") -> List[TrialItem]:
    """
    Parse the dataset YAML and create a filtered list of trials that:
      - have target in {Gt,Vx,Dr,Bs}
      - construct presence vector over the same 4 classes
      - map file paths for mixed wav and EEG npy
    """
    with open(paths.meta_yaml, "r") as f:
        meta = yaml.safe_load(f)

    items: List[TrialItem] = []
    for subj_id, entries in meta.items():
        for key, info in entries.items():
            target = info.get("target")
            inst_list = info.get("instruments", [])
            wav_info = info.get("wav_info", {})
            eeg_info = info.get("eeg_info", {})
            if target not in INSTR2IDX:
                # skip non-target instruments (Co, Fl, Ob, Bo, Fh, etc.)
                continue

            # presence vector over the 4 allowed classes
            pres = np.zeros(4, dtype=np.float32)
            for inst in inst_list:
                if inst in INSTR2IDX:
                    pres[INSTR2IDX[inst]] = 1.0

            base = f"{subj_id}_{key}"  # matches filenames
            eeg_path   = os.path.join(paths.eeg_dir,   f"{base}_response.npy")
            mixed_path = os.path.join(paths.mixed_dir, f"{base}_stimulus.wav")

            if not (os.path.isfile(eeg_path) and os.path.isfile(mixed_path)):
                # silently skip missing pairs
                continue

            ch_names = eeg_info.get("ch_names", [])
            eeg_sfreq = float(eeg_info.get("sfreq", 256.0))
            wav_sfreq = int(wav_info.get("sfreq", 44100))

            items.append(TrialItem(
                base=base,
                eeg_ch_names=ch_names,
                eeg_sfreq=eeg_sfreq,
                wav_sfreq=wav_sfreq,
                present_vec=pres,
                target_idx=int(INSTR2IDX[target]),
                eeg_path=eeg_path,
                mixed_path=mixed_path,
            ))
    return items

# -----------------------------
# EEG preprocessing (single channel)
# -----------------------------
def _safe_channel_index(ch_names: List[str], want: str) -> int:
    if want in ch_names:
        return ch_names.index(want)
    # fallback: try upper-casing or common aliases
    want_alt = want.upper()
    for i, n in enumerate(ch_names):
        if n.upper() == want_alt:
            return i
    # last resort: first channel
    return 0

def notch_50hz(x: np.ndarray, fs: float, q: float = 30.0) -> np.ndarray:
    """Notch 50 Hz only (hard-coded per user’s spec)."""
    b, a = iirnotch(w0=50.0 / (fs / 2.0), Q=q)
    return filtfilt(b, a, x).astype(np.float32)

def butter_highpass(x: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff / (fs / 2.0), btype="highpass")
    return filtfilt(b, a, x).astype(np.float32)

def zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    sd = sd if sd > eps else eps
    return ((x - mu) / sd).astype(np.float32)

def preprocess_eeg_1ch(
    eeg_np: np.ndarray,
    fs: float,
    ch_names: List[str],
    channel_name: str = "F3",
) -> np.ndarray:
    """
    eeg_np: (T, C) or (C, T). We'll auto-fix orientation.
    Steps: Notch 50 -> High-pass -> Z-score. No clamp.
    Returns mono vector float32, shape (T,)
    """
    arr = np.asarray(eeg_np)
    if arr.ndim != 2:
        raise ValueError(f"EEG array must be 2D, got {arr.shape}")
    T0, C0 = arr.shape
    # Heuristic: pick the axis that equals #channels in YAML
    idx = _safe_channel_index(ch_names, channel_name)
    if C0 == len(ch_names):      # (T, C)
        x = arr[:, idx]
    elif T0 == len(ch_names):    # (C, T)
        x = arr[idx, :]
    else:
        # fallback assume (T, C)
        x = arr[:, idx]
    x = x.astype(np.float32)

    x = notch_50hz(x, fs)
    x = butter_highpass(x, fs, cutoff=0.5, order=4)
    x = zscore(x)
    return x

# -----------------------------
# Audio loading / resample
# -----------------------------
def load_mono_audio_resampled(path: str, target_sr: int = 256) -> np.ndarray:
    """
    Load .wav, downmix to mono, resample to target_sr using polyphase.
    Returns float32 in [-1, 1] approximately.
    """
    sr, data = wavfile.read(path)  # int16 or float
    x = data.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    # normalize if int
    if x.dtype.kind in ("i", "u"):
        maxv = np.max(np.abs(x)) or 1.0
        x = x / maxv

    # resample to target_sr
    if sr != target_sr:
        # resample_poly expects integers for up/down; use gcd
        from math import gcd
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        x = resample_poly(x, up=up, down=down).astype(np.float32)
    else:
        x = x.astype(np.float32)
    return x

# -----------------------------
# Dataset
# -----------------------------
class OneChanDataset:
    """
    Returns per-item dict:
      {
        "audio": float32 [T], resampled to 256 Hz,
        "eeg":   float32 [T], preprocessed, same 256 Hz grid,
        "present": float32 [4],
        "label": int [0..3],
      }
    We align by cropping both to min length per trial.
    """
    def __init__(self, items: List[TrialItem], eeg_channel_name: str = "F3"):
        self.items = items
        self.eeg_channel = eeg_channel_name

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> Dict:
        it = self.items[i]
        # load
        eeg_np = np.load(it.eeg_path, allow_pickle=False)
        eeg_vec = preprocess_eeg_1ch(eeg_np, fs=it.eeg_sfreq, ch_names=it.eeg_ch_names, channel_name=self.eeg_channel)
        # resample EEG to 256 Hz if needed (most are already 256)
        if it.eeg_sfreq != 256.0:
            from math import gcd
            g = gcd(int(it.eeg_sfreq), 256)
            up = 256 // g
            down = int(it.eeg_sfreq) // g
            eeg_vec = resample_poly(eeg_vec, up=up, down=down).astype(np.float32)

        aud_vec = load_mono_audio_resampled(it.mixed_path, target_sr=256)

        # align by cropping
        T = min(len(eeg_vec), len(aud_vec))
        eeg_vec = eeg_vec[:T].astype(np.float32)
        aud_vec = aud_vec[:T].astype(np.float32)

        return {
            "audio": aud_vec,
            "eeg": eeg_vec,
            "present": it.present_vec.astype(np.float32),
            "label": int(it.target_idx),
            "base": it.base,
        }


def _normalize_base(s: str) -> str:
    """Map '.../foo_stimulus.wav' or 'foo_response.npy' → 'foo'."""
    b = os.path.basename(s.strip())
    for suf in ("_stimulus.wav", "_response.npy", ".wav", ".npy"):
        if b.endswith(suf):
            b = b[: -len(suf)]
            break
    return b

def split_train_val_from_file(items: List[TrialItem], val_list_path: str):
    """
    Deterministic split:
    - Validation = exactly the trials listed in val_list_path
    - Training   = all remaining items (no leakage)
    """
    with open(val_list_path, "r") as f:
        val_bases = { _normalize_base(line) for line in f if line.strip() }

    va_items = [it for it in items if it.base in val_bases]
    tr_items = [it for it in items if it.base not in val_bases]

    # Optional sanity check:
    # unseen = sorted(val_bases - {it.base for it in items})
    # if unseen:
    #     print(f"[warn] {len(unseen)} val ids not found in items (e.g. {unseen[:5]})")

    return tr_items, va_items

def collate_crop_min(batch: List[Dict]) -> Dict:
    """Crop every sample in the batch to the minimum T (keeps shapes equal; avoids padding masks)."""
    T_min = min(len(b["audio"]) for b in batch)
    B = len(batch)
    audio = np.stack([b["audio"][:T_min] for b in batch], axis=0)  # (B, T)
    eeg   = np.stack([b["eeg"][:T_min]   for b in batch], axis=0)  # (B, T)
    present = np.stack([b["present"] for b in batch], axis=0)      # (B, 4)
    labels  = np.array([b["label"] for b in batch], dtype=np.int64)  # (B,)
    bases   = [b["base"] for b in batch]
    return {
        "audio": audio.astype(np.float32),
        "eeg": eeg.astype(np.float32),
        "present": present.astype(np.float32),
        "label": labels,
        "base": bases,
    }
