#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_dataset.py
Dataset loaders for EEG, Mixed Audio, and Solo Audio.
------------------------------------------------------
This file does the following:

1. Scans three source folders:
   - EEG response files  (.npy, suffix = "_response.npy")
   - Mixed-audio stimulus (.wav, suffix = "_stimulus.wav")
   - Solo-audio isolated (.wav, suffix = "_soli.wav")

2. For each file:
   - Strips off the suffix so we get a common "base name"
   - Splits the name by underscores "_" and looks at the last token
     → this is the instrument label (e.g. "Gt", "Vx", "Dr", "Bs")

3. Filtering rules:
   - Keep only if the instrument is inside INSTRUMENTS
   - Check if the base name is listed in the val split file
     → if yes → goes into validation set
     → if no  → goes into training set

4. EEG special case:
   - EEG files always have shape (20, 4864)
   - We only pick the subset of channels specified in CHANNELS
     by converting them to indices.

At the end:
- load_all_datasets() will return 6 lists:
  EEG_train, EEG_val, mixed_audio_train, mixed_audio_val, solo_audio_train, solo_audio_val
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import wavfile

# =====================
# User-set variables
# =====================
# EEG channels we want to load (out of 20 total)
CHANNELS = ["F3","F1","Fz","F2","F4","C3","C1","Cz","C2","C4","CPz","P3","P1","Pz","P2","P4","POz","O1","Oz","O2"]

# Instruments we are interested in
# Anything not in this list will be skipped
INSTRUMENTS = ["Gt", "Vx", "Dr", "Bs"]

# Paths for different data types
EEG_PATH = "../../../Dataset/MADEEG_normalized/response_npy"
MIXED_AUDIO_PATH = "../../../Dataset/MADEEG_normalized/stimulus_wav"
SOLO_AUDIO_PATH = "../../../Dataset/MADEEG_normalized/isolated_wav"

# File containing the list of validation split basenames
VAL_SPLIT_FILE = "../../../Models/EGMP/val_split_filenames.txt"

# =====================
# EEG channel mapping
# =====================
ALL_CHANNELS = ["F3","F1","Fz","F2","F4","C3","C1","Cz","C2","C4","CPz","P3","P1","Pz","P2","P4","POz","O1","Oz","O2"]
CHANNEL_TO_INDEX = {ch: i for i, ch in enumerate(ALL_CHANNELS)}


# =====================
# Helper functions
# =====================
def load_val_split():
    """Read validation split file into a set of base names (strings)."""
    with open(VAL_SPLIT_FILE, "r") as f:
        return set(line.strip() for line in f)


def split_files(folder, suffix, instruments, val_list):
    """
    Generic helper to scan a folder and split into train/val.

    Parameters:
    -----------
    folder: str
        Path to scan
    suffix: str
        File suffix to strip (e.g. "_response.npy")
    instruments: list
        Which instruments to keep
    val_list: set
        Basenames listed as validation

    Returns:
    --------
    train_set, val_set: list of file names
    """
    train_set, val_set = [], []

    # List only files that match the suffix
    all_files = sorted(f for f in os.listdir(folder) if f.endswith(suffix))

    for f in all_files:
        # Strip suffix → gives the "base name"
        base = f.replace(suffix, "")

        # Example: "0001_classique_morceau1_duo_CoFl_theme1_stereo_Co"
        # Split by "_"
        parts = base.split("_")

        # Last element is instrument (e.g. "Co", "Gt", "Vx", "Dr", "Bs")
        instr = parts[-1]

        # Skip if instrument not in our list
        if instr not in instruments:
            continue

        # Check if this base is in validation split file
        if base in val_list:
            val_set.append(f)
        else:
            train_set.append(f)

    return train_set, val_set


# =====================
# EEG Dataset
# =====================
class EEGDataset(Dataset):
    def __init__(self, channels=CHANNELS, instruments=INSTRUMENTS):
        # Convert channel names to indices (0–19)
        self.channels = [CHANNEL_TO_INDEX[ch] for ch in channels]
        self.instruments = instruments

        # Load validation split
        val_list = load_val_split()

        # Get train/val file lists
        self.train_files, self.val_files = split_files(
            EEG_PATH, "_response.npy", instruments, val_list
        )

        print(f"[EEGDataset] Loaded {len(self.train_files)} train and {len(self.val_files)} val files "
              f"(channels={channels}, instruments={instruments})")

    def load_file(self, file_name):
        """
        Load one EEG file:
        - Shape is always (20, 4864)
        - Select only chosen channels
        """
        file_path = os.path.join(EEG_PATH, file_name)
        eeg_full = np.load(file_path)   # full (20, 4864)
        eeg_selected = eeg_full[self.channels, :]  # keep only chosen channels
        return torch.tensor(eeg_selected, dtype=torch.float32)

    def get_train(self):
        """Return list of (tensor, filename) for train set."""
        return [(self.load_file(f), f) for f in self.train_files]

    def get_val(self):
        """Return list of (tensor, filename) for val set."""
        return [(self.load_file(f), f) for f in self.val_files]


# =====================
# Audio Dataset (generic → works for mixed or solo)
# =====================
class AudioDataset(Dataset):
    def __init__(self, audio_path, suffix, instruments=INSTRUMENTS):
        self.audio_path = audio_path
        self.instruments = instruments

        # Load validation split
        val_list = load_val_split()

        # Get train/val file lists
        self.train_files, self.val_files = split_files(
            audio_path, suffix, instruments, val_list
        )

        print(f"[AudioDataset:{os.path.basename(audio_path)}] Loaded {len(self.train_files)} train and {len(self.val_files)} val files "
              f"(instruments={instruments})")

    def load_file(self, file_name):
        """
        Load one audio file:
        - wavfile.read returns (sample_rate, np.array)
        """
        file_path = os.path.join(self.audio_path, file_name)
        _, audio = wavfile.read(file_path)  # audio can be (samples,) or (samples, 2)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        return audio_tensor

    def get_train(self):
        """Return list of (tensor, filename) for train set."""
        return [(self.load_file(f), f) for f in self.train_files]

    def get_val(self):
        """Return list of (tensor, filename) for val set."""
        return [(self.load_file(f), f) for f in self.val_files]


# =====================
# Convenience function
# =====================
def load_all_datasets():
    """
    Return all six sets in one call:
    - EEG train / val
    - Mixed audio train / val
    - Solo audio train / val
    - Each entry is a tuple (tensor, filename)
    """
    eeg = EEGDataset()
    mixed = AudioDataset(MIXED_AUDIO_PATH, "_stimulus.wav")
    solo = AudioDataset(SOLO_AUDIO_PATH, "_soli.wav")

    return (
        eeg.get_train(), eeg.get_val(),
        mixed.get_train(), mixed.get_val(),
        solo.get_train(), solo.get_val()
    )


# =====================
# Example usage
# =====================
if __name__ == "__main__":
    EEG_train, EEG_val, mixed_audio_train, mixed_audio_val, solo_audio_train, solo_audio_val = load_all_datasets()

    print("EEG train sample:", EEG_train[0][0].shape, EEG_train[0][1])
    print("Mixed audio train sample:", mixed_audio_train[0][0].shape, mixed_audio_train[0][1])
    print("Solo audio train sample:", solo_audio_train[0][0].shape, solo_audio_train[0][1])


    # print("-----EEG_train")
    # for eeg_tensor, fname in EEG_train:
    #     print(fname, eeg_tensor.shape)
    # print("-----EEG_val")
    # for eeg_tensor, fname in EEG_val:
    #     print(fname, eeg_tensor.shape)
    # print("-----mixed_audio_train")
    # for eeg_tensor, fname in mixed_audio_train:
    #     print(fname, eeg_tensor.shape)
    # print("-----mixed_audio_val")
    # for eeg_tensor, fname in mixed_audio_val:
    #     print(fname, eeg_tensor.shape)
    # print("-----solo_audio_train")
    # for eeg_tensor, fname in solo_audio_train:
    #     print(fname, eeg_tensor.shape)
    # print("-----solo_audio_val")
    # for eeg_tensor, fname in solo_audio_val:
    #     print(fname, eeg_tensor.shape)

    # print("EEG train sample:", EEG_train[0][0].shape, EEG_train[0][1])
    # print("Mixed audio train sample:", mixed_audio_train[0][0].shape, "sr:", mixed_audio_train[0][1], mixed_audio_train[0][2])
    # print("Solo audio train sample:", solo_audio_train[0][0].shape, "sr:", solo_audio_train[0][1], solo_audio_train[0][2])
