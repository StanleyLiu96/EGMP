#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_pre-processing.py
Preprocess MAD-EEG trials for EGMP (EEG + Gains + Mixed audio + Panning).

Inputs (fixed shapes per your note):
- EEG .npy:  (20, 4864) at 256 Hz
  /users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/response_npy
- Mixed .wav: (837900, 2) at 44100 Hz
  /users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/stimulus_wav

Optional per-trial YAML (if present, same basename with .yml/.yaml) containing:
  wav_info:
    gains:   [ ... ]  # strings or floats
    panning: [ ... ]  # in [0,1], 0=Left, 1=Right

Outputs:
- EGMP_preprocessed/<subject_id>/<basename>.npz
  with:
    audio_mel_lr:  (T, 2*nmels)           # log-Mel L & R
    audio_ild:     (T, nmels)             # interaural level difference (L-R)
    audio_ipd:     (T, nmels)             # interaural phase diff proxy (unwrap(phaseL-phaseR))
    eeg_proc:      (T, n_eeg_feat)        # EEG features resampled to T (after re-ref + zscore + conv-lag bank)
    gains:         (K,) or NaNs           # from YAML if available
    panning:       (K,) or NaNs
    pan_hist:      (3,)                   # [Left, Center, Right] histogram over stems (if panning found)
    gain_sorted:   (K_sorted,)            # sorted gains ascending (NaNs if missing)
    present_mask:  (4,)                   # which of {Gt,Vx,Dr,Bs} appear in filename (0/1)
    label_idx:     ()                     # y in {0:Gt, 1:Vx, 2:Dr, 3:Bs}
    sr_audio:      ()                     # 44100
    hop_len:       ()                     # STFT hop used (samples)
    nmels:         ()
    trial_id:      (str)
- EGMP_preprocessed/manifest.csv (one row per npz)

Notes:
- We parse label (attended instrument) from EEG filename suffix before "_response.npy".
  Example: "..._Dr_response.npy" => label = Dr
- We infer instruments present from mixed-audio filename tokens containing {Gt,Vx,Dr,Bs}.
- We align EEG to audio frame-rate by linear interpolation after temporal conv-lag features.
"""

import os
import re
import csv
import json
import glob
import math
import yaml
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import librosa

# -----------------------------
# Paths (edit if needed)
# -----------------------------
EEG_DIR = "/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/response_npy"
AUDIO_DIR = "/users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/stimulus_wav"
OUT_DIR = "./EGMP_preprocessed"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
SR_AUDIO = 44100
EEG_SR = 256
NMELS = 96
N_FFT = 1024
HOP = 512  # -> audio frames T ~= 837900/512 ≈ 1637.9
INSTR_CODES = ["Gt", "Vx", "Dr", "Bs"]
LABEL_TO_IDX = {"Gt": 0, "Vx": 1, "Dr": 2, "Bs": 3}

# EEG preprocessing
BP_LO, BP_HI = 1.0, 32.0   # band-pass Hz (optional but recommended)
USE_BANDPASS = True
# Temporal conv-lag bank: emulate 0..250ms neural latency with a tiny filter bank via shifted copies
LAG_MS = [0, 50, 100, 150, 200, 250]  # ms
# For re-referencing and z-score:
EPS = 1e-8

# -----------------------------
# Helpers
# -----------------------------
def parse_subject_and_label(eeg_path):
    """
    EEG filename examples:
      0001_pop_mixtape_trio_GtDrVx_theme2_stereo_Dr_response.npy
    We take leading 4 digits as subject id.
    Label = last instrument code before '_response.npy'
    """
    base = os.path.basename(eeg_path)
    m = re.match(r"^(\d{4})_.*_([A-Za-z]{2})_response\.npy$", base)
    if m:
        sid = m.group(1)
        label = m.group(2)
    else:
        # Fallback: try more permissive parse
        sid = base.split("_")[0]
        label = base.split("_")[-2] if base.endswith("_response.npy") else None
    if label not in LABEL_TO_IDX:
        raise ValueError(f"Cannot parse attended label from {base}")
    return sid, label

def find_audio_for_eeg(eeg_path):
    """
    We map EEG file to a mixed WAV by removing the trailing '_<Inst>_response.npy'
    and replacing directory + extension.
    Example:
      EEG:  .../response_npy/0001_pop_mixtape_trio_GtDrVx_theme2_stereo_Dr_response.npy
      WAV:  .../stimulus_wav/0001_pop_mixtape_trio_GtDrVx_theme2_stereo.wav
    """
    eeg_base = os.path.basename(eeg_path)
    core = re.sub(r"_[A-Za-z]{2}_response\.npy$", "", eeg_base)
    wav_path = os.path.join(AUDIO_DIR, core + ".wav")
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Mixed WAV not found for EEG {eeg_base}: {wav_path}")
    return wav_path, core

def parse_present_instruments_from_wav(wav_base):
    """
    From basename tokens, detect which of {Gt,Vx,Dr,Bs} are present.
    Example: "..._GtDrVx_..." => present Gt, Dr, Vx
    """
    pres = np.zeros(4, dtype=np.float32)
    for code, idx in LABEL_TO_IDX.items():
        if f"_{code}" in wav_base or code in wav_base.split("_"):
            pres[idx] = 1.0
    # Also look for bundled token like "..._GtDrVx_..."
    for token in wav_base.split("_"):
        for code, idx in LABEL_TO_IDX.items():
            if code in token and len(token) <= 6:
                pres[idx] = 1.0
    return pres

def maybe_read_yaml_sidecar(wav_path):
    """
    Try to read a YAML with same basename: *.yml or *.yaml
    Returns (gains, panning, pan_hist, gain_sorted)
    If missing, returns NaNs.
    """
    base = os.path.splitext(wav_path)[0]
    yml = None
    for ext in [".yml", ".yaml"]:
        cand = base + ext
        if os.path.exists(cand):
            yml = cand
            break
    if yml is None:
        K = 3  # duet/trio max; use NaNs
        return (np.full((K,), np.nan, dtype=np.float32),
                np.full((K,), np.nan, dtype=np.float32),
                np.array([np.nan, np.nan, np.nan], dtype=np.float32),
                np.full((K,), np.nan, dtype=np.float32))
    with open(yml, "r") as f:
        data = yaml.safe_load(f)

    try:
        gains = data.get("wav_info", {}).get("gains", [])
        panning = data.get("wav_info", {}).get("panning", [])
        gains = np.array([float(x) for x in gains], dtype=np.float32)
        panning = np.array([float(x) for x in panning], dtype=np.float32)
        # pan histogram: L (<0.33), C (0.33-0.66), R (>0.66)
        hist = np.zeros(3, dtype=np.float32)
        for p in panning:
            if p < 0.33: hist[0] += 1
            elif p > 0.66: hist[2] += 1
            else: hist[1] += 1
        gain_sorted = np.sort(gains)
        return gains, panning, hist, gain_sorted
    except Exception:
        K = 3
        return (np.full((K,), np.nan, dtype=np.float32),
                np.full((K,), np.nan, dtype=np.float32),
                np.array([np.nan, np.nan, np.nan], dtype=np.float32),
                np.full((K,), np.nan, dtype=np.float32))

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def eeg_preprocess(eeg_raw, per_subj_stats):
    """
    eeg_raw: (20, 4864)
    Steps:
      1) Common-average re-reference (subtract mean across channels)
      2) Optional band-pass 1-32 Hz
      3) Per-subject z-score (use provided stats; if missing, per-trial fallback)
      4) Build lag bank (0..250ms) by shifting along time (pad with edge)
    Returns: eeg_feats: (n_lags * 20, 4864)
    """
    x = eeg_raw.astype(np.float32)
    # common-average re-ref
    x = x - x.mean(axis=0, keepdims=True)

    # band-pass
    if USE_BANDPASS:
        b, a = butter_bandpass(BP_LO, BP_HI, EEG_SR, order=4)
        x = filtfilt(b, a, x, axis=1, method='gust')

    # z-score (per subject)
    if per_subj_stats is not None:
        mean, std = per_subj_stats
        std = np.where(std < 1e-6, 1.0, std)
        x = (x - mean) / std
    else:
        # fallback per-trial
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.std(axis=1, keepdims=True) + EPS
        x = (x - mu) / sigma

    # Lag bank (shift to the RIGHT in samples to emulate delayed brain)
    lags = [int(ms/1000.0 * EEG_SR) for ms in LAG_MS]
    feats = []
    for L in lags:
        if L == 0:
            feats.append(x)
        else:
            pad = np.repeat(x[:, :1], L, axis=1)
            shifted = np.concatenate([pad, x[:, :-L]], axis=1)
            feats.append(shifted)
    eeg_feats = np.concatenate(feats, axis=0)  # (20*len(lags), 4864)
    return eeg_feats

def resample_eeg_to_audio_frames(eeg_feats, T_audio):
    """
    eeg_feats: (C_eeg, 4864) -> transpose time first -> (4864, C_eeg)
    We linearly resample to T_audio frames (≈1638)
    Returns: (T_audio, C_eeg)
    """
    C, Teeg = eeg_feats.shape
    t_src = np.linspace(0.0, 1.0, num=Teeg, dtype=np.float32)
    t_dst = np.linspace(0.0, 1.0, num=T_audio, dtype=np.float32)
    out = np.empty((T_audio, C), dtype=np.float32)
    # vectorized linear interpolation per channel
    for c in range(C):
        out[:, c] = np.interp(t_dst, t_src, eeg_feats[c, :])
    return out

def audio_features(wav_stereo):
    """
    wav_stereo: (N, 2) float32 at 44100
    Returns:
      mel_lr: (T, 2*NMELS)
      ild:    (T, NMELS)
      ipd:    (T, NMELS)
      T: number of frames
    """
    L = wav_stereo[:,0].astype(np.float32)
    R = wav_stereo[:,1].astype(np.float32)

    # STFT
    SL = librosa.stft(L, n_fft=N_FFT, hop_length=HOP, window='hann', center=True)
    SR = librosa.stft(R, n_fft=N_FFT, hop_length=HOP, window='hann', center=True)

    # Mel filterbank
    mel_fb = librosa.filters.mel(sr=SR_AUDIO, n_fft=N_FFT, n_mels=NMELS, fmin=20.0, fmax=sr_to_fmax(SR_AUDIO))
    # Power spectrograms
    PL = np.abs(SL)**2
    PR = np.abs(SR)**2
    # Mel energies
    ML = np.dot(mel_fb, PL)
    MR = np.dot(mel_fb, PR)
    # log-mel
    logML = np.log(ML + 1e-8).T   # (T, NMELS)
    logMR = np.log(MR + 1e-8).T   # (T, NMELS)

    # ILD (level diff, mel-bandwise)
    ild = (logML - logMR)  # (T, NMELS)

    # IPD proxy: unwrap phase difference, then project to mel bands by energy-weighted avg
    phaseL = np.angle(SL)  # (F, T)
    phaseR = np.angle(SR)
    # raw IPD per linear bin
    ipd_lin = np.unwrap(phaseL - phaseR, axis=0)  # (F, T)
    # energy weights from (PL+PR)
    W = (PL + PR) + 1e-8
    # project to mel: (NMELS, F) @ (F, T) -> (NMELS, T) -> (T, NMELS)
    ipd_mel = (mel_fb @ (ipd_lin * W) ) / (mel_fb @ W)
    ipd = np.clip(ipd_mel.T, -np.pi, np.pi)

    mel_lr = np.concatenate([logML.T, logMR.T], axis=1)  # (T, 2*NMELS)
    T = mel_lr.shape[0]
    return mel_lr.astype(np.float32), ild.astype(np.float32), ipd.astype(np.float32), T

def sr_to_fmax(sr):
    # conservative upper mel limit
    return min(16000.0, 0.45*sr)

# -----------------------------
# Two-pass per-subject z-score
# -----------------------------
def collect_subject_stats(eeg_paths):
    """
    First pass: compute per-subject mean/std per channel AFTER re-ref (and band-pass if enabled).
    Returns dict sid -> (mean(20,), std(20,))
    """
    stats = {}
    acc_sum = {}
    acc_sq = {}
    count = {}
    for eeg_path in eeg_paths:
        sid, _ = parse_subject_and_label(eeg_path)
        x = np.load(eeg_path)  # (20, 4864)
        # re-ref & band-pass only (no z)
        xr = x - x.mean(axis=0, keepdims=True)
        if USE_BANDPASS:
            b, a = butter_bandpass(BP_LO, BP_HI, EEG_SR, order=4)
            xr = filtfilt(b, a, xr, axis=1, method='gust')
        mu = xr.mean(axis=1)     # (20,)
        var = xr.var(axis=1)     # (20,)
        if sid not in acc_sum:
            acc_sum[sid] = mu.copy()
            acc_sq[sid]  = var + mu**2
            count[sid] = 1
        else:
            acc_sum[sid] += mu
            acc_sq[sid]  += var + mu**2
            count[sid] += 1
    for sid in acc_sum:
        n = count[sid]
        m = acc_sum[sid] / n
        ex2 = acc_sq[sid] / n
        v = np.maximum(ex2 - m**2, 1e-6)
        s = np.sqrt(v)
        stats[sid] = (m.astype(np.float32), s.astype(np.float32))
    return stats

# -----------------------------
# Main
# -----------------------------
def main():
    eeg_paths = sorted(glob.glob(os.path.join(EEG_DIR, "*.npy")))
    if not eeg_paths:
        raise RuntimeError(f"No EEG .npy files found under {EEG_DIR}")

    # First pass for per-subject stats
    subj_stats = collect_subject_stats(eeg_paths)

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["trial_id","subject","npz_path","label_idx","present_mask","has_yaml","T_frames"])

        for eeg_path in eeg_paths:
            sid, label_code = parse_subject_and_label(eeg_path)
            label_idx = LABEL_TO_IDX[label_code]

            wav_path, core = find_audio_for_eeg(eeg_path)
            wav_stereo, sr = sf.read(wav_path, always_2d=True)
            if sr != SR_AUDIO:
                raise ValueError(f"Expected {SR_AUDIO} Hz WAV, got {sr} in {wav_path}")
            if wav_stereo.shape != (837900, 2):
                # enforce known length/shape
                wav_stereo = wav_stereo[:837900, :2]

            # audio features
            mel_lr, ild, ipd, T = audio_features(wav_stereo)

            # metadata
            gains, panning, pan_hist, gain_sorted = maybe_read_yaml_sidecar(wav_path)
            has_yaml = not np.isnan(gains).all()

            present_mask = parse_present_instruments_from_wav(os.path.basename(wav_path))

            # EEG
            eeg = np.load(eeg_path)  # (20, 4864)
            mean_std = subj_stats.get(sid, None)
            eeg_feats = eeg_preprocess(eeg, mean_std)        # (20*len(lags), 4864)
            eeg_resamp = resample_eeg_to_audio_frames(eeg_feats, T)  # (T, C_eeg)

            # Pack features
            trial_id = core + f"_{label_code}"
            out_dir = os.path.join(OUT_DIR, sid)
            os.makedirs(out_dir, exist_ok=True)
            out_npz = os.path.join(out_dir, trial_id + ".npz")
            np.savez_compressed(
                out_npz,
                audio_mel_lr=mel_lr,
                audio_ild=ild,
                audio_ipd=ipd,
                eeg_proc=eeg_resamp,
                gains=gains.astype(np.float32),
                panning=panning.astype(np.float32),
                pan_hist=pan_hist.astype(np.float32),
                gain_sorted=gain_sorted.astype(np.float32),
                present_mask=present_mask.astype(np.float32),
                label_idx=np.int64(label_idx),
                sr_audio=np.int64(SR_AUDIO),
                hop_len=np.int64(HOP),
                nmels=np.int64(NMELS),
                trial_id=trial_id
            )

            writer.writerow([trial_id, sid, out_npz, label_idx,
                             ";".join([str(int(x)) for x in present_mask.tolist()]),
                             int(has_yaml), T])

    print(f"Done. Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    main()
