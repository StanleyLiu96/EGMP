#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_pre-processing.py  (MAD-EEG-aligned timing)

- Align audio features to EEG at 256 Hz by using STFT:
    hop_length = round(44100 / 256) = 172 samples
    win_length = 2 * hop_length = 344 samples
  Then resample features to exactly 4864 frames (19 s * 256 Hz).

- No EEG upsampling. EEG stays at 256 Hz. We:
    * common-average re-reference
    * optional band-pass 1-32 Hz
    * per-subject z-score (two-pass)
    * build lag bank (0..250 ms)
    * (only if needed) tiny linear time-resample to match audio frames

Inputs:
  EEG .npy   (20, 4864) @ 256 Hz
    /users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/response_npy
  Mixed .wav (837900, 2) @ 44100 Hz
    /users/PAS2301/liu215229932/Music_Project/Dataset/MADEEG/processed_data/stimulus_wav
  Master YAML: madeeg_preprocessed.yaml
    keys: "<subject_id>": { "<trial_key>": { instruments, target, genre, ensemble, spatial,
                                             wav_info: {gains, panning, sfreq} } }

Outputs:
  EGMP_preprocessed/<subject>/<trial_id>.npz
    audio_mel_lr:  (4864, 2*nmels)
    audio_ild:     (4864, nmels)
    audio_ipd:     (4864, nmels)
    eeg_proc:      (4864, n_eeg_feat)    # n_eeg_feat = 20 * len(LAG_MS)
    gains:         (K,)                  # as floats from YAML
    panning:       (K,)
    pan_hist:      (3,)                  # [Left, Center, Right] counts
    gain_sorted:   (K,)
    present_mask:  (4,)                  # {Gt,Vx,Dr,Bs}
    label_idx:     ()                    # 0:Gt, 1:Vx, 2:Dr, 3:Bs
    sr_audio, hop_len, win_len, nmels, trial_id, genre, ensemble, spatial

Also writes EGMP_preprocessed/manifest.csv
"""

import os
import re
import csv
import glob
import math
import argparse
import numpy as np
import soundfile as sf
import librosa
import yaml
from scipy.signal import butter, filtfilt, iirnotch

# -----------------------------
# Paths (CLI overridable)
# -----------------------------
DEF_EEG_DIR = "../../Dataset/MADEEG/processed_data/response_npy"
DEF_AUDIO_DIR = "../../Dataset/MADEEG/processed_data/stimulus_wav"
DEF_YAML_PATH = "../../Dataset/MADEEG/madeeg_preprocessed.yaml"
OUT_DIR = "./EGMP_preprocessed"

# -----------------------------
# Constants / Config
# -----------------------------
SR_AUDIO = 44100
EEG_SR = 256
EEG_LEN = 4864                 # strict length per your note
NMELS = 96
HOP = 172                      # ≈ 44100 / 256
WIN = 344                      # 2 * HOP
N_FFT = 1024                   # was: N_FFT = WIN
FMIN = 20.0
FMAX = 16000.0

INSTR_CODES = ["Gt", "Vx", "Dr", "Bs"]
LABEL_TO_IDX = {"Gt": 0, "Vx": 1, "Dr": 2, "Bs": 3}
POP_ONLY = True  # skip trials whose attended label not in {Gt,Vx,Dr,Bs}

# EEG preprocessing
USE_BANDPASS = True
BP_LO, BP_HI = 1.0, 32.0
LAG_MS = [0, 50, 100, 150, 200, 250]  # neural latency modeling
EPS = 1e-8

# -----------------------------
# Helpers
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def parse_subject_and_trialkey(eeg_path):
    """
    EEG filename example:
      0001_pop_mixtape_trio_GtDrVx_theme2_stereo_Dr_response.npy
    subject = '0001'
    trialkey = 'pop_mixtape_trio_GtDrVx_theme2_stereo_Dr'
    """
    base = os.path.basename(eeg_path)
    m = re.match(r"^(\d{4})_(.+)_response\.npy$", base)
    if not m:
        raise ValueError(f"Unexpected EEG filename: {base}")
    sid = m.group(1)
    trialkey = m.group(2)
    return sid, trialkey

def wav_path_from_eeg(eeg_path, audio_dir):
    """
    EEG:  0001_pop_mixtape_duo_GtVx_theme2_stereo_Vx_response.npy
    WAV:  0001_pop_mixtape_duo_GtVx_theme2_stereo_Vx_stimulus.wav
    (keep the target instrument; just swap suffix)
    """
    eeg_base = os.path.basename(eeg_path)
    core_with_target = eeg_base.replace("_response.npy", "")  # keep the ..._stereo_<Inst>
    wav_name = core_with_target + "_stimulus.wav"
    wav_path = os.path.join(audio_dir, wav_name)
    return wav_path, core_with_target

def collect_subject_stats(eeg_paths):
    """
    Per-subject channel-wise mean/std AFTER re-ref (+band-pass if enabled), for z-scoring.
    Returns: dict[sid] -> (mean(20,), std(20,))
    """
    acc_sum, acc_sq, count = {}, {}, {}
    for eeg_path in eeg_paths:
        sid, _ = parse_subject_and_trialkey(eeg_path)
        x = np.load(eeg_path)  # (20,4864)
        xr = x - x.mean(axis=0, keepdims=True)
        xr = apply_notch_filter(xr, notch_freq=50.0, fs=EEG_SR)
        if USE_BANDPASS:
            b, a = butter_bandpass(BP_LO, BP_HI, EEG_SR, order=4)
            xr = filtfilt(b, a, xr, axis=1, method='gust')
        mu = xr.mean(axis=1)  # (20,)
        var = xr.var(axis=1)  # (20,)
        if sid not in acc_sum:
            acc_sum[sid] = mu.copy()
            acc_sq[sid]  = var + mu**2
            count[sid]   = 1
        else:
            acc_sum[sid] += mu
            acc_sq[sid]  += var + mu**2
            count[sid]   += 1
    stats = {}
    for sid in acc_sum:
        n = count[sid]
        m = acc_sum[sid] / n
        ex2 = acc_sq[sid] / n
        v = np.maximum(ex2 - m**2, 1e-6)
        s = np.sqrt(v)
        stats[sid] = (m.astype(np.float32), s.astype(np.float32))
    return stats

def eeg_preprocess(eeg_raw, mean_std):
    """
    eeg_raw: (20, 4864)
    Steps:
      - Common-average re-reference
      - Optional band-pass 1-32 Hz
      - Per-subject z-score
      - Lag bank (0..250 ms) via time shifts (pad at start)
    Return: eeg_feats (n_ch=20*len(LAG_MS), 4864)
    """
    x = eeg_raw.astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)  # re-ref
    x = apply_notch_filter(x, notch_freq=50.0, fs=EEG_SR)

    if USE_BANDPASS:
        b, a = butter_bandpass(BP_LO, BP_HI, EEG_SR, order=4)
        x = filtfilt(b, a, x, axis=1, method='gust')

    if mean_std is not None:
        mu, sd = mean_std
        sd = np.where(sd < 1e-6, 1.0, sd)
        x = (x - mu[:, None]) / sd[:, None]
    else:
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + EPS
        x = (x - mu) / sd

    lags = [int(ms/1000.0 * EEG_SR) for ms in LAG_MS]
    feats = []
    for L in lags:
        if L == 0:
            feats.append(x)
        else:
            pad = np.repeat(x[:, :1], L, axis=1)
            shifted = np.concatenate([pad, x[:, :-L]], axis=1)
            feats.append(shifted)
    eeg_feats = np.concatenate(feats, axis=0)  # (20*L, 4864)
    return eeg_feats

def linear_resample_time(X, T_target):
    """
    X: (T_src, D) or (D, T_src) -> returns same orientation with T_target along time
    This function auto-detects whether time is axis 0 or axis 1.
    """
    if X.ndim != 2:
        raise ValueError("linear_resample_time expects 2D array")
    time_axis = 0 if X.shape[0] < X.shape[1] else 1  # heuristic; we’ll resample along the smaller axis if ambiguous
    if time_axis == 0:
        T_src, D = X.shape
        if T_src == T_target:
            return X
        t_src = np.linspace(0.0, 1.0, T_src, dtype=np.float32)
        t_dst = np.linspace(0.0, 1.0, T_target, dtype=np.float32)
        Y = np.empty((T_target, D), dtype=np.float32)
        for d in range(D):
            Y[:, d] = np.interp(t_dst, t_src, X[:, d])
        return Y
    else:
        D, T_src = X.shape
        if T_src == T_target:
            return X
        t_src = np.linspace(0.0, 1.0, T_src, dtype=np.float32)
        t_dst = np.linspace(0.0, 1.0, T_target, dtype=np.float32)
        Y = np.empty((D, T_target), dtype=np.float32)
        for d in range(D):
            Y[d, :] = np.interp(t_dst, t_src, X[d, :])
        return Y

def sr_to_fmax(sr):
    return min(FMAX, 0.45 * sr)

def audio_features_aligned(wav_stereo):
    """
    Compute stereo log-Mel, ILD, IPD using hop=172, win=344.
    Then strictly resample to EEG_LEN=4864 frames.

    Returns:
      mel_lr: (4864, 2*NMELS)
      ild:    (4864, NMELS)
      ipd:    (4864, NMELS)
    """
    L = wav_stereo[:, 0].astype(np.float32)
    R = wav_stereo[:, 1].astype(np.float32)

    SL = librosa.stft(L, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window='hann', center=True)
    SR = librosa.stft(R, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window='hann', center=True)

    PL = np.abs(SL) ** 2  # (F, T_a)
    PR = np.abs(SR) ** 2

    mel_fb = librosa.filters.mel(sr=SR_AUDIO, n_fft=N_FFT, n_mels=NMELS, fmin=FMIN, fmax=sr_to_fmax(SR_AUDIO))
    ML = mel_fb @ PL        # (NMELS, T_a)
    MR = mel_fb @ PR        # (NMELS, T_a)

    logML = np.log(ML + 1e-8)  # (NMELS, T_a)
    logMR = np.log(MR + 1e-8)

    # ILD in mel domain
    ild = (logML - logMR)  # (NMELS, T_a)

    # IPD: mel-weighted phase difference
    phaseL = np.angle(SL)  # (F, T_a)
    phaseR = np.angle(SR)
    ipd_lin = np.unwrap(phaseL - phaseR, axis=0)
    # W = (PL + PR) + 1e-8
    # ipd_mel = (mel_fb @ (ipd_lin * W)) / (mel_fb @ W)  # (NMELS, T_a)
    # ipd_mel = np.clip(ipd_mel, -np.pi, np.pi)
    W = (PL + PR) + 1e-8
    den = (mel_fb @ W) + 1e-10           # <-- add epsilon here
    num = (mel_fb @ (ipd_lin * W))
    ipd_mel = num / den

    # Time-axis resample to exactly 4864 frames
    logML_T = linear_resample_time(logML.T, EEG_LEN)  # (4864, NMELS)
    logMR_T = linear_resample_time(logMR.T, EEG_LEN)  # (4864, NMELS)
    ild_T   = linear_resample_time(ild.T,   EEG_LEN)  # (4864, NMELS)
    ipd_T   = linear_resample_time(ipd_mel.T, EEG_LEN)

    mel_lr = np.concatenate([logML_T, logMR_T], axis=1).astype(np.float32)  # (4864, 2*NMELS)
    return mel_lr, ild_T.astype(np.float32), ipd_T.astype(np.float32)

def pan_histogram(panning_arr):
    hist = np.zeros(3, dtype=np.float32)
    for p in panning_arr:
        if p < 0.33: hist[0] += 1
        elif p > 0.66: hist[2] += 1
        else: hist[1] += 1
    return hist

def apply_notch_filter(eeg_data, notch_freq, fs=256.0, quality=30.0):
    b, a = iirnotch(notch_freq, quality, fs)
    return filtfilt(b, a, eeg_data, axis=1)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg_dir",   type=str, default=DEF_EEG_DIR)
    ap.add_argument("--audio_dir", type=str, default=DEF_AUDIO_DIR)
    ap.add_argument("--yaml_path", type=str, default=DEF_YAML_PATH)
    ap.add_argument("--out_dir",   type=str, default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load master YAML (subject -> trialkey -> dict)
    if not os.path.exists(args.yaml_path):
        raise FileNotFoundError(f"YAML not found: {args.yaml_path}")
    with open(args.yaml_path, "r") as f:
        meta_all = yaml.safe_load(f)

    eeg_paths = sorted(glob.glob(os.path.join(args.eeg_dir, "*.npy")))
    if not eeg_paths:
        raise RuntimeError(f"No EEG .npy found under {args.eeg_dir}")

    # Two-pass per-subject stats for z-score
    subj_stats = collect_subject_stats(eeg_paths)

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["trial_id","subject","npz_path","label_idx","present_mask",
                        "has_yaml","T_frames"])

        kept = skipped = 0

        for eeg_path in eeg_paths:
            base = os.path.basename(eeg_path)
            try:
                sid, trialkey = parse_subject_and_trialkey(eeg_path)
                if sid not in meta_all or trialkey not in meta_all[sid]:
                    raise KeyError(f"YAML missing entry for {sid}/{trialkey}")
                meta = meta_all[sid][trialkey]

                target = meta.get("target", None)
                if target is None:
                    raise KeyError(f"No 'target' in YAML for {sid}/{trialkey}")

                if POP_ONLY and target not in LABEL_TO_IDX:
                    skipped += 1
                    continue
                label_idx = LABEL_TO_IDX.get(target, -1)
                if label_idx < 0:
                    skipped += 1
                    continue

                instr_list = meta.get("instruments", [])
                present_mask = np.zeros(4, dtype=np.float32)
                for code in instr_list:
                    if code in LABEL_TO_IDX:
                        present_mask[LABEL_TO_IDX[code]] = 1.0

                wi = meta.get("wav_info", {})
                gains = np.array([float(x) for x in wi.get("gains", [])], dtype=np.float32)
                panning = np.array([float(x) for x in wi.get("panning", [])], dtype=np.float32)
                pan_hist = pan_histogram(panning)
                gain_sorted = np.sort(gains) if gains.size > 0 else np.array([], dtype=np.float32)
                has_yaml = 1

                # WAV
                wav_path, wav_core = wav_path_from_eeg(eeg_path, args.audio_dir)
                if not os.path.exists(wav_path):
                    raise FileNotFoundError(f"WAV not found: {wav_path}")
                wav_stereo, sr = sf.read(wav_path, always_2d=True)
                if sr != SR_AUDIO:
                    raise ValueError(f"Expected {SR_AUDIO} Hz, got {sr} in {wav_path}")
                if wav_stereo.shape[1] != 2:
                    raise ValueError(f"Expected stereo WAV with 2 channels: {wav_path}")
                # Enforce 19 s length
                if wav_stereo.shape[0] != 837900:
                    N = min(837900, wav_stereo.shape[0])
                    padN = 837900 - N
                    wav_stereo = wav_stereo[:N, :]
                    if padN > 0:
                        wav_stereo = np.pad(wav_stereo, ((0,padN),(0,0)), mode='constant')

                # Audio features (aligned to EEG_LEN)
                mel_lr, ild, ipd = audio_features_aligned(wav_stereo.astype(np.float32))

                # EEG preprocess (no resample; already 4864 samples)
                eeg = np.load(eeg_path).astype(np.float32)  # (20,4864)
                mean_std = subj_stats.get(sid, None)
                eeg_feats = eeg_preprocess(eeg, mean_std)   # (20*lags, 4864)
                # Convert to (4864, Ceeg)
                eeg_T = eeg_feats.T.astype(np.float32)

                # Safety: if any tiny mismatch, force exact 4864
                if eeg_T.shape[0] != EEG_LEN:
                    eeg_T = linear_resample_time(eeg_T, EEG_LEN)

                # Save
                trial_id = wav_core  # already includes the target (…_stereo_<Inst>)
                out_dir = os.path.join(args.out_dir, sid)
                os.makedirs(out_dir, exist_ok=True)
                out_npz = os.path.join(out_dir, trial_id + ".npz")
                np.savez_compressed(
                    out_npz,
                    audio_mel_lr=mel_lr,
                    audio_ild=ild,
                    audio_ipd=ipd,
                    eeg_proc=eeg_T,
                    gains=gains,
                    panning=panning,
                    pan_hist=pan_hist,
                    gain_sorted=gain_sorted,
                    present_mask=present_mask,
                    label_idx=np.int64(label_idx),
                    sr_audio=np.int64(SR_AUDIO),
                    hop_len=np.int64(HOP),
                    win_len=np.int64(WIN),
                    nmels=np.int64(NMELS),
                    trial_id=trial_id,
                    # genre=str(meta.get("genre","")),
                    # ensemble=str(meta.get("ensemble","")),
                    # spatial=str(meta.get("spatial",""))
                )

                writer.writerow([
                    trial_id, sid, out_npz, label_idx,
                    ";".join(str(int(x)) for x in present_mask.tolist()),
                    has_yaml, EEG_LEN
                ])
                kept += 1

            except Exception as e:
                print(f"[SKIP] {base} :: {e}")
                skipped += 1
                continue

    print(f"Done. Wrote manifest to {manifest_path}")
    print(f"Kept: {kept} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
