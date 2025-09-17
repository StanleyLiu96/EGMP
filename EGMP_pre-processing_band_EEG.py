#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_pre-processing.py  (cleaned, with EEG bandpower features)

What this script does:
----------------------
1. Reads paired EEG (.npy) and audio (.wav) files.
2. Extracts audio features:
      - Log-Mel spectrogram (L+R channels concatenated)
      - Interaural Level Difference (ILD)
      - Interaural Phase Difference (IPD)
   These are aligned to 256 Hz (4864 frames = 19 seconds).
3. Extracts EEG features:
      - Computes bandpower in 5 classical EEG bands:
          delta (0.5-4 Hz), theta (4-8 Hz),
          alpha (8-12 Hz), beta (13-30 Hz),
          gamma (30-70 Hz).
      - Returns tensor shape (73, 5, 20)
        = 73 time frames * 5 bands * 20 channels.
4. Stores all features and labels in compressed .npz files:
      audio_mel_lr, audio_ild, audio_ipd,
      eeg_band, present_mask, label_idx, trial_id
5. Writes a manifest.csv file with trial metadata.

This is a minimal version — unnecessary metadata and the old `eeg_proc` 
features are dropped.
"""

import os, re, csv, glob, argparse
import numpy as np
import soundfile as sf
import librosa, yaml
from scipy.signal import butter, filtfilt, iirnotch, welch

# -----------------------------
# Config
# -----------------------------
SR_AUDIO = 44100       # Audio sampling rate (Hz)
EEG_SR = 256           # EEG sampling rate (Hz)
EEG_LEN = 4864         # # of EEG samples per trial (19 s * 256 Hz)
NMELS = 96             # # of Mel bands for audio
HOP = 172              # Audio hop length (≈ SR_AUDIO/EEG_SR)
WIN = 344              # Audio window length = 2*HOP
N_FFT = 1024           # FFT size for audio STFT
FMIN, FMAX = 20.0, 16000.0  # Mel frequency range

INSTR_CODES = ["Gt", "Vx", "Dr", "Bs"]
LABEL_TO_IDX = {"Gt": 0, "Vx": 1, "Dr": 2, "Bs": 3}
POP_ONLY = True         # Skip non-Gt/Vx/Dr/Bs trials
EPS = 1e-8              # Small epsilon to avoid log(0)

# EEG frequency bands (in Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (13, 30),
    "gamma": (30, 70)
}

# -----------------------------
# Helper functions
# -----------------------------
def parse_subject_and_trialkey(eeg_path):
    """
    Extract subject ID and trial key from EEG filename.
    Example:
      EEG filename: 0001_pop_trio_GtVxDr_theme2_stereo_Dr_response.npy
      subject = "0001"
      trialkey = "pop_trio_GtVxDr_theme2_stereo_Dr"
    """
    base = os.path.basename(eeg_path)
    m = re.match(r"^(\d{4})_(.+)_response\.npy$", base)
    if not m:
        raise ValueError(f"Unexpected EEG filename: {base}")
    return m.group(1), m.group(2)

def wav_path_from_eeg(eeg_path, audio_dir):
    """
    Map EEG filename to corresponding WAV filename.
    Just replace "_response.npy" with "_stimulus.wav".
    """
    eeg_base = os.path.basename(eeg_path)
    core_with_target = eeg_base.replace("_response.npy", "")
    wav_name = core_with_target + "_stimulus.wav"
    return os.path.join(audio_dir, wav_name), core_with_target

def apply_notch_filter(x, fs=256.0, f0=50.0, Q=30.0):
    """
    Apply a notch filter at frequency f0 to remove line noise.
    Default: 50 Hz (Europe/Asia) or adjust to 60 Hz if needed.
    """
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, x, axis=1)

def audio_features_aligned(wav_stereo):
    """
    Compute audio features aligned to EEG frame rate.
    Returns:
      mel_lr: (4864, 2*NMELS) log-mel spectrogram (L+R)
      ild:    (4864, NMELS)   interaural level difference
      ipd:    (4864, NMELS)   interaural phase difference
    """
    # Separate L/R channels
    L = wav_stereo[:, 0].astype(np.float32)
    R = wav_stereo[:, 1].astype(np.float32)

    # STFT
    SL = librosa.stft(L, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    SR = librosa.stft(R, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    PL, PR = np.abs(SL)**2, np.abs(SR)**2  # Power spectra

    # Mel filterbank
    mel_fb = librosa.filters.mel(sr=SR_AUDIO, n_fft=N_FFT, n_mels=NMELS,
                                 fmin=FMIN, fmax=min(FMAX, 0.45*SR_AUDIO))

    # Mel energies (before log)
    ML, MR = mel_fb @ PL, mel_fb @ PR
    logML, logMR = np.log(ML+1e-8), np.log(MR+1e-8)

    # ILD = difference between log mel energies
    ild = logML - logMR

    # IPD = phase difference (mel-weighted)
    phaseL, phaseR = np.angle(SL), np.angle(SR)
    ipd_lin = np.unwrap(phaseL - phaseR, axis=0)
    W = (PL + PR) + 1e-8
    num, den = mel_fb @ (ipd_lin*W), (mel_fb @ W) + 1e-10
    ipd_mel = num/den

    # Resample along time axis to exactly EEG_LEN=4864
    logML_T = librosa.util.fix_length(logML.T, size=EEG_LEN)
    logMR_T = librosa.util.fix_length(logMR.T, size=EEG_LEN)
    ild_T   = librosa.util.fix_length(ild.T,   size=EEG_LEN)
    ipd_T   = librosa.util.fix_length(ipd_mel.T, size=EEG_LEN)

    mel_lr  = np.concatenate([logML_T, logMR_T], axis=1).astype(np.float32)
    return mel_lr, ild_T.astype(np.float32), ipd_T.astype(np.float32)

def eeg_bandpower(eeg_raw):
    """
    Compute EEG bandpower features using Welch PSD.
    Input:
      eeg_raw: (20, 4864)  EEG channels * time samples
    Output:
      feats:   (73, 5, 20) TimeFrames * Bands * Channels
    """
    chans, T = eeg_raw.shape

    # Common average reference
    eeg = eeg_raw - eeg_raw.mean(axis=0, keepdims=True)

    # Bandpass 1–70 Hz
    b, a = butter_bandpass(1.0, 70.0, EEG_SR, order=4)
    eeg = filtfilt(b, a, eeg, axis=1, method="gust")

    # Notch at 50 Hz (if mains leakage is strong)
    eeg = apply_notch_filter(eeg, fs=EEG_SR, f0=50.0) # Europe mains

    # Frame configuration (STFT-like)
    win, hop = 256, 64       # 1.0 s window, 0.25 s step
    frames = 1 + (T - win)//hop
    feats = np.zeros((frames, len(BANDS), chans), dtype=np.float32)

    # Welch PSD per channel
    for c in range(chans):
        f, Pxx = welch(eeg[c], fs=EEG_SR, window='hann',
                       nperseg=win, noverlap=win-hop, nfft=win)
        for bi, (lo, hi) in enumerate(BANDS.values()):
            mask = (f >= lo) & (f <= hi)
            # Integrate power in band, take log
            feats[:, bi, c] = np.log(np.trapz(Pxx[mask]) + EPS)
    return feats

def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create bandpass filter coefficients using a Butterworth filter.
    
    Args:
        lowcut: low cutoff frequency (Hz)
        highcut: high cutoff frequency (Hz)
        fs: sampling rate (Hz)
        order: filter order (higher = steeper roll-off)

    Returns:
        b, a: numerator/denominator filter coefficients
    """
    nyq = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg_dir",   type=str, default="../../Dataset/MADEEG/processed_data/response_npy")
    ap.add_argument("--audio_dir", type=str, default="../../Dataset/MADEEG/processed_data/stimulus_wav")
    ap.add_argument("--yaml_path", type=str, default="../../Dataset/MADEEG/madeeg_preprocessed.yaml")
    ap.add_argument("--out_dir",   type=str, default="./EGMP_preprocessed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load master YAML (trial metadata)
    with open(args.yaml_path, "r") as f:
        meta_all = yaml.safe_load(f)

    # Collect EEG paths
    eeg_paths = sorted(glob.glob(os.path.join(args.eeg_dir, "*.npy")))

    # Open manifest for writing
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["trial_id","subject","npz_path","label_idx","present_mask"])

        kept = skipped = 0
        for eeg_path in eeg_paths:
            base = os.path.basename(eeg_path)
            try:
                # Parse subject/trial
                sid, trialkey = parse_subject_and_trialkey(eeg_path)
                meta = meta_all[sid][trialkey]

                # Target instrument label
                target = meta["target"]
                if POP_ONLY and target not in LABEL_TO_IDX:
                    skipped += 1; continue
                label_idx = LABEL_TO_IDX[target]

                # Present mask (which instruments appear)
                instr_list = meta.get("instruments", [])
                present_mask = np.zeros(4, dtype=np.float32)
                for code in instr_list:
                    if code in LABEL_TO_IDX:
                        present_mask[LABEL_TO_IDX[code]] = 1.0

                # Load WAV
                wav_path, wav_core = wav_path_from_eeg(eeg_path, args.audio_dir)
                wav_stereo, sr = sf.read(wav_path, always_2d=True)
                if sr != SR_AUDIO: raise ValueError(f"Expected {SR_AUDIO}, got {sr}")
                if wav_stereo.shape[1] != 2: raise ValueError("Expected stereo WAV")
                if wav_stereo.shape[0] != 837900:
                    # Enforce 19 s length
                    wav_stereo = np.pad(wav_stereo[:837900,:],
                                        ((0,max(0,837900-wav_stereo.shape[0])),(0,0)))

                # Audio features
                mel_lr, ild, ipd = audio_features_aligned(wav_stereo.astype(np.float32))

                # EEG features (bandpower)
                eeg = np.load(eeg_path).astype(np.float32)  # (20,4864)
                eeg_band = eeg_bandpower(eeg)

                # Trial ID for saving
                trial_id = wav_core
                out_dir = os.path.join(args.out_dir, sid)
                os.makedirs(out_dir, exist_ok=True)
                out_npz = os.path.join(out_dir, trial_id + ".npz")

                # Save compressed npz
                np.savez_compressed(
                    out_npz,
                    audio_mel_lr=mel_lr,
                    audio_ild=ild,
                    audio_ipd=ipd,
                    eeg_band=eeg_band,
                    present_mask=present_mask,
                    label_idx=np.int64(label_idx),
                    trial_id=trial_id,
                )

                # Record in manifest
                writer.writerow([trial_id, sid, out_npz, label_idx,
                                 ";".join(str(int(x)) for x in present_mask.tolist())])
                kept += 1

            except Exception as e:
                print(f"[SKIP] {base} :: {e}")
                skipped += 1
                continue

    print(f"Done. Wrote manifest to {manifest_path}")
    print(f"Kept: {kept} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
