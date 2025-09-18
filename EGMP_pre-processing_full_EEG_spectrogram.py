#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EGMP_pre-processing.py  (Option B — Full EEG Spectrogram)

What this script does:
----------------------
1. Reads paired EEG (.npy) and audio (.wav) files.
2. Extracts audio features:
      - Log-Mel spectrogram (L+R channels concatenated)
      - Interaural Level Difference (ILD)
      - Interaural Phase Difference (IPD)
   These are aligned to 256 Hz (4864 frames = 19 seconds).
3. Extracts EEG features:
      - Computes full spectrogram (STFT, 1 s window, 0.25 s hop).
      - Keeps ALL 129 frequency bins (0–128 Hz).
      - Output tensor shape: (73, 129, 20)
        = 73 time frames × 129 frequency bins × 20 channels.
4. Stores all features and labels in compressed .npz files:
      audio_mel_lr, audio_ild, audio_ipd,
      eeg_spec, present_mask, label_idx, trial_id
5. Writes a manifest.csv file with trial metadata.

This replaces the earlier "bandpower" EEG features.
"""

import os, re, csv, glob, argparse
import numpy as np
import soundfile as sf
import librosa, yaml
from scipy.signal import butter, filtfilt, iirnotch, stft

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

# -----------------------------
# Helper functions
# -----------------------------
def parse_subject_and_trialkey(eeg_path):
    """Extract subject ID and trial key from EEG filename."""
    base = os.path.basename(eeg_path)
    m = re.match(r"^(\d{4})_(.+)_response\.npy$", base)
    if not m:
        raise ValueError(f"Unexpected EEG filename: {base}")
    return m.group(1), m.group(2)

def wav_path_from_eeg(eeg_path, audio_dir):
    """Map EEG filename to corresponding WAV filename."""
    eeg_base = os.path.basename(eeg_path)
    core_with_target = eeg_base.replace("_response.npy", "")
    wav_name = core_with_target + "_stimulus.wav"
    return os.path.join(audio_dir, wav_name), core_with_target

def apply_notch_filter(x, fs=256.0, f0=50.0, Q=30.0):
    """Apply a notch filter at f0 (default: 50 Hz for Europe mains)."""
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, x, axis=1)

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create bandpass filter coefficients (Butterworth)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    from scipy.signal import butter
    return butter(order, [low, high], btype="band")

def audio_features_aligned(wav_stereo):
    """
    Compute audio features aligned to EEG frame rate.
    Returns:
      mel_lr: (4864, 2*NMELS)
      ild:    (4864, NMELS)
      ipd:    (4864, NMELS)
    """
    L = wav_stereo[:, 0].astype(np.float32)
    R = wav_stereo[:, 1].astype(np.float32)

    SL = librosa.stft(L, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    SR = librosa.stft(R, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    PL, PR = np.abs(SL)**2, np.abs(SR)**2

    mel_fb = librosa.filters.mel(sr=SR_AUDIO, n_fft=N_FFT, n_mels=NMELS,
                                 fmin=FMIN, fmax=min(FMAX, 0.45*SR_AUDIO))

    ML, MR = mel_fb @ PL, mel_fb @ PR
    logML, logMR = np.log(ML+EPS), np.log(MR+EPS)
    ild = logML - logMR

    phaseL, phaseR = np.angle(SL), np.angle(SR)
    ipd_lin = np.unwrap(phaseL - phaseR, axis=0)
    W = (PL + PR) + EPS
    ipd_mel = (mel_fb @ (ipd_lin*W)) / ((mel_fb @ W) + EPS)

    logML_T = librosa.util.fix_length(logML.T, size=EEG_LEN)
    logMR_T = librosa.util.fix_length(logMR.T, size=EEG_LEN)
    ild_T   = librosa.util.fix_length(ild.T,   size=EEG_LEN)
    ipd_T   = librosa.util.fix_length(ipd_mel.T, size=EEG_LEN)

    mel_lr  = np.concatenate([logML_T, logMR_T], axis=1).astype(np.float32)
    return mel_lr, ild_T.astype(np.float32), ipd_T.astype(np.float32)

def eeg_spectrogram(eeg_raw):
    """
    Compute full EEG spectrogram using STFT.

    Input:
      eeg_raw: (20, 4864)  → 20 channels × 19s EEG at 256 Hz
    Output:
      feats:   (73, 129, 20)
        = 73 time frames × 129 frequency bins × 20 channels

    Notes:
      - We use 1 s window (256 samples), 0.25 s hop (64 samples).
      - With 4864 samples: frames = 1 + (4864 - 256)/64 = 73 exactly.
      - boundary=None, padded=False → prevents zero-padding, so
        output is always fixed length (73,129) instead of drifting
        to 77 frames.
    """
    chans, T = eeg_raw.shape

    # Common average reference
    eeg = eeg_raw - eeg_raw.mean(axis=0, keepdims=True)

    # Bandpass 1–70 Hz
    b, a = butter_bandpass(1.0, 70.0, EEG_SR, order=4)
    eeg = filtfilt(b, a, eeg, axis=1, method="gust")

    # Notch at 50 Hz
    eeg = apply_notch_filter(eeg, fs=EEG_SR, f0=50.0)

    # STFT parameters
    win, hop, nfft = 256, 64, 256
    frames = 1 + (T - win)//hop

    # Preallocate
    feats = np.zeros((frames, nfft//2 + 1, chans), dtype=np.float32)

    # Compute STFT power for each channel
    for c in range(chans):
        f, t, Zxx = stft(
            eeg[c],
            fs=EEG_SR,
            nperseg=win,
            noverlap=win-hop,
            nfft=nfft,
            boundary=None,   # disables reflection padding
            padded=False     # prevents zero-padding at the end
            )
        Pxx = np.abs(Zxx)**2
        feats[:, :, c] = np.log(Pxx.T + EPS)  # (73,129)

    return feats

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

    # Load trial metadata
    with open(args.yaml_path, "r") as f:
        meta_all = yaml.safe_load(f)

    eeg_paths = sorted(glob.glob(os.path.join(args.eeg_dir, "*.npy")))

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["trial_id","subject","npz_path","label_idx","present_mask"])

        kept = skipped = 0
        for eeg_path in eeg_paths:
            base = os.path.basename(eeg_path)
            try:
                sid, trialkey = parse_subject_and_trialkey(eeg_path)
                meta = meta_all[sid][trialkey]

                target = meta["target"]
                if POP_ONLY and target not in LABEL_TO_IDX:
                    skipped += 1; continue
                label_idx = LABEL_TO_IDX[target]

                instr_list = meta.get("instruments", [])
                present_mask = np.zeros(4, dtype=np.float32)
                for code in instr_list:
                    if code in LABEL_TO_IDX:
                        present_mask[LABEL_TO_IDX[code]] = 1.0

                wav_path, wav_core = wav_path_from_eeg(eeg_path, args.audio_dir)
                wav_stereo, sr = sf.read(wav_path, always_2d=True)
                if sr != SR_AUDIO: raise ValueError(f"Expected {SR_AUDIO}, got {sr}")
                if wav_stereo.shape[1] != 2: raise ValueError("Expected stereo WAV")
                if wav_stereo.shape[0] != 837900:
                    wav_stereo = np.pad(wav_stereo[:837900,:],
                                        ((0,max(0,837900-wav_stereo.shape[0])),(0,0)))

                mel_lr, ild, ipd = audio_features_aligned(wav_stereo.astype(np.float32))

                eeg = np.load(eeg_path).astype(np.float32)
                eeg_spec = eeg_spectrogram(eeg)  # (73,129,20)

                trial_id = wav_core
                out_dir = os.path.join(args.out_dir, sid)
                os.makedirs(out_dir, exist_ok=True)
                out_npz = os.path.join(out_dir, trial_id + ".npz")

                np.savez_compressed(
                    out_npz,
                    audio_mel_lr=mel_lr,
                    audio_ild=ild,
                    audio_ipd=ipd,
                    eeg_spec=eeg_spec,
                    present_mask=present_mask,
                    label_idx=np.int64(label_idx),
                    trial_id=trial_id,
                )

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
