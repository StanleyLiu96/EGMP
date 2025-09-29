#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
encoder.py

What this file provides
-----------------------
1) Two lightweight 1D-conv encoders (no attention inside):
   - EEGTimeAlignedEncoder   : EEG  -> [B, X, C_eeg]   (channels preserved)
   - AudioTimeAlignedEncoder : Audio-> [B, X, C_audio] (channels preserved, C_audio auto-detected: 1 or 2)

2) Temporal alignment with dynamic X
   X is NOT hardcoded. It is computed from duration and a fixed time window:
       X = round( (length_in_samples / sample_rate) / window_sec )
   with window_sec = 0.25 s (250 ms). ~19 s clips will naturally yield X ≈ 76.

3) Tuple output format for each sample:
       ( feature_tensor [X, C], insts_list, true_label )
   - insts_list: parsed from token right after 'duo'/'trio' by splitting every 2 chars
                 (len=4 -> 2 instruments, len=6 -> 3 instruments, etc.)
   - true_label: the token immediately following 'stereo' in the filename.

Assumptions
-----------
- This file is placed in the same folder as your load_dataset.py.
- Your loader returns lists of (tensor, filename) as shown earlier.
"""

import re
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

# Reuse your dataloader
from load_dataset import load_all_datasets


# =============================================================================
# Filename / label parsing utilities
# =============================================================================

# Valid 2-char instrument codes.
VALID_INSTS = {"Gt", "Vx", "Dr", "Bs"}
INST_PATTERN = re.compile(r"(Gt|Vx|Dr|Bs)")

def parse_insts_from_combo(combo: str) -> List[str]:
    """
    Parse a compact instrument combo like 'BsDr' or 'GtDrVx'.
    Rule: split every 2 chars; keep only valid 2-char codes.
      'BsDr'   -> ['Bs','Dr']      (len=4 -> 2 instruments)
      'GtDrVx' -> ['Gt','Dr','Vx'] (len=6 -> 3 instruments)
    """
    parts = [combo[i:i+2] for i in range(0, len(combo), 2)]
    return [p for p in parts if p in VALID_INSTS]

def parse_insts_and_true_label(fname: str) -> Tuple[List[str], str]:
    """
    From a filename like:
      '0001_pop_falldead_duo_BsDr_theme1_stereo_Dr_stimulus.wav'
    Return:
      insts      = ['Bs','Dr']  (taken from compact token right after 'duo'/'trio')
      true_label = 'Dr'         (the token right after 'stereo')

    Robust to different suffixes: '_stimulus.wav', '_soli.wav', '_response.npy', etc.
    """
    tokens = fname.split("_")
    insts: List[str] = []
    true_label: str = ""

    # 1) instruments: token after 'duo' or 'trio'
    for i, t in enumerate(tokens):
        if t in ("duo", "trio") and i + 1 < len(tokens):
            insts = parse_insts_from_combo(tokens[i + 1])
            break
    # Fallback: scan whole name for first occurrences (rarely needed)
    if not insts:
        pairs = INST_PATTERN.findall(fname)
        seen = set()
        insts = [p for p in pairs if not (p in seen or seen.add(p))]

    # 2) true label: token after 'stereo'
    for i, t in enumerate(tokens):
        if t == "stereo" and i + 1 < len(tokens):
            true_label = tokens[i + 1]
            break

    return insts, true_label

def strip_base(fname: str, suffix: str) -> Optional[str]:
    """Remove a known suffix to get the shared basename (for indexing/alignment)."""
    return fname[:-len(suffix)] if fname.endswith(suffix) else None

def to_map(pairs: List[Tuple[torch.Tensor, str]], suffix: str) -> Dict[str, Tuple[torch.Tensor, str]]:
    """
    Convert [(tensor, filename)] into a dict keyed by basename:
      { basename : (tensor, original_filename) }
    """
    m: Dict[str, Tuple[torch.Tensor, str]] = {}
    for tensor, fname in pairs:
        base = strip_base(fname, suffix)
        if base:
            m[base] = (tensor, fname)
    return m


# =============================================================================
# Batch → list of tuples (your requested order)
# =============================================================================

def make_tuple_outputs(filenames: List[str], batch_feat: torch.Tensor):
    """
    Convert batch features [B, X, C] into your desired tuples per item:
      (feature_tensor [X,C], insts_list, true_label)
    """
    outs = []
    for i, fname in enumerate(filenames):
        insts, label = parse_insts_and_true_label(fname)
        outs.append((batch_feat[i], insts, label))
    return outs


# =============================================================================
# Basic 1D Conv block (preserves channels by default)
# =============================================================================

class ConvBlock1D(nn.Module):
    """
    A minimal Conv1d -> BN -> ReLU block.
    - out_ch defaults to in_ch to preserve channel count.
    - Use stride 's' to progressively reduce temporal length.
    """
    def __init__(self, in_ch: int, out_ch: Optional[int] = None, k: int = 15, s: int = 2, p: Optional[int] = None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        if p is None:
            p = k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, L] -> [B, C', L']
        return self.act(self.bn(self.conv(x)))


# =============================================================================
# EEG encoder (dynamic X, channel-preserving)
# =============================================================================

class EEGTimeAlignedEncoder(nn.Module):
    """
    EEG encoder:
      - Input  : [B, C_eeg, L]
      - Output : [B, X, C_eeg]
      - X is computed from duration (L / eeg_sr) and window_sec (default 0.25 s).
      - Channel count is preserved end-to-end.
    """
    def __init__(self, in_ch: int = 1, eeg_sr: float = 256.0, window_sec: float = 0.25):
        super().__init__()
        self.eeg_sr = float(eeg_sr)
        self.window_sec = float(window_sec)

        # Keep channels unchanged; stride shrinks time.
        self.net = nn.Sequential(
            ConvBlock1D(in_ch, in_ch, k=31, s=4),  # ~L/4
            ConvBlock1D(in_ch, in_ch, k=15, s=2),  # ~L/8
            ConvBlock1D(in_ch, in_ch, k=15, s=2),  # ~L/16
        )

    def _compute_x(self, L: int) -> int:
        """X = round( (L / eeg_sr) / window_sec )."""
        duration = L / self.eeg_sr
        return max(int(round(duration / self.window_sec)), 1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Args:
          eeg: [B, C_eeg, L_eeg]
        Returns:
          [B, X, C_eeg]
        """
        B, C, L = eeg.shape
        X = self._compute_x(L)
        x = self.net(eeg)                   # [B, C, L’]
        x = nn.AdaptiveAvgPool1d(X)(x)      # [B, C, X]
        return x.permute(0, 2, 1).contiguous()  # [B, X, C]


# =============================================================================
# Audio encoder (dynamic X, auto channel detect, channel-preserving)
# =============================================================================

class AudioTimeAlignedEncoder(nn.Module):
    """
    Audio encoder:
      - Accepts stereo (2ch) or mono (1ch) inputs and auto-detects channel count on first forward.
      - Preserves channel count end-to-end.
      - Input  : [B, T, C_audio] with C_audio∈{1,2} or [B, T] for mono.
      - Output : [B, X, C_audio]
      - X is computed from duration (T / audio_sr) and window_sec (default 0.25 s).

    Important:
      This instance "locks" to the channel count observed on the first forward.
      If you need to process mono and stereo in the same script, create two instances.
    """
    def __init__(self, in_ch: Optional[int] = None, audio_sr: float = 44100.0, window_sec: float = 0.25):
        super().__init__()
        self.audio_sr   = float(audio_sr)
        self.window_sec = float(window_sec)
        self.in_ch      = in_ch    # None -> determine at first forward
        self.net: Optional[nn.Sequential] = None

    @staticmethod
    def _to_BCT(audio: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Normalize input to [B, C, T] and return (x_BCT, detected_channels).
          [B, T, 2] -> [B, 2, T]
          [B, T]    -> [B, 1, T]
          [B, T, 1] -> [B, 1, T]
        """
        if audio.dim() == 3 and audio.size(-1) == 2:
            B, T, _ = audio.shape
            return audio.permute(0, 2, 1).contiguous(), 2
        if audio.dim() == 2:
            B, T = audio.shape
            return audio.unsqueeze(1), 1
        if audio.dim() == 3 and audio.size(-1) == 1:
            B, T, _ = audio.shape
            return audio.permute(0, 2, 1).contiguous(), 1
        raise ValueError(f"Unsupported audio shape {tuple(audio.shape)}; expected [B,T,2] or [B,T] or [B,T,1].")

    def _compute_x(self, T: int) -> int:
        """X = round( (T / audio_sr) / window_sec )."""
        duration = T / self.audio_sr
        return max(int(round(duration / self.window_sec)), 1)

    def _build_if_needed(self, detected_ch: int, device: torch.device):
        """
        Lazily build a small Conv1d stack once we know the channel count.
        The stack preserves channels (in_ch -> in_ch).
        """
        if self.net is not None:
            if detected_ch != self.in_ch:
                raise ValueError(
                    f"AudioTimeAlignedEncoder was built for {self.in_ch} channels, "
                    f"but got a batch with {detected_ch} channels. "
                    f"Create separate instances per channel mode."
                )
            return

        if self.in_ch is not None and self.in_ch != detected_ch:
            raise ValueError(f"in_ch={self.in_ch} mismatches detected channels {detected_ch}.")
        self.in_ch = detected_ch

        # Channel-preserving stack; first layer uses large stride to shrink long raw wave quickly.
        self.net = nn.Sequential(
            ConvBlock1D(self.in_ch, self.in_ch, k=401, s=160),  # fast reduction
            ConvBlock1D(self.in_ch, self.in_ch, k=15,  s=4),
            ConvBlock1D(self.in_ch, self.in_ch, k=15,  s=4),
        ).to(device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
          audio: mixed -> [B, T, 2]; solo -> [B, T] or [B, T, 1]
        Returns:
          [B, X, C_audio]  (C_audio preserved and auto-detected)
        """
        x, C_in = self._to_BCT(audio)          # [B, C_in, T]
        device = x.device
        B, _, T = x.shape

        self._build_if_needed(C_in, device)    # lazy-build on first call
        X = self._compute_x(T)                 # dynamic X

        x = self.net(x)                        # [B, C_in, L’]
        x = nn.AdaptiveAvgPool1d(X)(x)         # [B, C_in, X]
        return x.permute(0, 2, 1).contiguous() # [B, X, C_in]


# =============================================================================
# Demo: call each modality separately and emit tuples (feature, insts, label)
# =============================================================================

if __name__ == "__main__":
    # Load all datasets (lists of (tensor, filename))
    EEG_tr, EEG_val, MIX_tr, MIX_val, SOLO_tr, SOLO_val = load_all_datasets()

    # Build quick maps (basename → (tensor, filename)) for convenience
    eeg_map  = to_map(EEG_tr,  "_response.npy")
    mix_map  = to_map(MIX_tr,  "_stimulus.wav")
    solo_map = to_map(SOLO_tr, "_soli.wav")

    # -------- EEG: pick a few different files if available --------
    if eeg_map:
        print("=== EEG examples ===")
        eeg_keys = list(eeg_map.keys())[:3]  # try up to 3 different items
        eeg_enc = None
        for k in eeg_keys:
            eeg_tensor, eeg_fname = eeg_map[k]      # [C_eeg, L_eeg]
            eeg_batch = eeg_tensor.unsqueeze(0)     # [1, C_eeg, L_eeg]
            if eeg_enc is None:
                eeg_enc = EEGTimeAlignedEncoder(in_ch=eeg_batch.shape[1], eeg_sr=256.0, window_sec=0.25)
            eeg_feat = eeg_enc(eeg_batch)           # [1, X, C_eeg]
            tuples = make_tuple_outputs([eeg_fname], eeg_feat)
            feat, insts, label = tuples[0]
            print("file:", eeg_fname)
            print("  shape:", tuple(feat.shape), "insts:", insts, "true_label:", label)

    # -------- Mixed Audio: try a few different files --------
    if mix_map:
        print("\n=== Mixed audio examples ===")
        mix_keys = list(mix_map.keys())[:3]
        aud_m_enc = AudioTimeAlignedEncoder(in_ch=None, audio_sr=44100.0, window_sec=0.25)  # auto-detect (2ch)
        for k in mix_keys:
            mix_tensor, mix_fname = mix_map[k]      # [T, 2]
            mix_batch = mix_tensor.unsqueeze(0)     # [1, T, 2]
            mix_feat = aud_m_enc(mix_batch)         # [1, X, 2]
            tuples = make_tuple_outputs([mix_fname], mix_feat)
            feat, insts, label = tuples[0]
            print("file:", mix_fname)
            print("  shape:", tuple(feat.shape), "insts:", insts, "true_label:", label)

    # -------- Solo Audio: try a few different files --------
    if solo_map:
        print("\n=== Solo audio examples ===")
        solo_keys = list(solo_map.keys())[:3]
        aud_s_enc = AudioTimeAlignedEncoder(in_ch=None, audio_sr=44100.0, window_sec=0.25)  # auto-detect (1ch)
        for k in solo_keys:
            solo_tensor, solo_fname = solo_map[k]   # [T] mono
            solo_batch = solo_tensor.unsqueeze(0)   # [1, T]
            solo_feat = aud_s_enc(solo_batch)       # [1, X, 1]
            tuples = make_tuple_outputs([solo_fname], solo_feat)
            feat, insts, label = tuples[0]
            print("file:", solo_fname)
            print("  shape:", tuple(feat.shape), "insts:", insts, "true_label:", label)
