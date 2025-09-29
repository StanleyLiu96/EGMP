#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
paic_model.py
-------------
PAIC = Presence-Assisted Instrument Classifier

Combines:
- Cross-attention classifier (EEG + audio)
- Presence prior (audio only)

Returns logits for both heads, which are trained with:
- CrossEntropyLoss for attended instrument (cross_logits)
- BCEWithLogitsLoss for instrument presence (pres_logits)
"""

import torch
import torch.nn as nn

from cross_attention import CrossAttentionClassifier  # returns softmax probs [B,4]
from presence_prior import PresencePrior          # your presence prior head


class PAIC(nn.Module):
    def __init__(
        self,
        audio_in_ch: int,        # C_audio  (e.g., 2 for stereo)
        eeg_in_ch: int,          # C_eeg    (e.g., number of EEG channels after selection)
        pres_hidden_dim: int=128,
        ca_hidden_ch: int=64,    # hidden channels inside cross-attention stack
        n_classes: int=4,
        ca_layers: int=2,
        ca_kernel: int=3,
        ca_dilation: int=1,
    ):
        super().__init__()
        self.n_classes = n_classes

        # Cross-attention classifier over encoded audio+EEG.
        # It expects [B, T, C] inputs and outputs raw logits [B, n_classes].
        self.cross_cls = CrossAttentionClassifier(
            audio_in_ch=audio_in_ch,
            eeg_in_ch=eeg_in_ch,
            hidden_ch=ca_hidden_ch,
            num_classes=n_classes,
            layer=ca_layers,
            kernel_size=ca_kernel,
            dilation=ca_dilation,
        )

        # Presence prior (audio-only). Expects [B, 76, audio_in_ch].
        # internally projects input_dim=2 → hidden_dim,
        # but it works with any last-dim = audio_in_ch (2 recommended).
        self.pres_head = PresencePrior(hidden_dim=pres_hidden_dim, n_classes=n_classes)

    def forward(self, encoded_eeg: torch.Tensor, encoded_audio: torch.Tensor):
        """
        Parameters
        ----------
        encoded_eeg   : [B, 76, C_eeg]
        encoded_audio : [B, 76, C_audio]

        Returns
        -------
        cross_probs : [B, 4]  # softmax from cross-attention classifier
        pres_probs  : [B, 4]  # sigmoid probs from presence prior
        pres_mask   : [B, 4]  # 0/1 mask (>=0.5 threshold applied inside PresencePrior)
        """

        cross_logits = self.cross_cls(encoded_audio, encoded_eeg)   # [B,4]
        pres_logits = self.pres_head(encoded_audio)      # [B,4], [B,4]

        return cross_logits, pres_logits
