#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
presence_prior.py
-----------------
This file defines a small sub-model that predicts the *presence prior*:
- Input  : mixed_audio_encoded tensor of shape [B, 76, 2]
           (B = batch size, 76 = fixed time steps, 2 = audio features per step)
- Output : presence logits of shape [B, 4]
           (4 instruments: Gt, Vx, Dr, Bs)

How it works:
1. Project the 2-dim input features at each time step into a higher hidden dimension.
2. Pass through a small temporal encoder (1D conv).
3. Average over the 76 time steps → global clip-level representation.
4. Linear layer maps to 4 scores, one per instrument.
5. Apply sigmoid → each score becomes a probability in [0,1].

These 4 outputs are the probabilities for whether each instrument is present
in the mixture audio, independent of EEG.
"""

import torch
import torch.nn as nn


class PresencePrior(nn.Module):
    def __init__(self, hidden_dim=128, n_classes=4):
        super().__init__()
        self.n_classes = n_classes

        # Step 1. Project 2 → hidden_dim
        # Input : [B, 76, 2]
        # Output: [B, 76, hidden_dim]
        self.input_proj = nn.Linear(2, hidden_dim)

        # Step 2. Temporal encoder (Conv1d along time)
        # Conv1d expects [B, C, T], so we will transpose later
        # Keeps hidden_dim dimension the same
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # Step 3. Final linear layer: hidden_dim → 4 instruments
        self.output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, mixed_audio_encoded):
        """
        Input:
            mixed_audio_encoded : [B, 76, 2]
        Output:
            logits : [B, 4]
        """

        # [B, 76, 2] → [B, 76, hidden_dim]
        x = self.input_proj(mixed_audio_encoded)

        # Rearrange for Conv1d: [B, 76, hidden_dim] → [B, hidden_dim, 76]
        x = x.transpose(1, 2)

        # Apply temporal encoder: [B, hidden_dim, 76] → [B, hidden_dim, 76]
        x = self.temporal_encoder(x)

        # Global average pooling over time: [B, hidden_dim, 76] → [B, hidden_dim]
        x = x.mean(dim=2)

        # Final linear layer: [B, hidden_dim] → [B, 4]
        logits = self.output_layer(x)

        # return pres_logits
        return logits


# ============================
# Example usage
# ============================
# if __name__ == "__main__":
#     B = 2
#     mixed_audio_encoded = torch.randn(B, 76, 2)  # fake input
#     model = PresencePrior()
    
#     pres_probs, pres_mask = model(mixed_audio_encoded)  # unpack both

#     INSTRUMENTS = ["Gt", "Vx", "Dr", "Bs"]

#     print("Input shape :", mixed_audio_encoded.shape)   # [2, 76, 2]
#     print("Probabilities shape:", pres_probs.shape)     # [2, 4]
#     print("Mask shape:", pres_mask.shape)               # [2, 4]
#     # print("Probabilities:", pres_probs)                 # e.g. [[0.87, 0.90, 0.03, 0.05]]
#     # print("Mask:", pres_mask)                           # e.g. [[1, 1, 0, 0]]
#     # Pretty-print with instrument names
#     for i in range(B):
#         probs_dict = {inst: f"{pres_probs[i,j].item():.2f}" for j, inst in enumerate(INSTRUMENTS)}
#         mask_dict  = {inst: int(pres_mask[i,j].item()) for j, inst in enumerate(INSTRUMENTS)}
#         print(f"\nSample {i}:")
#         print("  Probabilities:", probs_dict)
#         print("  Mask:", mask_dict)