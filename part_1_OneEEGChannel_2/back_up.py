import torch
import torch.nn as nn

class EEGAudioCrossAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4, use_proj=True):
        super().__init__()
        
        # NOTE:
        # Inputs are already *feature maps* produced by convolution encoders
        #   EEG_feature_map:   (B, C_eeg=20, T)   -> result of EEG 1D conv
        #   Audio_feature_map: (B, C_audio=2, T)  -> result of Audio 1D conv
        #
        # These are *not raw signals* anymore. They are higher-level features.
        #
        # Problem:
        #   - nn.MultiheadAttention expects query, key, value to have the SAME embedding dim D
        #   - But here EEG has 20 channels, Audio has 2 channels → mismatch
        #
        # Solution:
        #   - Optionally project both into a common d_model (e.g., 64) dimension
        #   - This can be done using 1x1 Conv (fast) or Linear (equivalent)
        self.use_proj = use_proj
        if use_proj:
            self.eeg_proj   = nn.Conv1d(in_channels=20, out_channels=d_model, kernel_size=1)
            self.audio_proj = nn.Conv1d(in_channels=2,  out_channels=d_model, kernel_size=1)
            final_dim = d_model
        else:
            # If your EEG and Audio encoders *already* output the same D, skip projection
            final_dim = 20  # <- only works if C_eeg == C_audio == 20 (example)

        # Cross-attention: EEG queries into Audio sequence
        self.cross_attn = nn.MultiheadAttention(embed_dim=final_dim,
                                                num_heads=n_heads,
                                                batch_first=True)

    def forward(self, eeg_feat, audio_feat):
        """
        eeg_feat:   (B, 20, T)   EEG feature map from conv
        audio_feat: (B,  2, T)   Audio feature map from conv
        """

        # Step 1. (Optional) Project to common dimension d_model
        # - After this, both EEG and Audio will have the same "channel size"
        # - Shape stays (B, D, T)
        if self.use_proj:
            eeg_proj   = self.eeg_proj(eeg_feat)     # (B, d_model, T)
            audio_proj = self.audio_proj(audio_feat) # (B, d_model, T)
        else:
            eeg_proj, audio_proj = eeg_feat, audio_feat  # (B, C, T)

        # Step 2. MultiheadAttention expects input as (B, T, D), not (B, D, T)
        # So we transpose the last two dims
        eeg_seq   = eeg_proj.transpose(1, 2)    # (B, T, D)
        audio_seq = audio_proj.transpose(1, 2)  # (B, T, D)

        # Step 3. Cross-attention
        # - Query = EEG sequence  (B, T, D)
        # - Key   = Audio sequence (B, T, D)
        # - Value = Audio sequence (B, T, D)
        #
        # Interpretation:
        #   For each EEG time step, we ask:
        #     "Which audio frames are most relevant to this EEG frame?"
        attn_out, attn_w = self.cross_attn(query=eeg_seq,
                                           key=audio_seq,
                                           value=audio_seq,
                                           need_weights=True)
        # attn_out: (B, T, D)  = EEG-guided Audio features (time-aligned)
        # attn_w:   (B, T, T)  = Attention map (EEG time → Audio time)

        return attn_out, attn_w
