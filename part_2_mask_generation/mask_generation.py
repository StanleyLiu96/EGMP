# mask_generation.py
# Audio-only mask generation network (Conv-TasNet style)
# Always outputs exactly 1 mask

import torch
import torch.nn as nn
from tcn import TCN  # import TCN directly from tcn.py


class MASK_GENERATION(nn.Module):
    """
    Audio-only mask generation network (Conv-TasNet style).
    Takes raw stereo mixture as input and outputs ONE mask
    for the chosen instrument.

    Args:
        enc_channel (int): channels of audio encoder
        feature_channel (int): hidden channels in the TCN
        encoder_kernel_size (int): kernel size of audio encoder
        layer_per_stack (int): layers per TCN stack
        stack (int): number of stacks in TCN
        kernel (int): kernel size in TCN
    """

    def __init__(self,
                 enc_channel=64,
                 feature_channel=32,
                 encoder_kernel_size=32,
                 layer_per_stack=8,
                 stack=3,
                 kernel=3):
        super(MASK_GENERATION, self).__init__()

        # ====== hyper parameters ======
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_instruments = 1   # fixed to 1
        self.encoder_kernel_size = encoder_kernel_size
        self.stride = 4
        self.layer = layer_per_stack
        self.stack = stack
        self.kernel = kernel

        # ====== Audio encoder ======
        self.audio_encoder = nn.Conv1d(
            in_channels=2,
            out_channels=self.enc_channel,
            kernel_size=self.encoder_kernel_size,
            stride=self.stride,
            bias=False,
        )

        # ====== TCN separation network (Conv-TasNet style) ======
        self.TCN = TCN(
            input_dim=self.enc_channel,
            output_dim=self.enc_channel * self.num_instruments,
            BN_dim=self.enc_channel,
            hidden_dim=self.feature_channel,
            layer=self.layer,
            stack=self.stack,
            kernel=self.kernel,
            skip=True,
            dilated=True,
        )

        self.receptive_field = self.TCN.receptive_field

        # ====== Decoder ======
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_channel,
            out_channels=1,
            kernel_size=self.encoder_kernel_size,
            stride=self.stride,
            bias=False,
        )

    def forward(self, input_audio):
        """
        Args:
            input_audio: (B, 2, T) raw stereo mixture
        Returns:
            outputs: (B, 1, T) separated waveform
        """
        batch_size = input_audio.size(0)

        # Encode raw audio
        enc_output = self.audio_encoder(input_audio)  # (B, enc_channel, T')

        # Apply TCN -> masks
        masks = torch.sigmoid(self.TCN(enc_output))  # (B, enc_channel, T')
        masks = masks.view(batch_size, 1, self.enc_channel, -1)

        # Apply mask
        masked_output = enc_output.unsqueeze(1) * masks  # (B, 1, enc_channel, T')

        # Decode waveform
        masked_output = masked_output.view(batch_size, self.enc_channel, -1)
        decoded = self.decoder(masked_output)  # (B, 1, T)

        return decoded
