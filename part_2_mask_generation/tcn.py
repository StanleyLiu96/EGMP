# tcn.py
import torch
import torch.nn as nn
from depthconv1d import DepthConv1d


class TCN(nn.Module):
    """
    Audio-only TCN for mask estimation.
    input_dim  = encoder channels
    output_dim = encoder channels * num_instruments
    """

    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, dilated=True):
        super(TCN, self).__init__()

        self.skip = skip
        self.dilated = dilated
        self.layer = layer

        self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        self.BN = nn.Conv1d(input_dim, BN_dim, kernel_size=1)

        self.TCN = nn.ModuleList()
        self.receptive_field = 0
        for s in range(stack):
            for i in range(layer):
                dil = 2 ** i if self.dilated else 1
                pad = dil if self.dilated else 1
                self.TCN.append(
                    DepthConv1d(BN_dim, hidden_dim, kernel=kernel,
                                dilation=dil, padding=pad, skip=skip)
                )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    self.receptive_field += (kernel - 1) * dil

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(BN_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, input_dim, T)
        returns: (B, output_dim, T)
        """
        y = self.LN(x)
        y = self.BN(y)

        if self.skip:
            skip_sum = 0.0
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](y)
                y = y + residual
                skip_sum = skip_sum + skip
            y = self.output(skip_sum)
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](y)
                y = y + residual
            y = self.output(y)

        return y
