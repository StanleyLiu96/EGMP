# depthconv1d.py
import torch
import torch.nn as nn


class DepthConv1d(nn.Module):
    """
    One TCN block:
      1x1 conv -> PReLU+GN -> depthwise dilated conv -> PReLU+GN -> 1x1 conv (residual)
      Optional 1x1 skip connection.
    """

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True):
        super(DepthConv1d, self).__init__()
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, kernel_size=1)
        self.dconv1d = nn.Conv1d(
            in_channels=hidden_channel,
            out_channels=hidden_channel,
            kernel_size=kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=padding,
        )
        self.res_out = nn.Conv1d(hidden_channel, input_channel, kernel_size=1)
        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, kernel_size=1)

        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-8)

    def forward(self, x):
        h = self.conv1d(x)
        h = self.nonlinearity1(h)
        h = self.reg1(h)

        h = self.dconv1d(h)
        h = self.nonlinearity2(h)
        h = self.reg2(h)

        residual = self.res_out(h)
        if self.skip:
            skip = self.skip_out(h)
            return residual, skip
        else:
            return residual
