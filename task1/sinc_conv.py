# __init__.py

from torch import nn



class SincConv(nn.Module):
    """An implementation of a 1d convolution layer with sinc filters."""

    def __init__(self):
        pass

    def forward(self, x):
        return x