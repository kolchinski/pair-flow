import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.real_nvp import RealNVP
from models.real_nvp.d2d_real_nvp import D2DRealNVP

class PairedNVP(nn.Module):
    """Double RealNVP Model with domain-to-domain and domain-to-latent maps


    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """
    def __init__(self, *args, **kwargs):
        super(PairedNVP, self).__init__()

        # Assume the two flows are taking the same arguments
        self.d2d = D2DRealNVP(*args, **kwargs)
        self.rnvp = RealNVP(*args, **kwargs)

    def forward(self, input, double_flow, reverse=False):
        if reverse:
            z = input
            x, _ = self.rnvp(z, reverse=True)

            if double_flow:
                x2, _ = self.d2d(x, reverse=True)
                return x2, None
            else:
                return x, None

        else:
            if double_flow:
                x2 = input
                x, sldj = self.d2d(x2)
            else:
                x = input
                sldj = None

            z, sldj = self.rnvp(x, sldj)
            return z, sldj
