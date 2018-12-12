import torch
import torch.nn as nn

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

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(PairedNVP, self).__init__()

        # Assume the two flows are taking the same arguments
        self.d2d = D2DRealNVP(num_scales=num_scales, in_channels=in_channels, mid_channels=mid_channels,
                              num_blocks=num_blocks)
        self.rnvp = RealNVP(num_scales=num_scales, in_channels=in_channels, mid_channels=mid_channels,
                            num_blocks=num_blocks)

    def forward(self, input, double_flow, reverse=False):
        if reverse:
            z = input
            x, _ = self.rnvp(z, reverse=True)

            if double_flow:
                x = torch.sigmoid(x)
                x2, _ = self.d2d(x, reverse=True)
                return x2, None
            else:
                return x, None

        else:
            if double_flow:
                x2 = input
                x, g_sldj = self.d2d(x2)

                # Don't backprop gradients from dF(G(x2))/dx2 into F, since z is independent of
                # x2 when conditioned on x. I.e. we should only train the mapping F: z<->x on samples
                # from x, not x2, and only train the mapping G: x<->x2 on samples from x2
                x = x.detach()
                g_sldj = g_sldj.detach()
            else:
                x = input
                g_sldj = None

            z, sldj = self.rnvp(x, g_sldj=g_sldj)

            print('\n\n Sldj mean:', torch.mean(sldj))
            print('double_flow:', double_flow)
            if g_sldj is not None:
                print('G_sldj mean: ', torch.mean(g_sldj))
            return z, sldj
