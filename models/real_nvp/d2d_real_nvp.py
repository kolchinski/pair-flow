import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.coupling import Coupling, MaskType
from models.real_nvp.real_nvp import RealNVP
from models.real_nvp.squeezing import Squeezing, Unsqueezing
from util import depth_to_space, space_to_depth


class D2DRealNVP(RealNVP):
    """Domain-to-Domain RealNVP Model

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """

    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(D2DRealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))

        self.x_channels = in_channels
        # self.z_channels = 4 ** (num_scales - 1) * in_channels

        # Get inner layers
        layers = []

        # Squeeze part
        for scale in range(num_scales):
            layers += [Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)]

            in_channels *= 4  # Account for the squeeze
            mid_channels *= 2  # When squeezing, double the number of hidden-layer features in s and t
            layers += [Squeezing(),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)]

        # Unsqueeze part
        for scale in range(num_scales):
            layers += [Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)]

            in_channels = int(in_channels / 4)  # Account for the unsqueeze
            mid_channels = int(mid_channels / 2)  # When unsqueezing, half the number of hidden-layer features in
            # s and t
            layers += [Unsqueezing(),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, reverse=False):
        if reverse:
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                                 .format(x.min(), x.max()))

            y, sldj = self.pre_process(x)

            # Apply inverse flows
            z_split = None
            for layer in reversed(self.layers):
                y, z_split = layer.backward(y, z_split)

            assert(z_split is None)

            x2 = y
            assert(x.shape == x2.shape)

            return x2, None

        else:
            x2 = x
            # Expect inputs in [0, 1]
            if x2.min() < 0 or x2.max() > 1:
                raise ValueError('Expected x2 in [0, 1], got x2 with min/max {}/{}'
                                 .format(x2.min(), x2.max()))

            # Dequantize and convert to logits
            y, sldj = self.pre_process(x2)

            # Apply forward flows
            z_split = None
            for layer in self.layers:
                y, sldj, z_split = layer.forward(y, sldj, z_split)

            # This model shouldn't have split layers => z should stay None
            assert(z_split is None)

            x = self.pre_process(y)

            # Shape should stay constant - hourglass architecture image-to-image
            assert x.shape == x2.shape, f'x and x2 have different shapes: {x.shape}, {x2.shape}'

            return x, sldj
