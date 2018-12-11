import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256, lambda_max=float("inf")):
        super(RealNVPLoss, self).__init__()
        self.k = k
        self.lambda_max = lambda_max

    def forward(self, z, sldj, double_flow):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()

        if double_flow is None or double_flow is False:
            jacobian_loss = (max(sldj.mean(), self.lambda_max) - self.lambda_max) ** 2
        elif double_flow is True:
            # TODO: Make this a parameter
            double_lambda_max = 9000
            jacobian_loss = (max(sldj.mean(), double_lambda_max) - self.lambda_max) ** 2
        else:
            raise Exception(f'Double is neither boolen or none: {double_flow}')
        # TODO: Make hyperparam
        return nll, jacobian_loss
