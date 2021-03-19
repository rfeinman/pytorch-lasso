import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LipBoundConv2d(nn.Module):
    """A module to estimate the (squared) Lipschitz constant of a Conv2d layer.

    The Lipschitz constant of a Conv2d is the largest singular value of the
    corresponding Toeplitz matrix. This module computes a close lower bound of
    the constant that is very fast.

    For ISTA we need the max eigenvalue of W^T W, corresponding to the square
    of the max singular value of W. Hence we set squared=True by default for
    the classical ISTA use case.

    Paper: https://arxiv.org/abs/2006.08391
    Code: https://github.com/MILES-PSL/Upper-Bound-Lipschitz-Convolutional-Layers
    """
    def __init__(self, kernel_size, padding, stride=1, sample=50, squared=True):
        super().__init__()
        assert len(kernel_size) == 4
        if not kernel_size[-1] == kernel_size[-2]:
            raise ValueError("The last 2 dim of the kernel must be equal.")
        if not kernel_size[-1] % 2 == 1:
            raise ValueError("The dimension of the kernel must be odd.")
        if not stride == 1:
            raise NotImplementedError("LipBound not implemented for stride > 1.")
        ksize = kernel_size[-1]
        self.ksize = ksize
        self.squared = squared

        # define frequencies \omega0 and \omega1
        x = torch.linspace(0, 2*math.pi, sample)
        w0, w1 = torch.meshgrid(x, x)
        w0 = w0.reshape(-1,1)
        w1 = w1.reshape(-1,1)

        # define location indices h0 and h1
        p_index = 1.0 + torch.arange(padding-ksize, padding)
        H0, H1 = torch.meshgrid(p_index, p_index)
        H0 = H0.reshape(1,-1)
        H1 = H1.reshape(1,-1)
        buf = (w0 * H0 + w1 * H1).T
        self.register_buffer('buf', buf)

    def forward(self, kernel):
        assert kernel.dim() == 4
        assert kernel.size(2) == kernel.size(3) == self.ksize
        if kernel.size(0) > kernel.size(1):
            kernel = kernel.transpose(0,1)

        real = torch.cos(self.buf)  # [K**2, S**2]
        imag = torch.sin(self.buf)

        kernel = kernel.flatten(2) # [Co, Ci, K**2]
        poly_real = torch.matmul(kernel, real)  # [Co, Ci, S**2]
        poly_imag = torch.matmul(kernel, imag)  # [Co, Ci, S**2]
        poly1 = poly_real.square().sum(1)  # [Co, S**2]
        poly2 = poly_imag.square().sum(1)  # [Co, S**2]
        poly = poly1 + poly2
        bound = poly.max(-1)[0].sum()  # maximum eigenvalue of W^T W
        if not self.squared:
            bound = bound.sqrt()  # maximum singular value of W

        return bound