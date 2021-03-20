import math
import torch.nn.functional as F

from .lip_bound import lip_bound_conv2d


def ista_conv2d(x, z0, weight, alpha=1.0, stride=1, padding=0, fast=True,
                maxiter=10, lr='auto', tol=1e-5, verbose=False):
    if lr == 'auto':
        if stride != 1:
            raise NotImplementedError("auto lr not implemented for stride > 1.")
        # set lr based on the maximum eigenvalue of W^T @ W
        L = lip_bound_conv2d(weight, padding, squared=True)
        lr = 1 / L
    tol = z0.numel() * tol

    def rss_grad(zk):
        x_hat = F.conv_transpose2d(zk, weight, stride=stride, padding=padding)
        return F.conv2d(x_hat-x, weight, stride=stride, padding=padding)

    def loss_fn(zk):
        x_hat = F.conv_transpose2d(zk, weight, stride=stride, padding=padding)
        loss = 0.5 * (x - x_hat).pow(2).sum() + alpha * zk.abs().sum()
        return loss / x.size(0)

    # ista step function
    def step(zk):
        return F.softshrink(zk - lr * rss_grad(zk), alpha * lr)

    # optimize
    z = z0
    if fast:
        y = z0
        t = 1
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))
        z_next = step(y) if fast else step(z)
        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break
        z = z_next

    return z