import math
import torch
import torch.nn.functional as F


def _freeze_grad(model):
    requires_grad = {}
    for p in model.parameters():
        requires_grad[p] = p.requires_grad
        p.requires_grad_(False)
    return requires_grad


def _unfreeze_grad(model, requires_grad):
    for p in model.parameters():
        if requires_grad[p]:
            p.requires_grad_(True)


def ista_nonlinear(x, z0, decoder, alpha=1.0, fast=True, maxiter=10, lr=0.01,
                   tol=1e-5, eval_mode=True):
    # configure decoder
    requires_grad = _freeze_grad(decoder)
    training = decoder.training
    if eval_mode:
        # set decoder to eval mode (in case dropout, batchnorm etc.)
        decoder.eval()

    tol = z0.numel() * tol

    # derivative of the residual sum-of-squares (rss) objective
    def rss_grad_fn(zk):
        zk = zk.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = 0.5 * (x - decoder(zk)).pow(2).sum()
        grad, = torch.autograd.grad(loss, zk)
        return grad

    # ista step function
    def step(zk):
        return F.softshrink(zk - lr * rss_grad_fn(zk), alpha * lr)

    # optimize
    z = z0.detach()
    if fast:
        y = z0.detach()
        t = 1
    for _ in range(maxiter):
        z_next = step(y) if fast else step(z)
        # check for convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break
        # update state
        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next

    # re-configure decoder
    _unfreeze_grad(decoder, requires_grad)
    if eval_mode and training:
        decoder.train()

    return z.detach()