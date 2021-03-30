import warnings
import math
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigsh


def _lipschitz_constant(W):
    #L = torch.linalg.norm(W, ord=2) ** 2
    WtW = torch.matmul(W.t(), W)
    #L = torch.linalg.eigvalsh(WtW)[-1]
    L = eigsh(WtW.detach().cpu().numpy(), k=1, which='LM',
              return_eigenvectors=False).item()
    return L


def backtracking(z, x, weight, alpha, lr0, eta=1.5, maxiter=1000, verbose=False):
    if eta <= 1:
        raise ValueError('eta must be > 1.')

    # store initial values
    resid_0 = torch.matmul(z, weight.T) - x
    fval_0 = 0.5 * resid_0.pow(2).sum()
    fgrad_0 = torch.matmul(resid_0, weight)

    def calc_F(z_1):
        resid_1 = torch.matmul(z_1, weight.T) - x
        return 0.5 * resid_1.pow(2).sum() + alpha * z_1.abs().sum()

    def calc_Q(z_1, t):
        dz = z_1 - z
        return (fval_0
                + (dz * fgrad_0).sum()
                + (0.5 / t) * dz.pow(2).sum()
                + alpha * z_1.abs().sum())

    lr = lr0
    z_next = None
    for i in range(maxiter):
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)
        F_next = calc_F(z_next)
        Q_next = calc_Q(z_next, lr)
        if verbose:
            print('iter: %4d,  t: %0.5f,  F-Q: %0.5f' % (i, lr, F_next-Q_next))
        if F_next <= Q_next:
            break
        lr = lr / eta
    else:
        warnings.warn('backtracking line search failed. Reverting to initial '
                      'step size')
        lr = lr0
        z_next = F.softshrink(z - lr * fgrad_0, alpha * lr)

    return z_next, lr


def ista(x, z0, weight, alpha=1.0, fast=True, lr='auto', maxiter=10,
         tol=1e-5, backtrack=False, eta_backtrack=1.5, verbose=False):
    if lr == 'auto':
        # set lr based on the maximum eigenvalue of W^T @ W; i.e. the
        # Lipschitz constant of \grad f(z), where f(z) = ||Wz - x||^2
        L = _lipschitz_constant(weight)
        lr = 1 / L
    tol = z0.numel() * tol

    def loss_fn(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        loss = 0.5 * resid.pow(2).sum() + alpha * z_k.abs().sum()
        return loss / x.size(0)

    def rss_grad(z_k):
        resid = torch.matmul(z_k, weight.T) - x
        return torch.matmul(resid, weight)

    # optimize
    z = z0
    if fast:
        y, t = z0, 1
    for _ in range(maxiter):
        if verbose:
            print('loss: %0.4f' % loss_fn(z))

        # ista update
        z_prev = y if fast else z
        if backtrack:
            # perform backtracking line search
            z_next, _ = backtracking(z_prev, x, weight, alpha, lr, eta_backtrack)
        else:
            # constant step size
            z_next = F.softshrink(z_prev - lr * rss_grad(z_prev), alpha * lr)

        # check convergence
        if (z - z_next).abs().sum() <= tol:
            z = z_next
            break

        # update variables
        if fast:
            t_next = (1 + math.sqrt(1 + 4 * t**2)) / 2
            y = z_next + ((t-1)/t_next) * (z_next - z)
            t = t_next
        z = z_next

    return z