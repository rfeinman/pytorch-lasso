import warnings
from torch import Tensor
import torch
from scipy.optimize import minimize_scalar
from scipy.optimize.optimize import _status_message

from ...conjgrad import conjgrad
from ..utils import batch_cholesky_solve


def iterative_ridge(z0, x, weight, alpha=1.0, tol=1e-5, tikhonov=1e-4, eps=None,
                    maxiter=10, line_search=True, cg=False, cg_options=None,
                    verbose=False):
    """Iterated Ridge Regression method for Lasso problems

    Explained in section 2.5 of Mark Schmidt, 2005:
        "Least Squares Optimization with L1-Norm Regularization."

    Parameters
    ----------
    z0 : Tensor of shape [batch, code_size]
        Initial code vectors
    x : Tensor of shape [batch, inp_size]
        Reconstruction target
    weight : Tensor of shape [inp_size, code_size]
        Dictionary matrix (i.e. decoder weights)
    alpha : float
        Sparsity weight of the Lasso problem
    tol : float
        Tolerance for absolute change in parameter value
    tikhonov : float
        Small value added to diagonal of the Gram matrix for stability
    eps : float
        Threshold for generalized inverse
    maxiter : int, optional
        Maximum number of iterations
    line_search : bool
        Whether to use line search optimization (as opposed to fixed step size)
    cg : bool
        Whether to use conjugate gradient to solve the ridge problem at each
        iteration. When `False` (default), Cholesky factorization is used.
    cg_options : dict, optional
        Options to pass to conjugate gradient solver. Ignored if `cg=False`
    verbose : bool
        Verbosity indicator

    Returns
    -------
    z : Tensor
        Optimize code vectors

    """
    if tikhonov < 1e-5:
        warnings.warn('small regularization value %0.4e may lead to '
                      'inprecise results.' % tikhonov)
    if cg and cg_options is None:
        cg_options = {}
    if eps is None:
        eps = torch.finfo(weight.dtype).eps
    tol = z0.numel() * tol

    def f(z):
        x_hat = torch.mm(z, weight.T)
        loss = 0.5 * (x_hat - x).pow(2).sum() + alpha * z.abs().sum()
        return loss

    # initialize
    z = z0
    fval = f(z)
    if verbose:
        print('initial fval: %0.4f' % fval)

    # right hand side of the residual sum of squares (RSS) problem. [B,D]
    rhs = torch.mm(x, weight)  # [B,D] = [B,K] @ [K,D]

    # batch gram matrix W^T @ W. [D,D]
    A = torch.mm(weight.T, weight)

    if not cg:
        A = A.expand(z.size(0), -1, -1)  # [B,D,D]

    for k in range(1, maxiter + 1):
        # compute diagonal factor
        zmag = z.abs()
        is_zero = zmag < eps
        diag = (alpha / zmag).masked_fill(is_zero, 0)
        rhs_k = rhs.masked_fill(is_zero, 0.)

        # solve ridge sub-problem
        if cg:
            # use conjugate gradient method
            def Adot(v):
                Av = torch.mm(v.masked_fill(is_zero, 0.), A)
                Av.masked_fill_(is_zero, 0.)
                Av.addcmul_(diag + tikhonov, v)
                return Av
            dot = lambda u, v: torch.sum(u*v, 1, keepdim=True)
            z_sol = conjgrad(rhs_k, Adot, dot, **cg_options)
        else:
            # use cholesky factorization
            A_k = A.masked_fill((is_zero.unsqueeze(1) | is_zero.unsqueeze(2)), 0.)
            A_k.diagonal(dim1=1, dim2=2).add_(diag + tikhonov)
            z_sol = batch_cholesky_solve(rhs_k, A_k)  # [B,D]

        if line_search:
            # line search optimization
            p = z_sol - z
            line_obj = lambda t: float(f(z.add(p, alpha=t)))
            res = minimize_scalar(line_obj, bounds=(0,10), method='bounded')
            t = res.x
            fval = torch.tensor(res.fun)
            update = p.mul(t)
            z = torch.where(is_zero, z, z + update)
        else:
            # fixed step size
            update = z_sol - z
            z = torch.where(is_zero, z, z_sol)
            fval = f(z)

        if verbose:
            print('iter %3d - fval: %0.4f' % (k, fval))

        # check for convergence
        if update.abs().sum() <= tol:
            msg = _status_message['success']
            break

        # check for NaN
        if (fval.isnan() | update.isnan().any()):
            msg = _status_message['nan']
            break

    else:
        msg = "Warning: " + _status_message['maxiter']

    if verbose:
        print(msg)
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)

    return z