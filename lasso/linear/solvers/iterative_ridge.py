from torch import Tensor
import torch
from scipy.optimize import minimize_scalar
from scipy.optimize.optimize import _status_message

from ...conjgrad import conjgrad
from ..utils import batch_cholesky_solve


def iterative_ridge(z0, x, weight, alpha=1.0, tol=1e-5, tikhonov=1e-5, eps=None,
                    maxiter=100, line_search=True, cg=False, cg_options=None,
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
        Whether to use strong-wolfe line search (as opposed to fixed step size)
    cg : bool
        Whether to use conjugate gradient to solve the ridge problem at each
        iteration. When `False` (default), Cholesky factorization is used.
    cg_options : dict, optional
        Options to pass to conjugate gradient solver. Ignored if `cg=False`
    verbose : bool
        Verbosity indicator

    Returns
    -------
    zk : Tensor
        Optimize code vectors
    success : bool
        Boolean indicating if the optimization succeeded

    """
    def f(z):
        x_hat = torch.matmul(z, weight.T)
        loss = 0.5 * (x_hat - x).pow(2).sum() + alpha * z.abs().sum()
        return loss

    # initialize
    if cg and cg_options is None:
        cg_options = {}
    if eps is None:
        eps = torch.finfo(weight.dtype).eps
    batch_size = z0.size(0)
    tol = z0.numel() * tol
    zk = z0
    fval = f(zk)

    # right hand side of the residual sum of squares (RSS) problem
    rhs = torch.matmul(x, weight)  # [B,D] = [B,K] @ [K,D]

    if not cg:
        # batch gram matrix W^T @ W
        A = torch.matmul(weight.T, weight) # [D,D] = [D,K] @ [K,D]
        A = A[None].expand(batch_size, -1, -1)  # [B,D,D]

    for k in range(1, maxiter + 1):
        # compute ridge diagonal factor
        zmag = zk.abs()
        zmag_inv = zmag.reciprocal().masked_fill(zmag < eps, 0)

        # solve ridge problem
        if cg:
            # use conjugate gradient method
            def Adot(v):
                Av = torch.mm(torch.mm(v, weight.T), weight)
                Av += (2 * alpha * zmag_inv + tikhonov) * v
                return Av
            dot = lambda u, v: torch.sum(u*v, 1, keepdim=True)
            zk1 = conjgrad(rhs, Adot, dot, **cg_options)
        else:
            # use cholesky factorization
            Ak = A + 2 * alpha * torch.diag_embed(zmag_inv)  # [B,D,D]
            if tikhonov > 0:
                Ak.diagonal(dim1=1, dim2=2).add_(tikhonov)
            zk1 = batch_cholesky_solve(rhs, Ak)  # [B,D]

        if line_search:
            # line search optimization
            pk = zk1 - zk
            line_obj = lambda t: float(f(zk.add(pk, alpha=t)))
            res = minimize_scalar(line_obj, bounds=(0,10), method='bounded')
            t = res.x
            fval = torch.tensor(res.fun)
            update = pk.mul(t)
            zk = zk + update
        else:
            # fixed step size
            update = zk1 - zk
            zk = zk1
            fval = f(zk)

        if verbose:
            print('fval: %0.4f' % fval)

        # check for termination
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

    return zk