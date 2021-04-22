from torch import Tensor
import torch
import torch.autograd as autograd
from scipy.optimize import minimize_scalar

from ..linear.utils import batch_cholesky_solve

outer = lambda u,v: u.unsqueeze(-1) * v.unsqueeze(-2)
inner = lambda u,v: torch.sum(u*v, 1, keepdim=True)


class BFGS(object):
    def __init__(self, x, g):
        self.H = torch.diag_embed(torch.ones_like(x))  # [B,D,D]
        self.x_prev = x.clone(memory_format=torch.contiguous_format)
        self.g_prev = g.clone(memory_format=torch.contiguous_format)

    def update(self, x, g):
        s = x - self.x_prev
        y = g - self.g_prev
        # update the BFGS hessian approximation
        rho_inv = inner(y, s)
        valid = rho_inv.abs().gt(1e-10)
        rho = torch.where(valid,
                          rho_inv.reciprocal(),
                          torch.full_like(rho_inv, 1000.))

        HssH = torch.bmm(self.H, torch.bmm(outer(s, s), self.H.transpose(1,2)))
        sHs = inner(s, torch.bmm(self.H, s.unsqueeze(2)).squeeze(2))
        self.H = torch.where(
            valid.unsqueeze(2),
            self.H + rho.unsqueeze(2) * outer(y, y) - HssH / sHs.unsqueeze(2),
            self.H
        )
        self.x_prev.copy_(x, non_blocking=True)
        self.g_prev.copy_(g, non_blocking=True)


@torch.no_grad()
def iterative_ridge_bfgs(f, x0, alpha=1.0, lr=1.0, xtol=1e-5, tikhonov=1e-4,
                         eps=None, line_search=True, maxiter=None,
                         verbose=0):
    """A BFGS analogue to Iterative Ridge for nonlinear reconstruction terms.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    alpha : float
        Sparsity weight of the Lasso problem
    lr : float
        Initial step size (learning rate) for each line search.
    xtol : float
        Convergence tolerance on changes to parameter x
    eps : float
        Threshold for non-zero identification
    line_search : bool
        Whether to use line search optimization (as opposed to fixed step size)
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * num_params
    verbose : int
        Verbosity level

    """
    assert x0.dim() == 2
    if maxiter is None:
        maxiter = x0.size(1) * 5
    if eps is None:
        eps = torch.finfo(x0.dtype).eps
    verbose = int(verbose)

    def evaluate(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        # NOTE: do not include l1 penalty term in the gradient
        grad, = autograd.grad(fval, x)
        fval = fval.detach() + alpha * x.norm(p=1)
        return fval, grad

    # initialize
    x = x0.detach()
    fval, grad = evaluate(x)
    if verbose:
        print('initial loss: %0.4f' % fval)
    t = torch.clamp(lr / grad.norm(p=1), max=lr)
    delta_x = x.new_tensor(float('inf'))
    bfgs = BFGS(x, grad)

    # begin main loop
    for k in range(1, maxiter + 1):
        # locate zeros
        xmag = x.abs()
        is_zero = xmag < eps

        # compute step direction
        diag = (alpha / xmag).masked_fill(is_zero, 0)
        rhs = (grad + diag*x).masked_fill(is_zero, 0)
        H = bfgs.H.masked_fill((is_zero.unsqueeze(1) | is_zero.unsqueeze(2)), 0.)
        H.diagonal(dim1=1, dim2=2).add_(diag + tikhonov)
        d = batch_cholesky_solve(rhs, H)

        # optional strong-wolfe line search
        if line_search:
            def line_obj(tt):
                x_new = x.add(d, alpha=-tt)
                return float(f(x_new) + alpha * x_new.norm(p=1))
            t = minimize_scalar(line_obj, bounds=(0,10), method='bounded').x

        # update x
        x_new = torch.where(is_zero, x, x.add(d, alpha=-t))
        torch.norm(x_new - x, p=2, out=delta_x)
        x = x_new

        # re-evaluate f and grad
        fval, grad = evaluate(x)
        if verbose > 1:
            print('iter %3d - loss: %0.4f - dx: %0.4e' % (k, fval, delta_x))

        # stopping check
        if (delta_x <= xtol) | ~fval.isfinite():
            break

        # update hessian estimate
        bfgs.update(x, grad)
        t = lr

    if verbose:
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)

    return x
