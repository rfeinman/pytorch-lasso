import warnings
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
from torch import Tensor
import torch
import torch.autograd as autograd
from torch.optim.lbfgs import _strong_wolfe

from ..linear.utils import batch_cholesky_solve

Inf = float('inf')


def pinv(x, eps=1e-8):
    return x.reciprocal().masked_fill(x < eps, 0)


@torch.no_grad()
def iterative_ridge_bfgs(f, x0, alpha=1.0, gtol=1e-5, lr=1.0, lambd=1e-4,
                         line_search=True, normp=Inf, maxiter=None,
                         return_losses=False, disp=False):
    """A BFGS analogue to Iterative Ridge for nonlinear reconstruction terms.

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    lr : float
        Initial step size (learning rate) for each line search.
    line_search : bool
        Whether or not to perform line search for optimal step size.
    normp : int, inf
        The norm type to use for calculating gradient magnitude.
    maxiter : int, optional
        Maximum number of iterations to perform. Defaults to 200 * num_params
    disp : bool
        Set to True to print convergence messages.
    """
    assert x0.dim() == 2
    xshape = x0.shape
    x = x0.detach()
    if maxiter is None:
        maxiter = x.size(1) * 200


    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % Fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % nfev)
        result = OptimizeResult(fun=Fval, jac=grad, nfev=nfev,
                                status=warnflag, success=(warnflag==0),
                                message=msg, x=x, nit=k)
        if return_losses:
            return result, losses
        return result

    def evaluate(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            fval = f(x)
        grad, = autograd.grad(fval, x)
        # add tikhonov regularization (still on f)
        if lambd > 0:
            fval = fval + 0.5 * lambd * x.pow(2).sum()
            grad = grad + lambd * x
        # add L1 regularization (now we are on F)
        if alpha == 0:
            Fval, gradF = fval, grad
        else:
            Fval = fval + alpha * x.abs().sum()
            gradF = grad + alpha * x.sign()
        return fval.detach(), Fval.detach(), grad, gradF

    def dir_evaluate(x, t, d):
        """used for strong-wolfe line search"""
        x = (x + t * d).reshape(xshape)
        fval, Fval, grad, gradF = evaluate(x)
        return Fval, gradF.flatten()

    # compute initial f(x) and f'(x)
    _, Fval, grad, gradF = evaluate(x)
    gradF_norm = gradF.norm(normp)
    nfev = 1
    if disp > 1:
        print('initial loss: %0.4f' % Fval)
    if return_losses:
        losses = [Fval.item()]

    # initialize BFGS
    H = torch.diag_embed(torch.ones_like(x))  # [B,D,D]

    # BFGS iterations
    for k in range(1, maxiter + 1):
        # set the initial step size
        if k == 1:
            # use sample-specific learning rate for the first step,
            # unless we're doing a line search.
            t = (lr / gradF.abs().sum(1, keepdim=True)).clamp(None, lr)
            if line_search:
                t = t.mean().item()
        else:
            t = lr

        # compute newton direction
        if k == 1:
            # use - grad_F for the first step
            d = gradF.neg()
        else:
            # use - H^{-1} @ grad_F for remaining steps
            Hk = H
            if alpha > 0:
                Hk = Hk + torch.diag_embed(alpha * pinv(x.abs()))
            d = batch_cholesky_solve(gradF.neg(), Hk)

        # optional strong-wolfe line search
        if line_search:
            gtd = torch.sum(gradF * d)
            _, _, t, ls_nevals = \
                _strong_wolfe(dir_evaluate, x.flatten(), t, d.flatten(), Fval,
                              gradF.flatten(), gtd)
            nfev += ls_nevals

        # update variables and gradient
        x_new = x + t * d
        _, Fval_new, grad_new, gradF_new = evaluate(x_new)
        nfev += 1

        if disp > 1:
            print('iter %3d - loss: %0.4f' % (k, Fval_new))
        if return_losses:
            losses.append(Fval_new.item())

        # update \delta x and \delta f'(x)
        s = x_new - x
        y = grad_new - grad
        # now update current state
        x = x_new
        grad = grad_new
        Fval = Fval_new

        # stopping check
        gradF_norm = gradF.norm(normp)
        if gradF_norm <= gtol:
            return terminate(0, _status_message['success'])
        if Fval.isinf() or Fval.isnan():
            return terminate(2, _status_message['pr_loss'])

        # update the BFGS hessian approximation
        rho_inv = y.mul(s).sum(1, keepdim=True)
        valid = rho_inv.abs() > 1e-10
        if not valid.all():
            warnings.warn("Divide-by-zero encountered: rho assumed large")
        rho = torch.where(valid,
                          rho_inv.reciprocal(),
                          torch.full_like(rho_inv, 1000.))

        Hs = torch.bmm(H, s.unsqueeze(-1))
        H = torch.where(
            valid.unsqueeze(-1),
            torch.addcdiv(
                torch.baddbmm(H, (rho*y).unsqueeze(-1), y.unsqueeze(-2)),  # H - rho * y @ y^T
                torch.bmm(Hs, Hs.transpose(-1,-2)),  # Hs @ (Hs)^T
                torch.bmm(s.unsqueeze(-2), Hs),  # s^T @ Hs
                value=-1),
            H)

    # final sanity check
    if gradF_norm.isnan() or Fval.isnan() or x.isnan().any():
        return terminate(3, _status_message['nan'])

    # if we get to the end, the maximum num. iterations was reached
    return terminate(1, "Warning: " + _status_message['maxiter'])