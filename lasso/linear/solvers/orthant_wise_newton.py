import warnings
import torch
from scipy.optimize import minimize_scalar


def project(u, v):
    return u.masked_fill(u.sign() != v.sign(), 0)


def pseudo_grad(z, grad_f, alpha):
    grad_r = alpha * z.sign()
    grad_right = grad_f + grad_r.masked_fill(z == 0, alpha)
    grad_left = grad_f + grad_r.masked_fill(z == 0, -alpha)
    pgrad = torch.zeros_like(z)
    pgrad = torch.where(grad_right < 0, grad_right, pgrad)
    pgrad = torch.where(grad_left > 0, grad_left, pgrad)
    return pgrad


def backtracking(dir_evaluate, z, v, t, d, f, tol=0.1, decay=0.95, maxiter=500):
    for n_iter in range(1, maxiter + 1):
        z_new, f_new = dir_evaluate(z, t, d)
        if f_new <= f - tol * v.mul(z_new - z).sum():
            break
        t = t * decay
    else:
        warnings.warn('line search did not converge.')

    return t, n_iter


@torch.no_grad()
def orthant_wise_newton(
        weight, x, z0, alpha=1., lr=1., maxiter=20, xtol=1e-5,
        line_search='brent', ls_options=None, verbose=0):
    """Orthant-wise Newton

    This is a modification of the "Orthant-wise limited-memory quasi-newton"
    method originally designed for nonlinear lasso problems. In this
    version we use the explicit RSS hessian matrix, computed in analytical form

    Parameters
    ----------
    weight : torch.Tensor
        Dictionary matrix. Shape [n_features, n_components]
    x : torch.Tensor
        Regression target vectors. Shape [n_samples, n_features]
    z0 : torch.Tensor
        Initial code vectors. Shape [n_samples, n_components]
    alpha : float
        Sparsity weight of the Lasso problem
    lr : float
        Learning rate (default = 1)
    maxiter : int
        Maximum number of iterations (default = 20)
    xtol : float
        Termination tolerance on parameter changes
    line_search : str
        Line search specifier
    ls_options : dict, optional
        Dictionary of keyword arguments for backtracking line search. Ignored
        unless line_search='backtrack'.

    Returns
    -------
    z : Tensor
        Final value of the parameters after optimization.

    """
    assert z0.dim() == 2
    verbose = int(verbose)
    if ls_options is None:
        ls_options = {}
    if not line_search in ['brent', 'backtrack', 'none']:
        raise ValueError("line_search must be one of {'brent', 'backtrack', 'none'}.")

    def evaluate(z):
        resid = torch.mm(z, weight.T) - x
        f = 0.5 * resid.square().sum() + alpha * z.norm(p=1)
        # NOTE: do not include l1 penalty term in the gradient
        grad = torch.mm(resid, weight)
        grad_pseudo = pseudo_grad(z, grad, alpha)
        return f, grad, grad_pseudo

    hess = torch.mm(weight.T, weight)
    hess.diagonal().add_(1e-4)
    hess_inv = torch.cholesky_inverse(torch.cholesky(hess))

    # evaluate initial f(x) and f'(x)
    z = z0.detach()
    f, g, g_pseudo = evaluate(z)
    if verbose:
        print('initial f: %0.4f' % f)

    # initialize
    delta_z = z.new_tensor(float('inf'))

    # optimize for a max of max_iter iterations
    for n_iter in range(1, maxiter + 1):

        # descent direction
        v = g_pseudo.neg()

        # compute quasi-newton direction
        d = torch.mm(v, hess_inv.T)

        # project the quasi-newton direction
        d = project(d, v)

        # compute eta
        eta = torch.where(z == 0, v.sign(), z.sign())

        # perform line search to determine step size
        if line_search == 'brent':
            def line_obj(t):
                z_new = project(z.add(d, alpha=t), eta)
                resid = torch.mm(z_new, weight.T) - x
                f_new = 0.5 * resid.square().sum() + alpha * z_new.norm(p=1)
                return float(f_new)

            res = minimize_scalar(line_obj, bounds=(0,10), method='bounded')
            t = res.x
            ls_iters = res.nfev

        elif line_search == 'backtrack':
            def dir_evaluate(z, t, d):
                z_new = project(z.add(d, alpha=t), eta)
                resid = torch.mm(z_new, weight.T) - x
                f_new = 0.5 * resid.square().sum() + alpha * z_new.norm(p=1)
                return z_new, f_new

            t, ls_iters = backtracking(dir_evaluate, z, v, lr, d, f, **ls_options)

        elif line_search == 'none':
            t = lr
            ls_iters = 0

        else:
            raise RuntimeError('invalid line_search parameter encountered.')

        # update z
        z_new = project(z.add(d, alpha=t), eta)
        torch.norm(z_new - z, p=2, out=delta_z)
        z = z_new

        # re-evaluate
        f, g, g_pseudo = evaluate(z)
        if verbose > 1:
            print('iter %3d - ls_iters: %3d - f: %0.4f - dz: %0.3e'
                  % (n_iter, ls_iters, f, delta_z))

        # check for convergence
        if delta_z <= xtol:
            break

    if verbose:
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)

    return z