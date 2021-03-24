import warnings
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
import torch
import torch.autograd as autograd
from torch.optim.lbfgs import _strong_wolfe
from ptkit.optim.conjgrad import conjgrad, batch_conjgrad

Inf = float('inf')

status_message = {
    0: _status_message['success'],
    1: _status_message['maxiter'],
    2: "CG iterations didn't converge. The Hessian is not positive definite.",
    3: _status_message['nan']
}


def pinv(x, eps=1e-8):
    return x.reciprocal().masked_fill(x < eps, 0)


def _ncg_step(x, gradF, f, alpha, cg_options, twice_diffable, batch=True):
    # store hessian diagonal
    diag = alpha * pinv(x.abs())

    # Compute search direction with conjugate gradient (GG).
    # Instead of explicitly computing the hessian matrix, we use the
    # hessian-vector product, which is much more efficient
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        # compute f'(xk)
        gx = autograd.grad(f(x), x, create_graph=True)[0]

    if twice_diffable:
        hvp_ = lambda v: autograd.grad(gx, x, v, retain_graph=True)[0]
    else:
        with torch.enable_grad():
            grad_gx = torch.zeros_like(gx, requires_grad=True)
            grad_x = autograd.grad(gx, x, grad_gx, create_graph=True)[0]
        hvp_ = lambda v: autograd.grad(grad_x, grad_gx, v, retain_graph=True)[0]

    hvp = lambda v: hvp_(v) + diag * v

    if batch:
        d, cg_iters, cg_status = batch_conjgrad(
            gradF.neg(), hvp, return_info=True, **cg_options)
    else:
        dot = lambda u,v: torch.sum(u*v, 1, keepdim=True)
        d, cg_iters, cg_status = conjgrad(
            gradF.neg(), hvp, dot, return_info=True, **cg_options)
        cg_status = x.new_full((x.size(0), 1), cg_status, dtype=torch.long)

    return d.detach(), cg_iters, (cg_status == 4)


def backtracking(dir_evaluate, x, g, t, d, fval, tol=0.1, decay=0.98,
                 maxiter=1000):
    stop = x.new_zeros((x.shape[0], 1), dtype=torch.bool)
    for i in range(maxiter):
        x_new, fval_new = dir_evaluate(x, t, d)
        df = torch.sum(g * (x_new - x), 1, keepdim=True)
        stop = stop | (fval_new <= fval - tol * df)
        t = torch.where(stop, t, t * decay)
        if stop.all() or (t < 1e-4).any():
            break
    else:
        warnings.warn('backtracking did not converge.')

    return t, i


@torch.no_grad()
def iterative_ridge_ncg(f, x0, alpha=1.0, lambd=1e-4, lr=1.0, xtol=1e-5,
                        gtol=1e-5, normp=Inf, maxiter=None, cg_options=None,
                        twice_diffable=True, line_search=None, disp=0):
    """Newton Conjugate Gradient (CG)

    This version uses a "batch" variant that can be thought of as solving
    a single optimization problem for each entry in the batch

    Status messages:
        0: Succesful termination
        1: CG failure
        2: NaN or Inf encountered
        3: Maximum iterations reached

    Parameters
    ----------
    f : callable
        Scalar objective function to minimize
    x0 : Tensor
        Initialization point
    """
    if not x0.dim() == 2:
        raise NotImplementedError('batch NewtonCG is only supported for 2D '
                                  'input tensors at the time being.')

    # add tikhonov regularization to the objective
    f_ = f
    f = lambda x: f_(x) + 0.5 * lambd * x.pow(2).sum()
    F = lambda x: f(x) + alpha * x.abs().sum()

    def eval(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            Fval = F(x)
        gradF, = autograd.grad(Fval, x)
        return Fval.detach(), gradF

    def dir_eval(x, t, d):
        """used for backtracking line search"""
        x_new = x + t * d
        return x_new, F(x_new)

    def dir_eval_flat(x, t, d):
        """used for strong-wolfe line search"""
        x = (x + t * d).reshape(x0.shape)
        Fval, gradF = eval(x)
        return Fval, gradF.flatten()

    numel = x0[0].numel()
    if maxiter is None:
        maxiter = numel * 200
    if cg_options is None:
        cg_options = {}
    cg_options.setdefault('maxiter', numel * 20)
    cg_options.setdefault('rtol', 1.)
    if 'batch' in cg_options:
        cg_batch = cg_options['batch']
        del cg_options['batch']
    else:
        cg_batch = True

    # initialize
    x = x0.detach().clone(memory_format=torch.contiguous_format)
    Fval, gradF = eval(x)
    nfev = 0  # number of function/gradient evaluations
    ncg = 0   # number of cg iterations
    k = 0
    status = x.new_full((x.size(0), 1), 1, dtype=torch.long)
    delta_x = torch.full_like(x, xtol)

    if disp > 1:
        print('initial loss: %0.4f' % Fval)

    # begin optimization loop
    for k in range(1, maxiter + 1):
        if not torch.any(status == 1):
            break

        # Newton direction (computed with CG)
        d, cg_iters, cg_fail = \
            _ncg_step(x, gradF, f, alpha, cg_options, twice_diffable, cg_batch)

        # check for CG failure
        status.masked_fill_(torch.logical_and(cg_fail, status==1), 2)

        # compute update.
        if line_search is not None:
            if line_search == 'strong_wolfe':
                gtd = torch.sum(gradF * d)
                _, _, t, ls_nevals = \
                    _strong_wolfe(dir_eval_flat, x.flatten(), lr, d.flatten(),
                                  Fval, gradF.flatten(), gtd)
            elif line_search == 'backtrack':
                t = x.new_full((x.shape[0], 1), lr)
                t, ls_nevals = backtracking(dir_eval, x, gradF, t, d, Fval)
            else:
                raise ValueError('invalid line search specifier')
            dx = t * d
            nfev += ls_nevals
        else:
            dx = lr * d

        infinite = torch.logical_or(dx.isnan(), dx.isinf()).any(1, keepdim=True)
        status.masked_fill_(torch.logical_and(infinite, status==1), 3)

        # update variables
        do_update = status == 1
        delta_x = torch.where(do_update, dx, delta_x)
        x = torch.where(do_update, x + dx, x)
        Fval, gradF = eval(x)
        nfev += 1
        ncg += cg_iters

        if disp > 1:
            print('iter %3d - loss: %0.4f' % (k, Fval))

        converged = torch.logical_or(
            (delta_x.abs().mean(1, keepdim=True) <= xtol),
            (gradF.norm(normp, 1, keepdim=True) <= gtol))
        status.masked_fill_(torch.logical_and(converged, status==1), 0)


    if disp:
        print("    Current function value: %f" % Fval)
        print("    Iterations: %d" % k)
        print("    Function evaluations: %d" % nfev)
        print("    CG iterations: %d" % ncg)
        print("    Status:")
        s_val, s_count = status.unique(return_counts=True)
        for val, count in zip(s_val, s_count):
            print("        " + ("%i: " % count) + status_message[val.item()])

    return OptimizeResult(x=x, fun=Fval, niter=k, nfev=nfev, ncg=ncg,
                          status=status)