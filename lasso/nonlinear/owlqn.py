import warnings
import torch
from torch import Tensor, autograd
from scipy.optimize import minimize_scalar


class L_BFGS(object):
    def __init__(self, x, g, history_size=100):
        super().__init__()
        self.y = []
        self.s = []
        self.rho = []
        self.H_diag = 1.
        self.alpha = x.new_empty(history_size)
        self.history_size = history_size
        self.x_prev = x.clone(memory_format=torch.contiguous_format)
        self.g_prev = g.clone(memory_format=torch.contiguous_format)
        self.n_updates = 0

    def solve(self, d):
        mem_size = len(self.y)
        dshape = d.shape
        d = d.view(-1).clone(memory_format=torch.contiguous_format)
        for i in reversed(range(mem_size)):
            self.alpha[i] = self.s[i].dot(d) * self.rho[i]
            d.add_(self.y[i], alpha=-self.alpha[i])
        d.mul_(self.H_diag)
        for i in range(mem_size):
            beta_i = self.y[i].dot(d) * self.rho[i]
            d.add_(self.s[i], alpha=self.alpha[i] - beta_i)

        return d.view(dshape)

    def update(self, x, g):
        s = (x - self.x_prev).view(-1)
        y = (g - self.g_prev).view(-1)
        rho_inv = y.dot(s)
        if rho_inv <= 1e-10:
            # curvature is negative; do not update
            return
        if len(self.y) == self.history_size:
            self.y.pop(0)
            self.s.pop(0)
            self.rho.pop(0)
        self.y.append(y)
        self.s.append(s)
        self.rho.append(rho_inv.reciprocal())
        self.H_diag = rho_inv / y.dot(y)
        self.x_prev.copy_(x, non_blocking=True)
        self.g_prev.copy_(g, non_blocking=True)
        self.n_updates += 1


def project(x, y):
    return x.masked_fill(x.sign() != y.sign(), 0)


def pseudo_grad(x, grad_f, alpha):
    grad_r = alpha * x.sign()
    grad_right = grad_f + grad_r.masked_fill(x == 0, alpha)
    grad_left = grad_f + grad_r.masked_fill(x == 0, -alpha)
    pgrad = torch.zeros_like(x)
    pgrad = torch.where(grad_right < 0, grad_right, pgrad)
    pgrad = torch.where(grad_left > 0, grad_left, pgrad)
    return pgrad


def backtracking(dir_evaluate, x, v, t, d, f, tol=0.1, decay=0.95, maxiter=500):
    for n_iter in range(1, maxiter + 1):
        x_new, f_new = dir_evaluate(x, t, d)
        if f_new <= f - tol * v.mul(x_new-x).sum():
            break
        t = t * decay
    else:
        warnings.warn('line search did not converge.')

    return t, n_iter


@torch.no_grad()
def owlqn(fun, x0, alpha=1., lr=1, max_iter=20, xtol=1e-5, history_size=100,
          line_search='brent', ls_options=None, verbose=0):
    """Orthant-wise limited-memory quasi-newton

    Parameters
    ----------
    fun : callable
        Objective function. Must output a scalar with grad_fn
    x0 : Tensor
        Initial value of the parameters
    alpha : float
        Sparsity weight of the Lasso problem
    lr : float
        Learning rate (default = 1)
    max_iter : int
        Maximum number of iterations (default = 20)
    xtol : float
        Termination tolerance on parameter changes
    history_size : int
        History size for L-BFGS memory updates (default = 100)
    line_search : str, optional
        Optional line search specifier

    Returns
    -------
    x : Tensor
        Final value of the parameters after optimization.

    """
    assert x0.dim() == 2
    verbose = int(verbose)
    if ls_options is None:
        ls_options = {}

    def evaluate(x):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = fun(x)
        # NOTE: do not include l1 penalty term in the gradient
        grad = autograd.grad(f, x)[0]
        f = f.detach() + alpha * x.norm(p=1)
        grad_pseudo = pseudo_grad(x, grad, alpha)
        return f, grad, grad_pseudo

    # evaluate initial f(x) and f'(x)
    x = x0.detach()
    f, g, g_pseudo = evaluate(x)
    if verbose:
        print('initial f: %0.4f' % f)

    # initialize
    lbfgs = L_BFGS(x, g, history_size)
    t = torch.clamp(lr / g_pseudo.norm(p=1), max=lr)
    delta_x = x.new_tensor(float('inf'))

    # optimize for a max of max_iter iterations
    for n_iter in range(1, max_iter + 1):

        # descent direction
        v = g_pseudo.neg()

        # compute quasi-newton direction
        d = lbfgs.solve(v)

        # project the quasi-newton direction
        d = project(d, v)

        # compute eta
        eta = torch.where(x == 0, v.sign(), x.sign())

        # perform line search to determine step size
        if line_search == 'brent':
            def line_obj(t):
                x_new = project(x.add(d, alpha=t), eta)
                f_new = fun(x_new) + alpha * x_new.norm(p=1)
                return float(f_new)

            res = minimize_scalar(line_obj, bounds=(0,10), method='bounded')
            t = res.x
            ls_iters = res.nfev

        elif line_search == 'backtrack':
            def dir_evaluate(x, t, d):
                x_new = project(x.add(d, alpha=t), eta)
                f_new = fun(x_new) + alpha * x_new.norm(p=1)
                return x_new, f_new

            t, ls_iters = backtracking(dir_evaluate, x, v, t, d, f, **ls_options)

        elif line_search == 'none':
            ls_iters = 0

        else:
            raise RuntimeError

        # update x
        x_new = project(x.add(d, alpha=t), eta)
        torch.norm(x_new - x, p=2, out=delta_x)
        x = x_new

        # re-evaluate
        f, g, g_pseudo = evaluate(x)
        if verbose > 1:
            print('iter %3d - ls_iters: %3d - f: %0.4f - dx: %0.3e'
                  % (n_iter, ls_iters, f, delta_x))

        # check for convergence
        if delta_x <= xtol:
            break

        # update hessian estimate
        lbfgs.update(x, g)
        t = lr

    if verbose:
        print("         Current function value: %f" % f)
        print("         Iterations: %d" % n_iter)

    return x