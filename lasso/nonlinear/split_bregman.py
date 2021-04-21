import warnings
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch._vmap_internals import _vmap

# TODO: remove dependence on ptkit
try:
    from ptkit.linalg.sparse import LinearOperator, JacobianLinearOperator, cg
except:
    warnings.warn('cannot find package ptkit! split_bregman cannot be used '
                  'with option solver="cg".')


def _lstsq_exact(fun_with_jac, x, d, b, max_iter=5, mu=1., lambd=1., lr=1.):
    for _ in range(max_iter):
        f, J = fun_with_jac(x)
        grad = mu * J.T.mv(f)
        grad.add_(d - x - b, alpha=-lambd)
        JtJ = mu * J.T.mm(J)
        JtJ.diagonal().add_(lambd)
        try:
            p = torch.cholesky_solve(grad.unsqueeze(1),
                                     torch.cholesky(JtJ)).squeeze(1)
        except RuntimeError as exc:
            if 'singular' not in exc.args[0]:
                raise
            p = torch.linalg.solve(JtJ, grad)
        x.add_(p, alpha=-lr)

    return x


def _lstsq_cg(fun, x, d, b, max_iter=5, mu=1., lambd=1., lr=1., **cg_kwargs):
    for _ in range(max_iter):
        J = JacobianLinearOperator(fun, x)
        f = J.f.detach()
        grad = mu * J.rmv(f)
        grad.add_(d - x - b, alpha=-lambd)
        JtJ = LinearOperator(shape=(x.numel(), x.numel()),
                             mv=lambda v: mu * J.rmv(J.mv(v)) + lambd * v)
        p = cg(JtJ, grad, **cg_kwargs)[0]
        x.add_(p, alpha=-lr)

    return x


@torch.no_grad()
def split_bregman(
        fun, x0, lr=1.,
        alpha=1., lambd=1., tau=1.,
        max_iter=None, inner_iter=5, lstsq_iter=5,
        xtol=1e-5, disp=0,
        solver='cg', cg_kwargs=None):
    f0 = fun(x0)
    input_size = x0.numel()
    output_size = f0.numel()
    lr = float(lr)
    disp = int(disp)
    xtol = input_size * xtol
    if max_iter is None:
        max_iter = min(input_size, output_size)
    if cg_kwargs is None:
        cg_kwargs = {}

    def cost_fn(x):
        x = x.view_as(x0)
        return 0.5 * fun(x).square().sum() + alpha * x.abs().sum()

    if solver == 'exact':
        I = torch.eye(output_size, dtype=x0.dtype, device=x0.device)

        def fun_with_jac(x):
            x = x.view_as(x0).detach().requires_grad_(True)
            with torch.enable_grad():
                f = fun(x).view(output_size)
            J = _vmap(lambda v: autograd.grad(f, x, v)[0])(I)
            J = J.view(output_size, input_size)
            return f.detach(), J

        def lstsq_subproblem(x, d, b):
            return _lstsq_exact(fun_with_jac, x, d, b, max_iter=lstsq_iter,
                                mu=1/alpha, lambd=lambd, lr=lr)

    elif solver == 'cg':
        def lstsq_subproblem(x, d, b):
            return _lstsq_cg(fun, x, d, b, max_iter=lstsq_iter,
                            mu=1/alpha, lambd=lambd, lr=lr, **cg_kwargs)

    else:
        raise ValueError('Expected `solver` to be one of "exact" or "cg" '
                         'but got {}'.format(solver))

    # initial settings
    x = x0.detach().clone(memory_format=torch.contiguous_format)
    if solver == 'exact':
        x = x.view(-1)
    cost = cost_fn(x)
    if disp:
        print('initial cost: %0.4f' % cost)

    # bregman parameters
    b = torch.zeros_like(x)
    d = torch.zeros_like(x)

    nit_inner = 0
    update = x.new_tensor(float('inf'))
    for nit in range(1, max_iter + 1):
        if ((update <= xtol) | ~cost.isfinite()):
            break

        xold = x.clone()
        for _ in range(inner_iter):
            nit_inner += 1

            # Regularized nonlinear least squares sub-problem
            x = lstsq_subproblem(x, d, b)

            # Shrinkage
            d = F.softshrink(x + b, 1 / lambd)

            if disp > 2:
                print('   iter %3d - cost: %0.4f' % (nit_inner, cost_fn(x)))

        # Bregman update
        b.add_(x - d, alpha=tau)

        # update norm
        torch.norm(x - xold, p=2, out=update)

        # re-compute cost
        cost = cost_fn(x)

        if disp > 1:
            print('iter %3d - cost: %0.4f' % (nit, cost))

    if disp:
        print('final cost: %0.4f' % cost)

    return x.view_as(x0)