import math
import torch
import torch.nn.functional as F
import torch.autograd as autograd


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


def softshrink(x, theta):
    """modified variant of F.softshrink that supports non-scalar theta"""
    return x.sign() * F.relu(x.abs() - theta)


def hessian_2norm(fun, x, niter=10):
    """Estimate the Hessian 2-norm using power iteration."""
    assert x.dim() >= 2
    feature_dim = list(range(1, x.dim()))

    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        f = fun(x)
        assert f.numel() == 1
        g = autograd.grad(f, x, create_graph=True)[0]

    # create jvp and vjp functions
    grad_g = torch.zeros_like(g, requires_grad=True)
    with torch.enable_grad():
        grad_x = autograd.grad(g, x, grad_g, create_graph=True)[0]
    jvp = lambda r: autograd.grad(grad_x, grad_g, r, retain_graph=True)[0]
    vjp = lambda r: autograd.grad(g, x, r, retain_graph=True)[0]

    # power iterations
    u = F.normalize(torch.randn_like(x), dim=feature_dim, eps=1e-8)
    for _ in range(niter):
        v = F.normalize(jvp(u), dim=feature_dim, eps=1e-8)
        u = F.normalize(vjp(v), dim=feature_dim, eps=1e-8)

    sigma = torch.sum(v * jvp(u), dim=feature_dim)

    return sigma


def ista_nl(x, z0, decoder, alpha=1.0, fast=True, maxiter=10, lr='auto',
            power_iters=10, tol=1e-5, eval_mode=True, verbose=0):
    # check arguments
    if not (lr == 'auto' or isinstance(lr, float)):
        raise ValueError('expected `lr` to be either float or "auto".')

    # configure decoder
    requires_grad = _freeze_grad(decoder)
    training = decoder.training
    if eval_mode:
        # set decoder to eval mode (in case dropout, batchnorm etc.)
        decoder.eval()
    verbose = int(verbose)
    tol = z0.numel() * tol

    def rss_loss(zk):
        return 0.5 * (decoder(zk) - x).pow(2).sum()

    def lasso_loss(zk):
        return rss_loss(zk) + alpha * zk.abs().sum()

    # derivative of the residual sum-of-squares (rss) objective
    def rss_grad_fn(zk):
        zk = zk.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = 0.5 * (x - decoder(zk)).pow(2).sum()
        grad, = autograd.grad(loss, zk)
        return grad

    # ista step function
    def step(zk):
        zk_grad = rss_grad_fn(zk)
        if lr == 'auto':
            # set lr based on lipschitz constant of \grad rss(z)
            L = hessian_2norm(rss_loss, zk, niter=power_iters)
            t = 0.98 / L
            for _ in range(1, zk.dim()):
                t = t.unsqueeze(-1)
        else:
            t = lr
        return softshrink(zk - t * zk_grad, alpha * t)

    if verbose:
        print('initial loss: %0.4f' % lasso_loss(z0))

    # optimize
    z = z0.detach()
    if fast:
        y = z0.detach()
        t = 1
    for niter in range(1, maxiter + 1):
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
        if verbose > 1:
            print('iter %3d - loss: %0.4f' % (niter, lasso_loss(z)))

    if verbose:
        print('final loss: %0.4f' % lasso_loss(z))

    # re-configure decoder
    _unfreeze_grad(decoder, requires_grad)
    if eval_mode and training:
        decoder.train()

    return z.detach()