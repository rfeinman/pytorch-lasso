import torch
import torch.nn.functional as F

_status_messages = {
    0: 'Absolute tolerance reached.',
    1: 'Relative tolerance reached.',
    2: 'Curvature has converged.',
    3: 'Curvature is negative.',
    4: 'Maximum iterations reached.'
}


def conjgrad(b, Adot, dot, maxiter=None, tol=1e-10, rtol=1e-1, verbose=False):
    if maxiter is None:
        maxiter = 20 * (b.numel() if b.dim() == 1 else b[0].numel())
    b_abs = b.abs().sum()
    termcond = rtol * b_abs * b_abs.sqrt().clamp(0, 0.5)
    float_eps = torch.finfo(b.dtype).eps

    # initialize x
    x = torch.zeros_like(b)

    def terminate(warnflag):
        if verbose:
            print(_status_messages[warnflag])
        return x

    # iterate
    r = -b
    p = b
    rs_old = dot(r, r)
    for i in range(maxiter):
        if r.abs().sum() <= termcond:
            return terminate(1)
        Ap = Adot(p)
        curv = dot(p, Ap)
        curv_sum = curv.sum()
        if 0 <= curv_sum <= 3 * float_eps:
            return terminate(2)
        elif curv_sum < 0:
            if i == 0:
                # fall back to steepest descent direction
                x = - rs_old / curv * b
            return terminate(3)
        alpha = rs_old / curv
        x = x + alpha * p
        r = r + alpha * Ap
        rs_new = dot(r, r)
        if rs_new.sum().sqrt() < tol:
            return terminate(0)
        p = - r + (rs_new / rs_old) * p
        rs_old = rs_new
        if verbose:
            print('iter: %i - rs: %0.4f' % (i, rs_new.sum().sqrt()))

    return terminate(4)


def cg(A, b, maxiter=None, tol=1e-10, rtol=1.0, verbose=False):
    assert A.dim() == 2
    assert b.dim() == 1
    if maxiter is None:
        maxiter = 20 * len(b)

    Adot = lambda v: A.matmul(v)
    dot = lambda u,v: u.dot(v)

    return conjgrad(b, Adot, dot, maxiter, tol, rtol, verbose)


def batch_cg(A, b, maxiter=None, tol=1e-10, rtol=1.0, verbose=False):
    assert A.dim() == 2
    assert b.dim() == 2
    if maxiter is None:
        maxiter = 20 * b.size(1)

    Adot = lambda v: torch.mm(v, A.T)
    dot = lambda u,v: torch.sum(u*v, 1, keepdim=True)

    return conjgrad(b, Adot, dot, maxiter, tol, rtol, verbose)


def batch_cg_conv2d(kernel, b, tik=0, maxiter=None, tol=1e-10, rtol=1.0,
                    verbose=False, **conv_kwargs):
    """
    Assume:
        A = W.T @ W + tik * I

    where W is the toeplitz matrix of the conv-transpose operation:
        y = Wx = conv_transpose2d(x, kernel, **conv_kwargs)
    """
    assert kernel.dim() == 4
    assert b.dim() == 4
    if maxiter is None:
        maxiter = 20 * b[0].numel()

    def Adot(v):
        Av = F.conv_transpose2d(v, kernel, **conv_kwargs)
        Av = F.conv2d(Av, kernel, **conv_kwargs)
        if tik > 0:
            Av = Av + tik * v
        return Av

    dot = lambda u,v: torch.sum(u*v, [1,2,3], keepdim=True)

    return conjgrad(b, Adot, dot, maxiter, tol, rtol, verbose)