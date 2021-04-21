import math
import torch
import torch.nn.functional as F


def split_bregman(A, y, x0=None, alpha=1.0, lambd=1.0, maxiter=20, niter_inner=5,
                  tol=1e-10, tau=1., verbose=False):
    """Split Bregman for L1-regularized least squares.

    Parameters
    ----------
    A : torch.Tensor
        Linear transformation marix. Shape [n_features, n_components]
    y : torch.Tensor
        Reconstruction targets. Shape [n_samples, n_features]
    x0 : torch.Tensor, optional
        Initial guess at the solution. Shape [n_samples, n_components]
    alpha : float
        L1 Regularization strength
    lambd : float
        Dampening term; constraint penalty strength
    maxiter : int
        Number of iterations of outer loop
    niter_inner : int
        Number of iterations of inner loop
    tol : float, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    tau : float, optional
        Scaling factor in the Bregman update (must be close to 1)

    Returns
    -------
    x : torch.Tensor
        Sparse coefficients. Shape [n_samples, n_components]
    itn_out : int
        Iteration number of outer loop upon termination

    """
    assert y.dim() == 2
    assert A.dim() == 2
    assert y.shape[1] == A.shape[0]
    n_features, n_components = A.shape
    n_samples = y.shape[0]
    y = y.T.contiguous()
    mu = 1 / alpha

    # Rescale dampings
    epsR = lambd / mu
    if x0 is None:
        x = y.new_zeros(n_components, n_samples)
    else:
        assert x0.shape == (n_samples, n_components)
        x = x0.T.clone(memory_format=torch.contiguous_format)

    # reg buffers
    b = torch.zeros_like(x)
    d = torch.zeros_like(x)

    # normal equations
    Aty = torch.matmul(A.T, y)
    AtA = torch.matmul(A.T, A)
    AtA.diagonal(dim1=-2, dim2=-1).add_(epsR)
    L = torch.cholesky(AtA)

    update = y.new_tensor(float('inf'))
    for itn in range(maxiter):
        if update <= tol:
            break

        xold = x.clone()
        for _ in range(niter_inner):
            # Regularized sub-problem
            Aty_i = Aty.add(d - b, alpha=epsR)
            torch.cholesky_solve(Aty_i, L, out=x)

            # Shrinkage
            d = F.softshrink(x + b, lambd)

        # Bregman update
        b.add_(x - d, alpha=tau)

        # update norm
        torch.norm(x - xold, out=update)

        if verbose:
            cost = 0.5 * (A.matmul(x) - y).square().sum() + alpha * x.abs().sum()
            print('iter %3d - cost: %0.4f' % (itn, cost))

    x = x.T.contiguous()

    return x, itn