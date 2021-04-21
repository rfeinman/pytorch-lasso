import math
import torch
import torch.nn.functional as F


def split_bregman(A, b, x0=None, alpha=1.0, eps=1.0, maxiter=20, niter_inner=5,
                  tol=1e-10, tau=1., verbose=False):
    """Split Bregman for L1-regularized least squares.

    Parameters
    ----------
    A : torch.Tensor
        Linear transformation marix. Shape [n_features, n_components]
    b : torch.Tensor
        Reconstruction targets. Shape [n_samples, n_features]
    x0 : torch.Tensor, optional
        Initial guess at the solution. Shape [n_samples, n_components]
    alpha : float
        L1 Regularization strength
    eps : float
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
    assert b.dim() == 2
    assert A.dim() == 2
    assert b.shape[1] == A.shape[0]
    n_features, n_components = A.shape
    n_samples = b.shape[0]
    b = b.T.contiguous()
    mu = 1 / alpha

    # Rescale dampings
    epsR = eps / mu
    if x0 is None:
        x = b.new_zeros(n_components, n_samples)
    else:
        assert x0.shape == (n_samples, n_components)
        x = x0.T.clone(memory_format=torch.contiguous_format)

    # reg buffers
    c = torch.zeros_like(x)
    d = torch.zeros_like(x)

    # normal equations
    Atb = torch.matmul(A.T, b)
    AtA = torch.matmul(A.T, A)
    AtA.diagonal(dim1=-2, dim2=-1).add_(epsR)
    L = torch.cholesky(AtA)

    update = b.new_tensor(float('inf'))
    for itn in range(maxiter):
        if update <= tol:
            break

        xold = x.clone()
        for _ in range(niter_inner):
            # Regularized sub-problem
            Atb_i = Atb.add(d - c, alpha=epsR)
            torch.cholesky_solve(Atb_i, L, out=x)

            # Shrinkage
            d = F.softshrink(x + c, eps)

        # Bregman update
        c.add_(x - d, alpha=tau)

        # update norm
        torch.norm(x - xold, out=update)

        if verbose:
            cost = 0.5 * (A.matmul(x) - b).square().sum() + alpha * x.abs().sum()
            print('iter %3d - cost: %0.4f' % (itn, cost))

    x = x.T.contiguous()

    return x, itn