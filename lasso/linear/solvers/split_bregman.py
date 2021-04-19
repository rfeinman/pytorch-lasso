import math
import torch
import torch.nn.functional as F


def split_bregman(A, b, x0=None, alpha=1.0, maxiter=20, niter_inner=5,
                  mu=1., tol=1e-10, tau=1.):
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
    maxiter : int
        Number of iterations of outer loop
    niter_inner : int
        Number of iterations of inner loop
    mu : float, optional
         Data term damping
    tol : float, optional
        Tolerance. Stop outer iterations if difference between inverted model
        at subsequent iterations is smaller than ``tol``
    tau : float, optional
        Scaling factor in the Bregman update (must be close to 1)

    Returns
    -------
    xinv : torch.Tensor
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

    # Rescale dampings
    epsR = math.sqrt(alpha / 2) / math.sqrt(mu / 2)
    if x0 is None:
        xinv = b.new_zeros(n_components, n_samples)
    else:
        assert x0.shape == (n_samples, n_components)
        xinv = x0.T.clone(memory_format=torch.contiguous_format)

    # reg buffers
    c = torch.zeros_like(xinv)
    d = torch.zeros_like(xinv)

    # normal equations
    Atb = torch.matmul(A.T, b)
    AtA = torch.matmul(A.T, A)
    AtA.diagonal(dim1=-2, dim2=-1).add_(epsR ** 2)
    L = torch.cholesky(AtA)

    update = b.new_tensor(float('inf'))
    for itn in range(maxiter):
        if update <= tol:
            break

        xold = xinv.clone()
        for _ in range(niter_inner):
            # Regularized sub-problem
            Atb_i = Atb.add(d - c, alpha=epsR ** 2)
            torch.cholesky_solve(Atb_i, L, out=xinv)

            # Shrinkage
            d = F.softshrink(xinv + c, alpha)

        # Bregman update
        c.add_(xinv - d, alpha=tau)

        # update norm
        torch.norm(xinv - xold, out=update)

    xinv = xinv.T.contiguous()

    return xinv, itn