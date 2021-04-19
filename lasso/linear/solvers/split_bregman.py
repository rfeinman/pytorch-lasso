import math
import torch
import torch.nn.functional as F


def split_bregman(A, b, x0=None, alpha=1.0, niter_outer=3, niter_inner=5,
                  mu=1., tol=1e-10, tau=1.):
    """Split Bregman for L1-regularized least squares.

    Parameters
    ----------
    A : torch.Tensor
        Linear transformation marix
    b : torch.Tensor
        Reconstruction target
    x0 : torch.Tensor, optional
        Initial guess at the solution
    alpha : float
         L1 Regularization strength
    niter_outer : int
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
        Inverted model
    itn_out : int
        Iteration number of outer loop upon termination

    """
    assert b.shape[0] == A.shape[0]
    squeeze = b.dim() == 1
    if squeeze:
        b = b.unsqueeze(-1)
    batch_shape = b.shape[1:]

    # Rescale dampings
    epsR = math.sqrt(alpha / 2) / math.sqrt(mu / 2)
    if x0 is None:
        xinv = b.new_zeros(A.shape[1], *batch_shape)
    else:
        assert x0.shape == (A.shape[1], ) + batch_shape
        xinv = x0.clone()

    # reg buffers
    c = torch.zeros_like(xinv)
    d = torch.zeros_like(xinv)

    # normal equations
    Atb = torch.matmul(A.T, b)
    AtA = torch.matmul(A.T, A)
    AtA.diagonal(dim1=-2, dim2=-1).add_(epsR ** 2)
    L = torch.cholesky(AtA)

    itn_out = 0
    update = b.new_tensor(float('inf'))
    while update > tol and itn_out < niter_outer:
        xold = xinv
        for _ in range(niter_inner):
            # Regularized sub-problem
            Atb_i = Atb.add(d - c, alpha=epsR ** 2)
            xinv = torch.cholesky_solve(Atb_i, L)

            # Shrinkage
            d = F.softshrink(xinv + c, alpha)

        # Bregman update
        c.add_(xinv - d, alpha=tau)
        itn_out += 1

        # update norm
        torch.norm(xinv - xold, out=update)

    if squeeze:
        xinv = xinv.squeeze(-1)

    return xinv, itn_out