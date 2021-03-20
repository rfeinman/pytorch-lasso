from torch import Tensor
import torch
import torch.nn.functional as F

from ..utils import lstsq, batch_cholesky_solve


def _check_inputs(x, weight, z0):
    assert torch.is_tensor(x) and x.dim() == 2
    assert torch.is_tensor(weight) and weight.dim() == 2
    batch_size, input_size = x.shape
    if z0 is not None:
        assert torch.is_tensor(z0) and z0.dim() == 2
        batch_size1, code_size = z0.shape
        assert batch_size1 == batch_size
        assert weight.shape == (input_size, code_size)
    else:
        input_size1, code_size = weight.shape
        assert input_size1 == input_size
    return batch_size, input_size, code_size


def _general_inverse(x, eps):
    #return x.reciprocal().masked_fill(x.abs() < eps, eps)
    #return x.reciprocal().clamp(-1/eps, 1/eps)
    return x.reciprocal().masked_fill(x.abs() < eps, 0)


def interior_point(
        x, weight, z0=None, alpha=1.0, maxiter=20, barrier_init=0.1,
        tol=1e-8, tol_change=1e-8, eps=1e-5, verbose=False):
    """Interior point method, non-negative variables with log barrier

    Explained in section 2.3 of Mark Schmidt, 2005:
        "Least Squares Optimization with L1-Norm Regularization."

    Originally proposed in Chen et al., 2001:
        "Atomic Decomposition by Basis Pursuit"

    Parameters
    ----------
    x : Tensor
        Initial code vectors of shape [batch, inp_size].
    weight : Tensor
        Dictionary matrix of shape [inp_size, code_size].
    z0 : Tensor, optional
        Initial code vectors of shape [batch, code_size]. If not provided,
        the least-norm solution will be used for initial point.
    alpha : float
        Sparsity weight of the Lasso problem.
    maxiter : int
        Maximum number of iterations. Default=20
    barrier_init : float
        Initial barrier weight. Will be decreased gradually.
    tol : float
        Tolerance for primal/dual feasibility and duality gap (optimality).
    tol_change : float
        Tolerance for variable changes.
    eps : float
        Minimum value threshold for generalized inverse
    verbose : bool
        Verbosity control. Set to True for loss tracking

    Returns
    -------
    zf : Tensor
        Final code vectors after optimization.
    success : bool
        Boolean indicator for whether the optimization completed successfully.
    """
    batch_size, input_size, code_size = _check_inputs(x, weight, z0)
    if z0 is None:
        z0 = lstsq(x.T, weight).T

    # expand z and weight to pos/neg components
    z0 = torch.cat([F.relu(z0), F.relu(-z0)], 1)  # [B,2K]
    weight = torch.cat([weight, -weight], 1)  # [D,2K]

    # barrier parameter
    mu = barrier_init * x.new_ones(batch_size, 1)  # [B,1]

    # primal variable
    z = z0  # [D,2K]

    # dual variable
    lmbda = x - torch.matmul(z, weight.T)  # [B,D]

    # dual slack variable
    s = alpha - torch.matmul(lmbda, weight)  # [B,2K]

    def f(z_k, lmbda_k):
        return alpha * z_k.sum() + 0.5 * lmbda_k.pow(2).sum()

    # Optimize
    success = False
    for i in range(maxiter):
        if verbose:
            print('loss_%0.2i: %0.4f' % (i, f(z, lmbda)))

        # ---------------------------
        #   KKT condition values
        # ---------------------------

        ra = - torch.matmul(lmbda, weight) - s + alpha  # [B,2K]
        rb = x - torch.matmul(z, weight.T) - lmbda  # [B,D]
        rc = mu - z * s  # [B,2K]


        # ---------------------------
        #     Newton directions
        # ---------------------------
        s_inv = _general_inverse(s, eps)
        d = s_inv * z  # [B, 2K]

        # direction for lambda (use cholesky solve)
        rhs = s_inv * rc - d * ra
        rhs = rb - torch.matmul(rhs, weight.T)  # [B,D]
        # compute d_lmbda = (I + WDW^T)^{-1} rhs
        #M = torch.matmul(weight, d.unsqueeze(2) * weight.T.unsqueeze(0))  # [B,D,D]
        #M.diagonal(dim1=1, dim2=2).add_(1)
        #d_lmbda = batch_cholesky_solve(rhs, M)  # [B,D]
        d_inv = s * _general_inverse(z, eps)
        M = torch.matmul(weight.T, weight)
        M = M.repeat(len(d_inv), 1, 1)
        M.diagonal(dim1=1, dim2=2).add_(d_inv)
        d_lmbda = torch.matmul(rhs, weight)
        d_lmbda = batch_cholesky_solve(d_lmbda, M)
        d_lmbda = rhs - torch.matmul(d_lmbda, weight.T)

        # direction for s
        d_s = ra - torch.matmul(d_lmbda, weight)

        # direction for z
        d_z = s_inv * (rc - z * d_s)


        # --------------------------
        #     Variable updates
        # --------------------------
        # step sizes
        z_ratio = -z / d_z
        z_ratio.masked_fill_(d_z >= 0, float('inf'))
        beta_z = z_ratio.min(1, keepdim=True)[0]  # [B,1]
        s_ratio = -s / d_s
        s_ratio.masked_fill_(d_s >= 0, float('inf'))
        beta_sl = s_ratio.min(1, keepdim=True)[0]  # [B,1]

        # include possibility of a full Newton step
        beta_z.clamp_(float('-inf'), 1)
        beta_sl.clamp_(float('-inf'), 1)

        # update variables
        update_z = 0.99 * beta_z * d_z
        update_lmbda = 0.99 * beta_sl * d_lmbda
        update_s = 0.99 * beta_sl * d_s
        z += update_z
        lmbda += update_lmbda
        s += update_s

        if max(update_z.abs().max(), update_lmbda.abs().max(),
               update_s.abs().max()) < tol_change:
            break

        # update barrier weight
        mu *= 1 - torch.min(beta_z, beta_sl).clamp(float('-inf'), 0.99)


        # -------------------------------
        #     Check stopping criteria
        # -------------------------------

        # TODO: seperate convergence checks for each batch entry?
        z_norm = z.norm()
        lmbda_norm = lmbda.norm()
        primal_feas = rb.norm() / (1 + z_norm)
        dual_feas = ra.norm() / (1 + lmbda_norm)
        duality_gap = (z*s).sum() / (1 + z_norm * lmbda_norm)
        if (primal_feas < tol) and (dual_feas < tol) and (duality_gap < tol):
            success = True
            break

    z_pos, z_neg = z.chunk(2, dim=1)
    zf = F.relu(z_pos) - F.relu(z_neg)

    return zf, success