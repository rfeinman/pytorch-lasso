import torch
import torch.nn.functional as F


def coord_descent(x, W, z0=None, alpha=1.0, maxiter=1000, tol=1e-6, verbose=False):
    input_dim, code_dim = W.shape  # [D,K]
    batch_size, input_dim1 = x.shape  # [N,D]
    assert input_dim1 == input_dim
    tol = tol * code_dim
    if z0 is None:
        z = x.new_zeros(batch_size, code_dim)  # [N,K]
    else:
        assert z0.shape == (batch_size, code_dim)
        z = z0

    # initialize b
    # TODO: how should we initialize b when 'z0' is provided?
    b = torch.mm(x, W)  # [N,K]

    # precompute S = I - W^T @ W
    S = - torch.mm(W.T, W)  # [K,K]
    S.diagonal().add_(1.)

    # loss function
    def fn(z):
        x_hat = torch.matmul(z, W.T)
        loss = 0.5 * (x_hat - x).pow(2).sum() + alpha * z.abs().sum()
        return loss

    # update function
    def cd_update(z, b):
        z_next = F.softshrink(b, alpha)  # [N,K]
        z_diff = z_next - z  # [N,K]
        k = z_diff.abs().argmax(1)  # [N]
        kk = k.unsqueeze(1)  # [N,1]
        b = b + S[:,k].T * z_diff.gather(1, kk)  # [N,K] += [N,K] * [N,1]
        z = z.scatter(1, kk, z_next.gather(1, kk))
        return z, b

    active = torch.arange(batch_size, device=W.device)
    for i in range(maxiter):
        if len(active) == 0:
            break
        z_old = z[active]
        z_new, b[active] = cd_update(z_old, b[active])
        update = (z_new - z_old).abs().sum(1)
        z[active] = z_new
        active = active[update > tol]
        if verbose:
            print('iter %i - loss: %0.4f' % (i, fn(F.softshrink(b, alpha))))

    z = F.softshrink(b, alpha)

    return z


def coord_descent_mod(x, W, z0=None, alpha=1.0, max_iter=1000, tol=1e-4):
    """Modified variant of the CD algorithm

    Based on `enet_coordinate_descent` from sklearn.linear_model._cd_fast

    This version is much slower, but it produces more reliable results
    as compared to the above.

    x : Tensor of shape [n_samples, n_features]
    W : Tensor of shape [n_features, n_components]
    z : Tensor of shape [n_samples, n_components]
    """
    n_features, n_components = W.shape
    n_samples = x.shape[0]
    assert x.shape[1] == n_features
    if z0 is None:
        z = x.new_zeros(n_features, n_components)  # [N,K]
    else:
        assert z0.shape == (n_features, n_components)
        z = z0

    gap = z.new_full((n_samples,), tol + 1.)
    converged = z.new_zeros(n_samples, dtype=torch.bool)
    d_w_tol = tol
    tol = tol * x.pow(2).sum(1)  # [N,]

    # compute squared norms of the columns of X
    norm_cols_X = W.pow(2).sum(0)  # [K,]

    # function to check convergence state (per sample)
    def _check_convergence(z_, x_, R_, tol_):
        XtA = torch.mm(R_, W)  # [N,K]
        dual_norm_XtA = XtA.abs().max(1)[0]  # [N,]
        R_norm2 = R_.pow(2).sum(1)  # [N,]

        small_norm = dual_norm_XtA <= alpha
        const = (alpha / dual_norm_XtA).masked_fill(small_norm, 1.)
        gap = torch.where(small_norm, R_norm2, 0.5 * R_norm2 * (1 + const.pow(2)))

        gap = gap + alpha * z_.abs().sum(1) - const * (R_ * x_).sum(1)
        converged = gap < tol_

        return converged, gap

    # initialize residual
    R = x - torch.matmul(z, W.T) # [N,D]

    for n_iter in range(max_iter):
        if converged.all():
            break
        active_ix, = torch.where(~converged)
        z_max = z.new_zeros(len(active_ix))
        d_z_max = z.new_zeros(len(active_ix))
        for i in range(n_components):  # Loop over components
            if norm_cols_X[i] == 0:
                continue

            atom_i = W[:,i].contiguous()

            z_i = z[active_ix, i].clone()
            nonzero = z_i != 0
            R[active_ix[nonzero]] += torch.outer(z_i[nonzero], atom_i)

            z[active_ix, i] = F.softshrink(R[active_ix].matmul(atom_i), alpha)
            z[active_ix, i] /= norm_cols_X[i]

            z_new_i = z[active_ix, i]
            nonzero = z_new_i != 0
            R[active_ix[nonzero]] -= torch.outer(z_new_i[nonzero], atom_i)

            # update the maximum absolute coefficient update
            d_z_max = torch.maximum(d_z_max, (z_new_i - z_i).abs())
            z_max = torch.maximum(z_max, z_new_i.abs())

        ### check convergence ###
        check = (z_max == 0) | (d_z_max / z_max < d_w_tol) | (n_iter == max_iter-1)
        if not check.any():
            continue
        check_ix = active_ix[check]
        converged[check_ix], gap[check_ix] = \
            _check_convergence(z[check_ix], x[check_ix], R[check_ix], tol[check_ix])

    return z, gap