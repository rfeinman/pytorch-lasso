import torch
import torch.nn.functional as F


def lasso_cd(x, W, z0=None, alpha=1.0, maxiter=1000, tol=1e-6, verbose=False):
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