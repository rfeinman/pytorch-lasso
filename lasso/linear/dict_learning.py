from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_encode import sparse_encode



def lasso_loss(X, Z, weight, alpha=1.0):
    X_hat = torch.matmul(Z, weight.T)
    loss = 0.5 * (X - X_hat).pow(2).sum() + alpha * Z.abs().sum()
    return loss / X.size(0)


def dict_evaluate(X, weight, alpha, **kwargs):
    X = X.to(weight.device)
    Z = sparse_encode(X, weight, alpha, **kwargs)
    loss = lasso_loss(X, Z, weight, alpha)
    return loss


def dict_learning(X, n_components, alpha=1.0, constrained=True, persist=False,
                  lambd=1e-2, steps=60, device='cpu', progbar=True,
                  **solver_kwargs):
    n_samples, n_features = X.shape
    X = X.to(device)
    weight = torch.empty(n_features, n_components, device=device)
    nn.init.orthogonal_(weight)
    if constrained:
        weight = F.normalize(weight, dim=0)
    Z0 = None

    losses = torch.zeros(steps, device=device)
    if progbar:
        progress_bar = tqdm(total=steps)
    for i in range(steps):
        # infer sparse coefficients and compute loss
        Z = sparse_encode(X, weight, alpha, Z0, **solver_kwargs)
        losses[i] = lasso_loss(X, Z, weight, alpha)
        if persist:
            Z0 = Z

        # update dictionary
        if constrained:
            weight = update_dict(weight, X, Z)
        else:
            weight = update_dict_ridge(X, Z, lambd=lambd)

        # update progress bar
        if progbar:
            progress_bar.set_postfix(loss=losses[i].item())
            progress_bar.update(1)

    if progbar:
        progress_bar.close()

    return weight, losses


def update_dict(dictionary, X, Z, random_seed=None, positive=False,
                eps=1e-10):
    """Update the dense dictionary factor in place.

    Modified from `_update_dict` in sklearn.decomposition._dict_learning

    Parameters
    ----------
    dictionary : Tensor of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.
    X : Tensor of shape (n_samples, n_features)
        Data matrix.
    code : Tensor of shape (n_samples, n_components)
        Sparse coding of the data against which to optimize the dictionary.
    random_seed : int
        Seed for randomly initializing the dictionary.
    positive : bool
        Whether to enforce positivity when finding the dictionary.
    eps : float
        Minimum vector norm before considering "degenerate"
    """
    n_components = dictionary.size(1)
    if random_seed is not None:
        torch.manual_seed(random_seed)

    # Residuals
    R = X - torch.matmul(Z, dictionary.T)  # (n_samples, n_features)
    for k in range(n_components):
        # Update k'th atom
        R += torch.outer(Z[:, k], dictionary[:, k])
        dictionary[:, k] = torch.matmul(Z[:, k], R)
        if positive:
            dictionary[:, k].clamp_(0, None)

        # Re-scale k'th atom
        atom_norm = dictionary[:, k].norm()
        if atom_norm < eps:
            dictionary[:, k].normal_()
            if positive:
                dictionary[:, k].clamp_(0, None)
            dictionary[:, k] /= dictionary[:, k].norm()
            # Set corresponding coefs to 0
            Z[:, k].zero_()  # TODO: is this necessary?
        else:
            dictionary[:, k] /= atom_norm
            R -= torch.outer(Z[:, k], dictionary[:, k])

    return dictionary


def update_dict_ridge(x, z, lambd=1e-4):
    """Update an (unconstrained) dictionary with ridge regression

    This is equivalent to a Newton step with the (L2-regularized) squared
    error objective:
        f(V) = (1/2N) * ||Vz - x||_2^2 + (lambd/2) * ||V||_2^2

    x : a batch of observations with shape (n_samples, n_features)
    z : a batch of code vectors with shape (n_samples, n_components)
    lambd : weight decay parameter
    """
    rhs = torch.mm(z.T, x)
    M = torch.mm(z.T, z)
    M.diagonal().add_(lambd * x.size(0))
    L = torch.linalg.cholesky(M)
    V = torch.cholesky_solve(rhs, L).T

    return V


