import warnings
import torch

from .utils import lstsq, ridge
from .solvers import (coord_descent, gpsr_basic, iterative_ridge, ista,
                      interior_point)


def sparse_encode(x, weight, alpha=1.0, z0=None, algorithm='ista', init='zero',
                  **kwargs):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if z0 is not None:
        assert z0.shape == (n_samples, n_components)
    elif init == 'zero':
        if algorithm == 'iter-ridge':
            warnings.warn("IteratedRidge should not be zero-initialized.")
        z0 = x.new_zeros(n_samples, n_components)
    elif init == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1)
    elif init == 'lstsq':
        z0 = lstsq(x.T, weight).T
    elif init == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    elif init == 'transpose':
        z0 = torch.matmul(x, weight)
    else:
        raise ValueError("invalid init parameter '{}'.".format(init))

    if algorithm == 'cd':
        z = coord_descent(x, weight, z0, alpha, **kwargs)
    elif algorithm == 'gpsr':
        A = lambda v: torch.mm(v, weight.T)
        AT = lambda v: torch.mm(v, weight)
        z = gpsr_basic(x, A, tau=alpha, AT=AT, x0=z0, **kwargs)
    elif algorithm == 'iter-ridge':
        z = iterative_ridge(z0, x, weight, alpha, **kwargs)
    elif algorithm == 'ista':
        z = ista(x, z0, weight, alpha, **kwargs)
    elif algorithm == 'interior-point':
        z, _ = interior_point(x, weight, z0, alpha, **kwargs)
    else:
        raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))

    return z