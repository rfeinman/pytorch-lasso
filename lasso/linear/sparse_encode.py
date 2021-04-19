import warnings
import torch

from .utils import lstsq, ridge
from .solvers import (coord_descent, gpsr_basic, iterative_ridge, ista,
                      interior_point, split_bregman)

_init_defaults = {
    'ista': 'zero',
    'cd': 'zero',
    'gpsr': 'zero',
    'iter-ridge': 'ridge',
    'interior-point': 'ridge',
    'split-bregman': 'zero'
}


def initialize_code(x, weight, alpha, mode):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        z0 = x.new_zeros(n_samples, n_components)
    elif mode == 'unif':
        z0 = x.new(n_samples, n_components).uniform_(-0.1, 0.1)
    elif mode == 'lstsq':
        z0 = lstsq(x.T, weight).T
    elif mode == 'ridge':
        z0 = ridge(x.T, weight, alpha=alpha).T
    elif mode == 'transpose':
        z0 = torch.matmul(x, weight)
    else:
        raise ValueError("invalid init parameter '{}'.".format(mode))

    return z0


def sparse_encode(x, weight, alpha=1.0, z0=None, algorithm='ista', init=None,
                  **kwargs):
    n_samples = x.size(0)
    n_components = weight.size(1)

    # initialize code variable
    if z0 is not None:
        assert z0.shape == (n_samples, n_components)
    else:
        if init is None:
            init = _init_defaults.get(algorithm, 'zero')
        elif init == 'zero' and algorithm == 'iter-ridge':
            warnings.warn("Iterative Ridge should not be zero-initialized.")
        z0 = initialize_code(x, weight, alpha, mode=init)

    # perform inference
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
    elif algorithm == 'split-bregman':
        z, _ = split_bregman(weight, x, z0, alpha, **kwargs)
    else:
        raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))

    return z