import warnings
import multiprocessing as mp
from functools import partial
from scipy import optimize
from scipy.sparse.linalg import LinearOperator
import numpy as np
import torch

__all__ = ['scipy_inference']


def _scipy_constr(
        x, weight, z0=None, method='slsqp', rss_lim=0.1,
        tol=None, **options):
    method = method.lower()
    assert method in ['trust-constr', 'slsqp', 'cobyla']
    assert x.ndim == 1
    assert weight.ndim == 2
    if z0 is None:
        z0 = np.linalg.lstsq(weight, x[:,None], rcond=None)[0][:,0]
    assert z0.ndim == 1

    # objective
    f = lambda z: np.sum(np.abs(z))
    if method in ['trust-constr', 'slsqp']:
        jac = lambda z: np.sign(z)
    else:
        jac = None
    if method == 'trust-constr':
        hess_ = np.zeros((z0.size, z0.size))
        hess = lambda z: hess_
    else:
        hess = None

    # constraints
    if method == 'trust-constr':
        constr_hess_ = weight.T @ weight
        constr = optimize.NonlinearConstraint(
            fun=lambda z: 0.5 * np.sum((weight.dot(z) - x)**2),
            lb=-np.inf,
            ub=rss_lim,
            jac=lambda z: weight.T @ (weight.dot(z) - x),
            hess=lambda x, v: v[0] * constr_hess_
        )
    else:
        constr = {
            'type': 'ineq',
            'fun': lambda z: rss_lim - 0.5 * np.sum((weight.dot(z) - x)**2)
        }
        if method == 'slsqp':
            constr['jac'] = lambda z: - weight.T @ (weight.dot(z) - x)

    z0 = z0.flatten()
    res = optimize.minimize(
        f, z0, method=method, tol=tol, jac=jac, hess=hess,
        constraints=constr, options=options)

    zf = res.x

    return zf


def _scipy_constr_bound(
        x, weight, z0=None, method='slsqp', rss_lim=0.1,
        tol=None, **options):
    method = method.lower()
    assert method in ['trust-constr', 'slsqp']
    assert x.ndim == 1
    assert weight.ndim == 2
    assert weight.shape[0] == x.shape[0]
    if z0 is None:
        z0 = np.linalg.lstsq(weight, x[:,None], rcond=None)[0][:,0]
    assert z0.ndim == 1

    # store batch_size and code_size
    n_components = z0.shape[0]

    # expand pos/neg
    z0 = np.concatenate([np.maximum(z0, 0), np.maximum(-z0, 0)])
    def weight_dot(v):
        return weight.dot(v[:n_components]) - weight.dot(v[n_components:])
    def weightT_dot(v):
        Wtv = np.zeros(2*n_components)
        Wtv[:n_components] = weight.T.dot(v)
        Wtv[n_components:] = -Wtv[:n_components]
        return Wtv

    # objective
    f = lambda z: np.sum(z)
    jac = lambda z: np.ones_like(z)
    if method == 'trust-constr':
        hess_ = np.zeros((z0.size, z0.size))
        hess = lambda z: hess_
    else:
        hess = None

    # constraints
    if method == 'trust-constr':
        H = weight.T @ weight
        def constr_hess(x, v):
            def matvec(p):
                Hp = np.zeros_like(p)
                Hp[:n_components] = H.dot(p[:n_components]) - H.dot(p[n_components:])
                Hp[n_components:] = -Hp[:n_components]
                return v[0] * Hp
            return LinearOperator((2*n_components, 2*n_components), matvec=matvec)
        constr = optimize.NonlinearConstraint(
            fun=lambda z: 0.5 * np.sum((weight_dot(z) - x)**2),
            lb=-np.inf,
            ub=rss_lim,
            jac=lambda z: weightT_dot(weight_dot(z) - x),
            hess=constr_hess
        )
    else:
        constr = {
            'type': 'ineq',
            'fun': lambda z: rss_lim - 0.5 * np.sum((weight_dot(z) - x)**2),
            'jac': lambda z: - weightT_dot(weight_dot(z) - x)
        }

    # bounds
    bounds = optimize.Bounds(np.zeros(z0.size), np.full(z0.size, np.inf))

    z0 = z0.flatten()
    res = optimize.minimize(
        f, z0, method=method, tol=tol, jac=jac, hess=hess,
        constraints=constr, bounds=bounds, options=options)

    # reverse pos/neg expansion
    zf = res.x[:n_components] - res.x[n_components:]

    return zf


# ===================================
#  batch mode (with multiprocessing)
# ===================================

def _check_input(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            warnings.warn('GPU is not supported for scipy-based inference. '
                          'Data will be moved to CPU.')
        x = x.detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def scipy_inference(
        x, weight, z0=None, bound=True, method='slsqp', rss_lim=0.1,
        tol=None, **options):
    if bound:
        assert method in ['trust-constr', 'slsqp']
        inference_fn = _scipy_constr_bound
    else:
        assert method in ['trust-constr', 'slsqp', 'cobyla']
        inference_fn = _scipy_constr

    # convert torch tensors to numpy arrays
    is_tensor = torch.is_tensor(x)
    device = x.device if is_tensor else None
    x = _check_input(x)
    weight = _check_input(weight)
    if z0 is not None:
        z0 = _check_input(z0)
        assert z0.ndim == x.ndim
    assert weight.ndim == 2

    # single-input case
    if x.ndim == 1:
        return inference_fn(x, weight, z0, method=method, rss_lim=rss_lim,
                            tol=tol, **options)

    # batch case
    assert x.ndim == 2
    if z0 is not None:
        assert z0.shape[0] == x.shape[0]

    p = mp.Pool()
    try:
        z = p.starmap(
            partial(inference_fn, method=method, rss_lim=rss_lim,
                    tol=tol, **options),
            [(x[i].copy(), weight.copy(), z0 if z0 is None else z0[i].copy())
             for i in range(x.shape[0])]
        )
        p.close()
        p.join()
    except:
        p.close()
        p.terminate()
        raise

    z = np.stack(z)
    if is_tensor:
        z = torch.from_numpy(z).to(device)

    return z