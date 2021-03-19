from typing import Callable
import warnings
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F


def _gpsr_basic(
        x, y, u, v, tau, A, AT, dot, Ay=None, mu=0.1, lambda_backtrack=0.5,
        maxiter=1000, miniter=5, tol=1e-2, stop_criterion=3, start_iter=0,
        verbose=0):
    # initialize Ay
    if Ay is None:
        Ay = AT(y)

    # Compute and store initial values
    resid = y - A(x)  # residual
    resid_base = y - resid  # "base" residual
    f = 0.5 * dot(resid, resid) + tau * (u.sum() + v.sum())  # objective
    nz_x = x != 0  # num nonzero components
    n_iter = start_iter  # num iterations
    if verbose:
        print('Initial obj = %10.6e, nz = %d\n' % (f, nz_x.long().sum()))

    # loop for GP algorithm
    while True:
        # compute gradient
        tmp = AT(resid_base) - Ay
        gradu, gradv = (tmp + tau), (-tmp + tau)
        old_u, old_v = u, v

        # calculate unconstrained minimizer along this direction, use this
        # as the first guess of steplength parameter lambda
        #    lambd0 = - (gradu^T du + gradv^T dv) / dGd
        condgradu = gradu.masked_fill((old_u <= 0) & (gradu >= 0), 0.)
        condgradv = gradv.masked_fill((old_v <= 0) & (gradv >= 0), 0.)
        auv_cond = A(condgradu - condgradv)
        lambd = (dot(gradu, condgradu) + dot(gradv, condgradv))
        lambd /= dot(auv_cond, auv_cond) + 1e-7

        # line search, starting with the initial guess from above
        while True:
            # calculate step for this lambda and candidate point
            du = F.relu(u - lambd * gradu) - u
            u_new = u + du
            dv = F.relu(v - lambd * gradv) - v
            v_new = v + dv
            dx = du - dv
            x_new = x + dx

            # evaluate function at the candidate point
            resid_base = A(x_new)
            resid = y - resid_base
            f_new = 0.5 * dot(resid, resid) + tau * (u_new.sum() + v_new.sum())

            # test sufficient decrease condition
            if f_new <= f + mu * (dot(gradu,du) + dot(gradv, dv)):
                break

            lambd = lambd * lambda_backtrack
            if verbose > 1:
                print('    line-search reducing lambda to %6.2e' % lambd)

        u, v = u_new, v_new
        prev_f, f = f, f_new
        uvmin = torch.min(u, v)
        u, v = (u - uvmin), (v - uvmin)
        x = u - v

        # calculate nonzero pattern and number of nonzeros (do this *always*)
        nz_x_prev = nz_x
        nz_x = x != 0
        num_nz_x = nz_x.long().sum()

        # update n_iter
        n_iter += 1

        # print out stuff
        if verbose:
            print('It = %4d, obj = %9.5e, lambda = %6.2e, nz = %d' %
                  (n_iter, f, lambd, num_nz_x))

        # check stop criterion
        if stop_criterion == 0:
            # compute the stopping criterion based on the change
            # of the number of non-zero components of the estimate
            if num_nz_x >= 1:
                criterion = (nz_x != nz_x_prev).long().sum()
            else:  # definitely terminate
                criterion = float('-inf')  # tol / 2
            criterion_name = 'd_nz'
        elif stop_criterion == 1:
            # compute the stopping criterion based on the relative
            # variation of the objective function.
            criterion = (f - prev_f).abs() / prev_f
            criterion_name = 'd_f'
        elif stop_criterion == 2:
            # stopping criterion based on relative norm of step taken
            criterion = dx.norm() / x.norm()
            criterion_name = '||d_x|| / ||x||'
        elif stop_criterion == 3:
            # compute the "LCP" stopping criterion - again based on the
            # previous iterate. Make it "relative" to the norm of x.
            inf = float('inf')
            tmp_u = torch.min(gradu, old_u)
            tmp_v = torch.min(gradv, old_v)
            numer = torch.max(tmp_u.norm(inf), tmp_v.norm(inf))
            denom = torch.max(old_u.norm(inf), old_v.norm(inf))
            criterion = numer / denom.clamp_(1e-6, None)
            criterion_name = 'LCP'
        elif stop_criterion == 4:
            # continue if not yet reached target value tol
            criterion = f
            criterion_name = 'f'
        else:
            raise ValueError('Unknwon stopping criterion')

        if verbose:
            print(4*' ' + criterion_name + ' = %e (target = %e)' %
                  (criterion, tol))

        if n_iter > miniter and criterion <= tol:
            break

        if n_iter >= maxiter:
            break

    return x, f, resid, n_iter


def _debias(x, y, tau, A, AT, dot=None, tol=1e-4, start_iter=0, miniter=0,
            maxiter=500, verbose=0):
    if dot is None:
        dot = lambda u,v: torch.sum(u*v)

    # calculate initial residual
    resid = A(x) - y

    # check num nonzero components
    num_nz_x = (x != 0).long().sum()
    if num_nz_x > y.numel() or num_nz_x == 0:
        msg = 'Debiasing requested but not performed. '
        if num_nz_x == 0:
            msg = msg + 'x has no nonzeros.'
        else:
            msg = msg + 'There are too many nonzeros in x.'
        warnings.warn(msg)
        f = 0.5 * dot(resid, resid) + tau * x.abs().sum()
        return x, f, resid, start_iter

    is_zero = x == 0
    n_iter = start_iter

    # initialize CG r
    rvec = AT(resid)

    # mask out the zeros
    rvec = rvec.masked_fill(is_zero, 0.)
    rTr_cg = dot(rvec, rvec)

    # set convergence threshold for the residual || RW x - y ||_2
    tol = tol * dot(rvec, rvec)

    # initialize pvec
    pvec = - rvec

    # main loop
    while True:
        # calculate A*p = Wt * Rt * R * W * pvec
        RWpvec = A(pvec)
        Apvec = AT(RWpvec)

        # mask out the zero terms
        Apvec = Apvec.masked_fill(is_zero, 0.)

        # calculate alpha for CG
        alpha_cg = rTr_cg / dot(pvec, Apvec)

        # take the step
        x = x + alpha_cg * pvec
        resid = resid + alpha_cg * RWpvec
        rvec  = rvec + alpha_cg * Apvec

        rTr_cg_plus = dot(rvec, rvec)
        beta_cg = rTr_cg_plus / rTr_cg
        pvec = -rvec + beta_cg * pvec

        rTr_cg = rTr_cg_plus

        n_iter += 1
        f = 0.5 * dot(resid, resid) + tau * x.abs().sum()

        if verbose:
            # in the debiasing CG phase, always use convergence criterion
            # based on the residual (this is standard for CG)
            print(' Iter = %5d, resid = %13.8e, convergence = '
                  '%8.3e' % (n_iter, dot(resid,resid), rTr_cg / tol))

        db_iter = n_iter - start_iter
        continue_cg = (db_iter <= miniter) or \
                         ((rTr_cg > tol) and (db_iter <= maxiter))
        if not continue_cg:
            break

    return x, f, resid, n_iter


def gpsr_basic(y, A, tau, AT=None, x0=None, stop_criterion=3, tol=1e-2,
               maxiter=1000, miniter=5, init=0, continuation=False,
               debias=False, verbose=0, **kwargs):
    """GPSR-Basic algorithm for sparse reconstruction

    This function solves the convex problem
        arg min_x = 0.5*|| y - A x ||_2^2 + tau || x ||_1

    using the algorithm `GPSR-Basic` described in the paper "Gradient
    Projection for Sparse Reconstruction: Application to Compressed Sensing
    and Other Inverse Problems" by Figueiredo et al. (2007).


    Parameters
    ----------
    y : Tensor
        Observations. Either 1D vector or 2D array.

    A : Tensor | Callable
        If y and x are both 1D vectors, "A" can be a [d x k] matrix (where
        d is the size of y and k the size of x) or a callable function that
        computes products of the form A @ v for some vector v. In any other
        case (e.g. y and/or x are 2D tensors), A must be a callable, and
        another callable "AT" must be provided that computes A^T @ v.

    tau : float
        Sparsity controlling parameter.

    """
    verbose = int(verbose)

    if not stop_criterion in [0, 1, 2, 3, 4]:
       raise ValueError('Unknown stopping criterion')

    # build matrix multiplication operators
    if callable(A):
        if not callable(AT):
            raise ValueError('The function handle for transpose of A is missing')
    else:
        # if A is a matrix, we find out dimensions of y and x,
        # and create function handles for multiplication by A and A',
        # so that the code below doesn't have to distinguish between
        # the handle/not-handle cases
        A_mat = A
        AT = lambda x: torch.matmul(A_mat.T, x)
        A = lambda x: torch.matmul(A_mat, x)

    # build dot product operator
    dot = lambda u,v: torch.sum(u*v)

    # Precompute A^T @ y since we'll use it a lot
    Ay = AT(y)

    # Initialize parameters
    if x0 is not None:
        x = x0
    elif init == 0:
        x = torch.zeros_like(Ay)
    elif init == 1:
        x = torch.randn_like(Ay)
    elif init == 2:
        x = Ay
    else:
        raise ValueError('Unknown initialization option')

    # check the value of tau; if it's large enough,
    # the optimal solution is the zero vector
    max_tau = Ay.abs().max()
    if tau >= max_tau:
        warnings.warn('tau is too small; solution is zero vector')
        return torch.zeros_like(Ay)

    # set continuation factors
    if continuation:
        # first check top see if the first tau factor is
        # too large (i.e., large enough to make the first
        # solution all zeros). If so, make it a little smaller than that.
        # Also set to that value as default
        cont_steps = kwargs.get('cont_steps', 5)
        first_tau_factor = kwargs.get('first_tau_factor', None)
        if first_tau_factor is None or first_tau_factor * tau >= max_tau:
            warnings.warn('parameter FirstTauFactor too large; changing')
            first_tau_factor = 0.8 * max_tau / tau
        cont_factors = 10 ** np.linspace(np.log10(first_tau_factor), 0, cont_steps)
    else:
        cont_steps = 1
        cont_factors = [1]

    # initialize u and v
    u, v = F.relu(x), F.relu(-x)

    # line search settings
    mu = kwargs.get('mu', 0.1)  # sufficient decrease
    lambda_backtrack = kwargs.get('lambda_backtrack', 0.5)  # step size

    # counter for number of iterations
    n_iter = 0

    # loop for continuation
    for i in range(cont_steps):
        # set tau
        tau_i = tau * cont_factors[i]
        if verbose > 1:
            print('Setting tau = %8.4f\n' % tau_i)

        # set tol and stop criterion (use defaults for all steps other than last)
        is_last = i + 1 == cont_steps
        tol_i = tol if is_last else 1e-3
        stop_criterion_i = stop_criterion if is_last else 3

        # core GP algorithm step (inner loop)
        x, f, resid, n_iter = _gpsr_basic(
            x, y, u, v, tau_i, A, AT, dot, Ay, mu, lambda_backtrack,
            maxiter, miniter, tol_i, stop_criterion_i, start_iter=n_iter,
            verbose=verbose)

    # print results
    if verbose:
        num_nz_x = (x != 0.0).long().sum()
        if verbose == 1:
            print('\nFinal obj = %10.6e, nz = %d' % (f, num_nz_x))
        elif verbose > 1:
            print('\nFinished the main algorithm.\nResults:')
            print('    ||A x - y ||_2^2 = %10.3e' % dot(resid, resid))
            print('    ||x||_1 = %10.3e' % x.abs().sum())
            print('     Obj. function: %10.3e' % f)
            print('     Num. non-zero components: %d' % num_nz_x)

    # de-biasing not yet implemented
    if not debias:
        return x

    # collect default values
    tol_db = kwargs.get('tol_debias', 1e-4)
    maxiter_db = kwargs.get('maxiter_debias', 500)
    miniter_db = kwargs.get('miniter_debias', 0)

    if verbose:
        print('\nStarting the debiasing phase...\n')

    # run de-biasing
    x, f, resid, n_iter = _debias(
        x, y, tau, A, AT, dot, tol_db, n_iter,
        miniter_db, maxiter_db, verbose)

    if verbose:
        num_nz_x = (x != 0.0).long().sum()
        if verbose == 1:
            print('\nFinal obj = %10.6e, nz = %d' % (f, num_nz_x))
        elif verbose > 1:
            print('\nFinished the debiasing phase.\nResults:')
            print('    ||A x - y ||_2^2 = %10.3e' % dot(resid, resid))
            print('    ||x||_1 = %10.3e' % x.abs().sum())
            print('     Obj. function: %10.3e' % f)
            print('     Num. non-zero components: %d' % num_nz_x)

    return x
