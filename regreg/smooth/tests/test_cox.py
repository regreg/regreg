"""
Tests of `cox_loglike`, the wrapper of `coxdev` for the Cox partial likelihood.

The objective, gradient and Hessian come from `coxdev`; these tests check
how `cox_loglike` uses them (scalings, offsets, case weights, `coef`).
"""
from copy import copy

import numpy as np
import pytest
from coxdev import CoxDeviance

from ..cox import cox_loglike
from .. import glm

def _coxdev(eta, times, status, w, tie_breaking='breslow'):
    """
    Negative log partial likelihood, gradient and Hessian from `coxdev`.
    """
    cd = CoxDeviance(event=times, status=status, tie_breaking=tie_breaking)
    result = cd(eta, w)
    info = cd.information(eta, w)
    hess = info @ np.identity(eta.shape[0])
    return result.deviance / 2 - result.loglik_sat, result.gradient / 2, hess

def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    times = np.round(rng.exponential(size=n), 1) + 0.1 # with ties
    status = rng.binomial(1, 0.7, size=n)
    eta = 0.5 * rng.standard_normal(n)
    w = rng.uniform(0.5, 2, size=n)
    return times, status, eta, w

@pytest.mark.parametrize('tie_breaking', ['breslow', 'efron'])
def test_cox(tie_breaking):

    times, status, eta, w = _data()
    n = eta.shape[0]
    value, grad, hess = _coxdev(eta, times, status, w, tie_breaking)

    L = cox_loglike((n,), times, status, case_weights=w,
                    tie_breaking=tie_breaking)
    f, g = L.smooth_objective(eta, 'both')
    np.testing.assert_allclose(f, value)
    np.testing.assert_allclose(g, grad)
    np.testing.assert_allclose(L.smooth_objective(eta, 'func'), value)
    np.testing.assert_allclose(L.smooth_objective(eta, 'grad'), grad)

    v = np.random.default_rng(1).standard_normal(n)
    np.testing.assert_allclose(L.hessian_mult(eta, v), hess @ v)

    # gradient is the derivative of the objective
    d = 1e-6
    numerical = (L.smooth_objective(eta + d * v, 'func') -
                 L.smooth_objective(eta - d * v, 'func')) / (2 * d)
    np.testing.assert_allclose(numerical, g @ v, rtol=1e-6)

def test_case_weights_argument():
    # weights passed as an argument combine with the loss's own weights

    times, status, eta, w = _data()
    n = eta.shape[0]
    v = np.linspace(1, 2, n)
    L = cox_loglike((n,), times, status, case_weights=w)
    value, grad, _ = _coxdev(eta, times, status, w * v)
    f, g = L.smooth_objective(eta, 'both', case_weights=v)
    np.testing.assert_allclose(f, value)
    np.testing.assert_allclose(g, grad)
    np.testing.assert_allclose(L.case_weights, w)

def test_offset_coef():
    # coef scales the objective, gradient and Hessian

    times, status, eta, w = _data()
    n = eta.shape[0]
    off = np.linspace(-1, 1, n)
    L = cox_loglike((n,), times, status, coef=3., offset=off)
    value, grad, hess = _coxdev(eta - off, times, status, np.ones(n))
    f, g = L.smooth_objective(eta, 'both')
    np.testing.assert_allclose(f, 3 * value)
    np.testing.assert_allclose(g, 3 * grad)
    v = np.random.default_rng(2).standard_normal(n)
    np.testing.assert_allclose(L.hessian_mult(eta, v), 3 * hess @ v)

def test_subsample_copy():

    times, status, eta, w = _data()
    n = eta.shape[0]
    L = cox_loglike((n,), times, status, case_weights=w, tie_breaking='efron')
    idx = np.arange(0, n, 2)
    keep = np.zeros(n)
    keep[idx] = 1
    value, _, _ = _coxdev(eta, times, status, w * keep, 'efron')
    S = L.subsample(idx)
    np.testing.assert_allclose(S.smooth_objective(eta, 'func'), value)
    np.testing.assert_allclose(L.case_weights, w)   # original untouched

    C = copy(L)
    assert C.tie_breaking == 'efron'
    np.testing.assert_allclose(C.smooth_objective(eta, 'func'),
                               L.smooth_objective(eta, 'func'))

def test_glm_cox():

    rng = np.random.default_rng(3)
    times, status, _, w = _data()
    n, p = times.shape[0], 3
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal(p) * 0.3
    for tie_breaking in ['breslow', 'efron']:
        G = glm.glm.cox(X, times, status, case_weights=w,
                        tie_breaking=tie_breaking)
        value, grad, hess = _coxdev(X @ beta, times, status, w, tie_breaking)
        np.testing.assert_allclose(G.objective(beta), value)
        np.testing.assert_allclose(G.gradient(beta), X.T @ grad)
        np.testing.assert_allclose(G.hessian(beta), X.T @ hess @ X)
