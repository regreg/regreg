"""
Regression tests for the smooth-loss bugs in gh-108.
"""
from copy import copy

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import norm
from scipy.special import log_ndtr

import regreg.api as rr
from regreg.identity_quadratic import identity_quadratic as iq
from regreg.smooth.binary import (logistic_loglike, probit_loglike,
                                  cloglog_loglike, huber_svm)
from regreg.smooth.glm import poisson_loglike
from regreg.smooth.mglm import (mglm, multinomial_loglike,
                                multinomial_baseline_loglike)
from regreg.smooth.quadratic import quadratic_loss
from regreg.problems.composite import smooth_conjugate

def _fd_grad(loss, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(x.shape[0]):
        e = np.zeros_like(x); e[i] = h
        g[i] = (loss.smooth_objective(x + e, 'func') -
                loss.smooth_objective(x - e, 'func')) / (2 * h)
    return g

@pytest.mark.parametrize('eta', [-30., -8., -6., -1., 0., 2., 4., 6.])
def test_cloglog_no_clipping(eta):
    # eta was clipped to [-5, 5]: wrong values outside, zero gradient for a
    # confidently wrong prediction, and an infinite Hessian from eta ~ 3.7
    y = np.array([0., 1.])
    L = cloglog_loglike((2,), y)
    x = np.array([eta, eta])
    e = np.exp(eta)
    expected = -(np.log(-np.expm1(-e)) - e)    # y = 1 term + y = 0 term
    np.testing.assert_allclose(L.smooth_objective(x, 'func'), expected, rtol=1e-12)
    g = L.smooth_objective(x, 'grad')
    np.testing.assert_allclose(g, _fd_grad(L, x, h=1e-6 * max(1, abs(eta))),
                               rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(g[0], e, rtol=1e-12)      # y = 0: d/deta of e^eta
    assert np.all(np.isfinite(L.hessian(x)))
    np.testing.assert_allclose(L.mean_function(eta), -np.expm1(-e), rtol=1e-12)

@pytest.mark.parametrize('eta', [-40., -25., -10., 0., 10., 25., 40.])
def test_probit_no_clipping(eta):
    # the CDF / SF were clipped at +-20
    y = np.array([0., 1.])
    L = probit_loglike((2,), y)
    x = np.array([eta, eta])
    expected = -(log_ndtr(eta) + log_ndtr(-eta))
    np.testing.assert_allclose(L.smooth_objective(x, 'func'), expected, rtol=1e-12)
    g = L.smooth_objective(x, 'grad')
    np.testing.assert_allclose(g, _fd_grad(L, x, h=1e-6 * max(1, abs(eta))),
                               rtol=1e-5, atol=1e-8)
    assert np.all(np.isfinite(L.hessian(x)))

def test_binary_hessians_unchanged():
    # in the moderate range the Fisher information is what it was
    y = np.array([0, 1, 0, 1, 1.])
    eta = np.array([-3., -1, 0, 1, 2.])
    pi = norm.cdf(eta)
    np.testing.assert_allclose(probit_loglike((5,), y).hessian(eta),
                               norm.pdf(eta)**2 / (pi * (1 - pi)))
    pi = 1 - np.exp(-np.exp(eta))
    pdf = np.exp(eta - np.exp(eta))
    np.testing.assert_allclose(cloglog_loglike((5,), y).hessian(eta),
                               pdf**2 / (pi * (1 - pi)))

def test_smooth_solve_optimum_includes_quadratic():
    # the optimal value omitted the quadratic passed to solve, so the value
    # of smooth_conjugate for a smooth atom was wrong
    ql = quadratic_loss(3, offset=np.array([1., 2, 3]))
    value, x = ql.solve(quadratic=iq(1, 0, -np.ones(3), 0), return_optimum=True, tol=1e-14)
    np.testing.assert_allclose(value, -0.25, atol=1e-10)

    from scipy.optimize import minimize
    a = np.array([1., -2, 0.5, 3])
    ql = quadratic_loss(4, offset=a)
    sc = smooth_conjugate(ql, iq(1, 0, 0, 0))
    v = np.array([.3, -1, 2, .5])
    sup = -minimize(lambda u: -(v @ u - ql.smooth_objective(u, 'func') - 0.5 * u @ u),
                    np.zeros(4)).fun
    np.testing.assert_allclose(sc.smooth_objective(v, 'func'), sup, rtol=1e-6)

@pytest.mark.parametrize('make', [
    lambda y: logistic_loglike((10,), y, trials=2 * np.ones(10)),
    lambda y: probit_loglike((10,), y, trials=2 * np.ones(10)),
    lambda y: cloglog_loglike((10,), y, trials=2 * np.ones(10)),
    lambda y: poisson_loglike((10,), y, quadratic=iq(1., 0, 0, 3.)),
    lambda y: multinomial_loglike((5, 2), y.reshape((5, 2)) + 1),
    lambda y: multinomial_baseline_loglike((5, 1), y.reshape((5, 2)) + 1),
    ])
def test_copy_keeps_saturated_constant(make):
    # __copy__ passed the quadratic, which already holds the saturated
    # constant, to __init__, which added it again
    y = np.random.default_rng(0).integers(0, 3, 10).astype(float)
    L = make(y)
    c2 = copy(copy(L))
    np.testing.assert_allclose(c2.quadratic.constant_term, L.quadratic.constant_term)
    np.testing.assert_allclose(c2.quadratic.coef, L.quadratic.coef)

def test_multinomial_subsample_and_set_data():
    rng = np.random.default_rng(1)
    n, J = 7, 4
    Y = np.zeros((n, J)); Y[np.arange(n), rng.integers(0, J, n)] = 1
    cw = rng.uniform(.5, 2, n)
    # no __copy__, so subsample multiplied the original's case weights
    L = multinomial_loglike((n, J), Y, case_weights=cw.copy())
    S = L.subsample([0, 1])
    np.testing.assert_allclose(L.case_weights, cw)
    assert np.all(S.case_weights[2:] == 0)
    # set_data only stored the data
    X = rng.standard_normal((n, 3))
    Y2 = np.zeros((n, J)); Y2[np.arange(n), rng.integers(0, J, n)] = 1
    b = rng.standard_normal((3, J))
    M = mglm.multinomial(X, Y)
    M.data = (X, Y2)
    np.testing.assert_allclose(M.smooth_objective(b, 'func'),
                               mglm.multinomial(X, Y2).smooth_objective(b, 'func'))

@pytest.mark.parametrize('family', ['gaussian', 'logistic', 'probit', 'cloglog', 'poisson', 'cox'])
def test_glm_hessian_sparse(family):
    # W[:, None] * X is a matrix product for scipy.sparse X
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 4)); X[X < .3] = 0
    beta = rng.standard_normal(4) * .3
    y = (rng.uniform(size=30) < .5).astype(float)
    T, S = rng.exponential(size=30), (rng.uniform(size=30) < .7).astype(float)
    make = {'gaussian': lambda Z: rr.glm.gaussian(Z, y),
            'logistic': lambda Z: rr.glm.logistic(Z, y),
            'probit': lambda Z: rr.glm.probit(Z, y),
            'cloglog': lambda Z: rr.glm.cloglog(Z, y),
            'poisson': lambda Z: rr.glm.poisson(Z, 3 * y),
            'cox': lambda Z: rr.glm.cox(Z, T, S)}[family]
    np.testing.assert_allclose(make(sparse.csr_matrix(X)).hessian(beta),
                               make(X).hessian(beta), atol=1e-12)

def test_huber_svm_labels():
    # +-1 labels silently gave the wrong loss (2y - 1 = -3 for y = -1)
    huber_svm((3,), np.array([1, 0, 1.]), 0.1)
    with pytest.raises(ValueError):
        huber_svm((3,), np.array([1, -1, 1.]), 0.1)
