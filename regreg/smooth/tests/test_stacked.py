"""
Tests of the stacked losses and their factory methods (gh-72).
"""
import numpy as np
import pytest

from ..glm import (stacked_loglike,
                   gaussian_loglike,
                   poisson_loglike,
                   huber_loss,
                   cox_loglike)
from ..binary import (logistic_loglike,
                      probit_loglike,
                      cloglog_loglike,
                      huber_svm)
from ..mglm import stacked_common_loglike, mglm

n, K = 6, 2

def _responses():
    rng = np.random.default_rng(0)
    binary = [rng.binomial(1, .5, size=n).astype(float) for _ in range(K)]
    trials = [rng.integers(1, 4, size=n).astype(float) for _ in range(K)]
    successes = [rng.binomial(t.astype(int), .5).astype(float) for t in trials]
    counts = [rng.poisson(2, size=n).astype(float) for _ in range(K)]
    gaussian = [rng.standard_normal(n) for _ in range(K)]
    times = [rng.exponential(size=n) for _ in range(K)]
    status = [rng.binomial(1, .7, size=n) for _ in range(K)]
    return dict(binary=binary, trials=trials, successes=successes,
                counts=counts, gaussian=gaussian, times=times, status=status)

def _cases(R, coef):
    # (factory call, matching component losses)
    yield (lambda klass: klass.gaussian(R['gaussian'], coef=coef),
           [gaussian_loglike((n,), y, coef=coef) for y in R['gaussian']])
    for name, comp in [('logistic', logistic_loglike),
                       ('probit', probit_loglike),
                       ('cloglog', cloglog_loglike)]:
        yield (lambda klass, name=name: getattr(klass, name)(R['binary'], coef=coef),
               [comp((n,), y, coef=coef) for y in R['binary']])
        yield (lambda klass, name=name: getattr(klass, name)(R['successes'], trials=R['trials'], coef=coef),
               [comp((n,), y, trials=t, coef=coef) for y, t in zip(R['successes'], R['trials'])])
    yield (lambda klass: klass.poisson(R['counts'], coef=coef),
           [poisson_loglike((n,), y, coef=coef) for y in R['counts']])
    yield (lambda klass: klass.huber(R['gaussian'], .5, coef=coef),
           [huber_loss((n,), y, .5, coef=coef) for y in R['gaussian']])
    yield (lambda klass: klass.huber_svm(R['binary'], .5, coef=coef),
           [huber_svm((n,), y, .5, coef=coef) for y in R['binary']])
    yield (lambda klass: klass.cox(R['times'], R['status'], coef=coef),
           [cox_loglike((n,), t, s, coef=coef) for t, s in zip(R['times'], R['status'])])

@pytest.mark.parametrize('coef', [1., 2.5])
def test_stacked_loglike_factories(coef):

    R = _responses()
    eta = np.random.default_rng(1).standard_normal(n * K) * .5
    for make, comps in _cases(R, coef):
        L = make(stacked_loglike)
        f, g = L.smooth_objective(eta, 'both')
        expected_f = sum(c.smooth_objective(eta[i * n:(i + 1) * n], 'func')
                         for i, c in enumerate(comps))
        expected_g = np.hstack([c.smooth_objective(eta[i * n:(i + 1) * n], 'grad')
                                for i, c in enumerate(comps)])
        np.testing.assert_allclose(f, expected_f)
        np.testing.assert_allclose(g, expected_g)

@pytest.mark.parametrize('coef', [1., 2.5])
def test_stacked_common_loglike_factories(coef):

    R = _responses()
    eta = np.random.default_rng(2).standard_normal((n, K)) * .5
    for make, comps in _cases(R, coef):
        try:
            L = make(stacked_common_loglike)
        except AttributeError: # no probit / cloglog factories
            continue
        f, g = L.smooth_objective(eta, 'both')
        expected_f = sum(c.smooth_objective(eta[:, i], 'func')
                         for i, c in enumerate(comps))
        expected_g = np.array([c.smooth_objective(eta[:, i], 'grad')
                               for i, c in enumerate(comps)]).T
        np.testing.assert_allclose(f, expected_f)
        np.testing.assert_allclose(g, expected_g)

def test_stacked_loglike_case_weights():
    # case weights (stored, or passed in) reach the component losses

    R = _responses()
    eta = np.random.default_rng(3).standard_normal(n * K)
    w = np.random.default_rng(4).uniform(0, 2, size=n * K)
    comps = [gaussian_loglike((n,), y) for y in R['gaussian']]

    zero = stacked_loglike(comps, case_weights=np.zeros(n * K))
    np.testing.assert_allclose(zero.smooth_objective(eta, 'func'), 0)

    expected = sum(c.smooth_objective(eta[i * n:(i + 1) * n], 'func',
                                      case_weights=w[i * n:(i + 1) * n])
                   for i, c in enumerate(comps))
    stored = stacked_loglike(comps, case_weights=w)
    np.testing.assert_allclose(stored.smooth_objective(eta, 'func'), expected)
    passed = stacked_loglike(comps)
    np.testing.assert_allclose(passed.smooth_objective(eta, 'func', case_weights=w), expected)
    f, g = passed.smooth_objective(eta, 'both', case_weights=w)
    np.testing.assert_allclose(g, w * (eta - np.hstack(R['gaussian'])))

def test_mglm_stacked():

    R = _responses()
    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, 3))
    B = rng.standard_normal((3, K))
    comps = [gaussian_loglike((n,), y) for y in R['gaussian']]
    M = mglm.stacked(X, comps)
    eta = X @ B
    expected = sum(c.smooth_objective(eta[:, i], 'func') for i, c in enumerate(comps))
    np.testing.assert_allclose(M.smooth_objective(B, 'func'), expected)
