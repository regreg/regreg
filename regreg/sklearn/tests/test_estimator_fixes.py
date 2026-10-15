"""
Regression tests for the sklearn estimator bugs in gh-108 (and the plain
classifiers' solve_args in gh-104).
"""
import numpy as np
import pytest

sklearn = pytest.importorskip('sklearn')
from sklearn.base import clone

import regreg.api as rr
from regreg.sklearn import api as A

def _data(seed=0, n=60, p=6):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.standard_normal((n, p - 1))]
    y = 2 + X[:, 1] + rng.standard_normal(n)
    return X, y, (y > 2).astype(float)

def _weighted_constructor(p):
    w = np.r_[0., np.ones(p - 1)]          # unpenalized intercept
    return lambda score, frac: rr.weighted_l1norm(w, lagrange=frac * np.fabs(score[1:]).max())

@pytest.mark.parametrize('name', ['gaussian_lagrange', 'logistic_lagrange', 'huber_lagrange',
                                  'huber_svm_lagrange'])
def test_lagrange_unpenalized(name):
    # the null fit set the atom's lagrange to inf, so the objective was
    # inf * 0 = nan ("objective is NaN")
    X, y, yb = _data()
    args = (1.,) if 'huber' in name else ()
    est = getattr(A, name)(*args, _weighted_constructor(X.shape[1]), {'frac': .3},
                           unpenalized=True, solve_args={'tol': 1e-12, 'min_its': 100})
    est.fit(X, yb if ('logistic' in name or 'svm' in name) else y)
    assert np.all(np.isfinite(est._coefs))

def test_lagrange_unpenalized_null_gradient():
    # with the intercept unpenalized the null fit is the mean, and the
    # Lagrange parameter comes from the gradient there
    X, y, _ = _data()
    seen = {}
    def constructor(score, frac):
        seen['score'] = score.copy()
        return _weighted_constructor(X.shape[1])(score, frac)
    A.gaussian_lagrange(constructor, {'frac': .3}, unpenalized=True).fit(X, y)
    null = np.zeros(X.shape[1])
    null[0] = y.mean()
    np.testing.assert_allclose(seen['score'], X.T @ (X @ null - y), atol=1e-3)

@pytest.mark.parametrize('name', ['logistic_classifier', 'probit_classifier',
                                  'cloglog_classifier', 'multinomial_classifier'])
def test_classifier_solve_args(name):
    # gh-104: the plain classifiers did not accept solve_args
    X, y, yb = _data()
    p = X.shape[1]
    if name == 'multinomial_classifier':
        Y = np.zeros((len(yb), 2)); Y[np.arange(len(yb)), yb.astype(int)] = 1
        est = getattr(A, name)(rr.l1norm, {'shape': (p, 2), 'lagrange': 1.}, solve_args={'tol': 1e-12})
        est.fit(X, Y)
    else:
        est = getattr(A, name)(rr.l1norm, {'shape': p, 'lagrange': 1.}, solve_args={'tol': 1e-12})
        est.fit(X, yb)
    assert clone(est).get_params()['solve_args'] == {'tol': 1e-12}

@pytest.mark.parametrize('name', ['huber', 'huber_svm', 'huber_lagrange', 'huber_svm_lagrange'])
def test_huber_parameters(name):
    # the Huber estimators dropped solve_args, quadratic, initial (and
    # unpenalized for the Lagrange forms)
    X, y, yb = _data()
    p = X.shape[1]
    q = rr.identity_quadratic(1., 0, 0, 0)
    if 'lagrange' in name:
        est = getattr(A, name)(1., _weighted_constructor(p), {'frac': .3}, quadratic=q,
                               solve_args={'tol': 1e-12}, unpenalized=True)
    else:
        est = getattr(A, name)(1., rr.l1norm, {'shape': p, 'lagrange': 1.}, quadratic=q,
                               solve_args={'tol': 1e-12})
    params = clone(est).get_params()
    assert params['solve_args'] == {'tol': 1e-12} and params['quadratic'] is not None
    est.fit(X, yb if 'svm' in name else y)

def test_enet_keeps_atom_quadratic():
    # enet_alpha replaced, rather than added to, a quadratic on the atom
    p = 6
    q = rr.identity_quadratic(2., 0, 0, 0)
    est = A.gaussian_lagrange(lambda score: rr.l1norm(p, lagrange=1., quadratic=q), {}, enet_alpha=.5)
    atom = est._construct_atom(np.ones(p))
    np.testing.assert_allclose(atom.quadratic.coef, 2.5)
    np.testing.assert_allclose(atom.lagrange, .5)
