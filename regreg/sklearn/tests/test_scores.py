"""
Regression tests for the sklearn estimators' scores, predictions and
Lagrange-form mixins (gh-75, gh-76).
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

sklearn = pytest.importorskip('sklearn')
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

import regreg.api as rr
from regreg.sklearn import api as A
from regreg.sklearn.base import (base_mixin,
                                 lagrange_mixin,
                                 survival_mixin_lagrange,
                                 sklearn_classifier_lagrange,
                                 sklearn_survival_lagrange)

def _binary_data(seed=0, n=80, p=5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] - X[:, 1] + 0.5 * rng.standard_normal(n) > 0).astype(float)
    w = rng.uniform(0.2, 2, n)
    return X, y, w

def _lagrange_atom(score, frac=0.5):
    return rr.l1norm(score.shape, lagrange=frac * np.fabs(score).max())

# gh-75

def test_classifier_lagrange_mixin():
    # the mixin called lagrange_mixin(self, ...) and passed an undefined name
    est = sklearn_classifier_lagrange(_lagrange_atom, {'frac': 0.5})
    assert est.enet_alpha == 0 and est.offset is False

@pytest.mark.parametrize('family', ['logistic', 'probit', 'cloglog'])
def test_classifier_lagrange(family):
    # the *_classifier_lagrange estimators were the plain classifiers, with
    # no Lagrange-form atom construction
    X, y, _ = _binary_data()
    klass = getattr(A, family + '_classifier_lagrange')
    assert issubclass(klass, lagrange_mixin)
    est = klass(_lagrange_atom, {'frac': 0.5})
    assert {'enet_alpha', 'unpenalized', 'quadratic'} <= set(est.get_params())
    est.fit(X, y)

    # same fit as the plain classifier with the Lagrange parameter worked out
    loss = est._loglike
    lagrange = 0.5 * np.fabs(loss.smooth_objective(np.zeros(X.shape[1]), 'grad')).max()
    plain = getattr(A, family + '_classifier')(rr.l1norm, {'shape': X.shape[1],
                                                           'lagrange': lagrange})
    plain.fit(X, y)
    np.testing.assert_allclose(est._coefs, plain._coefs, atol=1e-6)

    params = {'atom_params': [{'frac': f} for f in [0.3, 0.6]]}
    GridSearchCV(clone(est), params, cv=3).fit(X, y)

def test_multinomial_classifier_lagrange():
    assert issubclass(A.multinomial_classifier_lagrange, lagrange_mixin)

def test_survival_lagrange():
    # sklearn_survival_lagrange inherited classifier_mixin
    assert issubclass(sklearn_survival_lagrange, survival_mixin_lagrange)
    est = sklearn_survival_lagrange(_lagrange_atom, {'frac': 0.5})
    assert 'enet_alpha' in est.get_params()

def test_loglike_factory_not_implemented():
    with pytest.raises(NotImplementedError):
        base_mixin(rr.l1norm, {})._loglike_factory(None, None)

def test_score_sample_weight():
    # sample_weight was ignored unless y had a case weight column
    rng = np.random.default_rng(1)
    n, p = 50, 4
    X = rng.standard_normal((n, p))
    Y = X[:, 0] + rng.standard_normal(n)
    w = rng.uniform(0.2, 2, n)
    g = A.gaussian(rr.l1norm, {'shape': p, 'lagrange': .1}).fit(X, Y)
    gw = A.gaussian(rr.l1norm, {'shape': p, 'lagrange': .1}, case_weights=True)
    gw._coefs = g._coefs
    assert not np.isclose(g.score(X, Y, sample_weight=w), g.score(X, Y))
    np.testing.assert_allclose(g.score(X, Y, sample_weight=w),
                               gw.score(X, np.c_[Y, w]))
    # with a case weight column, the two are multiplied
    np.testing.assert_allclose(gw.score(X, np.c_[Y, w], sample_weight=w),
                               gw.score(X, np.c_[Y, w**2]))

@pytest.mark.parametrize('lagrange_form', [False, True])
def test_quadratic(lagrange_form):
    # the quadratic argument was not used in fit
    rng = np.random.default_rng(2)
    n, p = 50, 4
    X = rng.standard_normal((n, p))
    Y = X[:, 0] + rng.standard_normal(n)
    q = rr.identity_quadratic(20., 0, rng.standard_normal(p), 0)
    solve_args = {'tol': 1e-12, 'min_its': 200}
    if lagrange_form:
        est = A.gaussian_lagrange(_lagrange_atom, {'frac': 0.3}, quadratic=q,
                                  solve_args=solve_args)
    else:
        est = A.gaussian(rr.l1norm, {'shape': p, 'lagrange': 3.}, quadratic=q,
                         solve_args=solve_args)
    est.fit(X, Y)
    loss = est._loglike
    if lagrange_form:
        null_grad = (loss.smooth_objective(np.zeros(p), 'grad') +
                     q.objective(np.zeros(p), 'grad'))
        lagrange = 0.3 * np.fabs(null_grad).max()
    else:
        lagrange = 3.
    direct = rr.simple_problem(loss, rr.l1norm(p, lagrange=lagrange)).solve(q, **solve_args)
    np.testing.assert_allclose(est._coefs, direct, atol=1e-8)
    assert 'quadratic' in clone(est).get_params()

# gh-76

@pytest.mark.parametrize('family', ['logistic', 'probit', 'cloglog'])
@pytest.mark.parametrize('kind', ['classifier', 'regression'])
def test_accuracy(family, kind):
    # accuracy compared the linear predictor with the labels
    X, y, w = _binary_data()
    if kind == 'classifier':
        make = lambda **kw: getattr(A, family + '_classifier')(
            rr.l1norm, {'shape': X.shape[1], 'lagrange': .5}, **kw)
    else:
        make = lambda **kw: getattr(A, family)(
            rr.l1norm, {'shape': X.shape[1], 'lagrange': .5}, score_method='accuracy', **kw)
    est = make().fit(X, y)
    # the label is predicted where the fitted probability exceeds 1/2
    threshold = np.log(np.log(2)) if family == 'cloglog' else 0
    labels = X.dot(est._coefs) > threshold
    if kind == 'classifier':
        np.testing.assert_array_equal(est.predict(X), labels)
    np.testing.assert_allclose(est.score(X, y), np.mean(labels == y))
    assert est.score(X, y) > 0.7

    # weighted accuracy, with case weights or with sample_weight
    estw = make(case_weights=True)
    estw._coefs = est._coefs
    np.testing.assert_allclose(estw.score(X, np.c_[y, w]),
                               np.average(labels == y, weights=w))
    np.testing.assert_allclose(est.score(X, y, sample_weight=w),
                               np.average(labels == y, weights=w))

def test_cloglog_predict():
    # predict thresholded the linear predictor at 0, i.e. a probability of
    # 1 - exp(-1), not 1/2
    X, y, _ = _binary_data()
    clf = A.cloglog_classifier(rr.l1norm, {'shape': X.shape[1], 'lagrange': .5}).fit(X, y)
    np.testing.assert_array_equal(clf.predict(X), clf.predict_proba(X) > 0.5)

def _multinomial_data(seed=3, n=70, p=4, J=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = rng.standard_normal((p, J))
    eta = X.dot(beta)
    probs = np.exp(eta) / np.exp(eta).sum(1)[:, None]
    labels = np.array([rng.choice(J, p=pr) for pr in probs])
    counts = np.zeros((n, J))
    counts[np.arange(n), labels] = 1
    return X, counts, rng.uniform(0.2, 2, n)

def test_multinomial_classifier():
    # predict_proba failed to broadcast, and accuracy compared the linear
    # predictor with the counts
    X, counts, w = _multinomial_data()
    J = counts.shape[1]
    clf = A.multinomial_classifier(rr.l1norm, {'shape': (X.shape[1], J), 'lagrange': .1}).fit(X, counts)
    probs = clf.predict_proba(X)
    np.testing.assert_allclose(probs.sum(1), 1)
    eta = X.dot(clf._coefs)
    np.testing.assert_allclose(probs, np.exp(eta) / np.exp(eta).sum(1)[:, None])
    labels = clf.predict(X)
    np.testing.assert_array_equal(labels, probs.argmax(1))
    np.testing.assert_allclose(clf.score(X, counts), np.mean(labels == counts.argmax(1)))
    np.testing.assert_allclose(clf.score(X, counts, sample_weight=w),
                               np.average(labels == counts.argmax(1), weights=w))

def _families():
    X, y, w = _binary_data()
    n, p = X.shape
    out = {}
    for fam in ['logistic', 'probit', 'cloglog']:
        out[fam] = (getattr(A, fam), (rr.l1norm, {'shape': p, 'lagrange': .5}), X, y, w)
    rng = np.random.default_rng(4)
    Y = X[:, 0] + rng.standard_normal(n)
    out['gaussian'] = (A.gaussian, (rr.l1norm, {'shape': p, 'lagrange': .5}), X, Y, w)
    T = rng.exponential(size=n)
    S = (rng.uniform(size=n) < 0.7).astype(float)
    out['cox'] = (A.cox, (rr.l1norm, {'shape': p, 'lagrange': .5}), X, np.c_[T, S], w)
    Xm, counts, wm = _multinomial_data()
    out['multinomial'] = (A.multinomial, (rr.l1norm, {'shape': (Xm.shape[1], counts.shape[1]), 'lagrange': .1}),
                          Xm, counts, wm)
    return out

@pytest.mark.parametrize('family', ['logistic', 'probit', 'cloglog', 'gaussian', 'cox', 'multinomial'])
def test_mean_deviance(family):
    # cloglog's mean deviance was not divided by n
    klass, args, X, y, _ = _families()[family]
    dev = klass(*args, score_method='deviance').fit(X, y)
    mdev = klass(*args, score_method='mean_deviance').fit(X, y)
    np.testing.assert_allclose(mdev.score(X, y) * X.shape[0], dev.score(X, y))

@pytest.mark.parametrize('family', ['logistic', 'probit', 'cloglog', 'gaussian'])
@pytest.mark.parametrize('weighted', [False, True])
def test_R2_null_model(family, weighted):
    # the R2 of the best constant linear predictor is 0: cloglog used the
    # wrong inverse link, and the null mean ignored case weights
    klass, args, X, y, w = _families()[family]
    est = klass(*args, score_method='R2')
    cw = w if weighted else None
    R2 = lambda c: est._saturated_score(c * np.ones(X.shape[0]), y, case_weights=cw)
    best = minimize_scalar(lambda c: -R2(c), bracket=(-1, 1), tol=1e-10)
    np.testing.assert_allclose(-best.fun, 0, atol=1e-7)

def test_R2_binomial_trials():
    # with a (successes, trials) response the null probability averaged the
    # two columns
    rng = np.random.default_rng(5)
    n = 40
    trials = rng.integers(1, 6, n).astype(float)
    successes = rng.binomial(trials.astype(int), 0.3).astype(float)
    y = np.c_[successes, trials]
    for fam in ['logistic', 'probit', 'cloglog']:
        est = getattr(A, fam)(rr.l1norm, {'shape': 2, 'lagrange': 1.}, score_method='R2')
        R2 = lambda c: est._saturated_score(c * np.ones(n), y)
        best = minimize_scalar(lambda c: -R2(c), bracket=(-1, 1), tol=1e-10)
        np.testing.assert_allclose(-best.fun, 0, atol=1e-7)

def test_R2_cox_multinomial_null():
    # Cox: SST was missing the factor of 2, so the null model scored -1;
    # multinomial: the null predictor had the wrong shape
    klass, args, X, y, w = _families()['cox']
    est = klass(*args, score_method='R2')
    np.testing.assert_allclose(est._saturated_score(np.zeros(X.shape[0]), y), 0)

    klass, args, X, counts, w = _families()['multinomial']
    est = klass(*args, score_method='R2')
    for cw in [None, w]:
        weights = np.ones(X.shape[0]) if cw is None else cw
        props = (weights[:, None] * counts).sum(0) / weights.sum()
        null = np.ones((X.shape[0], 1)) * np.log(props)[None, :]
        np.testing.assert_allclose(est._saturated_score(null, counts, case_weights=cw), 0, atol=1e-12)
        # other constant predictors do worse
        assert est._saturated_score(null + [0.3, 0, 0], counts, case_weights=cw) < 0
    est.fit(X, counts)
    assert 0 < est.score(X, counts) < 1
