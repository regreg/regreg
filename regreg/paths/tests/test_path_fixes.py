"""
Regression tests for the path bugs in gh-108.
"""
import numpy as np
import pytest

import regreg.api as rr
from regreg.identity_quadratic import identity_quadratic as iq
from regreg.smooth.quadratic import quadratic_loss
from regreg.smooth.glm import gaussian_loglike, logistic_loglike
from regreg.paths import lasso, strong_rules, warm_start
from regreg.paths.lasso import lasso_path
from regreg.paths.basil import basil, basil_inner_loop
from regreg.paths import basil as basil_module
from regreg.paths import cross_validate as cv_module
from regreg.paths.mixed_lasso import mixed_lasso_path, UNPENALIZED

def test_mixed_lasso_path_elastic_net():
    # the elastic net entered grad() and the KKT checks but not the
    # restricted problems that were solved, so it had no effect on the fit
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    Y = X[:, 0] + rng.standard_normal(80)
    coef = 10.
    path = mixed_lasso_path.gaussian(X, Y, nstep=6, lagrange_proportion=0.1,
                                     elastic_net=iq(coef, 0, 0, 0))
    sol = path.main(inner_tol=1e-12)
    beta = np.asarray(sol['beta'].todense())
    ps = path.penalty_structure
    for k, lam in enumerate(sol['lagrange']):
        w = coef * lam * (ps != UNPENALIZED)
        loss = rr.smooth_sum([path.loss, quadratic_loss(w.shape, w, Qdiag=True)])
        direct = rr.simple_problem(loss, rr.mixed_lasso(ps, lam, weights=path.group_weights)).solve(tol=1e-14, min_its=500)
        np.testing.assert_allclose(beta[:, k], direct, atol=1e-5)

def test_cross_validate_first_lagrange():
    # each fold started from the full-data null fit (the intercept fitted on
    # the test fold too), so the first Lagrange parameter's score was wrong
    model_selection = pytest.importorskip('sklearn.model_selection')
    rng = np.random.default_rng(0)
    n, p = 60, 5
    X = np.c_[np.ones(n), rng.standard_normal((n, p - 1))]
    Y = 3 + X[:, 1] + rng.standard_normal(n)
    w = np.r_[0., np.ones(p - 1)]
    path = lasso.gaussian(X, Y, w)
    lams = lasso.default_lagrange_sequence(path.penalty, path.grad_solution, nstep=5)
    kf = model_selection.KFold(4)
    mean, _ = cv_module.cross_validate(path, lams, cv=kf, n_jobs=1, verbose=False, inner_tol=1e-12)
    scores = []
    for tr, te in kf.split(np.arange(n)):
        tp = lasso_path(gaussian_loglike((len(tr),), Y[tr], coef=n / len(tr)), X[tr], w)
        beta = strong_rules(tp, lams, (tp.solution, tp.grad_solution), inner_tol=1e-12)['beta']
        lin = X[te] @ beta.T
        scores.append([np.linalg.norm(lin[:, i] - Y[te])**2 / len(te) for i in range(lin.shape[1])])
    np.testing.assert_allclose(mean, np.mean(scores, 0), rtol=1e-5)

def test_cross_validate_basil():
    # basil returns an array, which cross_validate indexed as result['beta']
    model_selection = pytest.importorskip('sklearn.model_selection')
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 5))
    Y = X[:, 0] + rng.standard_normal(60)
    path = lasso.gaussian(X, Y, np.ones(5))
    lams = lasso.default_lagrange_sequence(path.penalty, path.grad_solution, nstep=5)
    mean, sd = cv_module.cross_validate(path, lams, cv=model_selection.KFold(3), n_jobs=1,
                                        verbose=False, path_fitter=basil)
    assert mean.shape == (5,) and np.all(np.isfinite(mean))

def _review_data(family, n=100, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    eta = X[:, :3] @ np.array([1., -1, .5])
    if family == 'gaussian':
        return X, gaussian_loglike((n,), eta + rng.standard_normal(n))
    Y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, logistic_loglike((n,), Y)

@pytest.mark.parametrize('family,unpenalized', [('gaussian', True), ('logistic', False), ('logistic', True)])
def test_basil_check_active_terminates(family, unpenalized, monkeypatch):
    # a batch that solved nothing without enlarging the candidate set made
    # basil repeat the same call forever
    calls = []
    inner = basil_module.basil_inner_loop
    def counted(*args, **kwargs):
        calls.append(1)
        if len(calls) > 200:
            raise RuntimeError('basil is not making progress')
        return inner(*args, **kwargs)
    monkeypatch.setattr(basil_module, 'basil_inner_loop', counted)
    X, L = _review_data(family)
    w = np.ones(12)
    if unpenalized:
        w[0] = 0
    path = lasso_path(L, X, w)
    lams = lasso.default_lagrange_sequence(path.penalty, path.grad_solution, nstep=5)
    sol = basil(path, lams, (path.solution.copy(), path.grad_solution.copy()),
                check_active=True, inner_tol=1e-10)
    X, L = _review_data(family)
    p2 = lasso_path(L, X, w)
    ref = warm_start(p2, lams, (p2.solution.copy(), p2.grad_solution.copy()), inner_tol=1e-12)['beta']
    np.testing.assert_allclose(sol, ref, atol=1e-3)
