"""
Tests of the path drivers `basil`, `warm_start` and `cross_validate` (gh-82).
"""
import numpy as np
import pytest

from .. import (lasso,
                group_lasso,
                sparse_group_block,
                sparse_group_lasso,
                warm_start,
                strong_rules)
from .. import basil as basil_mod
from .. import cross_validate as cv_mod

def _lasso_path(seed=0, n=100, p=20):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Y = X[:, :3].sum(1) + rng.standard_normal(n)
    return lasso.gaussian(X, Y, np.ones(p))

def _block_path(seed=0, n=100, p=20, q=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    Y = X[:, :3].dot(np.ones((3, q))) + rng.standard_normal((n, q))
    return sparse_group_block.multiresponse_gaussian(X, Y, 1, np.sqrt(q))

@pytest.mark.parametrize('make_path', [_lasso_path, _block_path])
def test_basil_warm_start(make_path, monkeypatch):
    # basil computed the next warm start but never used it, and dropped
    # initial_step, verbose and check_active

    path = make_path()
    lagrange_seq = lasso.default_lagrange_sequence(path.penalty,
                                                   path.grad_solution,
                                                   nstep=12)
    calls = []
    inner_loop = basil_mod.basil_inner_loop
    def spy(path_obj, lagrange_subseq, initial_data, **kw):
        calls.append((initial_data[0].copy(), kw))
        result = inner_loop(path_obj, lagrange_subseq, initial_data, **kw)
        calls[-1] += (result[1],)
        return result
    monkeypatch.setattr(basil_mod, 'basil_inner_loop', spy)

    n_power_L = []
    power_L = basil_mod.power_L
    monkeypatch.setattr(basil_mod, 'power_L',
                        lambda *a, **k: n_power_L.append(1) or power_L(*a, **k))

    sol = basil_mod.basil(path,
                          lagrange_seq,
                          (path.solution.copy(), path.grad_solution.copy()),
                          inner_tol=1.e-12,
                          step_nvar=5,
                          step_lagrange=4,
                          check_active=True)
    assert len(calls) >= 3
    assert len(n_power_L) == 1
    for i, (initial, kw, solved) in enumerate(calls):
        assert kw['check_active'] and kw['initial_step'] is not None
        if i > 0 and len(calls[i-1][2]) > 0:
            # starts from the last solution of the previous batch
            np.testing.assert_allclose(initial, calls[i-1][2][-1].T)
    assert np.any(calls[-1][0] != 0)

    ref = warm_start(make_path(),
                     lagrange_seq,
                     (path.solution.copy(), path.grad_solution.copy()),
                     inner_tol=1.e-12)['beta']
    np.testing.assert_allclose(sol, ref, atol=1.e-4)

def test_basil_inner_loop_keeps_gradient():
    # basil_inner_loop wrote into the caller's gradient array
    path = _lasso_path()
    lagrange_seq = lasso.default_lagrange_sequence(path.penalty,
                                                   path.grad_solution,
                                                   nstep=10)
    grad = path.grad_solution.copy()
    basil_mod.basil_inner_loop(path,
                               lagrange_seq[:5],
                               (path.solution, path.grad_solution),
                               step_nvar=5)
    np.testing.assert_array_equal(path.grad_solution, grad)

def test_warm_start_solve_args():
    # warm_start wrote tol and start_step into the caller's (or the default)
    # solve_args
    path = _lasso_path()
    lagrange_seq = lasso.default_lagrange_sequence(path.penalty,
                                                   path.grad_solution,
                                                   nstep=3)
    solve_args = {'min_its': 5}
    warm_start(path, lagrange_seq, (path.solution, path.grad_solution),
               solve_args=solve_args)
    assert solve_args == {'min_its': 5}
    assert warm_start.__defaults__[-1] is None

def test_make_folds():
    nsample = 30
    folds = cv_mod._make_folds(3, nsample)
    assert len(folds) == 3
    for train, test in folds:
        assert train.dtype == bool and np.all(train == ~test)
    np.testing.assert_array_equal(sum(test for _, test in folds), 1)

    # a sklearn splitter is used as given
    model_selection = pytest.importorskip('sklearn.model_selection')
    folds = cv_mod._make_folds(model_selection.KFold(5), nsample)
    assert len(folds) == 5
    np.testing.assert_array_equal(folds[0][1], np.arange(nsample) < 6)

    # group labels keep each group in a single fold
    labels = np.repeat(np.arange(6), 5)
    folds = cv_mod._make_folds(3, nsample, group_labels=labels)
    assert len(folds) == 3
    for _, test in folds:
        for g in np.unique(labels[test]):
            assert np.all(test[labels == g])

def test_cross_validate_splitter():
    # cross_validate ignored sklearn, so cv had to be an int
    model_selection = pytest.importorskip('sklearn.model_selection')
    path = _lasso_path()
    lagrange_seq = lasso.default_lagrange_sequence(path.penalty,
                                                   path.grad_solution,
                                                   nstep=5)
    mean1, sd1 = cv_mod.cross_validate(path,
                                       lagrange_seq,
                                       cv=model_selection.KFold(4),
                                       n_jobs=1,
                                       verbose=False)
    # the same folds, by hand
    nsample = path.saturated_loss.shape[0]
    scores = []
    for train, test in model_selection.KFold(4).split(np.arange(nsample)):
        train_path = path.subsample(train)
        train_path.saturated_loss.coef *= nsample / train.shape[0]
        beta = strong_rules(train_path,
                            lagrange_seq,
                            (path.solution, path.grad_solution))['beta']
        linpred = path.X.dot(beta.T)[test]
        y = path.saturated_loss.data[test]
        scores.append([np.linalg.norm(linpred[:, i] - y)**2 / y.shape[0]
                       for i in range(linpred.shape[1])])
    np.testing.assert_allclose(mean1, np.mean(scores, 0), rtol=1.e-4)
    np.testing.assert_allclose(sd1, np.std(scores, 0), rtol=1.e-3)

    labels = np.repeat(np.arange(10), 10)
    mean2, _ = cv_mod.cross_validate(path,
                                     lagrange_seq,
                                     cv=4,
                                     group_labels=labels,
                                     n_jobs=1,
                                     verbose=False)
    assert mean2.shape == (5,)

def _unpenalized_paths():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 9))
    Y = X[:, :3].sum(1) + rng.standard_normal(100)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    weights = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1.])
    return {'lasso': lambda: lasso.gaussian(X, Y, weights),
            'group_lasso': lambda: group_lasso.gaussian(X, Y, groups,
                                                        weights={0: 0., 1: 1., 2: 1.}),
            'sparse_group_lasso': lambda: sparse_group_lasso.gaussian(X, Y, groups, weights,
                                                                      weights={0: 0., 1: 1., 2: 1.})}

@pytest.mark.parametrize('kind', ['lasso', 'group_lasso', 'sparse_group_lasso'])
def test_strong_rules_unpenalized(kind):
    # strong_rules left the unpenalized groups out of every subproblem, so
    # their coefficients stayed at the null model's values along the path
    make_path = _unpenalized_paths()[kind]
    path = make_path()
    lagrange_seq = lasso.default_lagrange_sequence(path.penalty,
                                                   path.grad_solution,
                                                   nstep=5)
    beta = strong_rules(path,
                        lagrange_seq,
                        (path.solution, path.grad_solution),
                        inner_tol=1.e-12)['beta']
    path = make_path()
    ref = warm_start(path,
                     lagrange_seq,
                     (path.solution, path.grad_solution),
                     inner_tol=1.e-14)['beta']
    # the first row is the null solution passed in
    np.testing.assert_allclose(beta[1:], ref[1:], atol=1.e-6)
