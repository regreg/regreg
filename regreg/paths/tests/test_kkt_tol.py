"""
Tests of the KKT tolerance for groups at zero (gh-99).

The path algorithms accepted a solution once the KKT conditions held to
within 1%, so a group at the edge of the active set could be left out.
"""
import io
import contextlib

import numpy as np
import pytest

import regreg.api as rr
from regreg.atoms.mixed_lasso import check_KKT as mixed_lasso_check_KKT
from regreg.paths import lasso, group_lasso, sparse_group_lasso, sparse_group_block
from regreg.paths import mixed_lasso, strong_rules, warm_start
from regreg.paths.basil import basil
from regreg.paths.sparse_group_lasso import _gauge_function_dual_strong

def _scale_to(grad, value, lagrange):
    # rescale grad so its dual norm is value * lagrange
    return grad * value * lagrange

def test_lasso_check_KKT():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    path = lasso.gaussian(X, rng.standard_normal(20), np.ones(4))
    grad = np.zeros(4)
    grad[2] = 1.005 * 2
    _, ranks = path.check_KKT(grad, np.zeros(4), 2.)
    assert ranks[2] >= 0
    _, ranks = path.check_KKT(grad, np.zeros(4), 2., kkt_tol=1e-2)
    assert np.all(ranks < 0)
    path.kkt_tol = 1e-2
    _, ranks = path.check_KKT(grad, np.zeros(4), 2.)
    assert np.all(ranks < 0)

def test_group_lasso_check_KKT():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    groups = np.repeat(np.arange(3), 2)
    path = group_lasso.gaussian(X, rng.standard_normal(20), groups, weights={})
    grad = np.zeros(6)
    grad[2:4] = 1.
    w = path.penalty.weights[1]
    grad = grad / np.linalg.norm(grad) * 1.005 * 2 * w
    _, ranks = path.check_KKT(grad, np.zeros(6), 2.)
    assert ranks[1] >= 0
    _, ranks = path.check_KKT(grad, np.zeros(6), 2., kkt_tol=1e-2)
    assert np.all(ranks < 0)

def test_sparse_group_lasso_check_KKT():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    groups = np.repeat(np.arange(3), 2)
    path = sparse_group_lasso.gaussian(X, rng.standard_normal(20), groups,
                                       np.ones(6), weights={})
    grad = np.zeros(6)
    grad[2:4] = [1., 3.]
    l1w, l2w = path.penalty.lasso_weights[2:4], path.penalty.weights[1]
    grad /= _gauge_function_dual_strong(-grad[2:4], l1w, l2w)[0]
    grad *= 1.005 * 2
    _, ranks = path.check_KKT(grad, np.zeros(6), 2.)
    assert ranks[1] >= 0
    _, ranks = path.check_KKT(grad, np.zeros(6), 2., kkt_tol=1e-2)
    assert np.all(ranks < 0)

def test_sparse_group_block_check_KKT():
    from ..sparse_group_block import _check_KKT
    grad = np.zeros((3, 2))
    grad[1] = [1., 3.]
    grad /= _gauge_function_dual_strong(-grad[1], 1., 1.)[0]
    grad *= 1.005 * 2
    _, ranks = _check_KKT(grad, np.zeros((3, 2)), 1., 1., 2., kkt_tol=1e-6)
    assert ranks[1] >= 0
    _, ranks = _check_KKT(grad, np.zeros((3, 2)), 1., 1., 2., kkt_tol=1e-2)
    assert np.all(ranks < 0)

def test_mixed_lasso_atom_check_KKT():
    # l1 coordinate 0, group {1, 2}, positive part coordinate 3
    pen = rr.mixed_lasso([mixed_lasso.L1_PENALTY, 0, 0, mixed_lasso.POSITIVE_PART],
                         lagrange=2.)
    w = pen._weight_array[0]
    for grad, idx in [(np.array([1.005 * 2, 0, 0, 0]), [0]),
                      (np.array([0, 0.6, 0.8, 0]) * 1.005 * 2 * w, [1, 2]),
                      (np.array([0, 0, 0, -1.005 * 2]), [3])]:
        failing = mixed_lasso_check_KKT(pen, grad, np.zeros(4), 2.)
        assert not np.any(failing)
        failing = mixed_lasso_check_KKT(pen, grad, np.zeros(4), 2., kkt_tol=1e-6)
        assert np.all(failing[idx])
        assert np.sum(failing) == len(idx)

def _mixed_lasso_problem(seed):
    rng = np.random.default_rng(seed)
    n = 100
    X = rng.standard_normal((n, 10))
    U = rng.standard_normal((n, 2))
    Y = rng.standard_normal(n)
    Y += (X @ np.array([3, 4, 5] + [0] * 7) + U @ np.array([10, -5])) * 5
    Xn = np.hstack([np.ones((n, 1)), (X - X.mean(0)) / X.std(0, ddof=1)])
    return Xn, Y

@pytest.mark.parametrize('seed', [22, 27])
def test_mixed_lasso_path_edge_group(seed):
    # seeds at which a group just entering the model was left out
    Xn, Y = _mixed_lasso_problem(seed)
    structure = [0] * 7 + [1] * 3
    loss = rr.glm.gaussian(Xn, Y)

    def max_error(**main_args):
        path = mixed_lasso.mixed_lasso_path.gaussian(Xn[:, 1:], Y,
                                                     penalty_structure=structure,
                                                     nstep=10)
        with contextlib.redirect_stdout(io.StringIO()):
            sol = path.main(inner_tol=1.e-12, **main_args)
        beta = np.asarray(sol['beta'].todense())
        error = 0
        for k, lam in enumerate(sol['lagrange'][1:], 1):
            pen = rr.mixed_lasso([mixed_lasso.UNPENALIZED] + structure, lagrange=lam)
            ref = rr.simple_problem(loss, pen).solve(tol=1.e-16, min_its=2000,
                                                     max_its=50000)
            error = max(error, np.fabs(beta[:, k] - ref).max())
        return error

    assert max_error() < 1e-6
    # the old tolerance reproduces the error
    assert max_error(kkt_tol=1e-2) > 1e-3

@pytest.mark.parametrize('maker', ['lasso', 'group_lasso', 'sparse_group_lasso'])
@pytest.mark.parametrize('algorithm', ['strong_rules', 'basil'])
def test_paths_match_warm_start(maker, algorithm):
    rng = np.random.default_rng(1001)
    n, p = 300, 60
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = [3, 4, 5, -3]
    Y = X @ beta / np.sqrt(n) + rng.standard_normal(n)
    groups = np.repeat(np.arange(p // 3), 3)

    def make():
        if maker == 'lasso':
            return lasso.gaussian(X, Y, np.ones(p))
        elif maker == 'group_lasso':
            return group_lasso.gaussian(X, Y, groups, weights={})
        return sparse_group_lasso.gaussian(X, Y, groups, np.ones(p), weights={})

    path1 = make()
    lagrange_seq = lasso.default_lagrange_sequence(path1.penalty, path1.grad_solution,
                                                   nstep=30, lagrange_proportion=0.1)
    initial = (path1.solution.copy(), path1.grad_solution.copy())
    with contextlib.redirect_stdout(io.StringIO()):
        if algorithm == 'strong_rules':
            beta1 = strong_rules(path1, lagrange_seq, initial, inner_tol=1e-12)['beta']
        else:
            beta1 = basil(path1, lagrange_seq, initial, inner_tol=1e-12,
                          step_nvar=10, step_lagrange=10)
    path2 = make()
    beta2 = warm_start(path2, lagrange_seq,
                       (path2.solution.copy(), path2.grad_solution.copy()),
                       inner_tol=1e-14)['beta']
    beta1, beta2 = np.asarray(beta1), np.asarray(beta2)
    # with the old 1% tolerance these differed by 3e-4 to 1.5e-3
    assert np.linalg.norm(beta1 - beta2) / np.linalg.norm(beta2) < 1e-4
