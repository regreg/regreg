"""
Solving basis pursuit with TFOCS
"""
import numpy as np

import regreg.api as rr

def test_tfocs():
    rng = np.random.default_rng(4200)
    n, p = 100, 200
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 3
    Y = rng.standard_normal(n) + np.dot(X, beta)

    lscoef = np.dot(np.linalg.pinv(X), Y)
    minimum_l2 = np.linalg.norm(Y - np.dot(X, lscoef))
    maximum_l2 = np.linalg.norm(Y)

    l2bound = (minimum_l2 + maximum_l2) * 0.5

    constraint = rr.l2norm.affine(X,-Y,bound=l2bound)
    T, conjugate = constraint.dual
    l1 = rr.l1norm(p, lagrange=1)

    primal, dual = rr.tfocs(l1, T, conjugate, coef_tol=1.e-4)
    assert np.fabs(np.linalg.norm(Y - np.dot(X, primal)) - l2bound) <= l2bound * 1.e-3

def _basis_pursuit(seed=0, n=60, p=150, k=6):
    from scipy.optimize import linprog
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:k] = rng.standard_normal(k) * 5
    Y = X @ beta
    lp = linprog(np.ones(2 * p), A_eq=np.hstack([X, -X]), b_eq=Y,
                 bounds=(0, None), method='highs')
    return X, Y, lp.x[:p] - lp.x[p:]

import pytest, warnings

@pytest.mark.parametrize('eps', [10., 1., 0.1, 0.01])
def test_tfocs_accuracy(eps):
    # each smoothed dual problem was solved only to relative tolerance
    # max(eps, tol), so the answer depended on eps (errors of 1e-3 to 0.7);
    # the smoothing parameter should only affect the speed
    X, Y, bp = _basis_pursuit()
    p = X.shape[1]
    l1 = rr.l1norm(p, lagrange=1)
    transform, zero = rr.zero_constraint.affine(X, -Y).dual
    with warnings.catch_warnings():
        warnings.simplefilter('error')        # converged: no 'not solved' warning
        primal, dual = rr.tfocs(l1, transform, zero, epsilon=[eps] * 200)
    np.testing.assert_allclose(primal, bp, atol=1e-6)

@pytest.mark.parametrize('eps', [10., 1., 0.1])
def test_nesta_accuracy(eps):
    X, Y, bp = _basis_pursuit()
    p = X.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        primal, dual = rr.nesta(None, rr.l1norm(p, lagrange=1),
                                rr.zero_constraint.affine(X, -Y), epsilon=[eps] * 200)
    np.testing.assert_allclose(primal, bp, atol=1e-5)
    # the multiplier: X^T u is a subgradient of the l1 norm at the solution
    g = X.T @ dual
    active = np.abs(bp) > 1e-6
    np.testing.assert_allclose(np.fabs(g[active]), 1, atol=1e-3)

def test_tfocs_no_state_from_atom():
    # the inner problems shared their coefficients with the atoms passed in,
    # so whatever an atom held from an earlier solve became the starting point
    X, Y, bp = _basis_pursuit()
    p = X.shape[1]
    l1 = rr.l1norm(p, lagrange=1)
    transform, zero = rr.zero_constraint.affine(X, -Y).dual
    zero.coefs[:] = 1e3
    primal, _ = rr.tfocs(l1, transform, zero, epsilon=[1.] * 200)
    np.testing.assert_allclose(primal, bp, atol=1e-6)
    l1n = rr.l1norm(p, lagrange=1)
    l1n.coefs[:] = 1e3
    primal, _ = rr.nesta(None, l1n, rr.zero_constraint.affine(X, -Y), epsilon=[1.] * 200)
    np.testing.assert_allclose(primal, bp, atol=1e-5)

def test_not_solved_warning():
    # the warning fired partway through whenever len(epsilon) > max_iters;
    # it is for running out of outer iterations without converging
    X, Y, bp = _basis_pursuit()
    p = X.shape[1]
    l1 = rr.l1norm(p, lagrange=1)
    transform, zero = rr.zero_constraint.affine(X, -Y).dual
    with pytest.warns(UserWarning, match='not solved'):
        rr.tfocs(l1, transform, zero, epsilon=[1e4] * 3)
