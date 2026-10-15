"""
Regression tests for the problem / solver bugs in gh-108.
"""
import numpy as np

import regreg.api as rr
from regreg.identity_quadratic import identity_quadratic as iq
from regreg.problems.separable import has_overlap

def test_container_uses_its_quadratics():
    # container.proximal solved the dual with the total quadratic but rebuilt
    # the primal from the proximal quadratic alone
    y = np.array([1., 2, -1, -3, 2])
    v = np.array([.5, -1, .3, .2, -.4])
    exact = np.sign(y - v) * np.maximum(np.fabs(y - v) - 1, 0) / 3
    c = rr.container(rr.squared_error(np.identity(5), y),
                     rr.l1norm.linear(np.identity(5), lagrange=1))
    np.testing.assert_allclose(c.solve(quadratic=iq(2, 0, v, 0), tol=1e-12), exact, atol=1e-8)
    loss = rr.squared_error(np.identity(5), y)
    loss.quadratic = iq(2, 0, v, 0)
    c = rr.container(loss, rr.l1norm.linear(np.identity(5), lagrange=1))
    np.testing.assert_allclose(c.solve(tol=1e-12), exact, atol=1e-8)

def test_has_overlap_identical_groups():
    assert has_overlap((6,), [slice(0, 3), slice(0, 3)])
    assert not has_overlap((6,), [slice(0, 3), slice(3, 6)])

def test_separable_shapes_with_center():
    # with `shapes`, only the linear term was reshaped, which failed for a
    # quadratic with a center (and reshaped the caller's array)
    rng = np.random.default_rng(7)
    sep = rr.separable((6,), [rr.nuclear_norm((2, 2), lagrange=1.), rr.l1norm(2, lagrange=1.)],
                       [slice(0, 4), slice(4, 6)], shapes=[(2, 2), None])
    for q in [iq(2, rng.standard_normal(6), 0, 0),
              iq(2, rng.standard_normal(6), rng.standard_normal(6), 0)]:
        c = q.collapsed()
        z = -c.linear_term / c.coef
        ref = np.r_[rr.nuclear_norm((2, 2), lagrange=1.).proximal(iq(c.coef, z[:4].reshape(2, 2), 0, 0)).reshape(-1),
                    rr.l1norm(2, lagrange=1.).proximal(iq(c.coef, z[4:], 0, 0))]
        np.testing.assert_allclose(sep.proximal(q), ref, atol=1e-12)

def test_fista_returns_final_step():
    # on convergence the solution returned was the previous iterate, not the
    # final (accepted) proximal step. With X = I one proximal step from 0
    # (step 1) is the exact lasso solution, soft(Y, lambda); a huge tol
    # makes FISTA stop right after that step
    Y = np.array([3., -1, .5, -4, 2])
    lam = 1.
    problem = rr.simple_problem(rr.squared_error(np.identity(5), Y), rr.l1norm(5, lagrange=lam))
    solver = rr.FISTA(problem)
    solver.perform_backtrack = False
    problem.lipschitz = 1.
    solver.fit(tol=1e10, min_its=0, max_its=10)
    np.testing.assert_allclose(problem.coefs, np.sign(Y) * np.maximum(np.fabs(Y) - lam, 0))
