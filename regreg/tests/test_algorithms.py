"""
Tests of the arguments of `regreg.algorithms.FISTA.fit` (gh-77).
"""
import numpy as np

import regreg.api as rr

def _problem(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((50, 10))
    Y = rng.standard_normal(50)
    return rr.simple_problem(rr.squared_error(X, Y), rr.l1norm(10, lagrange=1.))

def test_ista():
    # FISTA=False used Nesterov weights anyway

    solver = rr.FISTA(_problem())
    solver.fit(FISTA=False, max_its=5, min_its=5, tol=1e-20)
    assert solver.weight_old == 1.

    solver = rr.FISTA(_problem())
    solver.fit(FISTA=True, max_its=5, min_its=5, tol=1e-20)
    assert solver.weight_old > 1.

    # both converge to the same point
    a = _problem()
    rr.FISTA(a).fit(FISTA=False, tol=1e-14, min_its=100, max_its=20000)
    b = _problem()
    rr.FISTA(b).fit(FISTA=True, tol=1e-14, min_its=100)
    np.testing.assert_allclose(a.coefs, b.coefs, atol=1e-6)

def test_zero_tolerance_and_iterations():
    # tol=0, min_its=0 and max_its=0 were replaced by the defaults

    solver = rr.FISTA(_problem())
    hist = solver.fit(tol=0, max_its=300, min_its=1, monotonicity_restart=False)
    assert len(hist) == 300

    solver = rr.FISTA(_problem())
    hist = solver.fit(max_its=0)
    assert len(hist) == 0

def test_prox_control():
    # prox_control was discarded, and backtrack referred to an undefined name

    solver = rr.FISTA(_problem())
    solver.fit(prox_control={'max_its': 3}, max_its=3)
    assert solver.prox_control == {'max_its': 3}
    coefs, smooth = solver.backtrack(0)
    assert np.all(np.isfinite(coefs))

    # the default is not shared between fits
    s1, s2 = rr.FISTA(_problem()), rr.FISTA(_problem())
    s1.fit(max_its=2)
    s2.fit(max_its=2)
    assert s1.prox_control == {} and s1.prox_control is not s2.prox_control

def test_restart():
    # restart reset an unused local instead of the Nesterov weight

    solver = rr.FISTA(_problem())
    solver.fit(restart=3, max_its=2, min_its=2, tol=1e-20,
               monotonicity_restart=False)  # restart at itercount = 2
    assert solver.weight_old > 1.
    solver = rr.FISTA(_problem())
    solver.fit(restart=3, max_its=3, min_its=3, tol=1e-20,
               monotonicity_restart=False)
    # the weight was reset to 1 at the start of the third iteration, then
    # updated once
    assert np.isclose(solver.weight_old, 0.5 * (1 + np.sqrt(5)))
