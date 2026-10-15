from copy import copy
from itertools import product

import numpy as np

import regreg.api as rr
from regreg.atoms.tests.test_seminorms import all_close


def test_class():
    '''
    runs several class methods on generic instance
    '''
    rng = np.random.default_rng(3900)

    n, p = 100, 20
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    loss = rr.squared_error(X, Y)
    pen = rr.l1norm(p, lagrange=1.)
    problem = rr.simple_problem(loss, pen)

    problem.latexify()

    for debug, coef_stop, max_its in product([True, False], [True, False], [5, 100]):
        rr.gengrad(problem, rr.power_L(X)**2, max_its=max_its, debug=debug, coef_stop=coef_stop)

def test_gengrad():
    rng = np.random.default_rng(3901)
    Z = rng.standard_normal(100) * 4
    p = rr.l1norm(100, lagrange=0.13)
    L = 0.14

    loss = rr.quadratic_loss.shift(Z, coef=L)
    problem = rr.simple_problem(loss, p)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10, debug=True)

    simple_coef = solver.composite.coefs
    prox_coef = p.proximal(rr.identity_quadratic(L, Z, 0, 0))

    p2 = rr.l1norm(100, lagrange=0.13)
    p2 = copy(p)
    p2.quadratic = rr.identity_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, debug=True)
    simple_nonsmooth_coef = solver.composite.coefs

    p = rr.l1norm(100, lagrange=0.13)
    p.quadratic = rr.identity_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p)
    simple_nonsmooth_gengrad = rr.gengrad(problem, L, tol=1.0e-10)

    p = rr.l1norm(100, lagrange=0.13)
    problem = rr.separable_problem.singleton(p, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10)
    separable_coef = solver.composite.coefs

    loss2 = rr.quadratic_loss.shift(Z, coef=0.6*L)
    loss2.quadratic = rr.identity_quadratic(0.4*L, Z, 0, 0)
    p.coefs *= 0
    problem2 = rr.simple_problem(loss2, p)
    loss2_coefs = problem2.solve(coef_stop=True)
    solver2 = rr.FISTA(problem2)
    solver2.fit(tol=1.0e-10, debug=True, coef_stop=True)

    all_close(prox_coef, simple_nonsmooth_gengrad, 'prox to nonsmooth gengrad', None)
    all_close(prox_coef, separable_coef, 'prox to separable', None)
    all_close(prox_coef, simple_nonsmooth_coef, 'prox to simple_nonsmooth', None)
    all_close(prox_coef, simple_coef, 'prox to simple', None)
    all_close(prox_coef, loss2_coefs, 'simple where loss has quadratic 1', None)
    all_close(prox_coef, solver2.composite.coefs, 'simple where loss has quadratic 2', None)

def test_gengrad_blocknorms():
    rng = np.random.default_rng(3902)
    Z = rng.standard_normal((10,10)) * 4
    p = rr.l1_l2((10,10), lagrange=0.13)
    dual = p.conjugate
    L = 0.23

    loss = rr.quadratic_loss.shift(Z, coef=L)
    problem = rr.simple_problem(loss, p)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10, debug=True)
    simple_coef = solver.composite.coefs

    q = rr.identity_quadratic(L, Z, 0, 0)
    prox_coef = p.proximal(q)

    p2 = copy(p)
    p2.quadratic = rr.identity_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p2)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-14, debug=True)
    simple_nonsmooth_coef = solver.composite.coefs

    p = rr.l1_l2((10,10), lagrange=0.13)
    p.quadratic = rr.identity_quadratic(L, Z, 0, 0)
    problem = rr.simple_problem.nonsmooth(p)
    simple_nonsmooth_gengrad = rr.gengrad(problem, L, tol=1.0e-10)

    p = rr.l1_l2((10,10), lagrange=0.13)
    problem = rr.separable_problem.singleton(p, loss)
    solver = rr.FISTA(problem)
    solver.fit(tol=1.0e-10)
    separable_coef = solver.composite.coefs

    all_close(prox_coef, simple_coef, 'prox to simple', None)
    all_close(prox_coef, simple_nonsmooth_gengrad, 'prox to nonsmooth gengrad', None)
    all_close(prox_coef, separable_coef, 'prox to separable', None)
    all_close(prox_coef, simple_nonsmooth_coef, 'prox to simple_nonsmooth', None)



def test_gengrad_smooth_loss():
    # gh-84: gengrad did a full solve per step, minimizing 2 * loss + penalty

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    Y = rng.standard_normal(100)
    pen = rr.l1norm(5, lagrange=5.)
    ref = rr.simple_problem(rr.squared_error(X, Y), pen).solve(tol=1e-14, min_its=500)
    soln = rr.gengrad(rr.simple_problem(rr.squared_error(X, Y), pen),
                      np.linalg.norm(X, 2)**2, tol=1e-14, max_its=10000)
    np.testing.assert_allclose(soln, ref, atol=1e-8)

def test_gengrad_negative_objective():
    # gh-84: the relative tolerance used max(vnew, 1) rather than
    # max(|vnew|, 1), so a large negative objective got an absolute
    # tolerance instead of a (looser) relative one

    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 5))
    Y = rng.standard_normal(100)

    def iterations(constant):
        problem = rr.simple_problem(rr.squared_error(X, Y), rr.l1norm(5, lagrange=5.))
        problem.quadratic = rr.identity_quadratic(0, 0, 0, constant)
        calls = []
        smooth_objective = problem.smooth_objective
        def counted(*args, **kwargs):
            calls.append(1)
            return smooth_objective(*args, **kwargs)
        problem.smooth_objective = counted
        rr.gengrad(problem, np.linalg.norm(X, 2)**2, tol=1e-6, max_its=5000)
        return len(calls)

    assert iterations(-1e6) <= iterations(0)
