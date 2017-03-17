from __future__ import print_function, division, absolute_import

from copy import copy

import numpy as np

import scipy.optimize

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_l1prox():
    '''
    this test verifies that the l1 prox in lagrange form can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, lagrange=0.3)
    ww = np.random.standard_normal(4)*3
    ab = l1.proximal(rr.identity_quadratic(0.5, ww, 0,0))

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.simple_problem.nonsmooth(l1c)
    solver = rr.FISTA(a)
    solver.fit(tol=1.e-10, min_its=100)

    ad = a.coefs

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.dual_problem.fromprimal(l1c)

    # to ensure code is tested
    a.latexify()
    a.quadratic
    a.solve(return_optimum=True)

    solver = rr.FISTA(a)
    solver.fit(tol=1.0e-14, min_its=100)

    ac = a.primal

    np.testing.assert_allclose(ac + 0.1, ab + 0.1, rtol=1.e-4)
    np.testing.assert_allclose(ac + 0.1, ad + 0.1, rtol=1.e-4)

@set_seed_for_test()
def test_l1prox_bound():
    '''
    this test verifies that the l1 prox in bound form can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, bound=2.)
    ww = np.random.standard_normal(4)*2
    ab = l1.proximal(rr.identity_quadratic(0.5, ww, 0, 0))

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.simple_problem.nonsmooth(l1c)
    solver = rr.FISTA(a)
    solver.fit(min_its=100)

    l1c = copy(l1)
    l1c.quadratic = rr.identity_quadratic(0.5, ww, None, 0.)
    a = rr.dual_problem.fromprimal(l1c)
    solver = rr.FISTA(a)
    solver.fit(min_its=100)

    ac = a.primal

    np.testing.assert_allclose(ac + 0.1, ab + 0.1, rtol=1.e-4)


@set_seed_for_test()
def test_lasso():
    '''
    this test verifies that the l1 prox can be solved
    by a primal/dual specification 

    obviously, we don't to solve the l1 prox this way,
    but it verifies that specification is working correctly

    '''

    l1 = rr.l1norm(4, lagrange=2.)
    l1.quadratic = rr.identity_quadratic(0.5, 0, None, 0.)

    X = np.random.standard_normal((10,4))
    Y = np.random.standard_normal(10) + 3
    
    loss = rr.quadratic_loss.affine(X, -Y, coef=0.5)

    p2 = rr.separable_problem.singleton(l1, loss)
    solver2 = rr.FISTA(p2)
    solver2.fit(tol=1.0e-14, min_its=100)


    f = p2.objective
    ans = scipy.optimize.fmin_powell(f, np.zeros(4), ftol=1.0e-12, xtol=1.e-10)

    print(f(solver2.composite.coefs), f(ans))
    np.testing.assert_allclose(ans + 0.1, solver2.composite.coefs + 0.1, rtol=1.e-3)

