from copy import copy
import numpy as np
import nose.tools as nt

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_conjugate_l1norm():
    '''
    this test verifies that numerically computing the conjugate
    is essentially the same as using the smooth_conjugate
    of the atom
    '''

    q = rr.identity_quadratic(1.2,0,0,0)
    l1 = rr.l1norm(4, lagrange=0.3)
    pen2 = copy(l1)
    pen2.set_quadratic(q)

    v1 = rr.smooth_conjugate(l1, q)
    v2 = rr.conjugate(l1, q, tol=1.e-12, min_its=100)
    v3 = rr.conjugate(pen2, None, tol=1.e-12, min_its=100)
    w = np.random.standard_normal(4)

    u11, u12 = v1.smooth_objective(w)
    u21, u22 = v2.smooth_objective(w)
    u31, u32 = v3.smooth_objective(w)
    np.testing.assert_approx_equal(u11, u21)
    np.testing.assert_allclose(u12, u22, rtol=1.0e-05)
    np.testing.assert_approx_equal(u11, u31)
    np.testing.assert_allclose(u12, u32, rtol=1.0e-05)

    v2.smooth_objective(w, mode='func')
    v2.smooth_objective(w, mode='grad')
    nt.assert_raises(ValueError, v2.smooth_objective, w, 'blah')

@set_seed_for_test()
def test_conjugate_sqerror():
    """
    This verifies the conjugate class can compute the conjugate
    of a quadratic function.
    """

    ridge_coef = 0.4

    X = np.random.standard_normal((10,4))
    Y = np.random.standard_normal(10)
    l = rr.squared_error(X, Y)

    q = rr.identity_quadratic(ridge_coef,0,0,0)
    atom_conj = rr.conjugate(l, q, tol=1.e-12, min_its=100)
    w = np.random.standard_normal(4)
    u11, u12 = atom_conj.smooth_objective(w)

    # check that objective is half of squared error
    np.testing.assert_allclose(l.smooth_objective(w, mode='func'), 0.5 * np.linalg.norm(Y - np.dot(X, w))**2)
    np.testing.assert_allclose(atom_conj.atom.smooth_objective(w, mode='func'), 0.5 * np.linalg.norm(Y - np.dot(X, w))**2)

    XTX = np.dot(X.T, X) 
    XTXi = np.linalg.pinv(XTX)

    quadratic_term = XTX + ridge_coef * np.identity(4)
    linear_term = np.dot(X.T, Y) + w
    b = u22 = np.linalg.solve(quadratic_term, linear_term)
    u21 = (w*u12).sum() - l.smooth_objective(u12, mode='func') - q.objective(u12, mode='func')
    np.testing.assert_allclose(u12, u22, rtol=1.0e-05)
    np.testing.assert_approx_equal(u11, u21)
    
