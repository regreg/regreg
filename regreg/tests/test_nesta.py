"""
Solving a LASSO with linear constraints using NESTA
"""

import regreg.api as rr
import numpy as np
import nose.tools as nt

def test_nesta_nonnegative():

    n, p, q = 1000, 20, 5
    X = np.random.standard_normal((n, p))
    A = np.random.standard_normal((q,p))

    coef = 10 * np.fabs(np.random.standard_normal(q)) + 1
    coef[:2] = -0.2
    beta = np.dot(np.linalg.pinv(A), coef)

    Y = np.random.standard_normal(n) + np.dot(X, beta)

    loss = rr.squared_error(X,Y)
    penalty = rr.l1norm(p, lagrange=0.2)
    constraint = rr.nonnegative.linear(A)

    primal, dual = rr.nesta(loss, penalty, constraint, max_iters=300, coef_tol=1.e-4, tol=1.e-4)

    print np.dot(A, primal)
    assert_almost_nonnegative(np.dot(A,primal), tol=1.e-3)

def test_nesta_lasso():

    n, p = 1000, 20
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 30
    Y = np.random.standard_normal(n) + np.dot(X, beta)

    loss = rr.squared_error(X,Y)
    penalty = rr.l1norm(p, lagrange=2.)

    # using nesta
    z = rr.zero(p)
    primal, dual = rr.nesta(loss, z, penalty, tol=1.e-10,
                            epsilon=2.**(-np.arange(30)))

    # using simple problem

    problem = rr.simple_problem(loss, penalty)
    problem.solve()
    nt.assert_true(np.linalg.norm(primal - problem.coefs) / np.linalg.norm(problem.coefs) < 1.e-3)

def assert_almost_nonnegative(b, tol=1.e-6):
    nt.assert_true(np.linalg.norm(b[b<0]) <= tol * np.linalg.norm(b))

