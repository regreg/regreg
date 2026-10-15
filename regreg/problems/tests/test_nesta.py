"""
Solving a LASSO with linear constraints using NESTA
"""
from __future__ import print_function, division, absolute_import

import numpy as np

import regreg.api as rr

def test_nesta_nonnegative():

    rng = np.random.default_rng(3800)
    n, p, q = 1000, 20, 5
    X = rng.standard_normal((n, p))
    A = rng.standard_normal((q,p))

    coef = 10 * np.fabs(rng.standard_normal(q)) + 1
    coef[:2] = -0.2
    beta = np.dot(np.linalg.pinv(A), coef)
    print(r'\beta', beta)
    print(r'A\beta', np.dot(A, beta))

    Y = rng.standard_normal(n) + np.dot(X, beta)

    loss = rr.squared_error(X,Y)
    penalty = rr.l1norm(p, lagrange=0.2)
    constraint = rr.nonnegative.linear(A)

    primal, dual = rr.nesta(loss, penalty, constraint, max_iters=300, coef_tol=1.e-10, tol=1.e-10)

    print(r'A \hat{\beta}', np.dot(A, primal))
    assert_almost_nonnegative(np.dot(A,primal), tol=1.e-3)


def test_nesta_lasso():

    rng = np.random.default_rng(3801)
    n, p = 1000, 20
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 30
    Y = rng.standard_normal(n) + np.dot(X, beta)

    loss = rr.squared_error(X,Y)
    penalty = rr.l1norm(p, lagrange=2.)

    # using nesta
    z = rr.zero(p)
    primal, dual = rr.nesta(loss, z, penalty, tol=1.e-10,
                            epsilon=2.**(-np.arange(30)),
                            initial_dual=np.zeros(p))

    # using simple problem

    problem = rr.simple_problem(loss, penalty)
    problem.solve()
    assert np.linalg.norm(primal - problem.coefs) / np.linalg.norm(problem.coefs) < 1.e-3

    # test None as smooth_atom

    rr.nesta(None, z, penalty, tol=1.e-10,
             epsilon=2.**(-np.arange(30)),
             initial_dual=np.zeros(p))

    # using coefficients to stop

    rr.nesta(loss, z, penalty, tol=1.e-10,
             epsilon=2.**(-np.arange(30)),
             initial_dual=np.zeros(p),
             coef_stop=True)


def assert_almost_nonnegative(b, tol=1.e-6):
    assert np.linalg.norm(b[b<0]) <= tol * np.linalg.norm(b)

