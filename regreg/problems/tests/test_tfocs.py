"""
Solving basis pursuit with TFOCS
"""
import numpy as np
import nose.tools as nt

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_tfocs():
    n, p = 100, 200
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 3
    Y = np.random.standard_normal(n) + np.dot(X, beta)

    lscoef = np.dot(np.linalg.pinv(X), Y)
    minimum_l2 = np.linalg.norm(Y - np.dot(X, lscoef))
    maximum_l2 = np.linalg.norm(Y)

    l2bound = (minimum_l2 + maximum_l2) * 0.5

    constraint = rr.l2norm.affine(X,-Y,bound=l2bound)
    T, conjugate = constraint.dual
    l1 = rr.l1norm(p, lagrange=1)

    primal, dual = rr.tfocs(l1, T, conjugate, coef_tol=1.e-4)
    nt.assert_true(np.fabs(np.linalg.norm(Y - np.dot(X, primal)) - l2bound) <= l2bound * 1.e-3)
