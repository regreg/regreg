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
