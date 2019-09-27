import numpy as np
from sklearn.model_selection import cross_validate

import regreg.api as rr
from ..sklearn_mixin import (sklearn_regression,
                             sklearn_classifier)

def test_sklearn_regression():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    pen = rr.l1norm(p, lagrange=2 * np.sqrt(n))
    gaussian_lasso = sklearn_regression.gaussian(pen)
    huber_lasso = sklearn_regression.huber(pen)

    print(cross_validate(gaussian_lasso, X, y, cv=3))
    print(cross_validate(huber_lasso, X, y, cv=3))


def test_sklearn_classifier():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    pen = rr.l1norm(p, lagrange=2 * np.sqrt(n))
    ybin = y > 0

    logistic_lasso = sklearn_classifier.logistic(pen)
    print(cross_validate(logistic_lasso, X, ybin, cv=3))


