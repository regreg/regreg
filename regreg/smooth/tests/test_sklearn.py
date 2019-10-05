import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from ..sklearn_mixin import (sklearn_regression,
                                 sklearn_classifier)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm
from ...tests.decorators import set_seed_for_test

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_sklearn_regression_gaussian():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    pen = l1norm(p, lagrange=2 * np.sqrt(n))
    gaussian_lasso = sklearn_regression.gaussian(pen)
    print(cross_validate(gaussian_lasso, X, y, cv=10))

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_sklearn_regression_huber():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    pen = l1norm(p, lagrange=2 * np.sqrt(n))
    huber_lasso = sklearn_regression.huber(0.2, pen)
    print(cross_validate(huber_lasso, X, y, cv=10))

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_sklearn_logistic():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    pen = l1norm(p, lagrange=2 * np.sqrt(n))
    ybin = y > 0

    logistic_lasso = sklearn_classifier.logistic(pen)
    print(cross_validate(logistic_lasso, X, ybin, cv=10))


