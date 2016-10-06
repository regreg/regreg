from copy import copy

import nose.tools as nt
import numpy as np

from regreg.smooth.glm import glm, coxph

def test_logistic():

    for Y, T in [(np.random.binomial(1,0.5,size=(10,)),
                  np.ones(10)),
                 (np.random.binomial(1,0.5,size=(10,)),
                  None),
                 (np.random.binomial(3,0.5,size=(10,)),
                  3*np.ones(10))]:
        X = np.random.standard_normal((10,5))

        L = glm.logistic(X, Y, trials=T)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L.hessian(np.zeros(L.shape))

        Lcp = copy(L)

        if T is None:
            np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                       X.T.dot(0.5 - Y))
            np.testing.assert_allclose(L.hessian(np.zeros(L.shape)),
                                       X.T.dot(X) / 4.)
        else:
            L.gradient(np.zeros(L.shape))
            L.hessian(np.zeros(L.shape))

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data = (Y, T)
        L.saturated_loss.data

        L.data = (X, (Y, T))
        L.data

def test_poisson():

    X = np.random.standard_normal((10,5))
    Y = np.random.poisson(10, size=(10,))

    L = glm.poisson(X, Y)
    L.smooth_objective(np.zeros(L.shape), 'both')
    L.hessian(np.zeros(L.shape))

    Lcp = copy(L)

    np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                               X.T.dot(1 - Y))
    np.testing.assert_allclose(L.hessian(np.zeros(L.shape)),
                               X.T.dot(X))

    L.objective(np.zeros(L.shape))
    L.latexify()

    L.saturated_loss.data = Y
    L.saturated_loss.data

    L.data = (X, Y)
    L.data

def test_gaussian():

    X = np.random.standard_normal((10,5))
    Y = np.random.standard_normal(10)

    L = glm.gaussian(X, Y)
    L.hessian(np.zeros(L.shape))
    L.smooth_objective(np.zeros(L.shape), 'both')

    np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                               -X.T.dot(Y))
    np.testing.assert_allclose(L.hessian(np.zeros(L.shape)),
                               X.T.dot(X))

    Lcp = copy(L)

    L.objective(np.zeros(L.shape))
    L.latexify()

    L.saturated_loss.data = Y
    L.saturated_loss.data

    L.data = (X, Y)
    L.data

def test_huber():

    X = np.random.standard_normal((10,5))
    Y = np.random.standard_normal(10)

    L = glm.huber(X, Y, 0.1)
    L.smooth_objective(np.zeros(L.shape), 'both')

    Lcp = copy(L)

    L.gradient(np.zeros(L.shape))
    nt.assert_raises(NotImplementedError, L.hessian, np.zeros(L.shape))

    L.objective(np.zeros(L.shape))
    L.latexify()

    L.saturated_loss.data = Y
    L.saturated_loss.data

    L.data = (X, Y)
    L.data


def test_coxph():

    X = np.random.standard_normal((100,5))
    T = np.random.standard_exponential(100)
    S = np.random.binomial(1, 0.5, size=(100,))

    L = coxph(X, T, S)
    L.smooth_objective(np.zeros(L.shape), 'both')
    L.hessian(np.zeros(L.shape))

    L.gradient(np.zeros(L.shape))

    L.objective(np.zeros(L.shape))
    L.latexify()

    L.data = (X, T, S)
    L.data


