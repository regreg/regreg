import pytest
from copy import copy

import numpy as np

from .. import glm
from ...affine.block_maps import block_columns

def test_poisson():

    rng = np.random.default_rng(2300)
    X = rng.standard_normal((10,5))
    Y = rng.poisson(10, size=(10,))

    for case_weights in [np.ones(10), None]:
        L = glm.glm.poisson(X, Y, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L.hessian(np.zeros(L.shape))

        Lcp = copy(L)
        L_sub = L.subsample(np.arange(5))

        sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
        sat_sub.smooth_objective(np.zeros(sat_sub.shape))

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

        # check that subsample is getting correct answer

        Xsub = X[np.arange(5)]
        Ysub = Y[np.arange(5)]

        beta = np.ones(L.shape)
        if case_weights is not None:
            Lsub2 = glm.glm.poisson(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
            Lsub3 = glm.glm.poisson(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
        else:
            Lsub2 = glm.glm.poisson(Xsub, Ysub)
            Lsub3 = glm.glm.poisson(Xsub, Ysub)

        Lsub3.coef *= 2.

        f2, g2 = Lsub2.smooth_objective(beta, 'both')
        f3, g3 = Lsub3.smooth_objective(beta, 'both')

        np.testing.assert_allclose(f3, 2*f2)
        np.testing.assert_allclose(g3, 2*g2)

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Lsub2.gradient(beta))

        np.testing.assert_allclose(L.gradient(beta),
                                   X.T.dot(L.saturated_loss.mean_function(X.dot(beta)) - Y))

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Xsub.T.dot(L_sub.saturated_loss.mean_function(Xsub.dot(beta)) - Ysub))

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Xsub.T.dot(Lsub2.saturated_loss.mean_function(Xsub.dot(beta)) - Ysub))


def test_gaussian():

    rng = np.random.default_rng(2301)
    X = rng.standard_normal((10,5))
    Y = rng.standard_normal(10)

    for case_weights in [np.ones(10), None]:
        L = glm.glm.gaussian(X, Y, case_weights=case_weights)
        L.hessian(np.zeros(L.shape))
        L.smooth_objective(np.zeros(L.shape), 'both')

        sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
        sat_sub.smooth_objective(np.zeros(sat_sub.shape))

        L_sub = L.subsample(np.arange(5))

        Xs = X[np.arange(5)]
        Ys = Y[np.arange(5)]

        beta = np.ones(5)
        value_sub = 0.5 * np.linalg.norm(Ys - Xs.dot(beta))**2
        grad_sub = Xs.T.dot(Xs.dot(beta) - Ys)

        f, g= L_sub.smooth_objective(beta, 'both')

        np.testing.assert_allclose(value_sub, f)
        np.testing.assert_allclose(grad_sub, g)

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

        # check that subsample is getting correct answer

        Xsub = X[np.arange(5)]
        Ysub = Y[np.arange(5)]

        beta = np.ones(L.shape)
        if case_weights is not None:
            Lsub2 = glm.glm.gaussian(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
            Lsub3 = glm.glm.gaussian(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
        else:
            Lsub2 = glm.glm.gaussian(Xsub, Ysub)
            Lsub3 = glm.glm.gaussian(Xsub, Ysub)

        Lsub3.coef *= 2.

        f2, g2 = Lsub2.smooth_objective(beta, 'both')
        f3, g3 = Lsub3.smooth_objective(beta, 'both')

        np.testing.assert_allclose(f3, 2*f2)
        np.testing.assert_allclose(g3, 2*g2)

        beta = np.ones(L.shape)

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Lsub2.gradient(beta))

        np.testing.assert_allclose(L.gradient(beta),
                                   X.T.dot(L.saturated_loss.mean_function(X.dot(beta)) - Y))

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Xsub.T.dot(L_sub.saturated_loss.mean_function(Xsub.dot(beta)) - Ysub))

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Xsub.T.dot(Lsub2.saturated_loss.mean_function(Xsub.dot(beta)) - Ysub))

def test_huber():

    rng = np.random.default_rng(2302)
    X = rng.standard_normal((10,5))
    Y = rng.standard_normal(10)

    for case_weights in [np.ones(10), None]:
        L = glm.glm.huber(X, Y, 0.1, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')

        Lcp = copy(L)
        L_sub = L.subsample(np.arange(5))

        sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
        sat_sub.smooth_objective(np.zeros(sat_sub.shape))

        L.gradient(np.zeros(L.shape))
        with pytest.raises(NotImplementedError):
            L.hessian(np.zeros(L.shape))

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data = Y
        L.saturated_loss.data

        L.data = (X, Y)
        L.data

def test_huber_svm():

    rng = np.random.default_rng(2303)
    X = rng.standard_normal((10,5))
    Y = rng.binomial(1,0.5,size=(10,))

    for case_weights in [np.ones(10), None]:
        L = glm.glm.huber_svm(X, Y, 0.1, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')

        Lcp = copy(L)
        L_sub = L.subsample(np.arange(5))

        sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
        sat_sub.smooth_objective(np.zeros(sat_sub.shape))

        L.gradient(np.zeros(L.shape))
        with pytest.raises(NotImplementedError):
            L.hessian(np.zeros(L.shape))

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data = Y
        L.saturated_loss.data

        L.data = (X, Y)
        L.data


def test_coxph():

    rng = np.random.default_rng(2304)
    X = rng.standard_normal((100,5))
    T = rng.standard_exponential(100)
    S = rng.binomial(1, 0.5, size=(100,))

    for case_weights in [np.ones(100), None]:
        L = glm.glm.cox(X, T, S, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L.saturated_loss.hessian_mult(np.zeros(T.shape), np.ones(T.shape))
        L.hessian(np.zeros(L.shape))

        L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works

        L.gradient(np.zeros(L.shape))

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.data = (X, np.array([T, S]).T)
        L.data


def test_stacked():

    rng = np.random.default_rng(2305)
    X = rng.standard_normal((10,5))
    Y = rng.standard_normal((10,3))

    Xstack = block_columns([X for _ in range(3)])
    for case_weights in [np.ones(30), None]:
        sat = glm.stacked_loglike.gaussian(Y.T)
        L = glm.glm(Xstack, sat.data, sat, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')

        L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works

        np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                   -X.T.dot(Y))

        Lcp = copy(L)

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data

        L.data


