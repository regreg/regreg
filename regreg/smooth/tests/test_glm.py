from copy import copy

import nose.tools as nt
import numpy as np

from regreg.smooth.glm import glm

def test_logistic():

    for Y, T in [(np.random.binomial(1,0.5,size=(10,)),
                  np.ones(10)),
                 (np.random.binomial(1,0.5,size=(10,)),
                  None),
                 (np.random.binomial(3,0.5,size=(10,)),
                  3*np.ones(10))]:
        X = np.random.standard_normal((10,5))

        for case_weights in [None, np.ones(10)]:
            L = glm.logistic(X, Y, trials=T, case_weights=case_weights)
            L.smooth_objective(np.zeros(L.shape), 'both')
            L.hessian(np.zeros(L.shape))

            Lcp = copy(L)
            L_sub = L.subsample(np.arange(5))

            # check that subsample is getting correct answer

            Xsub = X[np.arange(5)]
            Ysub = Y[np.arange(5)]
            if T is not None:
                Tsub = T[np.arange(5)]
                T_num = T
            else:
                Tsub = np.ones(5)
                T_num = np.ones(10)

            beta = np.ones(L.shape)

            if case_weights is not None:
                Lsub2 = glm.logistic(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                Lsub3 = glm.logistic(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                case_cp = case_weights.copy() * 0
                case_cp[np.arange(5)] = 1
                Lsub4 = glm.logistic(X, Y, trials=T, case_weights=case_cp)
            else:
                Lsub2 = glm.logistic(Xsub, Ysub, trials=Tsub)
                Lsub3 = glm.logistic(Xsub, Ysub, trials=Tsub)

            Lsub3.coef *= 2.

            f2, g2 = Lsub2.smooth_objective(beta, 'both')
            f3, g3 = Lsub3.smooth_objective(beta, 'both')
            f4, g4 = Lsub2.smooth_objective(beta, 'both')

            np.testing.assert_allclose(f3, 2*f2)
            np.testing.assert_allclose(g3, 2*g2)

            np.testing.assert_allclose(f2, f4)
            np.testing.assert_allclose(g2, g4)

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       Lsub2.gradient(beta))

            np.testing.assert_allclose(L.gradient(beta),
                                       X.T.dot(L.saturated_loss.mean_function(X.dot(beta)) * T_num - Y))

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       Xsub.T.dot(L_sub.saturated_loss.mean_function(Xsub.dot(beta)) * Tsub - Ysub))

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       Xsub.T.dot(Lsub2.saturated_loss.mean_function(Xsub.dot(beta)) * Tsub - Ysub))

            # other checks on gradient

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

    for case_weights in [np.ones(10), None]:
        L = glm.poisson(X, Y, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L.hessian(np.zeros(L.shape))

        Lcp = copy(L)
        L_sub = L.subsample(np.arange(5))

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
            Lsub2 = glm.poisson(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
            Lsub3 = glm.poisson(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
        else:
            Lsub2 = glm.poisson(Xsub, Ysub)
            Lsub3 = glm.poisson(Xsub, Ysub)

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

    X = np.random.standard_normal((10,5))
    Y = np.random.standard_normal(10)

    for case_weights in [np.ones(10), None]:
        L = glm.gaussian(X, Y, case_weights=case_weights)
        L.hessian(np.zeros(L.shape))
        L.smooth_objective(np.zeros(L.shape), 'both')
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
            Lsub2 = glm.gaussian(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
            Lsub3 = glm.gaussian(Xsub, Ysub, case_weights=case_weights[np.arange(5)])
        else:
            Lsub2 = glm.gaussian(Xsub, Ysub)
            Lsub3 = glm.gaussian(Xsub, Ysub)

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

    X = np.random.standard_normal((10,5))
    Y = np.random.standard_normal(10)

    for case_weights in [np.ones(10), None]:
        L = glm.huber(X, Y, 0.1, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')

        Lcp = copy(L)
        L_sub = L.subsample(np.arange(5))

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

    for case_weights in [np.ones(100), None]:
        L = glm.cox(X, T, S, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L.saturated_loss.hessian_mult(np.zeros(T.shape), np.ones(T.shape))
        L.hessian(np.zeros(L.shape))

        L.gradient(np.zeros(L.shape))

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.data = (X, (T, S))
        L.data


