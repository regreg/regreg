from copy import copy

import nose.tools as nt
import numpy as np

from regreg.smooth import mglm, glm
from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_gaussian_common():

    X = np.random.standard_normal((10,5))
    Y = np.random.standard_normal((10,3))

    for case_weights in [np.ones(10), None]:
        sat = mglm.stacked_common_loglike.gaussian(Y.T)
        L = mglm.mglm(X, sat, case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L_sub = L.subsample(np.arange(5))

        Xs = X[np.arange(5)]
        Ys = Y[np.arange(5)]

        beta = np.ones(L.shape)
        value_sub = 0.5 * np.linalg.norm(Ys - Xs.dot(beta))**2
        grad_sub = Xs.T.dot(Xs.dot(beta) - Ys)

        f, g = L_sub.smooth_objective(beta, 'both')

        np.testing.assert_allclose(value_sub, f)
        np.testing.assert_allclose(grad_sub, g)

        np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                   -X.T.dot(Y))

        Lcp = copy(L)

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data

        L.data

        # check that subsample is getting correct answer

        Xsub = X[np.arange(5)]
        Ysub = Y[np.arange(5)]
        loss_sub = mglm.stacked_common_loglike.gaussian(Ysub.T)

        beta = np.ones(L.shape)
        if case_weights is not None:
            Lsub2 = mglm.mglm(Xsub, loss_sub, case_weights=case_weights[np.arange(5)])
            Lsub3 = mglm.mglm(Xsub, loss_sub, case_weights=case_weights[np.arange(5)])
        else:
            Lsub2 = mglm.mglm(Xsub, loss_sub)
            Lsub3 = mglm.mglm(Xsub, loss_sub)

        Lsub3.coef *= 2.

        f2, g2 = Lsub2.smooth_objective(beta, 'both')
        f3, g3 = Lsub3.smooth_objective(beta, 'both')

        np.testing.assert_allclose(f3, 2*f2)
        np.testing.assert_allclose(g3, 2*g2)

        beta = np.ones(L.shape)

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Lsub2.gradient(beta))

@set_seed_for_test()
def test_multinomial():

    """
    Test that multinomial regression with two categories is the same as logistic regression
    """

    n = 500
    p = 10
    J = 4


    X = np.random.standard_normal(n*p).reshape((n,p))
    counts = np.random.randint(0,10,n*J).reshape((n,J)) + 2
    for case_weights in [np.ones(n), None]:

        sat = mglm.multinomial_loglike(counts.shape, counts)
        L = mglm.mglm(X, 
                      sat,
                      case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L_sub = L.subsample(np.arange(100))

        np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                   -X.T.dot(counts - counts.mean(1)[:,None]))

        Lcp = copy(L)

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data

        L.data

        # check that subsample is getting correct answer

        Xsub = X[np.arange(100)]
        counts_sub = counts[np.arange(100)]
        loss_sub = mglm.multinomial_loglike(counts_sub.shape, counts_sub)

        beta = np.ones(L.shape)
        if case_weights is not None:
            Lsub2 = mglm.mglm(Xsub, loss_sub, case_weights=case_weights[np.arange(100)])
            Lsub3 = mglm.mglm(Xsub, loss_sub, case_weights=case_weights[np.arange(100)])
        else:
            Lsub2 = mglm.mglm(Xsub, loss_sub)
            Lsub3 = mglm.mglm(Xsub, loss_sub)

        Lsub3.coef *= 2.

        f2, g2 = Lsub2.smooth_objective(beta, 'both')
        f3, g3 = Lsub3.smooth_objective(beta, 'both')

        np.testing.assert_allclose(f3, 2*f2)
        np.testing.assert_allclose(g3, 2*g2)

        beta = np.ones(L.shape)

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Lsub2.gradient(beta))

@set_seed_for_test()
def test_multinomial_baseline():

    """
    Test that multinomial regression with two categories is the same as logistic regression
    """

    n = 500
    p = 10
    J = 4


    X = np.random.standard_normal(n*p).reshape((n,p))
    counts = np.random.randint(0,10,n*J).reshape((n,J)) + 2
    for case_weights in [np.ones(n), None]:

        sat = mglm.multinomial_baseline_loglike((n, J-1), counts)
        L = mglm.mglm(X, 
                      sat,
                      case_weights=case_weights)
        L.smooth_objective(np.zeros(L.shape), 'both')
        L_sub = L.subsample(np.arange(100))

        np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                   -X.T.dot(counts - counts.mean(1)[:,None])[:,:(J-1)])

        Lcp = copy(L)

        L.objective(np.zeros(L.shape))
        L.latexify()

        L.saturated_loss.data

        L.data

        # check that subsample is getting correct answer

        Xsub = X[np.arange(100)]
        counts_sub = counts[np.arange(100)]
        loss_sub = mglm.multinomial_baseline_loglike((counts_sub.shape[0], J-1), counts_sub)

        beta = np.ones(L.shape)
        if case_weights is not None:
            Lsub2 = mglm.mglm(Xsub, loss_sub, case_weights=case_weights[np.arange(100)])
            Lsub3 = mglm.mglm(Xsub, loss_sub, case_weights=case_weights[np.arange(100)])
        else:
            Lsub2 = mglm.mglm(Xsub, loss_sub)
            Lsub3 = mglm.mglm(Xsub, loss_sub)

        Lsub3.coef *= 2.

        f2, g2 = Lsub2.smooth_objective(beta, 'both')
        f3, g3 = Lsub3.smooth_objective(beta, 'both')

        np.testing.assert_allclose(f3, 2*f2)
        np.testing.assert_allclose(g3, 2*g2)

        beta = np.ones(L.shape)

        np.testing.assert_allclose(L_sub.gradient(beta),
                                   Lsub2.gradient(beta))



