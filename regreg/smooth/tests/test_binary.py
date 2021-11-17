from copy import copy

import nose.tools as nt
import numpy as np
from scipy.stats import norm as normal_dbn

from .. import glm

def test_logistic():

    for Y, T in [(np.random.binomial(1,0.5,size=(10,)),
                  np.ones(10)),
                 (np.random.binomial(1,0.5,size=(10,)),
                  None),
                 (np.random.binomial(3,0.5,size=(10,)),
                  3*np.ones(10))]:
        X = np.random.standard_normal((10,5))

        for case_weights in [None, np.ones(10)]:
            L = glm.glm.logistic(X, Y, trials=T, case_weights=case_weights)
            L.smooth_objective(np.zeros(L.shape), 'both')
            L.hessian(np.zeros(L.shape))

            sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
            sat_sub.smooth_objective(np.zeros(sat_sub.shape))

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
                Lsub2 = glm.glm.logistic(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                Lsub3 = glm.glm.logistic(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                case_cp = case_weights.copy() * 0
                case_cp[np.arange(5)] = 1
                Lsub4 = glm.glm.logistic(X, Y, trials=T, case_weights=case_cp)
            else:
                Lsub2 = glm.glm.logistic(Xsub, Ysub, trials=Tsub)
                Lsub3 = glm.glm.logistic(Xsub, Ysub, trials=Tsub)

            Lsub3.coef *= 2.

            f2, g2 = Lsub2.smooth_objective(beta, 'both')
            f3, g3 = Lsub3.smooth_objective(beta, 'both')
            f4, g4 = Lsub2.smooth_objective(beta, 'both')

            np.testing.assert_allclose(f3, 2*f2)
            np.testing.assert_allclose(g3, 2*g2)

            np.testing.assert_allclose(f2, f4)
            np.testing.assert_allclose(g2, g4)

            Lcp = copy(L)
            prev_value = L.smooth_objective(np.zeros(L.shape), 'func')
            L_sub = L.subsample(np.arange(5))
            L_sub.coef *= 45
            new_value = L.smooth_objective(np.zeros(L.shape), 'func')            
            assert(prev_value == new_value)
            

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       45 * Lsub2.gradient(beta))

            np.testing.assert_allclose(L.gradient(beta),
                                       X.T.dot(L.saturated_loss.mean_function(X.dot(beta)) * T_num - Y))

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       45 * Xsub.T.dot(L_sub.saturated_loss.mean_function(Xsub.dot(beta)) * Tsub - Ysub))

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       45 * Xsub.T.dot(Lsub2.saturated_loss.mean_function(Xsub.dot(beta)) * Tsub - Ysub))

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

def test_probit():

    for Y, T in [(np.random.binomial(1,0.5,size=(10,)),
                  np.ones(10)),
                 (np.random.binomial(1,0.5,size=(10,)),
                  None),
                 (np.random.binomial(3,0.5,size=(10,)),
                  3*np.ones(10))]:
        X = np.random.standard_normal((10,5))

        for case_weights in [None, np.ones(10)]:
            L = glm.glm.probit(X, Y, trials=T, case_weights=case_weights)
            L.smooth_objective(np.zeros(L.shape), 'both')
            L.hessian(np.zeros(L.shape))

            sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
            sat_sub.smooth_objective(np.zeros(sat_sub.shape))

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
                Lsub2 = glm.glm.probit(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                Lsub3 = glm.glm.probit(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                case_cp = case_weights.copy() * 0
                case_cp[np.arange(5)] = 1
                Lsub4 = glm.glm.probit(X, Y, trials=T, case_weights=case_cp)
            else:
                Lsub2 = glm.glm.probit(Xsub, Ysub, trials=Tsub)
                Lsub3 = glm.glm.probit(Xsub, Ysub, trials=Tsub)

            Lsub3.coef *= 2.

            f2, g2 = Lsub2.smooth_objective(beta, 'both')
            f3, g3 = Lsub3.smooth_objective(beta, 'both')
            f4, g4 = Lsub2.smooth_objective(beta, 'both')

            np.testing.assert_allclose(f3, 2*f2)
            np.testing.assert_allclose(g3, 2*g2)

            np.testing.assert_allclose(f2, f4)
            np.testing.assert_allclose(g2, g4)

            Lcp = copy(L)
            prev_value = L.smooth_objective(np.zeros(L.shape), 'func')
            L_sub = L.subsample(np.arange(5))
            L_sub.coef *= 45
            new_value = L.smooth_objective(np.zeros(L.shape), 'func')            
            assert(prev_value == new_value)
            

            np.testing.assert_allclose(L_sub.gradient(beta),
                                       45 * Lsub2.gradient(beta))

            linpred = X.dot(beta)
            np.testing.assert_allclose(L.gradient(beta),
                                       X.T.dot(-normal_dbn.pdf(linpred) * (Y / normal_dbn.cdf(linpred) - 
                                                                           (T_num - Y) / normal_dbn.sf(linpred))))

            linpred = Xsub.dot(beta)
            np.testing.assert_allclose(L_sub.gradient(beta),
                                       45 * Xsub.T.dot(-normal_dbn.pdf(linpred) * 
                                                        (Ysub / normal_dbn.cdf(linpred) - 
                                                         (Tsub - Ysub) / normal_dbn.sf(linpred))))

            # other checks on gradient

            if T is None:
                sat = L.saturated_loss
                np.testing.assert_allclose(sat.smooth_objective(np.zeros(sat.shape), 'grad'),
                                           (0.5 - Y) * normal_dbn.pdf(0) / 0.25)
                np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                           X.T.dot(0.5 - Y) * normal_dbn.pdf(0) / 0.25)
                np.testing.assert_allclose(L.hessian(np.zeros(L.shape)),
                                           X.T.dot(X) / 0.25 * normal_dbn.pdf(0)**2)
            else:
                L.gradient(np.zeros(L.shape))
                L.hessian(np.zeros(L.shape))

            L.objective(np.zeros(L.shape))
            L.latexify()

            L.saturated_loss.data = (Y, T)
            L.saturated_loss.data

            L.data = (X, (Y, T))
            L.data

def test_cloglog():

    for Y, T in [(np.random.binomial(1,0.5,size=(10,)),
                  np.ones(10)),
                 (np.random.binomial(1,0.5,size=(10,)),
                  None),
                 (np.random.binomial(3,0.5,size=(10,)),
                  3*np.ones(10))]:
        X = np.random.standard_normal((10,5))

        for case_weights in [None, np.ones(10)]:
            L = glm.glm.cloglog(X, Y, trials=T, case_weights=case_weights)
            L.smooth_objective(np.zeros(L.shape), 'both')
            L.hessian(np.zeros(L.shape))

            sat_sub = L.saturated_loss.subsample(np.arange(5)) # check that subsample of saturated loss at least works
            sat_sub.smooth_objective(np.zeros(sat_sub.shape))

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
                Lsub2 = glm.glm.cloglog(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                Lsub3 = glm.glm.cloglog(Xsub, Ysub, trials=Tsub, case_weights=case_weights[np.arange(5)])
                case_cp = case_weights.copy() * 0
                case_cp[np.arange(5)] = 1
                Lsub4 = glm.glm.cloglog(X, Y, trials=T, case_weights=case_cp)
            else:
                Lsub2 = glm.glm.cloglog(Xsub, Ysub, trials=Tsub)
                Lsub3 = glm.glm.cloglog(Xsub, Ysub, trials=Tsub)

            Lsub3.coef *= 2.

            f2, g2 = Lsub2.smooth_objective(beta, 'both')
            f3, g3 = Lsub3.smooth_objective(beta, 'both')
            f4, g4 = Lsub2.smooth_objective(beta, 'both')

            np.testing.assert_allclose(f3, 2*f2)
            np.testing.assert_allclose(g3, 2*g2)

            np.testing.assert_allclose(f2, f4)
            np.testing.assert_allclose(g2, g4)

            Lcp = copy(L)
            prev_value = L.smooth_objective(np.zeros(L.shape), 'func')
            L_sub = L.subsample(np.arange(5))
            L_sub.coef *= 45
            new_value = L.smooth_objective(np.zeros(L.shape), 'func')            
            assert(prev_value == new_value)
            
            np.testing.assert_allclose(L_sub.gradient(beta),
                                       45 * Lsub2.gradient(beta))

            linpred = X.dot(0.1 * beta)
            cdf = 1 - np.exp(-np.exp(linpred))
            sf = np.exp(-np.exp(linpred))
            pdf = np.exp(linpred - np.exp(linpred))

            np.testing.assert_allclose(L.gradient(0.1 * beta),
                                       X.T.dot(-pdf * (Y / cdf - 
                                                       (T_num - Y) / sf)))

            linpred = Xsub.dot(0.1 * beta)
            cdf = 1 - np.exp(-np.exp(linpred))
            sf = np.exp(-np.exp(linpred))
            pdf = np.exp(linpred - np.exp(linpred))

            np.testing.assert_allclose(L_sub.gradient(0.1 * beta),
                                       45 * Xsub.T.dot(-pdf * 
                                                        (Ysub / cdf - 
                                                         (Tsub - Ysub) / sf)))

            # other checks on gradient

            if T is None:
                sat = L.saturated_loss
                pdf0 = np.exp(-1)
                cdf = (1 - np.exp(-1))
                sf = np.exp(-1)
                denom = cdf * sf
                np.testing.assert_allclose(sat.smooth_objective(np.zeros(sat.shape), 'grad'),
                                           (cdf - Y) * pdf0 / denom)
                np.testing.assert_allclose(L.gradient(np.zeros(L.shape)),
                                           X.T.dot(cdf - Y) * pdf0 / denom)
                np.testing.assert_allclose(L.hessian(np.zeros(L.shape)),
                                           X.T.dot(X) / denom * pdf0**2)
            else:
                L.gradient(np.zeros(L.shape))
                L.hessian(np.zeros(L.shape))

            L.objective(np.zeros(L.shape))
            L.latexify()

            L.saturated_loss.data = (Y, T)
            L.saturated_loss.data

            L.data = (X, (Y, T))
            L.data

