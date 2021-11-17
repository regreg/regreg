import itertools

import numpy as np, regreg.api as rr
from .. import mixed_lasso as ml
from .. import group_lasso as gl
from ...tests.decorators import set_seed_for_test
from .test_seminorms import Solver, all_close, SolverFactory


@set_seed_for_test()
def test_mixed_lasso():

    prox_center = np.array([1,3,5,7,-9,3,4,6,7,8,9,11,13,4,-23,40], np.float)
    l1_penalty = np.array([0,1], np.intp)
    unpenalized = np.array([2,3], np.intp)
    positive_part = np.array([4,5], np.intp)
    nonnegative = np.array([], np.intp)
    mixed_spec = np.array([ml.L1_PENALTY,
                           ml.L1_PENALTY,
                           ml.UNPENALIZED,
                           ml.UNPENALIZED,
                           ml.POSITIVE_PART,
                           ml.POSITIVE_PART] + [0]*5 + [1]*5).astype(np.intp)
    weights = np.array([1.,1.])

    lagrange = 1.
    lipschitz = 0.5

    result = np.zeros_like(prox_center)

    result[unpenalized] = prox_center[unpenalized]
    result[positive_part] = (prox_center[positive_part] - lagrange / lipschitz) * np.maximum(prox_center[positive_part] - lagrange / lipschitz, 0)
    result[l1_penalty] = np.maximum(np.fabs(prox_center[l1_penalty]) - lagrange / lipschitz, 0) * np.sign(prox_center[l1_penalty])

    result[6:11] = prox_center[6:11] / np.linalg.norm(prox_center[6:11]) * max(np.linalg.norm(prox_center[6:11]) - weights[0] * lagrange/lipschitz, 0)
    result[11:] = prox_center[11:] / np.linalg.norm(prox_center[11:]) * max(np.linalg.norm(prox_center[11:]) - weights[1] * lagrange/lipschitz, 0)

    norms = np.zeros_like(weights)
    factors = np.zeros_like(weights)
    projection = np.zeros(prox_center.shape)

    prox_result = ml.mixed_lasso_lagrange_prox(prox_center, 
                                               lagrange, 
                                               lipschitz, 
                                               l1_penalty, 
                                               unpenalized, 
                                               positive_part, 
                                               nonnegative, 
                                               mixed_spec, 
                                               weights,
                                               norms,
                                               factors,
                                               projection)

    np.testing.assert_allclose(result, prox_result)

    norms = np.zeros_like(weights)
    factors = np.zeros_like(weights)
    projection = np.zeros(prox_center.shape)

    a3 = ml.mixed_lasso_lagrange_prox(prox_center, 
                                      lagrange, 
                                      1,
                                      l1_penalty, 
                                      unpenalized, 
                                      positive_part, 
                                      nonnegative, 
                                      mixed_spec, 
                                      weights,
                                      norms,
                                      factors,
                                      projection)

    norms = np.zeros_like(weights)
    factors = np.zeros_like(weights)
    projection = np.zeros(prox_center.shape)

    b3 = ml.mixed_lasso_dual_bound_prox(prox_center,
                                        lagrange,
                                        l1_penalty, 
                                        unpenalized, 
                                        positive_part, 
                                        nonnegative, 
                                        mixed_spec, 
                                        weights,
                                        norms,
                                        factors,
                                        projection)

    np.testing.assert_allclose(prox_center, a3 + b3)

    norms = np.zeros_like(weights)
    value = ml.seminorm_mixed_lasso(prox_center,
                                    l1_penalty, 
                                    unpenalized, 
                                    positive_part, 
                                    nonnegative, 
                                    mixed_spec, 
                                    weights,
                                    norms,
                                    0)

    test_value = 0
    test_value += np.fabs(prox_center[l1_penalty]).sum()
    test_value += np.fabs(np.maximum(prox_center[positive_part], 0)).sum()
    test_value += np.linalg.norm(prox_center[6:11])
    test_value += np.linalg.norm(prox_center[11:16])
    np.testing.assert_allclose(value, test_value)

    dual_value = ml.seminorm_mixed_lasso_dual(prox_center,
                                              l1_penalty, 
                                              unpenalized, 
                                              positive_part, 
                                              nonnegative, 
                                              mixed_spec, 
                                              weights,
                                              norms,
                                              0)


    norms = np.zeros_like(weights)
    factors = np.zeros_like(weights)
    projection = np.zeros(prox_center.shape)

    ml.mixed_lasso_bound_prox(prox_center,
                              0.5,
                              l1_penalty, 
                              unpenalized, 
                              positive_part, 
                              nonnegative, 
                              mixed_spec, 
                              weights,
                              norms,
                              factors,
                              projection)

    epigraph_center = np.zeros(prox_center.shape[0] + 1)
    epigraph_center[:-1] = prox_center
    epigraph_center[-1] = 0.5 * test_value

    norms = np.zeros_like(weights)
    factors = np.zeros_like(weights)
    projection = np.zeros(prox_center.shape)

    ml.mixed_lasso_epigraph(epigraph_center,
                            l1_penalty, 
                            unpenalized, 
                            positive_part, 
                            nonnegative, 
                            mixed_spec, 
                            weights,
                            norms,
                            factors,
                            projection)

@set_seed_for_test()
def test_compare_to_group_lasso():

    ps = np.array([0]*5 + [3]*3)
    weights = {3:2., 0:2.3}

    lagrange = 1.5
    lipschitz = 0.2
    p = gl.group_lasso(ps, weights=weights, lagrange=lagrange)
    z = 30 * np.random.standard_normal(8)
    q = rr.identity_quadratic(lipschitz, z, 0, 0)

    x = p.solve(q)

    norms = np.zeros(2)
    factors = np.zeros(2)
    projection = np.zeros(z.shape)

    a = ml.mixed_lasso_lagrange_prox(z, 
                                     lagrange, 
                                     lipschitz, 
                                     np.array([], np.intp), 
                                     np.array([], np.intp), 
                                     np.array([], np.intp), 
                                     np.array([], np.intp), 
                                     np.array([0,0,0,0,0,1,1,1]).astype(np.intp), 
                                     np.array([np.sqrt(5), 2]),
                                     norms,
                                     factors,
                                     projection)

    result = np.zeros_like(a)
    result[:5] = z[:5] / np.linalg.norm(z[:5]) * max(np.linalg.norm(z[:5]) - weights[0] * lagrange/lipschitz, 0)
    result[5:] = z[5:] / np.linalg.norm(z[5:]) * max(np.linalg.norm(z[5:]) - weights[3] * lagrange/lipschitz, 0)

    lipschitz = 1.
    q = rr.identity_quadratic(lipschitz, z, 0, 0)
    x2 = p.solve(q)
    pc = p.conjugate
    a2 = pc.solve(q)

    np.testing.assert_allclose(z-a2, x2)

    # make sure dtypes are correct

    norms = np.zeros(2)
    factors = np.zeros(2)
    projection = np.zeros(z.shape)

    a3 = ml.mixed_lasso_lagrange_prox(z, 
                                      lagrange, 
                                      1,
                                      np.array([], np.intp), 
                                      np.array([], np.intp), 
                                      np.array([], np.intp), 
                                      np.array([], np.intp), 
                                      np.array([0,0,0,0,0,1,1,1], np.intp), 
                                      np.array([np.sqrt(5), 2]),
                                      norms,
                                      factors,
                                      projection)

    norms = np.zeros(2)
    factors = np.zeros(2)
    projection = np.zeros(z.shape)

    b3 = ml.mixed_lasso_dual_bound_prox(z, 
                                        lagrange, 
                                        np.array([], np.intp), 
                                        np.array([], np.intp), 
                                        np.array([], np.intp), 
                                        np.array([], np.intp), 
                                        np.array([0,0,0,0,0,1,1,1], np.intp), 
                                        np.array([np.sqrt(5), 2]),
                                        norms,
                                        factors,
                                        projection)
    
    np.testing.assert_allclose(z, a3 + b3)


class MixedSolverFactory(SolverFactory):

    mixed_spec = [np.array([ml.L1_PENALTY,
                            ml.L1_PENALTY,
                            ml.UNPENALIZED,
                            ml.UNPENALIZED,
                            ml.POSITIVE_PART,
                            ml.POSITIVE_PART] + [0]*5 + [1]*5).astype(np.intp)]
    weights = [{0:1.,1:1.},{0:0.,1:1.}]
    lagrange = [1.1]
    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode):
        self.klass = klass
        self.mode = mode

    def __iter__(self):
        pen_choices = itertools.product(self.weights,
                                        self.lagrange,
                                        self.mixed_spec)
        for offset, FISTA, coef_stop, L, q, pen in itertools.product(self.offset_choices,
                                                                     self.FISTA_choices,
                                                                     self.coef_stop_choices,
                                                                     self.L_choices,
                                                                     self.quadratic_choices,
                                                                     pen_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            w, l, spec = pen
            atom = self.klass(spec, l, weights=w)

            if q: 
                atom.quadratic = rr.identity_quadratic(0, 0, np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver

@set_seed_for_test()
@np.testing.dec.slow
def test_proximal_maps():
    for klass, mode in zip([ml.mixed_lasso, 
                            ml.mixed_lasso_dual], 
                           ['lagrange', 'bound']):
        factory = MixedSolverFactory(klass, mode)
        for solver in factory:
            for t in solver.all_tests():
                yield t
        factory = MixedSolverFactory(klass, mode)
        for solver in factory:
            for t in solver.all_tests():
                yield t
