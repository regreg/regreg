import numpy as np, regreg.api as rr
from .. import mixed_lasso as ml
from .. import group_lasso as gl
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_mixed_lasso():
    prox_center = np.array([1,3,5,7,-9,3,4,6,7,8,9,11,13,4,-23,40], np.float)
    l1_penalty = np.array([0,1], np.intp)
    unpenalized = np.array([2,3], np.intp)
    positive_part = np.array([4,5], np.intp)
    nonnegative = np.array([], np.intp)
    groups = np.array([ml.L1_PENALTY,
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

    prox_result = ml.mixed_lasso_lagrange_prox(prox_center, 
                                               lagrange, 
                                               lipschitz, 
                                               l1_penalty, 
                                               unpenalized, 
                                               positive_part, 
                                               nonnegative, 
                                               groups, 
                                               weights)

    np.testing.assert_allclose(result, prox_result)

    a3 = ml.mixed_lasso_lagrange_prox(prox_center, 
                                      lagrange, 
                                      1,
                                      l1_penalty, 
                                      unpenalized, 
                                      positive_part, 
                                      nonnegative, 
                                      groups, 
                                      weights)

    b3 = ml.mixed_lasso_dual_bound_prox(prox_center,
                                        lagrange,
                                        l1_penalty, 
                                        unpenalized, 
                                        positive_part, 
                                        nonnegative, 
                                        groups, 
                                        weights)

    np.testing.assert_allclose(prox_center, a3 + b3)

    value = ml.seminorm_mixed_lasso(prox_center,
                                    l1_penalty, 
                                    unpenalized, 
                                    positive_part, 
                                    nonnegative, 
                                    groups, 
                                    weights,
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
                                              groups, 
                                              weights,
                                              0)


    ml.mixed_lasso_bound_prox(prox_center,
                              0.5,
                              l1_penalty, 
                              unpenalized, 
                              positive_part, 
                              nonnegative, 
                              groups, 
                              weights)

    epigraph_center = np.zeros(prox_center.shape[0] + 1)
    epigraph_center[:-1] = prox_center
    epigraph_center[-1] = 0.5 * test_value
    ml.mixed_lasso_epigraph(epigraph_center,
                            l1_penalty, 
                            unpenalized, 
                            positive_part, 
                            nonnegative, 
                            groups, 
                            weights)

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
    a = ml.mixed_lasso_lagrange_prox(z, 
                                     lagrange, 
                                     lipschitz, 
                                     np.array([], np.intp), 
                                     np.array([], np.intp), 
                                     np.array([], np.intp), 
                                     np.array([], np.intp), 
                                     np.array([0,0,0,0,0,1,1,1]).astype(np.intp), 
                                     np.array([np.sqrt(5), 2]))

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

    a3 = ml.mixed_lasso_lagrange_prox(z, 
                                      lagrange, 
                                      1,
                                      np.array([], np.intp), 
                                      np.array([], np.intp), 
                                      np.array([], np.intp), 
                                      np.array([], np.intp), 
                                      np.array([0,0,0,0,0,1,1,1], np.intp), 
                                      np.array([np.sqrt(5), 2]))

    b3 = ml.mixed_lasso_dual_bound_prox(z, 
                                        lagrange, 
                                        np.array([], np.intp), 
                                        np.array([], np.intp), 
                                        np.array([], np.intp), 
                                        np.array([], np.intp), 
                                        np.array([0,0,0,0,0,1,1,1], np.intp), 
                                        np.array([np.sqrt(5), 2]))
    
    np.testing.assert_allclose(z, a3 + b3)

