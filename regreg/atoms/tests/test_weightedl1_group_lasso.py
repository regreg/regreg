from __future__ import print_function, division, absolute_import

import itertools

import numpy as np

import regreg.api as rr
import regreg.atoms.weighted_atoms as WA
from regreg.tests.decorators import set_seed_for_test

from numpy import testing as npt

@set_seed_for_test()
def test_group_lasso_weightedl1_lagrange():
    n, p = 100, 50

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    loss = rr.glm.gaussian(X, Y)
    weights = np.ones(p)
    weights[-2:] = np.inf
    weights[:2] = 0 
    weight_dict = dict([(i, w) for i, w in enumerate(weights)])
    pen1 = rr.weighted_l1norm(weights, lagrange=0.5 * np.sqrt(n))
    pen2 = rr.group_lasso(np.arange(p), weights=weight_dict, lagrange=0.5 * np.sqrt(n))

    problem1 = rr.simple_problem(loss, pen1)
    problem2 = rr.simple_problem(loss, pen2)

    beta1 = problem1.solve(tol=1.e-14, min_its=500)
    beta2 = problem2.solve(tol=1e-14, min_its=500)

    npt.assert_allclose(beta1, beta2)

    bound_val = pen1.seminorm(beta1, lagrange=1)
    bound1 = rr.weighted_l1norm(weights, bound=bound_val)
    bound2 = rr.group_lasso(np.arange(p), weights=weight_dict, bound=bound_val)
    problem3 = rr.simple_problem(loss, bound1)
    problem4 = rr.simple_problem(loss, bound2)

    beta3 = problem3.solve(tol=1.e-14, min_its=500)
    beta4 = problem4.solve(tol=1.e-14, min_its=500)

    npt.assert_allclose(beta3, beta4)
    npt.assert_allclose(beta3, beta1)

@set_seed_for_test()
def test_group_lasso_weightedl1_bound():
    n, p = 100, 50

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    loss = rr.glm.gaussian(X, Y)
    weights = np.ones(p)
    weights[-2:] = np.inf
    weights[:2] = 0 
    weight_dict = dict([(i, w) for i, w in enumerate(weights)])
    bound1 = rr.weighted_l1norm(weights, bound=2)
    bound2 = rr.group_lasso(np.arange(p), weights=weight_dict, bound=2)

    problem1 = rr.simple_problem(loss, bound1)
    problem2 = rr.simple_problem(loss, bound2)

    beta1 = problem1.solve(tol=1.e-14, min_its=500)
    beta2 = problem2.solve(tol=1e-14, min_its=500)

    npt.assert_allclose(beta1, beta2)


