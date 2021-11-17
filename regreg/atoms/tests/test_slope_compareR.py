"""
Solve a problem with SLOPE and compare to R
"""

from __future__ import print_function

import numpy as np
import nose.tools as nt

try:
    import rpy2.robjects as rpy
    rpy2_available = True
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2 import robjects
except ImportError:
    rpy2_available = False

Rslope = True
try:
    SLOPE = importr('SLOPE')
except:
    Rslope = False
    

import regreg.api as rr
from regreg.atoms.slope import slope
from regreg.tests.decorators import set_seed_for_test

def fit_slope_R(X, Y):
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
    slope = function(X, Y, W=NA, choice_weights, tol_infeas = 1e-6,
       tol_rel_gap = 1e-6){

       result = SLOPE(X, Y, q = 0.1, lambda='bh', scale='l2',
                      intercept=FALSE,
                      tol_infeas = tol_infeas, tol_rel_gap = tol_rel_gap)
      
       print(result$alpha)
       return(list(beta = result$coefficients, E = result$selected, lambda_seq = result$lambda, sigma = result$sigma,
                   alpha=result$alpha))
    }''')

    r_slope = robjects.globalenv['slope']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)

    result = r_slope(r_X, r_Y)

    rpy2.robjects.numpy2ri.deactivate()

    return result[0], result[1], result[2], result[3], result[4]

@set_seed_for_test(10)
@np.testing.dec.skipif(True, msg="SLOPE parameterization in R has changed")
def test_using_SLOPE_weights():

    n, p = 500, 50

    X = np.random.standard_normal((n, p))
    #Y = np.random.standard_normal(n)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    beta = np.zeros(p)
    beta[:5] = 5.

    Y = X.dot(beta) + np.random.standard_normal(n)

    output_R = fit_slope_R(X, Y)
    r_beta = np.squeeze(output_R[0])[:,3]
    r_lambda_seq = np.array(output_R[2]).reshape(-1)
    alpha = output_R[-1]

    W = np.asarray(r_lambda_seq * alpha[3]).reshape(-1)
    pen = slope(W, lagrange=1.)

    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve(tol=1.e-14, min_its=500)

    # we get a better objective value
    nt.assert_true(problem.objective(soln) < problem.objective(np.asarray(r_beta)))
    nt.assert_true(np.linalg.norm(soln - r_beta) < 1.e-6 * np.linalg.norm(soln))


@set_seed_for_test(10)
@np.testing.dec.skipif(True, msg="SLOPE parameterization in R has changed")
def test_using_SLOPE_prox():

    n, p = 50, 50

    X = np.identity(n)
    beta = np.zeros(p)
    beta[:5] = 5.

    Y = X.dot(beta) + np.random.standard_normal(n)

    output_R = fit_slope_R(X, Y)
    r_beta = np.squeeze(output_R[0])[:,3]
    r_lambda_seq = np.array(output_R[2]).reshape(-1)
    alpha = output_R[-1]

    W = np.asarray(r_lambda_seq * alpha[3]).reshape(-1)
    pen = slope(W, lagrange=1.)

    soln = pen.lagrange_prox(Y)

    # test that the prox maps agree
    nt.assert_true(np.linalg.norm(soln - r_beta) < 1.e-10 * np.linalg.norm(soln))


