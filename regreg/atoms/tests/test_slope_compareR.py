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

def fit_slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian"):
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
    slope = function(X, Y, W=NA, normalize, choice_weights, fdr = NA, sigma = 1, tol_infeas = 1e-6,
       tol_rel_gap = 1e-6){

      if(is.na(sigma)){
      sigma = NULL}

      if(is.na(fdr)){
      fdr = 0.1 }

      if(normalize=="TRUE"){
       normalize = TRUE} else{
       normalize = FALSE}

      if(is.na(W))
      {
        if(choice_weights == "gaussian"){
        lambda = "gaussian"} else{
        lambda = "bhq"}
        result = SLOPE(X, Y, fdr = fdr, lambda = lambda, sigma = sigma, normalize = normalize,
                       tol_infeas = tol_infeas, tol_rel_gap = tol_rel_gap)
       } else{
        result = SLOPE(X, Y, fdr = fdr, lambda = W, sigma = sigma, normalize = normalize,
                       tol_infeas = tol_infeas, tol_rel_gap = tol_rel_gap)
      }

      return(list(beta = result$beta, E = result$selected, lambda_seq = result$lambda, sigma = result$sigma))
    }''')

    r_slope = robjects.globalenv['slope']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)

    if normalize is True:
        r_normalize = robjects.StrVector('True')
    else:
        r_normalize = robjects.StrVector('False')

    if W is None:
        r_W = robjects.NA_Logical
        if choice_weights is "gaussian":
            r_choice_weights  = robjects.StrVector('gaussian')
        elif choice_weights is "bhq":
            r_choice_weights = robjects.StrVector('bhq')
    else:
        r_W = robjects.r.matrix(W, nrow=p, ncol=1)
        r_choice_weights = robjects.StrVector('blah')
    result = r_slope(r_X, r_Y, r_W, r_normalize, r_choice_weights)

    rpy2.robjects.numpy2ri.deactivate()

    return result[0], result[1], result[2], result[3]

@set_seed_for_test(10)
@np.testing.dec.skipif(not rpy2_available or not Rslope, msg="rpy2 or SLOPE not available, skipping test")
def test_using_SLOPE_weights():

    n, p = 500, 50

    X = np.random.standard_normal((n, p))
    #Y = np.random.standard_normal(n)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    beta = np.zeros(p)
    beta[:5] = 5.

    Y = X.dot(beta) + np.random.standard_normal(n)

    output_R = fit_slope_R(X, Y, W = None, normalize = True, choice_weights = "bhq")
    r_beta = output_R[0]
    r_lambda_seq = output_R[2]

    W = r_lambda_seq
    pen = slope(W, lagrange=1.)

    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, pen)
    soln = problem.solve(tol=1.e-14, min_its=500)

    # we get a better objective value
    nt.assert_true(problem.objective(soln) < problem.objective(np.asarray(r_beta)))
    nt.assert_true(np.linalg.norm(soln - r_beta) < 1.e-6 * np.linalg.norm(soln))


@set_seed_for_test(10)
@np.testing.dec.skipif(not rpy2_available or not Rslope, msg="rpy2 or SLOPE not available, skipping test")
def test_using_SLOPE_prox():

    n, p = 50, 50

    X = np.identity(n)
    beta = np.zeros(p)
    beta[:5] = 5.

    Y = X.dot(beta) + np.random.standard_normal(n)

    output_R = fit_slope_R(X, Y, W = np.linspace(1, 0.1, n), normalize = True)
    r_beta = output_R[0]
    r_lambda_seq = output_R[2]

    W = r_lambda_seq
    pen = slope(W, lagrange=1.)

    soln = pen.lagrange_prox(Y)

    # test that the prox maps agree
    nt.assert_true(np.linalg.norm(soln - r_beta) < 1.e-10 * np.linalg.norm(soln))


