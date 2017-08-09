"""
Test that GLM agrees with R's glm
"""

from __future__ import print_function

import numpy as np
import nose.tools as nt

try:
    import rpy2.robjects as rpy
    rpy2_available = True
except ImportError:
    rpy2_available = False

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test(10)
@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_gaussian():

    rpy.r('set.seed(0)')
    rpy.r('X = matrix(rnorm(1000), 50, 20)')
    rpy.r('Y = rnorm(50)')
    rpy.r('C = coef(lm(Y~X-1))')
    C = np.asarray(rpy.r('C'))
    X = np.asarray(rpy.r('X'))
    Y = np.asarray(rpy.r('Y'))
    L = rr.glm.gaussian(X, Y)
    soln = L.solve(min_its=200)

    np.testing.assert_allclose(C, soln)

@set_seed_for_test(10)
@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_logistic():

    rpy.r('set.seed(0)')
    rpy.r('X = matrix(rnorm(1000), 50, 20)')
    rpy.r('TR = c(rep(1,20), rep(2,30))')
    rpy.r('Y = rbinom(50, TR, 0.5)')
    rpy.r('C = coef(glm(Y/TR~X-1, family=binomial(), weights=TR))')
    C = np.asarray(rpy.r('C'))
    X = np.asarray(rpy.r('X'))
    Y = np.asarray(rpy.r('Y'))
    TR = np.asarray(rpy.r('TR'))
    L = rr.glm.logistic(X, Y, trials=TR)
    soln = L.solve(min_its=200)

    np.testing.assert_allclose(C, soln, atol=1.e-5, rtol=1.e-5)

@set_seed_for_test(10)
@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_poisson():

    rpy.r('set.seed(0)')
    rpy.r('X = matrix(rnorm(1000), 50, 20)')
    rpy.r('Y = rpois(50, 5)')
    rpy.r('C = coef(glm(Y~X-1, family=poisson()))')
    C = np.asarray(rpy.r('C'))
    X = np.asarray(rpy.r('X'))
    Y = np.asarray(rpy.r('Y'))
    L = rr.glm.poisson(X, Y)
    soln = L.solve(min_its=200)

    # tolerance adjusted here because of an occasional failure
    # https://nipy.bic.berkeley.edu/builders/regreg-py2.6/builds/30/steps/shell_6/logs/stdio
    np.testing.assert_allclose(C, soln, atol=1.e-5, rtol=1.e-5)

@set_seed_for_test(10)
@np.testing.dec.skipif(not rpy2_available, msg="rpy2 not available, skipping test")
def test_coxph():

    rpy.r('''
    set.seed(0)
    sigma=1.
    X = matrix(rnorm(250), 50, 5)
    X = scale(X, TRUE, TRUE)
    beta = c(3,2,rep(0,3))
    tim = as.vector(X%*%beta + sigma*rnorm(50))
    tim = tim-min(tim)+1
    status = sample(c(0,1),size=50,replace=T)

    library(survival)
    C = coef(coxph(Surv(tim, status) ~ X))
    ''')
    C = np.asarray(rpy.r('C'))
    X = np.asarray(rpy.r('X'))
    T = np.asarray(rpy.r('tim'))
    S = np.asarray(rpy.r('status'))
    L = rr.coxph(X, T, S)
    soln = L.solve(min_its=200)

    np.testing.assert_allclose(C, soln, rtol=1.e-4, atol=1.e-4)
