import numpy as np

import regreg.api as rr 
import regreg.affine as ra
from regreg.problems.admm import admm_problem

def test_admm(n=100, p=10):

    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    loss = rr.squared_error(X, Y)
    D = np.identity(p)
    pen = rr.l1norm(p, lagrange=1.5)

    ADMM = admm_problem(loss, pen, ra.astransform(D), 0.5)
    ADMM.solve(niter=1000)

    coef1 = ADMM.atom_coefs
    problem2 = rr.simple_problem(loss, pen)
    coef2 = problem2.solve(tol=1.e-12, min_its=500)

    np.testing.assert_allclose(coef1, coef2, rtol=1.e-3, atol=1.e-4)
    
