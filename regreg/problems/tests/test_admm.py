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
    

def test_admm_nonsymmetric():
    # gh-85: the x-update used D instead of D^T

    rng = np.random.default_rng(0)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = np.identity(p) + np.triu(rng.standard_normal((p, p)), 1)
    A = admm_problem(rr.squared_error(X, Y), rr.l1norm(p, lagrange=1.5), ra.astransform(D), 0.5)
    A.solve(niter=5000)
    # w = Dx solves a lasso with design X D^{-1}
    w = rr.simple_problem(rr.squared_error(X @ np.linalg.inv(D), Y),
                          rr.l1norm(p, lagrange=1.5)).solve(tol=1e-14, min_its=1000)
    np.testing.assert_allclose(A.atom_coefs, w, atol=1e-8)

def test_admm_nonsquare():
    # gh-85: non-square D failed (aslinear swapped its shapes; the shape
    # assertion used the wrong space)

    rng = np.random.default_rng(1)
    n, p = 100, 5
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    D = np.diff(np.identity(p), axis=0)    # 4 x 5: fused lasso
    A = admm_problem(rr.squared_error(X, Y), rr.l1norm(p - 1, lagrange=2.), ra.astransform(D), 0.5)
    A.solve(niter=5000)
    # compare objectives with a direct solve of the fused lasso
    fused = rr.l1norm.linear(D, lagrange=2.)
    x_ref = rr.container(rr.squared_error(X, Y), fused).solve(tol=1e-14, min_its=2000)
    obj = lambda x: rr.squared_error(X, Y).smooth_objective(x, 'func') + 2 * np.fabs(D @ x).sum()
    np.testing.assert_allclose(obj(A.loss_coefs), obj(x_ref), rtol=1e-6)
