import numpy as np

from ...atoms.seminorms import l1norm
from ...smooth.glm import glm
from ..newton import quasi_newton
from ..simple import simple_problem

def test_lagrange():

    n, p, s = 1000, 50, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = 20 * np.random.standard_normal(s) / np.sqrt(n)
    eta = X.dot(beta)
    pi = np.exp(eta) / (1 + np.exp(eta))
    Y = np.random.binomial(1, pi)
    assert(Y.shape == pi.shape)
    loss = glm.logistic(X, Y)
    penalty = l1norm(p, lagrange=4)

    qn = quasi_newton(loss,
                      penalty,
                      X.T.dot(X) / 4.)
    soln_newton = qn.solve(niter=1000, tol=1.e-6,
                           maxfun=5, maxiter=5)

    problem = simple_problem(loss, penalty)
    soln_simple = problem.solve(min_its=200, tol=1.e-14)

    # the two solvers agree in objective to ~1e-10, but their coefficients
    # differ by up to ~1e-6 (relative) depending on the seed / BLAS
    assert(np.linalg.norm(soln_newton - soln_simple) / np.linalg.norm(soln_simple) < 1.e-5)
    assert(np.fabs(problem.objective(soln_newton) - problem.objective(soln_simple)) < 1.e-6)

def test_bound():

    n, p = 1000, 50
    X = np.random.standard_normal((n, p))
    Y = np.random.binomial(1, 0.5, size=(n,))
    loss = glm.logistic(X, Y)
    penalty = l1norm(p, bound=0.5)

    qn = quasi_newton(loss,
                      penalty,
                      X.T.dot(X) / 4.)
    soln_newton = qn.solve(niter=1000, tol=1.e-10,
                           maxfun=5, maxiter=5)

    problem = simple_problem(loss, penalty)
    soln_simple = problem.solve(tol=1.e-14)

    assert(np.linalg.norm(soln_newton - soln_simple) / max(np.linalg.norm(soln_simple), 1) < 1.e-5)
    assert(np.fabs(problem.objective(soln_newton) - problem.objective(soln_simple)) < 1.e-6)

def test_latexify_objective():
    # gh-84: latexify used undefined objective_vars; objective passed
    # 'func' as check_feasibility

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    Y = rng.standard_normal(50)
    from ...smooth.quadratic import squared_error
    loss = squared_error(X, Y)
    penalty = l1norm(5, lagrange=1.)
    qn = quasi_newton(loss, penalty, X.T.dot(X))
    assert 'Quasi-Newton' in qn.latexify()
    assert r'\gamma' in qn.latexify(var=r'\gamma')
    b = rng.standard_normal(5)
    np.testing.assert_allclose(qn.objective(b),
                               loss.smooth_objective(b, 'func') + penalty.nonsmooth_objective(b))
