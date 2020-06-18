import numpy as np

from ...atoms.seminorms import l1norm
from ...smooth.glm import glm
from ..newton import quasi_newton
from ..simple import simple_problem

def test_lagrange():

    n, p = 1000, 50
    X = np.random.standard_normal((n, p))
    Y = np.random.binomial(1, 0.5, size=(n,))
    loss = glm.logistic(X, Y)
    penalty = l1norm(p, lagrange=3)

    qn = quasi_newton(loss,
                      penalty,
                      X.T.dot(X))
    soln_newton = qn.solve(niter=20)

    problem = simple_problem(loss, penalty)
    soln_simple = problem.solve()

    assert(np.linalg.norm(soln_newton - soln_simple) / np.linalg.norm(soln_simple) < 1.e-6)

def test_bound():

    n, p = 1000, 50
    X = np.random.standard_normal((n, p))
    Y = np.random.binomial(1, 0.5, size=(n,))
    loss = glm.logistic(X, Y)
    penalty = l1norm(p, bound=0.2)

    qn = quasi_newton(loss,
                      penalty,
                      X.T.dot(X))
    soln_newton = qn.solve(niter=40)

    problem = simple_problem(loss, penalty)
    soln_simple = problem.solve()

    assert(np.fabs(soln_simple - soln_newton).sum() / 0.2)
