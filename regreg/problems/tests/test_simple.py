import numpy as np
from itertools import product
import regreg.api as rr

def test_class():
    '''
    runs several class methods on generic instance
    '''

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    loss = rr.squared_error(X, Y)
    pen = rr.l1norm(p, lagrange=1.)
    problem = rr.simple_problem(loss, pen)

    problem.latexify()

    for debug, coef_stop, max_its in product([True, False], [True, False], [5, 100]):
        rr.gengrad(problem, rr.power_L(X)**2, max_its=max_its, debug=debug, coef_stop=coef_stop)

