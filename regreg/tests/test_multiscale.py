import numpy as np
from regreg.affine.multiscale import multiscale
import regreg.api as rr
import regreg.affine as ra

import matplotlib.pyplot as plt

def _multiscale_matrix(p, minsize=None):
    minsize = minsize or int(p**(1/3.))
    rows = []
    for i in range(p):
        for j in range(i, p):
            if (j - i) >= minsize:
                row = np.zeros(p)
                row[i:j] = 1. / (j-i)
                rows.append(row)
    return np.array(rows)

def test_multiscale():

    M = _multiscale_matrix(200)
    Mtrans = multiscale(200)
    V = np.random.standard_normal(M.shape[1])
    W = np.random.standard_normal(M.shape[0])

    np.testing.assert_allclose(np.dot(M, V), Mtrans.linear_map(V))
    np.testing.assert_allclose(np.dot(M.T, W), Mtrans.adjoint_map(W))

def test_changepoint():

    p = 150
    M = multiscale(p)
    M.minsize = 10
    X = ra.adjoint(M)

    Y = np.random.standard_normal(p)
    Y[20:50] += 8
    Y += 2
    meanY = Y.mean()

    lammax = np.fabs(X.adjoint_map(Y)).max()

    penalty = rr.l1norm(X.input_shape, lagrange=0.5*lammax)
    loss = rr.squared_error(X, Y - meanY)
    problem = rr.simple_problem(loss, penalty)
    soln = problem.solve()
    Yhat = X.linear_map(soln)
    Yhat += meanY

    plt.scatter(np.arange(p), Y)
    plt.plot(np.arange(p), Yhat)

