import numpy as np

from regreg.affine.multiscale import multiscale, choose_tuning_parameter
import regreg.api as rr
import regreg.affine as ra
from regreg.tests.decorators import set_seed_for_test

INTERACTIVE = False

if INTERACTIVE:
    try:
        from matplotlib.pyplot import plt
    except ImportError:
        INTERACTIVE = False

def _multiscale_matrix(p, minsize=None):
    minsize = minsize or int(np.around(p**(1/3.)))
    rows = []
    for i in range(p):
        for j in range(i, p):
            if (j - i) >= minsize:
                row = np.zeros(p)
                row[i:j] = 1. / (j-i)
                rows.append(row)
    A = np.array(rows)
    A -= np.mean(A, 1)[:,None]
    return A

@set_seed_for_test()
def test_multiscale():

    M = _multiscale_matrix(200)
    Mtrans = multiscale(200)
    V = np.random.standard_normal(M.shape[1])
    W = np.random.standard_normal(M.shape[0])

    np.testing.assert_allclose(np.dot(M, V), Mtrans.linear_map(V))
    np.testing.assert_allclose(np.dot(M.T, W), Mtrans.adjoint_map(W))
    
    Mtrans.update_slices([(s,e) for s, e in Mtrans.slices])
    np.testing.assert_allclose(np.dot(M, V), Mtrans.linear_map(V))
    np.testing.assert_allclose(np.dot(M.T, W), Mtrans.adjoint_map(W))
    
    Mtrans = multiscale(200, slices=[(s,e) for s, e in Mtrans.slices])
    Mtrans.update_slices(Mtrans.slices)

    Mtrans = multiscale(200, slices=[(s,e) for s, e in Mtrans.slices], scaling=2)
    Mtrans.linear_map(V)
    Mtrans.affine_map(V)
    Mtrans.adjoint_map(W)

@set_seed_for_test()
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

    if INTERACTIVE:
        plt.scatter(np.arange(p), Y)
        plt.plot(np.arange(p), Yhat)
        plt.show()

@set_seed_for_test()
def test_changepoint_scaled():

    p = 150
    M = multiscale(p)
    M.minsize = 10
    X = ra.adjoint(M)

    Y = np.random.standard_normal(p)
    Y[20:50] += 8
    Y += 2
    meanY = Y.mean()

    lammax = np.fabs(np.sqrt(M.sizes) * X.adjoint_map(Y) / (1 + np.sqrt(np.log(M.sizes)))).max()

    penalty = rr.weighted_l1norm((1 + np.sqrt(np.log(M.sizes))) / np.sqrt(M.sizes), lagrange=0.5*lammax)
    loss = rr.squared_error(X, Y - meanY)
    problem = rr.simple_problem(loss, penalty)
    soln = problem.solve()
    Yhat = X.linear_map(soln)
    Yhat += meanY

    if INTERACTIVE:
        plt.scatter(np.arange(p), Y)
        plt.plot(np.arange(p), Yhat)
        plt.show()

def test_choose_parameter(delta=2, p=60):

    signal = np.zeros(p)
    signal[(p//2):] += delta
    Z = np.random.standard_normal(p) + signal
    p = Z.shape[0]
    M = multiscale(p)
    M.scaling = np.sqrt(M.sizes)
    lam = choose_tuning_parameter(M)
    weights = (lam + np.sqrt(2 * np.log(p / M.sizes))) / np.sqrt(p)

    Z0 = Z - Z.mean()
    loss = rr.squared_error(ra.adjoint(M), Z0)
    penalty = rr.weighted_l1norm(weights, lagrange=1.)
    problem = rr.simple_problem(loss, penalty)
    coef = problem.solve()
    active = coef != 0

    if active.sum():
        X = M.form_matrix(M.slices[active])[0]



