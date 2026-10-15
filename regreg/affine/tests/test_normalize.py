from itertools import product

import numpy as np
import scipy.sparse

import regreg.api as rr
from regreg.identity_quadratic import identity_quadratic as sq

def test_centering():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    rng = np.random.default_rng(5100)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = rng.normal(size=(N,P-1)) + offset
    X2 = X - X.mean(axis=0)[None,:]
    L = rr.normalize(X, center=True, scale=False)
    # coef for loss

    for _ in range(10):
        beta = rng.normal(size=(P,))
        v = L.linear_map(beta)
        v2 = np.dot(X, beta)
        v2 -= v2.mean()
        v3 = np.dot(X2, beta)
        v4 = L.affine_map(beta)
        np.testing.assert_almost_equal(v, v3)
        np.testing.assert_almost_equal(v, v2)
        np.testing.assert_almost_equal(v, v4)

        y = rng.standard_normal(N)
        u1 = L.adjoint_map(y)
        y2 = y - y.mean()
        u2 = np.dot(X.T, y2)
        np.testing.assert_almost_equal(u1, u2)

def test_scaling():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    rng = np.random.default_rng(5101)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = rng.normal(size=(N,P-1)) + offset

    L = rr.normalize(X, center=False, scale=True)
    # coef for loss

    scalings = np.sqrt((X**2).sum(0) / N)
    scaling_matrix = np.diag(1./scalings)
    
    for _ in range(10):

        beta = rng.normal(size=(P,))
        v = L.linear_map(beta)
        v2 = np.dot(X, np.dot(scaling_matrix, beta))
        v3 = L.affine_map(beta)
        np.testing.assert_almost_equal(v, v2)
        np.testing.assert_almost_equal(v, v3)

        y = rng.standard_normal(N)
        u1 = L.adjoint_map(y)
        u2 = np.dot(scaling_matrix, np.dot(X.T, y))
        np.testing.assert_almost_equal(u1, u2)

def test_scaling_and_centering():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    rng = np.random.default_rng(5102)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with no colum of ones!
    X = rng.normal(size=(N,P)) + offset

    L = rr.normalize(X, center=True, scale=True) # the default
    # coef for loss

    scalings = np.std(X, 0, ddof=1)
    scaling_matrix = np.diag(1./scalings)

    for _ in range(10):
        beta = rng.normal(size=(P,))
        v = L.linear_map(beta)
        v2 = np.dot(X, np.dot(scaling_matrix, beta))
        v2 -= v2.mean()
        np.testing.assert_almost_equal(v, v2)

        y = rng.standard_normal(N)
        u1 = L.adjoint_map(y)
        y2 = y - y.mean()
        u2 = np.dot(scaling_matrix, np.dot(X.T, y2))
        np.testing.assert_almost_equal(u1, u2)

def test_centering_fit(debug=False):

    rng = np.random.default_rng(5103)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X = rng.normal(size=(N,P)) + offset
    X2 = X - X.mean(axis=0)[None,:]

    # the normalizer
    L = rr.normalize(X, center=True, scale=False)

    # data
    Y = rng.normal(size=(N,)) + offset

    # coef for loss
    coef = 0.5
    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.quadratic_loss.affine(L, -Y, coef=coef)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P,), penalties,
                           groups)
    initial = rng.standard_normal(P)

    composite_form = rr.separable_problem.fromatom(penalty, loss)
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.quadratic_loss.affine(X2, -Y, coef=coef)

    initial2 = rng.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    for _ in range(10):
        beta = rng.standard_normal(P)
        g1 = loss.smooth_objective(beta, mode='grad')
        g2 = loss2.smooth_objective(beta, mode='grad')
        np.testing.assert_almost_equal(g1, g2)
        b1 = penalty.proximal(sq(1, beta, g1, 0))
        b2 = penalty.proximal(sq(1, beta, g1, 0))
        np.testing.assert_almost_equal(b1, b2)

        f1 = composite_form.objective(beta)
        f2 = composite_form2.objective(beta)
        np.testing.assert_almost_equal(f1, f2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    np.testing.assert_almost_equal(composite_form.objective(coefs), composite_form.objective(coefs2))
    np.testing.assert_almost_equal(composite_form2.objective(coefs), composite_form2.objective(coefs2))

    assert np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04

def test_scaling_fit(debug=False):

    rng = np.random.default_rng(5104)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = rng.normal(size=(N,P-1)) + offset
    X2 = X / (np.sqrt((X**2).mean(0)))[None,:]
    L = rr.normalize(X, center=False, scale=True)

    # data
    Y = rng.normal(size=(N,)) + offset

    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.squared_error(L, Y)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P,), penalties,
                           groups)
    initial = rng.standard_normal(P)
    composite_form = rr.separable_problem.fromatom(penalty, loss)

    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.squared_error(X2, Y)

    initial2 = rng.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = rng.standard_normal(P)
        g1 = loss.smooth_objective(beta, mode='grad')
        g2 = loss2.smooth_objective(beta, mode='grad')
        np.testing.assert_almost_equal(g1, g2)
        b1 = penalty.proximal(sq(1, beta, g1, 0))
        b2 = penalty.proximal(sq(1, beta, g2, 0))
        np.testing.assert_almost_equal(b1, b2)

        f1 = composite_form.objective(beta)
        f2 = composite_form2.objective(beta)
        np.testing.assert_almost_equal(f1, f2)


    np.testing.assert_almost_equal(composite_form.objective(coefs), composite_form.objective(coefs2))
    np.testing.assert_almost_equal(composite_form2.objective(coefs), composite_form2.objective(coefs2))

    assert np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04

def test_scaling_and_centering_fit(debug=False):

    rng = np.random.default_rng(5105)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = rng.normal(size=(N,P)) + 0 * offset
    X2 = X - X.mean(0)[None,:]
    X2 = X2 / np.std(X2,0,ddof=1)[None,:]

    L = rr.normalize(X, center=True, scale=True)
    # data
    Y = rng.normal(size=(N,)) + offset

    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.squared_error(L, Y)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P,), penalties,
                           groups)

    initial = rng.standard_normal(P)
    composite_form = rr.separable_problem.fromatom(penalty, loss)
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.squared_error(X2, Y)

    initial2 = rng.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = rng.standard_normal(P)
        g1 = loss.smooth_objective(beta, mode='grad')
        g2 = loss2.smooth_objective(beta, mode='grad')
        np.testing.assert_almost_equal(g1, g2)
        b1 = penalty.proximal(sq(1, beta, g1, 0))
        b2 = penalty.proximal(sq(1, beta, g2, 0))
        np.testing.assert_almost_equal(b1, b2)

        f1 = composite_form.objective(beta)
        f2 = composite_form2.objective(beta)
        np.testing.assert_almost_equal(f1, f2)

    np.testing.assert_almost_equal(composite_form.objective(coefs), composite_form.objective(coefs2))
    np.testing.assert_almost_equal(composite_form2.objective(coefs), composite_form2.objective(coefs2))

    assert np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04

def test_scaling_and_centering_intercept_fit(debug=False):

    rng = np.random.default_rng(5106)
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = rng.normal(size=(N,P)) + 0 * offset
    X2 = X - X.mean(0)[None,:]
    X2 = X2 / np.std(X2,0,ddof=1)[None,:]
    X2 = np.hstack([np.ones((X2.shape[0],1)), X2])
    
    L = rr.normalize(X, center=True, scale=True, intercept=True)
    # data
    Y = rng.normal(size=(N,)) + offset

    # lagrange for penalty
    lagrange = .1

    # Loss function (squared difference between fitted and actual data)
    loss = rr.squared_error(L, Y)

    penalties = [rr.constrained_positive_part(25, lagrange=lagrange),
                 rr.nonnegative(5)]
    groups = [slice(0,25), slice(25,30)]
    penalty = rr.separable((P+1,), penalties,
                           groups)

    initial = rng.standard_normal(P+1)
    composite_form = rr.separable_problem.fromatom(penalty, loss)
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.squared_error(X2, Y)

    initial2 = rng.standard_normal(P+1)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = rng.standard_normal(P+1)
        g1 = loss.smooth_objective(beta, mode='grad')
        g2 = loss2.smooth_objective(beta, mode='grad')
        np.testing.assert_almost_equal(g1, g2)
        b1 = penalty.proximal(sq(1, beta, g1, 0))
        b2 = penalty.proximal(sq(1, beta, g2, 0))
        np.testing.assert_almost_equal(b1, b2)

        f1 = composite_form.objective(beta)
        f2 = composite_form2.objective(beta)
        np.testing.assert_almost_equal(f1, f2)

    np.testing.assert_almost_equal(composite_form.objective(coefs), composite_form.objective(coefs2))
    np.testing.assert_almost_equal(composite_form2.objective(coefs), composite_form2.objective(coefs2))

    assert np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04

