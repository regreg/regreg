from itertools import product
import nose.tools as nt

import numpy as np
import scipy.sparse

import regreg.api as rr
from regreg.identity_quadratic import identity_quadratic as sq
from regreg.tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_centering():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    X2 = X - X.mean(axis=0)[np.newaxis,:]
    L = rr.normalize(X, center=True, scale=False)
    # coef for loss

    for _ in range(10):
        beta = np.random.normal(size=(P,))
        v = L.linear_map(beta)
        v2 = np.dot(X, beta)
        v2 -= v2.mean()
        v3 = np.dot(X2, beta)
        v4 = L.affine_map(beta)
        np.testing.assert_almost_equal(v, v3)
        np.testing.assert_almost_equal(v, v2)
        np.testing.assert_almost_equal(v, v4)

        y = np.random.standard_normal(N)
        u1 = L.adjoint_map(y)
        y2 = y - y.mean()
        u2 = np.dot(X.T, y2)
        np.testing.assert_almost_equal(u1, u2)

@set_seed_for_test()
def test_scaling():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset

    L = rr.normalize(X, center=False, scale=True)
    # coef for loss

    scalings = np.sqrt((X**2).sum(0) / N)
    scaling_matrix = np.diag(1./scalings)
    
    for _ in range(10):

        beta = np.random.normal(size=(P,))
        v = L.linear_map(beta)
        v2 = np.dot(X, np.dot(scaling_matrix, beta))
        v3 = L.affine_map(beta)
        np.testing.assert_almost_equal(v, v2)
        np.testing.assert_almost_equal(v, v3)

        y = np.random.standard_normal(N)
        u1 = L.adjoint_map(y)
        u2 = np.dot(scaling_matrix, np.dot(X.T, y))
        np.testing.assert_almost_equal(u1, u2)

@set_seed_for_test()
def test_scaling_and_centering():
    """
    This test verifies that the normalized transform
    of affine correctly implements the linear
    transform that multiplies first by X, then centers.
    """
    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with no colum of ones!
    X = np.random.normal(size=(N,P)) + offset

    L = rr.normalize(X, center=True, scale=True) # the default
    # coef for loss

    scalings = np.std(X, 0)
    scaling_matrix = np.diag(1./scalings)

    for _ in range(10):
        beta = np.random.normal(size=(P,))
        v = L.linear_map(beta)
        v2 = np.dot(X, np.dot(scaling_matrix, beta))
        v2 -= v2.mean()
        np.testing.assert_almost_equal(v, v2)

        y = np.random.standard_normal(N)
        u1 = L.adjoint_map(y)
        y2 = y - y.mean()
        u2 = np.dot(scaling_matrix, np.dot(X.T, y2))
        np.testing.assert_almost_equal(u1, u2)

@set_seed_for_test()
def test_centering_fit(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 50

    # design - with ones as last column
    X = np.ones((N,P))
    X = np.random.normal(size=(N,P)) + offset
    X2 = X - X.mean(axis=0)[np.newaxis,:]

    # the normalizer
    L = rr.normalize(X, center=True, scale=False)

    # data
    Y = np.random.normal(size=(N,)) + offset

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
    initial = np.random.standard_normal(P)

    composite_form = rr.separable_problem.fromatom(penalty, loss)
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.quadratic_loss.affine(X2, -Y, coef=coef)

    initial2 = np.random.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    for _ in range(10):
        beta = np.random.standard_normal(P)
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

    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@set_seed_for_test()
def test_scaling_fit(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    X2 = X / (np.sqrt((X**2).sum(0) / N))[np.newaxis,:]
    L = rr.normalize(X, center=False, scale=True)

    # data
    Y = np.random.normal(size=(N,)) + offset

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
    initial = np.random.standard_normal(P)
    composite_form = rr.separable_problem.fromatom(penalty, loss)

    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.quadratic_loss.affine(X2, -Y, coef=coef)

    initial2 = np.random.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = np.random.standard_normal(P)
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

    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@set_seed_for_test()
def test_scaling_and_centering_fit(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.random.normal(size=(N,P)) + offset
    X2 = X - X.mean(0)[np.newaxis,:]
    X2 = X2 / np.std(X2,0)[np.newaxis,:]

    L = rr.normalize(X, center=True, scale=True)
    # data
    Y = np.random.normal(size=(N,)) + offset

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

    initial = np.random.standard_normal(P)
    composite_form = rr.separable_problem.fromatom(penalty, loss)
    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X2
    loss2 = rr.quadratic_loss.affine(X2, -Y, coef=coef)

    initial2 = np.random.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = np.random.standard_normal(P)
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

    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@set_seed_for_test()
def test_scaling_and_centering_fit_inplace(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design
    X = np.random.normal(size=(N,P)) + offset
    L = rr.normalize(X, center=True, scale=True, inplace=True)

    # X should have been normalized in place
    np.testing.assert_almost_equal(np.sum(X**2, 0), N)
    np.testing.assert_almost_equal(np.sum(X, 0), 0)

    # data
    Y = np.random.normal(size=(N,)) + offset

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

    initial = np.random.standard_normal(P)
    composite_form = rr.separable_problem.fromatom(penalty, loss)

    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X, which has been normalized in place
    loss2 = rr.quadratic_loss.affine(X, -Y, coef=coef)

    initial2 = np.random.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = np.random.standard_normal(P)
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

    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@set_seed_for_test()
def test_scaling_fit_inplace(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.ones((N,P))
    X[:,:-1] = np.random.normal(size=(N,P-1)) + offset
    L = rr.normalize(X, center=False, scale=True, inplace=True)

    # X should have been normalized in place
    np.testing.assert_almost_equal(np.sum(X**2, 0), N)

    # data
    Y = np.random.normal(size=(N,)) + offset

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
    initial = np.random.standard_normal(P)
    composite_form = rr.separable_problem.fromatom(penalty, loss)

    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X, which has been normalized in place
    loss2 = rr.quadratic_loss.affine(X, -Y, coef=coef)

    initial2 = np.random.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = np.random.standard_normal(P)
        g1 = loss.smooth_objective(beta, mode='grad')
        g2 = loss2.smooth_objective(beta, mode='grad')
        np.testing.assert_almost_equal(g1, g2)
        b1 = penalty.proximal(sq(1, beta, g1,0))
        b2 = penalty.proximal(sq(1, beta, g2,0))
        np.testing.assert_almost_equal(b1, b2)

        f1 = composite_form.objective(beta)
        f2 = composite_form2.objective(beta)
        np.testing.assert_almost_equal(f1, f2)


    np.testing.assert_almost_equal(composite_form.objective(coefs), composite_form.objective(coefs2))
    np.testing.assert_almost_equal(composite_form2.objective(coefs), composite_form2.objective(coefs2))

    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@set_seed_for_test()
def test_centering_fit_inplace(debug=False):

    # N - number of data points
    # P - number of columns in design == number of betas
    N, P = 40, 30
    # an arbitrary positive offset for data and design
    offset = 2

    # design - with ones as last column
    X = np.random.normal(size=(N,P)) + offset
    L = rr.normalize(X, center=True, scale=False, inplace=True)

    # X should have been normalized in place
    np.testing.assert_almost_equal(np.sum(X, 0), 0)

    # data
    Y = np.random.normal(size=(N,)) + offset

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
    initial = np.random.standard_normal(P)

    composite_form = rr.separable_problem.fromatom(penalty, loss)

    solver = rr.FISTA(composite_form)
    solver.debug = debug
    solver.fit(tol=1.0e-12, min_its=200)
    coefs = solver.composite.coefs

    # Solve the problem with X, which has been normalized in place
    loss2 = rr.quadratic_loss.affine(X, -Y, coef=coef)

    initial2 = np.random.standard_normal(P)
    composite_form2 = rr.separable_problem.fromatom(penalty, loss2)

    solver2 = rr.FISTA(composite_form2)
    solver2.debug = debug
    solver2.fit(tol=1.0e-12, min_its=200)
    coefs2 = solver2.composite.coefs

    for _ in range(10):
        beta = np.random.standard_normal(P)
        g1 = loss.smooth_objective(beta, mode='grad')
        g2 = loss2.smooth_objective(beta, mode='grad')
        np.testing.assert_almost_equal(g1, g2)
        b1 = penalty.proximal(sq(1, beta, g1,0))
        b2 = penalty.proximal(sq(1, beta, g2,0))
        np.testing.assert_almost_equal(b1, b2)

        f1 = composite_form.objective(beta)
        f2 = composite_form2.objective(beta)
        np.testing.assert_almost_equal(f1, f2)


    np.testing.assert_almost_equal(composite_form.objective(coefs), composite_form.objective(coefs2))
    np.testing.assert_almost_equal(composite_form2.objective(coefs), composite_form2.objective(coefs2))

    nt.assert_true(np.linalg.norm(coefs - coefs2) / max(np.linalg.norm(coefs),1) < 1.0e-04)

@set_seed_for_test()
def test_normalize_intercept():

    for issparse, value, inplace, intercept_column, scale, center in product([False, True], 
                                                       [1,3], 
                                                       [False, True], 
                                                       [None, 2],
                                                       [True, False],
                                                       [True, False]):
        
        print (issparse, value, inplace, intercept_column, scale, center)
        if not (issparse and inplace):

            X = np.random.standard_normal((20,6))
            if intercept_column is not None:
                X[:,intercept_column] = 1
            Y = X.copy()

            if issparse:
                X = scipy.sparse.csr_matrix(X)

            Xn = rr.normalize(X, 
                              value=value, 
                              inplace=inplace, 
                              intercept_column=intercept_column,
                              scale=scale, 
                              center=center)

            if intercept_column is not None:
                v = np.zeros(Y.shape[1])
                v[intercept_column] = 4
                yield np.testing.assert_allclose, Xn.linear_map(v), 4 * np.ones(Y.shape[0])

            if scale and center:

                Y -= Y.mean(0)[None,:]
                Y /= Y.std(0)[None,:]
                Y *= np.sqrt(value)
                if intercept_column is not None:
                    Y[:,intercept_column] = 1
            
            elif scale and not center:

                Y /= (np.sqrt((Y**2).sum(0))[None,:] / np.sqrt(Y.shape[0]))
                Y *= np.sqrt(value)
                if intercept_column is not None:
                    Y[:,intercept_column] = 1

            elif center and not scale:

                Y -= Y.mean(0)[None,:]
                if intercept_column is not None:
                    Y[:,intercept_column] = 1

            V = np.random.standard_normal((20, 3))
            U = np.random.standard_normal((6,4))

            Xn.adjoint_map(V)
            yield np.testing.assert_allclose, np.dot(Y, U), Xn.linear_map(np.array(U))
            yield np.testing.assert_allclose, np.dot(Y, U), Xn.affine_map(np.array(U))
            yield np.testing.assert_allclose, np.dot(Y, U[:,0]), Xn.linear_map(np.array(U[:,0]))
            yield np.testing.assert_allclose, np.dot(Y, U[:,0]), Xn.affine_map(np.array(U[:,0]))
            yield np.testing.assert_allclose, np.dot(Y.T, V), Xn.adjoint_map(V)
            yield nt.assert_raises, ValueError, Xn.linear_map, np.zeros((6,4,3))

            X2 = Xn.slice_columns(list(range(3)))
            Y2 = Y[:,:3]
            U2 = np.random.standard_normal((3,4))
            V2 = np.random.standard_normal(20)

            yield np.testing.assert_allclose, np.dot(Y2, U2), X2.linear_map(np.array(U2))
            yield np.testing.assert_allclose, np.dot(Y2, U2), X2.affine_map(np.array(U2))
            yield np.testing.assert_allclose, np.dot(Y2, U2[:,0]), X2.linear_map(np.array(U2[:,0]))
            yield np.testing.assert_allclose, np.dot(Y2, U2[:,0]), X2.affine_map(np.array(U2[:,0]))
            yield np.testing.assert_allclose, np.dot(Y2.T, V2), X2.adjoint_map(V2)

            X2 = Xn.slice_columns(list(range(3,6)))
            Y2 = Y[:,3:]
            U2 = np.random.standard_normal((3,4))
            V2 = np.random.standard_normal(20)

            yield np.testing.assert_allclose, np.dot(Y2, U2), X2.linear_map(np.array(U2))
            yield np.testing.assert_allclose, np.dot(Y2, U2), X2.affine_map(np.array(U2))
            yield np.testing.assert_allclose, np.dot(Y2, U2[:,0]), X2.linear_map(np.array(U2[:,0]))
            yield np.testing.assert_allclose, np.dot(Y2, U2[:,0]), X2.affine_map(np.array(U2[:,0]))
            yield np.testing.assert_allclose, np.dot(Y2.T, V2), X2.adjoint_map(V2)

            keep = np.zeros(6, np.bool)
            keep[:3] = 1
            X2 = Xn.slice_columns(keep)
            Y2 = Y[:,:3]
            U2 = np.random.standard_normal((3,4))
            V2 = np.random.standard_normal(20)

            yield np.testing.assert_allclose, np.dot(Y2, U2), X2.linear_map(np.array(U2))
            yield np.testing.assert_allclose, np.dot(Y2, U2), X2.affine_map(np.array(U2))
            yield np.testing.assert_allclose, np.dot(Y2, U2[:,0]), X2.linear_map(np.array(U2[:,0]))
            yield np.testing.assert_allclose, np.dot(Y2, U2[:,0]), X2.affine_map(np.array(U2[:,0]))
            yield np.testing.assert_allclose, np.dot(Y2.T, V2), X2.adjoint_map(V2)

    yield nt.assert_raises, ValueError, rr.normalize, scipy.sparse.csr_matrix(Y), True, True, 1, True

