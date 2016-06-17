""" Testing D transform implementation
"""

from operator import add
import numpy as np

from regreg.affine import (broadcast_first, 
                           affine_transform, 
                           linear_transform,
                           AffineError, 
                           composition, 
                           adjoint,
                           astransform,
                           reshape,
                           selector,
                           vstack,
                           hstack,
                           product,
                           power_L,
                           posneg,
                           scalar_multiply,
                           todense)

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal)

import scipy.sparse
from nose.tools import assert_true, assert_equal, assert_raises

import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

def test_broad_first():
    # Test broadcasting over second axis
    a = np.arange(4) + 10
    b = np.arange(4).reshape(4,1)
    c = broadcast_first(a, b, add)
    res = a[:,None] + b
    assert_equal(res.shape, c.shape)
    assert_array_almost_equal(c, res)
    res1d = res.ravel()
    c = broadcast_first(b, a, add)
    assert_equal(res1d.shape, c.shape)
    assert_array_almost_equal(c, res1d)
    c = broadcast_first(a, b.ravel(), add)
    assert_equal(res1d.shape, c.shape)
    assert_array_almost_equal(c, res1d)

def test_dot():

    A = np.random.standard_normal((20, 12))
    B = np.random.standard_normal((10, 20))

    C = rr.astransform(B).dot(rr.astransform(A))

    assert_equal(C.shape, (10, 12))
    Z = np.random.standard_normal(12)
    assert_array_almost_equal(C.dot(Z), B.dot(A.dot(Z)))

    assert_equal(C.ndim, 2)

def test_affine_transform():
    # Test affine transform

    m = 20
    x1d = np.arange(m)
    x2d = x1d[:,None]
    x22d = np.c_[x2d, x2d]
    # Error if both of linear and affine components are None
    assert_raises(AffineError, affine_transform, None, None)
    assert_raises(AffineError, linear_transform, None)
    # With linear None and 0 affine offset - identity transform
    for x in (x1d, x2d, x22d):
        trans = affine_transform(None, np.zeros((m,1)))
        assert_array_almost_equal(trans.affine_map(x), x)
        assert_array_almost_equal(trans.linear_map(x), x)
        assert_array_almost_equal(trans.dot(x), x)
        assert_array_almost_equal(trans.adjoint_map(x), x)
        assert_array_almost_equal(trans.T.dot(x), x)
        # With linear eye and None affine offset - identity again
        trans = affine_transform(np.eye(m), None)
        assert_array_almost_equal(trans.affine_map(x), x)
        assert_array_almost_equal(trans.linear_map(x), x)
        assert_array_almost_equal(trans.adjoint_map(x), x)
        # affine_transform as input
        trans = affine_transform(trans, None)
        assert_array_almost_equal(trans.affine_map(x), x)
        assert_array_almost_equal(trans.linear_map(x), x)
        assert_array_almost_equal(trans.adjoint_map(x), x)
        # diag
        trans = affine_transform(np.ones(m), None, True)
        assert_array_almost_equal(trans.affine_map(x), x)
        assert_array_almost_equal(trans.linear_map(x), x)
        assert_array_almost_equal(trans.adjoint_map(x), x)

@set_seed_for_test()
def test_composition():
    X1 = np.random.standard_normal((20,30))
    X2 = np.random.standard_normal((30,10))
    b1 = np.random.standard_normal(20)
    b2 = np.random.standard_normal(30)
    L1 = affine_transform(X1, b1)
    L2 = affine_transform(X2, b2)

    z = np.random.standard_normal(10)
    w = np.random.standard_normal(20)
    comp = composition(L1,L2)

    assert_array_almost_equal(comp.linear_map(z), np.dot(X1, np.dot(X2, z)))
    assert_array_almost_equal(comp.adjoint_map(w), np.dot(X2.T, np.dot(X1.T, w)))
    assert_array_almost_equal(comp.affine_map(z), np.dot(X1, np.dot(X2, z)+b2)+b1)

@set_seed_for_test()
def test_composition2():
    X1 = np.random.standard_normal((20,30))
    X2 = np.random.standard_normal((30,10))
    X3 = np.random.standard_normal((10,20))

    b1 = np.random.standard_normal(20)
    b2 = np.random.standard_normal(30)
    b3 = np.random.standard_normal(10)

    L1 = affine_transform(X1, b1)
    L2 = affine_transform(X2, b2)
    L3 = affine_transform(X3, b3)

    z = np.random.standard_normal(20)
    w = np.random.standard_normal(20)
    comp = composition(L1,L2,L3)

    assert_array_almost_equal(comp.linear_map(z), 
                       np.dot(X1, np.dot(X2, np.dot(X3, z))))
    assert_array_almost_equal(comp.adjoint_map(w), 
                       np.dot(X3.T, np.dot(X2.T, np.dot(X1.T, w))))
    assert_array_almost_equal(
        comp.affine_map(z),
        np.dot(X1, np.dot(X2, np.dot(X3, z) + b3) + b2) + b1)

@set_seed_for_test()
def test_adjoint():
    X = np.random.standard_normal((20,30))
    b = np.random.standard_normal(20)
    L = affine_transform(X, b)

    z = np.random.standard_normal(30)
    w = np.random.standard_normal(20)
    A = adjoint(L)

    assert_array_almost_equal(A.linear_map(w), L.adjoint_map(w))
    assert_array_almost_equal(A.affine_map(w), L.adjoint_map(w))
    assert_array_almost_equal(A.adjoint_map(z), L.linear_map(z))

@set_seed_for_test()
def test_affine_sum():

    n = 100
    p = 25

    X1 = np.random.standard_normal((n,p))
    X2 = np.random.standard_normal((n,p))
    b = np.random.standard_normal(n)
    v = np.random.standard_normal(p)

    transform1 = rr.affine_transform(X1, b)
    transform2 = rr.linear_transform(X2)
    sum_transform = rr.affine_sum([transform1, transform2])

    yield assert_array_almost_equal, np.dot(X1,v) + np.dot(X2,v) + b, sum_transform.affine_map(v)
    yield assert_array_almost_equal, np.dot(X1,v) + np.dot(X2,v), sum_transform.linear_map(v)
    yield assert_array_almost_equal, np.dot(X1.T,b) + np.dot(X2.T,b), sum_transform.adjoint_map(b)
    yield assert_array_almost_equal, b, sum_transform.affine_offset

    sum_transform = rr.affine_sum([transform1, transform2], weights=[3,4])

    yield assert_array_almost_equal, 3*(np.dot(X1,v) + b) + 4*(np.dot(X2,v)), sum_transform.affine_map(v)
    yield assert_array_almost_equal, 3*np.dot(X1,v) + 4*np.dot(X2,v), sum_transform.linear_map(v)
    yield assert_array_almost_equal, 3*np.dot(X1.T,b) + 4*np.dot(X2.T,b), sum_transform.adjoint_map(b)
    yield assert_array_almost_equal, 3*b, sum_transform.affine_offset

@set_seed_for_test()
def test_affine_sparse():
    # test using sparse matrices for affine transforms

    n = 100
    p = 25

    X1 = scipy.sparse.csr_matrix(np.random.standard_normal((n,p)))
    b = scipy.sparse.csr_matrix(np.random.standard_normal(n))
    v = np.random.standard_normal(p)
    y = np.random.standard_normal(n)

    transform1 = rr.affine_transform(X1, b)

    transform1.linear_map(v)
    transform1.adjoint_map(y)
    transform1.affine_map(v)
    
    # should raise a warning about type of sparse matrix

    X1 = scipy.sparse.coo_matrix(np.random.standard_normal((n,p)))
    b = scipy.sparse.coo_matrix(np.random.standard_normal(n))
    v = np.random.standard_normal(p)
    y = np.random.standard_normal(n)

    transform2 = rr.affine_transform(X1, b)

    transform2.linear_map(v)
    transform2.adjoint_map(y)
    transform2.affine_map(v)
    
@set_seed_for_test()
def test_row_matrix():
    # make sure we can input a vector as a transform

    n, p = 20, 1
    x = np.random.standard_normal(n)
    b = np.random.standard_normal(p)
    v = np.random.standard_normal(n)
    y = np.random.standard_normal(p)

    transform1 = rr.linear_transform(x)
    transform2 = rr.affine_transform(x, b)

    transform1.linear_map(v)
    transform1.affine_map(v)
    transform1.adjoint_map(y)

    transform2.linear_map(v)
    transform2.affine_map(v)
    transform2.adjoint_map(y)

@set_seed_for_test()
def test_coefs_matrix():

    n, p, q = 20, 10, 5

    X = np.random.standard_normal((n, p))
    B = np.random.standard_normal((n, q))
    V = np.random.standard_normal((p, q))
    Y = np.random.standard_normal((n, q))

    transform1 = rr.linear_transform(X, input_shape=(p,q))
    assert_equal(transform1.linear_map(V).shape, (n,q))
    assert_equal(transform1.affine_map(V).shape, (n,q))
    assert_equal(transform1.adjoint_map(Y).shape, (p,q))

    transform2 = rr.affine_transform(X, B, input_shape=(p,q))
    assert_equal(transform2.linear_map(V).shape, (n,q))
    assert_equal(transform2.affine_map(V).shape, (n,q))
    assert_equal(transform2.adjoint_map(Y).shape, (p,q))

def test_selector():
    X = np.arange(30).reshape((6,5))
    offset = np.arange(6)
    transform = affine_transform(X, offset)
    apply_to_first5 = selector(slice(0,5), (20,), transform)
    apply_to_first5.affine_map(np.arange(20))
    apply_to_first5.linear_map(np.arange(20))
    apply_to_first5.adjoint_map(np.arange(6))

    just_select = selector(slice(0,5), (20,))
    just_select.affine_map(np.arange(20))
    just_select.linear_map(np.arange(20))
    just_select.adjoint_map(np.arange(5))
    np.testing.assert_allclose(np.arange(5), just_select.linear_map(np.arange(20)))

def test_reshape():
    reshape_ = reshape((30,), (6,5))
    assert_equal(reshape_.linear_map(np.arange(30)).shape, (6,5))
    assert_equal(reshape_.affine_map(np.arange(30)).shape, (6,5))
    assert_equal(reshape_.adjoint_map(np.zeros((6,5))).shape, (30,))

@set_seed_for_test()
def test_stack_product():
    X = np.random.standard_normal((5, 30))
    Y = np.random.standard_normal((5, 30))
    Z = np.random.standard_normal((5, 31))
    U = np.random.standard_normal((6, 30))
    stack = vstack([X, Y])

    assert_raises(ValueError, vstack, [X, Z])
    assert_raises(ValueError, hstack, [X, U])

    np.testing.assert_allclose(stack.linear_map(np.arange(30))[:5], np.dot(X, np.arange(30)))
    np.testing.assert_allclose(stack.linear_map(np.arange(30))[5:], np.dot(Y, np.arange(30)))

    np.testing.assert_allclose(stack.affine_map(np.arange(30))[:5], np.dot(X, np.arange(30)))
    np.testing.assert_allclose(stack.affine_map(np.arange(30))[5:], np.dot(Y, np.arange(30)))

    np.testing.assert_allclose(stack.adjoint_map(np.arange(10)), np.dot(X.T, np.arange(5)) + np.dot(Y.T, np.arange(5, 10)))

    _hstack = hstack([X, Y, Z])
    _hstack.linear_map(np.arange(91))
    _hstack.affine_map(np.arange(91))
    _hstack.adjoint_map(np.arange(5))

    b = np.random.standard_normal(5)
    XA = rr.affine_transform(X, b)
    _product = product([XA,Y])
    np.testing.assert_allclose(_product.linear_map(np.arange(60))[:5], np.dot(X, np.arange(30)))
    np.testing.assert_allclose(_product.linear_map(np.arange(60))[5:], np.dot(Y, np.arange(30, 60)))
    np.testing.assert_allclose(_product.affine_map(np.arange(60))[:5], np.dot(X, np.arange(30)) + b)
    np.testing.assert_allclose(_product.affine_map(np.arange(60))[5:], np.dot(Y, np.arange(30, 60)))

    np.testing.assert_allclose(_product.adjoint_map(np.arange(10))[:30], np.dot(X.T, np.arange(5)))
    np.testing.assert_allclose(_product.adjoint_map(np.arange(10))[30:], np.dot(Y.T, np.arange(5, 10)))

    scale_prod = scalar_multiply(_product, 2)
    np.testing.assert_allclose(scale_prod.linear_map(np.arange(60)), 2 * _product.linear_map(np.arange(60)))
    np.testing.assert_allclose(scale_prod.affine_map(np.arange(60)), 2 * _product.affine_map(np.arange(60)))
    np.testing.assert_allclose(scale_prod.adjoint_map(np.arange(60)), 2 * _product.adjoint_map(np.arange(60)))

@set_seed_for_test()
def test_posneg():

    X = np.random.standard_normal((40, 5))
    X_pn = posneg(X)
    
    V = np.random.standard_normal((2, 5))
    U = np.random.standard_normal(40)
    np.testing.assert_allclose(X_pn.linear_map(V), np.dot(X, V[0] - V[1]))
    np.testing.assert_allclose(X_pn.affine_map(V), np.dot(X, V[0] - V[1]))
    np.testing.assert_allclose(X_pn.adjoint_map(U)[0], np.dot(X.T, U))
    np.testing.assert_allclose(X_pn.adjoint_map(U)[1], -np.dot(X.T, U))

@set_seed_for_test()
def test_misc():
    X = np.random.standard_normal((40, 5))
    power_L(X)
    Xa = rr.astransform(X)
    np.testing.assert_allclose(todense(Xa), X)

    reshapeA = adjoint(reshape((30,), (6,5)))
    assert_raises(NotImplementedError, todense, reshapeA)
