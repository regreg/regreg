""" Testing D transform implementation
"""

import pytest
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

import regreg.api as rr

def test_broad_first():
    # Test broadcasting over second axis
    a = np.arange(4) + 10
    b = np.arange(4).reshape(4,1)
    c = broadcast_first(a, b, add)
    res = a[:,None] + b
    assert res.shape == c.shape
    assert_array_almost_equal(c, res)
    res1d = res.ravel()
    c = broadcast_first(b, a, add)
    assert res1d.shape == c.shape
    assert_array_almost_equal(c, res1d)
    c = broadcast_first(a, b.ravel(), add)
    assert res1d.shape == c.shape
    assert_array_almost_equal(c, res1d)

def test_dot():

    rng = np.random.default_rng(4800)
    A = rng.standard_normal((20, 12))
    B = rng.standard_normal((10, 20))

    C = rr.astransform(B).dot(rr.astransform(A))

    assert C.shape == (10, 12)
    Z = rng.standard_normal(12)
    assert_array_almost_equal(C.dot(Z), B.dot(A.dot(Z)))

    assert C.ndim == 2

def test_affine_transform():
    # Test affine transform

    m = 20
    x1d = np.arange(m)
    x2d = x1d[:,None]
    x22d = np.c_[x2d, x2d]
    # Error if both of linear and affine components are None
    with pytest.raises(AffineError):
        affine_transform(None, None)
    with pytest.raises(AffineError):
        linear_transform(None)
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

def test_composition():
    rng = np.random.default_rng(4801)
    X1 = rng.standard_normal((20,30))
    X2 = rng.standard_normal((30,10))
    b1 = rng.standard_normal(20)
    b2 = rng.standard_normal(30)
    L1 = affine_transform(X1, b1)
    L2 = affine_transform(X2, b2)

    z = rng.standard_normal(10)
    w = rng.standard_normal(20)
    comp = composition(L1,L2)

    assert_array_almost_equal(comp.linear_map(z), np.dot(X1, np.dot(X2, z)))
    assert_array_almost_equal(comp.adjoint_map(w), np.dot(X2.T, np.dot(X1.T, w)))
    assert_array_almost_equal(comp.affine_map(z), np.dot(X1, np.dot(X2, z)+b2)+b1)

def test_composition2():
    rng = np.random.default_rng(4802)
    X1 = rng.standard_normal((20,30))
    X2 = rng.standard_normal((30,10))
    X3 = rng.standard_normal((10,20))

    b1 = rng.standard_normal(20)
    b2 = rng.standard_normal(30)
    b3 = rng.standard_normal(10)

    L1 = affine_transform(X1, b1)
    L2 = affine_transform(X2, b2)
    L3 = affine_transform(X3, b3)

    z = rng.standard_normal(20)
    w = rng.standard_normal(20)
    comp = composition(L1,L2,L3)

    assert_array_almost_equal(comp.linear_map(z), 
                       np.dot(X1, np.dot(X2, np.dot(X3, z))))
    assert_array_almost_equal(comp.adjoint_map(w), 
                       np.dot(X3.T, np.dot(X2.T, np.dot(X1.T, w))))
    assert_array_almost_equal(
        comp.affine_map(z),
        np.dot(X1, np.dot(X2, np.dot(X3, z) + b3) + b2) + b1)

def test_adjoint():
    rng = np.random.default_rng(4803)
    X = rng.standard_normal((20,30))
    b = rng.standard_normal(20)
    L = affine_transform(X, b)

    z = rng.standard_normal(30)
    w = rng.standard_normal(20)
    A = adjoint(L)

    assert_array_almost_equal(A.linear_map(w), L.adjoint_map(w))
    assert_array_almost_equal(A.affine_map(w), L.adjoint_map(w))
    assert_array_almost_equal(A.adjoint_map(z), L.linear_map(z))

def test_affine_sum():

    rng = np.random.default_rng(4804)
    n = 100
    p = 25

    X1 = rng.standard_normal((n,p))
    X2 = rng.standard_normal((n,p))
    b = rng.standard_normal(n)
    v = rng.standard_normal(p)

    transform1 = rr.affine_transform(X1, b)
    transform2 = rr.linear_transform(X2)
    sum_transform = rr.affine_sum([transform1, transform2])

    assert_array_almost_equal(np.dot(X1,v) + np.dot(X2,v) + b, sum_transform.affine_map(v))
    assert_array_almost_equal(np.dot(X1,v) + np.dot(X2,v), sum_transform.linear_map(v))
    assert_array_almost_equal(np.dot(X1.T,b) + np.dot(X2.T,b), sum_transform.adjoint_map(b))
    assert_array_almost_equal(b, sum_transform.affine_offset)

    sum_transform = rr.affine_sum([transform1, transform2], weights=[3,4])

    assert_array_almost_equal(3*(np.dot(X1,v) + b) + 4*(np.dot(X2,v)), sum_transform.affine_map(v))
    assert_array_almost_equal(3*np.dot(X1,v) + 4*np.dot(X2,v), sum_transform.linear_map(v))
    assert_array_almost_equal(3*np.dot(X1.T,b) + 4*np.dot(X2.T,b), sum_transform.adjoint_map(b))
    assert_array_almost_equal(3*b, sum_transform.affine_offset)

def test_affine_sparse():
    rng = np.random.default_rng(4805)
    # test using sparse matrices for affine transforms

    n = 100
    p = 25

    X1 = scipy.sparse.csr_matrix(rng.standard_normal((n,p)))
    b = scipy.sparse.csr_matrix(rng.standard_normal(n))
    v = rng.standard_normal(p)
    y = rng.standard_normal(n)

    transform1 = rr.affine_transform(X1, b)

    transform1.linear_map(v)
    transform1.adjoint_map(y)
    transform1.affine_map(v)
    
    # should raise a warning about type of sparse matrix

    X1 = scipy.sparse.coo_matrix(rng.standard_normal((n,p)))
    b = scipy.sparse.coo_matrix(rng.standard_normal(n))
    v = rng.standard_normal(p)
    y = rng.standard_normal(n)

    transform2 = rr.affine_transform(X1, b)

    transform2.linear_map(v)
    transform2.adjoint_map(y)
    transform2.affine_map(v)
    
def test_row_matrix():
    rng = np.random.default_rng(4806)
    # make sure we can input a vector as a transform

    n, p = 20, 1
    x = rng.standard_normal(n)
    b = rng.standard_normal(p)
    v = rng.standard_normal(n)
    y = rng.standard_normal(p)

    transform1 = rr.linear_transform(x)
    transform2 = rr.affine_transform(x, b)

    transform1.linear_map(v)
    transform1.affine_map(v)
    transform1.adjoint_map(y)

    transform2.linear_map(v)
    transform2.affine_map(v)
    transform2.adjoint_map(y)

def test_coefs_matrix():

    rng = np.random.default_rng(4807)
    n, p, q = 20, 10, 5

    X = rng.standard_normal((n, p))
    B = rng.standard_normal((n, q))
    V = rng.standard_normal((p, q))
    Y = rng.standard_normal((n, q))

    transform1 = rr.linear_transform(X, input_shape=(p,q))
    assert transform1.linear_map(V).shape == (n,q)
    assert transform1.affine_map(V).shape == (n,q)
    assert transform1.adjoint_map(Y).shape == (p,q)

    transform2 = rr.affine_transform(X, B, input_shape=(p,q))
    assert transform2.linear_map(V).shape == (n,q)
    assert transform2.affine_map(V).shape == (n,q)
    assert transform2.adjoint_map(Y).shape == (p,q)

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
    assert reshape_.linear_map(np.arange(30)).shape == (6,5)
    assert reshape_.affine_map(np.arange(30)).shape == (6,5)
    assert reshape_.adjoint_map(np.zeros((6,5))).shape == (30,)

def test_stack_product():
    rng = np.random.default_rng(4808)
    X = rng.standard_normal((5, 30))
    Y = rng.standard_normal((5, 30))
    Z = rng.standard_normal((5, 31))
    U = rng.standard_normal((6, 30))
    stack = vstack([X, Y])

    with pytest.raises(ValueError):
        vstack([X, Z])
    with pytest.raises(ValueError):
        hstack([X, U])

    np.testing.assert_allclose(stack.linear_map(np.arange(30))[:5], np.dot(X, np.arange(30)))
    np.testing.assert_allclose(stack.linear_map(np.arange(30))[5:], np.dot(Y, np.arange(30)))

    np.testing.assert_allclose(stack.affine_map(np.arange(30))[:5], np.dot(X, np.arange(30)))
    np.testing.assert_allclose(stack.affine_map(np.arange(30))[5:], np.dot(Y, np.arange(30)))

    np.testing.assert_allclose(stack.adjoint_map(np.arange(10)), np.dot(X.T, np.arange(5)) + np.dot(Y.T, np.arange(5, 10)))

    _hstack = hstack([X, Y, Z])
    _hstack.linear_map(np.arange(91))
    _hstack.affine_map(np.arange(91))
    _hstack.adjoint_map(np.arange(5))

    b = rng.standard_normal(5)
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

def test_posneg():

    rng = np.random.default_rng(4809)
    X = rng.standard_normal((40, 5))
    X_pn = posneg(X)
    
    V = rng.standard_normal((2, 5))
    U = rng.standard_normal(40)
    np.testing.assert_allclose(X_pn.linear_map(V), np.dot(X, V[0] - V[1]))
    np.testing.assert_allclose(X_pn.affine_map(V), np.dot(X, V[0] - V[1]))
    np.testing.assert_allclose(X_pn.adjoint_map(U)[0], np.dot(X.T, U))
    np.testing.assert_allclose(X_pn.adjoint_map(U)[1], -np.dot(X.T, U))

def test_misc():
    rng = np.random.default_rng(4810)
    X = rng.standard_normal((40, 5))
    power_L(X)
    Xa = rr.astransform(X)
    np.testing.assert_allclose(todense(Xa), X)

    reshapeA = adjoint(reshape((30,), (6,5)))
    with pytest.raises(NotImplementedError):
        todense(reshapeA)

def test_aslinear_shapes():
    # gh-85 / gh-68: aslinear swapped input_shape and output_shape

    from regreg.affine import aslinear
    D = np.diff(np.identity(5), axis=0)    # 4 x 5
    L = aslinear(rr.astransform(D))
    assert L.input_shape == (5,)
    assert L.output_shape == (4,)
    x, u = np.arange(5.), np.arange(4.)
    np.testing.assert_allclose(L.linear_map(x), D @ x)
    np.testing.assert_allclose(L.adjoint_map(u), D.T @ u)

def test_composition_offset():
    # gh-68: the offset test in composition was inverted
    from regreg.affine import composition as comp
    c = comp(affine_transform(np.eye(3), np.ones(3)), np.eye(3))
    np.testing.assert_allclose(c.affine_offset, np.ones(3))
    assert comp(np.eye(3), np.eye(3)).affine_offset is None

def test_affine_atom_offset():
    # gh-68: affine_atom dropped the offset of a composition, failed for
    # a composition without one, and (not in the issue) added the
    # transform's offset to the atom's center rather than subtracting it
    import regreg.api as rr
    from regreg.atoms import affine_atom
    from regreg.affine import composition as comp

    rng = np.random.default_rng(0)
    D = rng.standard_normal((3, 4))
    o = rng.standard_normal(3)
    c = rng.standard_normal(3)
    x = rng.standard_normal(4)
    expected = np.fabs(D.dot(x) + o - c).sum()
    for T in [affine_transform(D, o),
              comp(affine_transform(np.eye(3), o), D)]:
        atom = rr.l1norm(3, lagrange=1., offset=c.copy())
        A = affine_atom(atom, T)
        np.testing.assert_allclose(A.nonsmooth_objective(x), expected)
        # the caller's atom is unchanged
        np.testing.assert_allclose(atom.offset, c)
        repr(A)

    A = affine_atom(rr.l1norm(3, lagrange=1.), comp(np.eye(3), D))
    np.testing.assert_allclose(A.nonsmooth_objective(x), np.fabs(D.dot(x)).sum())

def test_product_output_dtype():
    # gh-68: output_group_dtype used the input shapes
    from regreg.affine import product
    P = product([np.ones((2, 5)), np.ones((3, 4))])
    assert [P.output_group_dtype[n].shape for n in P.output_groups] == [(2,), (3,)]
    assert [P.input_group_dtype[n].shape for n in P.input_groups] == [(5,), (4,)]
