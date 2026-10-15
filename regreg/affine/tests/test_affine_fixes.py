"""
Regression tests for the affine-transform bugs in gh-108 (and the
selector aliasing in gh-104).
"""
import numpy as np
import pytest
from scipy import sparse

import regreg.api as rr
from regreg.affine import (affine_transform, linear_transform, scalar_multiply,
                           residual, reshape, scaler, add_intercept, vstack,
                           product, todense)
from regreg.affine.block_maps import block_columns
from regreg.affine.fused_lasso import trend_filter_inverse
from regreg.affine.factored_matrix import (nuclear_norm as fm_nuclear_norm,
                                           operator_norm as fm_operator_norm)

def _check_adjoint(T, rng, in_shape=None, out_shape=None):
    x = rng.standard_normal(in_shape or T.input_shape)
    u = rng.standard_normal(out_shape or T.output_shape)
    np.testing.assert_allclose(np.sum(T.linear_map(x) * u),
                               np.sum(x * T.adjoint_map(u)))

def test_affine_atom_offsets():
    # affine_atom trusted the affine_offset attribute, which was None for
    # scalar_multiply and residual and left out nested offsets
    rng = np.random.default_rng(0)
    X, b, c = rng.standard_normal((4, 3)), rng.standard_normal(4), rng.standard_normal(4)
    x, x4 = rng.standard_normal(3), rng.standard_normal(4)
    A = affine_transform(rng.standard_normal((4, 4)), b)
    for T, arg in [(scalar_multiply(affine_transform(X, b), 2.), x),
                   (residual(A), x4),
                   (affine_transform(affine_transform(X, b), c), x)]:
        atom = rr.linear_atom(rr.l1norm(4, lagrange=1.), T)
        np.testing.assert_allclose(atom.nonsmooth_objective(arg), np.fabs(T.affine_map(arg)).sum())
    # l1norm.affine with an affine transform kept only the outer offset
    np.testing.assert_allclose(rr.l1norm.affine(affine_transform(X, b), c, lagrange=1.).nonsmooth_objective(x),
                               np.fabs(X @ x + b + c).sum())
    # and the attributes the solvers read
    np.testing.assert_allclose(scalar_multiply(affine_transform(X, b), 2.).affine_offset, 2 * b)
    np.testing.assert_allclose(residual(A).affine_offset, -b)

def test_transforms_without_affine_offset():
    # scaler, reshape, add_intercept and block_columns never set affine_offset
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4)
    np.testing.assert_allclose(rr.l1norm.linear(scaler(np.array([1., 2, 3, 4])), lagrange=1.).nonsmooth_objective(x),
                               np.fabs(x * [1, 2, 3, 4]).sum())
    np.testing.assert_allclose(rr.linear_atom(rr.l1norm((2, 2), lagrange=1.), reshape((4,), (2, 2))).nonsmooth_objective(x),
                               np.fabs(x).sum())
    X = rng.standard_normal((5, 3))
    xi = rng.standard_normal(4)
    np.testing.assert_allclose(rr.linear_atom(rr.l1norm(5, lagrange=1.), add_intercept(X)).nonsmooth_objective(xi),
                               np.fabs(X @ xi[1:] + xi[0]).sum())
    B = block_columns([X, X])
    bx = rng.standard_normal((3, 2))
    np.testing.assert_allclose(rr.linear_atom(rr.l1norm(B.output_shape, lagrange=1.), B).nonsmooth_objective(bx),
                               np.fabs(B.affine_map(bx)).sum())

def test_selector_adjoint():
    rng = np.random.default_rng(2)
    # repeated indices were overwritten rather than summed
    S = rr.selector(np.array([0, 0, 1]), (3,))
    np.testing.assert_allclose(S.adjoint_map(np.array([1., 2, 3])), [3, 3, 0])
    for idx, shape, rs in [(slice(0, 3), (6,), None),
                           (np.array([0, 0, 1, 4]), (6,), None),
                           (slice(0, 6), (9,), (2, 3)),       # reshape failed
                           (np.array([True, False, True, True, False, False]), (6,), None),
                           ((slice(0, 2), slice(1, 3)), (3, 4), None)]:
        _check_adjoint(rr.selector(idx, shape, reshape=rs), rng)
    # gh-104: the adjoint returned its internal buffer
    S = rr.selector(slice(0, 3), (6,))
    a, b = S.adjoint_map(np.ones(3)), S.adjoint_map(2 * np.ones(3))
    np.testing.assert_allclose(a[:3], 1)
    # with reshape, the caller's array was reshaped in place
    S = rr.selector(slice(0, 9), (9,), reshape=(3, 3))
    x = np.arange(9.)
    S.linear_map(x)
    assert x.shape == (9,)

def test_trend_filter_inverse_2d_adjoint():
    rng = np.random.default_rng(3)
    T = trend_filter_inverse(np.arange(6.))
    V = rng.standard_normal((6, 3))
    np.testing.assert_allclose(T.adjoint_map(V),
                               np.column_stack([T.adjoint_map(V[:, j]) for j in range(3)]))
    _check_adjoint(T, rng, in_shape=(5, 3), out_shape=(6, 3))

@pytest.mark.parametrize('conv', [sparse.csr_matrix, sparse.csr_array, sparse.csc_array])
def test_sparse_arrays(conv):
    # scipy sparse arrays failed: isspmatrix is False for them, and `*` is
    # elementwise
    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, 4))
    X[X < 0] = 0
    y = rng.standard_normal(20)
    b = rr.simple_problem(rr.squared_error(conv(X), y), rr.l1norm(4, lagrange=1.)).solve(tol=1e-12)
    bd = rr.simple_problem(rr.squared_error(X, y), rr.l1norm(4, lagrange=1.)).solve(tol=1e-12)
    np.testing.assert_allclose(b, bd, atol=1e-10)
    for center, scale in [(True, True), (False, True), (True, False)]:
        # center=False, scale=True raised UnboundLocalError for sparse input
        N, Nd = rr.normalize(conv(X), center=center, scale=scale), rr.normalize(X, center=center, scale=scale)
        v, u = rng.standard_normal(4), rng.standard_normal(20)
        np.testing.assert_allclose(N.linear_map(v), Nd.linear_map(v), atol=1e-12)
        np.testing.assert_allclose(N.adjoint_map(u), Nd.adjoint_map(u), atol=1e-12)

def test_vstack_product_2d_outputs():
    rng = np.random.default_rng(5)
    R = reshape((6,), (2, 3))
    _check_adjoint(vstack([R, R]), rng)
    _check_adjoint(product([R, linear_transform(rng.standard_normal((2, 3)))]), rng)

@pytest.mark.parametrize('shape', [(10, 6), (6, 10), (12, 30), (30, 12)])
def test_factored_matrix_shapes(shape):
    # the default warm start failed when min(shape) < initial_rank, and
    # operator_norm failed for every wide shape
    A = np.random.default_rng(6).standard_normal(shape)
    for klass, ref in [(fm_nuclear_norm, rr.nuclear_norm), (fm_operator_norm, rr.operator_norm)]:
        out = todense(klass(shape, lagrange=1.).lagrange_prox(A))
        np.testing.assert_allclose(out, ref(shape, lagrange=1.).lagrange_prox(A), atol=2e-2)
