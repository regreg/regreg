import numpy as np

from ..block_maps import (block_columns, 
                          affine_transform,
                          block_rows)

def test_block_columns():

    X1 = np.random.standard_normal((20,10))
    X2 = np.random.standard_normal((23,10))

    B = block_columns([X1, X2])

    Z = np.random.standard_normal(B.input_shape)
    BL = B.linear_map(Z)
    BA = B.affine_map(Z)

    np.testing.assert_allclose(BL[:20], X1.dot(Z[:,0]))
    np.testing.assert_allclose(BL[20:], X2.dot(Z[:,1]))
    np.testing.assert_allclose(BA, BL)

    V = np.random.standard_normal(43)
    BTV = B.adjoint_map(V)

    np.testing.assert_allclose(np.column_stack([V[:20].dot(X1), V[20:].dot(X2)]), BTV)

def test_block_columns_affine():

    X1 = np.random.standard_normal((20,10))
    X2 = np.random.standard_normal((23,10))
    A1 = np.random.standard_normal(20)
    A2 = np.random.standard_normal(23)

    T1 = affine_transform(X1, A1)
    T2 = affine_transform(X2, A2)

    B = block_columns([T1, T2])

    Z = np.random.standard_normal(B.input_shape)
    BL = B.linear_map(Z)
    BA = B.affine_map(Z)

    np.testing.assert_allclose(BL[:20], X1.dot(Z[:,0]))
    np.testing.assert_allclose(BL[20:], X2.dot(Z[:,1]))
    #np.testing.assert_allclose(BA[:20], X1.dot(Z[:,0]) + A1)
    #np.testing.assert_allclose(BA[20:], X2.dot(Z[:,1]) + A2)
    np.testing.assert_allclose(BL + np.hstack([A1,A2]), BA)

    V = np.random.standard_normal(43)
    BTV = B.adjoint_map(V)

    np.testing.assert_allclose(np.column_stack([V[:20].dot(X1), V[20:].dot(X2)]), BTV)

def test_block_rows():

    X1 = np.random.standard_normal((20,10)).T
    X2 = np.random.standard_normal((23,10)).T

    B = block_rows([X1, X2])

    Z = np.random.standard_normal(B.input_shape)
    BL = B.linear_map(Z)
    BA = B.affine_map(Z)

    np.testing.assert_allclose(BL[:,0], X1.dot(Z[:20]))
    np.testing.assert_allclose(BL[:,1], X2.dot(Z[20:]))
    np.testing.assert_allclose(BA, BL)

    V = np.random.standard_normal(B.output_shape)
    BTV = B.adjoint_map(V)

    np.testing.assert_allclose(np.hstack([X1.T.dot(V[:,0]), X2.T.dot(V[:,1])]), BTV)

def test_block_rows_affine():

    X1 = np.random.standard_normal((20,10)).T
    X2 = np.random.standard_normal((23,10)).T
    A1 = np.random.standard_normal(10)
    A2 = np.random.standard_normal(10)

    T1 = affine_transform(X1, A1)
    T2 = affine_transform(X2, A2)

    B = block_rows([T1, T2])

    Z = np.random.standard_normal(B.input_shape)
    BL = B.linear_map(Z)
    BA = B.affine_map(Z)

    np.testing.assert_allclose(BL[:,0], X1.dot(Z[:20]))
    np.testing.assert_allclose(BL[:,1], X2.dot(Z[20:]))
    np.testing.assert_allclose(BA, BL + np.array([A1, A2]).T)

    V = np.random.standard_normal(B.output_shape)
    BTV = B.adjoint_map(V)

    np.testing.assert_allclose(np.hstack([X1.T.dot(V[:,0]), X2.T.dot(V[:,1])]), BTV)

