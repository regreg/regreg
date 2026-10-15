import pytest
import numpy as np
import regreg.affine.factored_matrix as FM
from regreg.affine import power_L, todense
from regreg.atoms.projl1_cython import projl1
from regreg.api import identity_quadratic
from regreg.atoms.tests.test_seminorms import all_close

def test_class():

    rng = np.random.default_rng(5000)
    for shape in [(100, 50), (50, 100)]:
        n, p = shape
        X = rng.standard_normal((n,p))
        fm = FM.factored_matrix(X, min_singular=1.e-2, affine_offset=0)
        fm.X
        fm.copy()

        U = rng.standard_normal((p, 10))
        V = rng.standard_normal((n, 20))
        with pytest.raises(ValueError):
            FM.factored_matrix(X, -1)

        fm.linear_map(U)
        fm.affine_map(U)
        fm.adjoint_map(V)

def test_rankone():
    rng = np.random.default_rng(5001)
    x1 = rng.standard_normal(100)
    x2 = rng.standard_normal(50)
    fm = FM.factored_matrix(np.multiply.outer(x1, x2), min_singular=1.e-2, affine_offset=0)
    fm.X
    fm.copy()

    X = rng.standard_normal((100, 50))
    X[:,:7] *= 5

    U = rng.standard_normal((50, 10))
    V = rng.standard_normal((100, 20))
    with pytest.raises(ValueError):
        FM.factored_matrix(X, -1)

    fm.linear_map(U)
    fm.affine_map(U)
    fm.adjoint_map(V)

def test_zero():

    fm = FM.factored_matrix(np.zeros((100, 50)))
    fm.X

def test_partial_svd():
    """
    Rank 10 partial SVD
    """
    rng = np.random.default_rng(5002)

    X = rng.standard_normal((100, 50))
    X[:,:7] *= 5

    rank = 5
    U, D, VT = FM.partial_svd(X, rank=rank, padding=10, stopping_rule=lambda D: False, tol=1.e-12)
    assert np.linalg.norm(np.dot(U.T, U) - np.identity(rank)) < 1.e-4
    assert np.linalg.norm(np.dot(VT, VT.T) - np.identity(rank)) < 1.e-4
    U_np, D_np, VT_np = np.linalg.svd(X, full_matrices=False)
    assert np.linalg.norm(U - np.dot(U, np.dot(U_np.T[:rank], U_np[:,:rank]))) < 1.e-4
    assert np.linalg.norm(VT - np.dot(VT, np.dot(VT_np[:rank].T, VT_np[:rank]))) < 1.e-4

    FM.partial_svd(X, rank=rank, padding=10, stopping_rule=lambda D: False, tol=1.e-12,
                   warm_start=rng.standard_normal(X.shape[0]))

def test_stopping_rule():
    '''
    use a stopping rule in compute_iterative_svd
    '''
    rng = np.random.default_rng(5003)

    def soft_threshold_rule(L):
        return lambda D: np.fabs(D).min() <= L

    L = 30
    svt_rule = soft_threshold_rule(L)

    X = rng.standard_normal((100, 50))
    X[:,:7] *= 5

    U, D, VT = FM.compute_iterative_svd(X, initial_rank=3, stopping_rule=svt_rule, tol=1.e-12, debug=True)

    D2 = (D - L) * (D > L)
    D1 = np.linalg.svd(X)[1]
    D1 = (D1 - L) * (D1 > L)
    rank = (D2 > 0).sum()
    all_close(D1[:rank], D2[:rank], 'stopping_rule', None)

@pytest.mark.skipif(True, reason="Proximal maps for factored_matrix are not fully worked out")
def test_proximal_maps():

    rng = np.random.default_rng(5004)
    X = rng.standard_normal((100, 50))
    X[:,:7] *= 5

    P = FM.nuclear_norm(X.shape, lagrange=1)
    RP = todense(P.lagrange_prox(X))

    B = FM.nuclear_norm(X.shape, bound=1)
    RB = todense(B.bound_prox(X))

    BO = FM.operator_norm(X.shape, bound=1)
    PO = FM.operator_norm(X.shape, lagrange=1)

    RPO = todense(PO.lagrange_prox(X))
    RBO = todense(BO.bound_prox(X))

    D = np.linalg.svd(X, full_matrices=0)[1]
    lD = np.linalg.svd(RP, full_matrices=0)[1]
    lagrange_rank = (lD > 1.e-10).sum()
    all_close(lD[:lagrange_rank] + P.lagrange, D[:lagrange_rank], 'proximal lagrange', None)

    bD = np.linalg.svd(RB, full_matrices=0)[1]
    bound_rank = (bD > 1.e-10).sum()

    all_close(bD[:bound_rank], projl1(D, B.bound)[:bound_rank], 'proximal bound', None)

    assert np.linalg.norm(RPO+RB-X) / np.linalg.norm(X) < 0.01
    assert np.linalg.norm(RBO+RP-X) / np.linalg.norm(X) < 0.01

    # running code to ensure it is tested

    P.conjugate
    P.quadratic = identity_quadratic(1, 0, 0, 0)
    P.conjugate

    BO.conjugate
    BO.quadratic = identity_quadratic(1, 0, 0, 0)
    BO.conjugate

    B.conjugate
    B.quadratic = identity_quadratic(1, 0, 0, 0)
    B.conjugate

    PO.conjugate
    PO.quadratic = identity_quadratic(1, 0, 0, 0)
    PO.conjugate

@pytest.mark.skipif(True, reason="Proximal maps for factored_matrix are not fully worked out")
def test_proximal_method():

    rng = np.random.default_rng(5005)
    X = rng.standard_normal((100, 50))
    X[:,:7] *= 5

    qX = identity_quadratic(1,X,0,0)
    P = FM.nuclear_norm(X.shape, lagrange=1)
    RP = todense(P.proximal(qX))

    B = FM.nuclear_norm(X.shape, bound=1)
    RB = todense(B.proximal(qX))

    BO = FM.operator_norm(X.shape, bound=1)
    PO = FM.operator_norm(X.shape, lagrange=1)

    RPO = todense(PO.proximal(qX))
    RBO = todense(BO.proximal(qX))

    D = np.linalg.svd(X, full_matrices=0)[1]
    lD = np.linalg.svd(RP, full_matrices=0)[1]
    lagrange_rank = (lD > 1.e-10).sum()
    all_close(lD[:lagrange_rank] + P.lagrange, D[:lagrange_rank], 'proximal method lagrange', None)

    bD = np.linalg.svd(RB, full_matrices=0)[1]
    bound_rank = (bD > 1.e-10).sum()

    all_close(bD[:bound_rank], projl1(D, B.bound)[:bound_rank], 'proximal method bound', None)

    assert np.linalg.norm(RPO+RB-X) / np.linalg.norm(X) < 0.01
    assert np.linalg.norm(RBO+RP-X) / np.linalg.norm(X) < 0.01

def test_max_rank():

    rng = np.random.default_rng(5006)
    X = rng.standard_normal((100, 200))
    with pytest.raises(ValueError):
        FM.compute_iterative_svd(X, max_rank=2)

    U, D, VT = FM.compute_iterative_svd(X)
    assert np.linalg.norm(np.dot(U.T, U) - np.identity(100)) < 1.e-6
    assert np.linalg.norm(np.dot(VT, VT.T) - np.identity(100)) < 1.e-6

    U, D, VT = FM.compute_iterative_svd(X, max_rank=200)
    assert np.linalg.norm(np.dot(U.T, U) - np.identity(100)) < 1.e-6
    assert np.linalg.norm(np.dot(VT, VT.T) - np.identity(100)) < 1.e-6


def test_warm_start_shape_error():
    # gh-68: the error message had one %s and two arguments
    from regreg.affine.factored_matrix import nuclear_norm as fm_nuclear_norm
    with pytest.raises(ValueError, match=r'\(5, 2\)'):
        fm_nuclear_norm((5, 4), lagrange=1., initial_rank=2,
                        warm_start=np.zeros((3, 3)))
