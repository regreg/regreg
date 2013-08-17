import nose.tools as nt
import numpy as np
import regreg.affine.factored_matrix as FM
from regreg.affine import power_L, todense
from regreg.atoms.projl1_cython import projl1
from regreg.api import identity_quadratic
from atoms.test_seminorms import all_close

X = np.random.standard_normal((100, 50))
X[:,:7] *= 5

def test_partial_svd():
    """
    Rank 10 parital SVD
    """
    rank = 5
    U, D, VT = FM.partial_svd(X, rank=rank, padding=10, stopping_rule=lambda D: False, tol=1.e-12)
    nt.assert_true(np.linalg.norm(np.dot(U.T, U) - np.identity(rank)) < 1.e-4)
    nt.assert_true(np.linalg.norm(np.dot(VT, VT.T) - np.identity(rank)) < 1.e-4)
    U_np, D_np, VT_np = np.linalg.svd(X, full_matrices=False)
    nt.assert_true(np.linalg.norm(U - np.dot(U, np.dot(U_np.T[:rank], U_np[:,:rank]))) < 1.e-4)
    nt.assert_true(np.linalg.norm(VT - np.dot(VT, np.dot(VT_np[:rank].T, VT_np[:rank]))) < 1.e-4)

def test_stopping_rule():
    '''
    use a stopping rule in compute_iterative_svd
    '''

    def soft_threshold_rule(L):
        return lambda D: np.fabs(D).min() <= L

    L = 30
    svt_rule = soft_threshold_rule(L)

    U, D, VT = FM.compute_iterative_svd(X, initial_rank=3, stopping_rule=svt_rule, tol=1.e-12)

    D2 = (D - L) * (D > L)
    D1 = np.linalg.svd(X)[1]
    D1 = (D1 - L) * (D1 > L)
    rank = (D2 > 0).sum()
    all_close(D1[:rank], D2[:rank], 'stopping_rule', None)

def test_proximal_maps():

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

    nt.assert_true(np.linalg.norm(RPO+RB-X) / np.linalg.norm(X) < 0.01)
    nt.assert_true(np.linalg.norm(RBO+RP-X) / np.linalg.norm(X) < 0.01)

def test_proximal_method():

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

    nt.assert_true(np.linalg.norm(RPO+RB-X) / np.linalg.norm(X) < 0.01)
    nt.assert_true(np.linalg.norm(RBO+RP-X) / np.linalg.norm(X) < 0.01)
