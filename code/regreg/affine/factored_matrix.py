"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
import numpy as np
import warnings
from ..affine import (linear_transform, composition, affine_sum, 
                      power_L, astransform, adjoint)

class factored_matrix(object):

    """
    A class for storing the SVD of a linear_tranform. 
    Has affine_transform attributes like linear_map.
    """

    def __init__(self,
                 linear_operator,
                 min_singular=0.,
                 tol=1e-5,
                 initial_rank=None,
                 initial = None,
                 affine_offset=None,
                 debug=False):

        self.affine_offset = affine_offset
        self.tol = tol
        self.initial_rank = initial_rank
        self.initial = initial
        self.debug = debug

        if min_singular >= 0:
            self.min_singular = min_singular
        else:
            raise ValueError("Minimum singular value must be non-negative")
        
        if type(linear_operator) in type([],()) and len(linear_operator) == 3:
            self.SVD = linear_operator
        else:
            self.X = linear_operator

    def copy(self):
        return factored_matrix([self.SVD[0].copy(), self.SVD[1].copy(), self.SVD[2].copy()])

    def _setX(self,transform):
        transform = astransform(transform)
        self.input_shape = transform.input_shape
        self.output_shape = transform.output_shape
        U, D, VT = compute_iterative_svd(transform, min_singular=self.min_singular, tol=self.tol, initial_rank = self.initial_rank, initial=self.initial, debug=self.debug)
        self.SVD = [U,D,VT]

    def _getX(self):
        if not self.rankone:
            return np.dot(self.SVD[0], np.dot(np.diag(self.SVD[1]), self.SVD[2]))
        else:
            return self.SVD[1][0,0] * np.dot(self.SVD[0], self.SVD[2])
    X = property(_getX, _setX)

    def _getSVD(self):
        return self._SVD
    def _setSVD(self, SVD):
        self.rankone = False
        if len(SVD[1].flatten()) == 1:
            SVD[0] = SVD[0].reshape((SVD[0].flatten().shape[0],1))
            SVD[1] = SVD[1].reshape((1,1))
            SVD[2] = SVD[2].reshape((1,SVD[2].flatten().shape[0]))
            self.rankone = True
        self.input_shape = (SVD[2].shape[1],)
        self.output_shape = (SVD[0].shape[0],)
        self._SVD = SVD
    SVD = property(_getSVD, _setSVD)

    def linear_map(self, x):
        if self.rankone:
            return self.SVD[1][0,0] * np.dot(self.SVD[0], np.dot(self.SVD[2], x))
        else:
            return np.dot(self.SVD[0], np.dot(np.diag(self.SVD[1]), np.dot(self.SVD[2], x)))

    def adjoint_map(self, x):
        if self.rankone:
            return self.SVD[1][0,0] * np.dot(self.SVD[2].T, np.dot(self.SVD[0].T, x))
        else:
            return np.dot(self.SVD[2].T, np.dot(np.diag(self.SVD[1]), np.dot(self.SVD[0].T, x)))

    def affine_map(self,x):
        if self.affine_offset is None:
            return self.offset_map(self.linear_map(x))
        else:
            return self.offset_map(self.linear_map(x))

    def offset_map(self, x, copy=False):
        if self.affine_offset is not None:
            return x + self.affine_offset
        else:
            return x

def compute_iterative_svd(transform,
                          initial_rank = None,
                          initial = None,
                          min_singular = 1e-16,
                          tol = 1e-5,
                          debug=False):

    """
    Compute the SVD of a matrix using partial_svd. If no initial
    rank is given, it assumes a rank of size min(n,p) / 10.

    Iteratively calls partial_svd until the singular_values are small enough.

    Parameters
    ----------

    transform : [linear_transform, ndarray]
        Linear_transform whose SVD is computed. If an
        ndarray, it is first cast with :func:`astransform()`

    initial_rank : None or int, optional
        A guess at the rank of the matrix.

    initial : np.ndarray(np.float), optional
        A guess at the left singular vectors of the matrix.

    min_singular : np.float, optional 
        Stop when the singular value has this relative tolerance.    
    
    tol: np.float, optional
        Tolerance at which the norm of the singular values are deemed
        to have converged.

    debug: bool, optional
        Print debugging statements.

    Returns
    -------

    U, D, VT : np.ndarray(np.float)
        An SVD of the transform.

    >>> np.random.seed(0)
    >>> X = np.random.standard_normal((100, 200))
    >>> U, D, VT = compute_iterative_svd(X)
    >>> np.linalg.norm(np.dot(U.T, U) - np.identity(100)) < 1.e-6
    True
    >>> np.linalg.norm(np.dot(VT, VT.T) - np.identity(100)) < 1.e-6
    True

    """

    if isinstance(transform, np.ndarray):
        transform = linear_transform(transform)

    n = transform.output_shape[0]
    p = transform.input_shape[0]
    
    need_to_transpose = False
    if n < p:
        transform = adjoint(transform)
        need_to_transpose = True

    if initial_rank is None:
        rank = np.round(np.min([n,p]) * 0.1) + 1
    else:
        rank = np.max([initial_rank,1])

    min_so_far = 1.
    D = [np.inf]
    while D[-1] >= min_singular * np.max(D):
        if debug:
            print "Trying rank", rank
        U, D, VT = partial_svd(transform, rank=rank, extra_rank=5, tol=tol, initial=initial, return_full=True, debug=debug)
        if D[0] < min_singular:
            return U[:,0], np.zeros((1,1)), VT[0,:]
        if len(D) < rank:
            break
        initial = 1. * U 
        rank *= 2

    ind = np.where(D >= min_singular)[0]
    if not need_to_transpose:
        return U[:,ind], D[ind],  VT[ind,:]
    else:
        return VT[ind,:].T, D[ind],  U[:,ind].T

def partial_svd(transform,
                rank=1,
                extra_rank=2,
                max_its = 1000,
                tol = 1e-8,
                initial=None,
                return_full = False,
                debug=False):

    """
    Compute the partial SVD of the linear_transform X using the Mazumder/Hastie 
    algorithm in (TODO: CITE)

    Parameters
    ----------

    transform : [linear_transform, ndarray]
        Linear_transform whose SVD is computed. If an
        ndarray, it is first cast with :func:`astransform()`

    rank : int, optional
        What rank of SVD to compute.

    padding : int, optional
        Compute a few further singular values / vectors. This results
        in a better estimator of the rank of interest.

    max_its : int, optional
        How many iterations of the full cycle to complete.

    tol : np.float, optional
        Tolerance at which the norm of the singular values are deemed
        to have converged.

    initial : np.ndarray(np.float), optional
        A guess at the left singular vectors of the matrix.

    return_full: bool, optional
        Return a singular values / vectors from padding?
    
    debug: bool, optional
        Print debugging statements.

    Returns
    -------

    U, D, VT : np.ndarray(np.float)
        An SVD up to `rank` of the transform.

    >>> np.random.seed(0)
    >>> X = np.random.standard_normal((100, 200))
    >>> U, D, VT = partial_svd(X, rank=10)
    >>> np.linalg.norm(np.dot(U.T, U) - np.identity(10)) < 1.e-4
    True
    >>> np.linalg.norm(np.dot(VT, VT.T) - np.identity(10)) < 1.e-4
    True
    >>> U_np, D_np, VT_np = np.linalg.svd(X, full_matrices=False)
    >>> np.linalg.norm(U - np.dot(U, np.dot(U_np.T[:10], U_np[:,:10]))) < 1.e-4
    True
    >>> np.linalg.norm(VT - np.dot(VT, np.dot(VT_np[:10].T, VT_np[:10]))) < 1.e-4
    True
    >>> 

    """

    if isinstance(transform, np.ndarray):
        transform = linear_transform(transform)

    n = np.product(transform.output_shape)
    p = np.product(transform.input_shape)

    rank = np.int(np.min([rank,p]))
    q = np.min([rank + padding, p])
    if initial is not None:
        if initial.shape == (n,q):
            U = initial
        elif len(initial.shape) == 1:
            U = np.hstack([initial.reshape((initial.shape[0],1)), np.random.standard_normal((n,q-1))])            
        else:
            U = np.hstack([initial, np.random.standard_normal((n,q-initial.shape[1]))])            
    else:
        U = np.random.standard_normal((n,q))

    if return_full:
        ind = np.arange(q)
    else:
        ind = np.arange(rank)
    old_singular_values = np.zeros(rank)
    change_ind = np.arange(rank)

    itercount = 0
    singular_rel_change = 1.

    while itercount < max_its and singular_rel_change > tol:
        if debug and itercount > 0:
            print itercount, singular_rel_change, np.sum(np.fabs(singular_values)>1e-12), np.fabs(singular_values[range(np.min([5,len(singular_values)]))])
        V, _ = np.linalg.qr(transform.adjoint_map(U))
        X_V = transform.linear_map(V)
        U, R = np.linalg.qr(X_V)
        singular_values = np.diagonal(R)[change_ind]
        singular_rel_change = np.linalg.norm(singular_values - old_singular_values)/np.linalg.norm(singular_values)
        old_singular_values[:] = singular_values
        itercount += 1
    singular_values = np.diagonal(R)[ind]

    nonzero = np.where(np.fabs(singular_values) > 1e-12)[0]
    if len(nonzero):
        return U[:,ind[nonzero]] * np.sign(singular_values[nonzero]), np.fabs(singular_values[nonzero]),  V[:,ind[nonzero]].T
    else:
        return U[:,ind[0]], np.zeros((1,1)),  V[:,ind[0]].T

def soft_threshold_svd(X, c=0.):

    """
    Soft-treshold the singular values of a matrix X
    """
    if not isinstance(X, factored_matrix):
        X = factored_matrix(X)

    singular_values = X.SVD[1]
    ind = np.where(singular_values >= c)[0]
    if len(ind) == 0:
        X.SVD = [np.zeros(X.output_shape[0]), np.zeros(1), np.zeros(X.input_shape[0])]
    else:
        X.SVD = [X.SVD[0][:,ind], np.maximum(singular_values[ind] - c,0), X.SVD[2][ind,:]]

    return X


