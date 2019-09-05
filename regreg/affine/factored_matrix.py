"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
from __future__ import print_function, division, absolute_import

import numpy as np

from ..affine import affine_sum, astransform, adjoint
from ..atoms.svd_norms import (svd_atom, nuclear_norm as nuclear_norm_atom,
                               operator_norm as operator_norm_atom)
from ..atoms.seminorms import l1norm, _work_out_conjugate
from ..problems.composite import smooth_conjugate

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import doc_template_user

@objective_doc_templater()
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
        
        if type(linear_operator) in [type([]),type(())] and len(linear_operator) == 3:
            U, D, VT = linear_operator
            self.factors = U, D, VT
        else:
            self.X = astransform(linear_operator)

    def copy(self):
        return factored_matrix([self.factors[0].copy(), self.factors[1].copy(), self.factors[2].copy()])

    def _setX(self,transform):
        transform = astransform(transform)
        self.input_shape = transform.input_shape
        self.output_shape = transform.output_shape
        U, D, VT = compute_iterative_svd(transform, 
                                         min_singular=self.min_singular, 
                                         tol=self.tol, 
                                         initial_rank=self.initial_rank, 
                                         warm_start=self.initial, 
                                         debug=self.debug)
        self.factors = [U,D,VT]

    def _getX(self):
        if not self.rankone:
            return np.dot(self.factors[0], np.dot(np.diag(self.factors[1]), self.factors[2]))
        else:
            return self.factors[1][0,0] * np.dot(self.factors[0], self.factors[2])
    X = property(_getX, _setX)

    def _get_factors(self):
        return self._factors
    def _set_factors(self, factors):
        U, D, VT = factors
        self.rankone = False
        if D.reshape(-1).shape == (1,):
            U = U.reshape((U.shape[0],1))
            D = D.reshape((1,1))
            VT = VT.reshape((1,VT.shape[-1]))
            self.rankone = True
        self.input_shape = (VT.shape[1],)
        self.output_shape = (U.shape[0],)
        self._factors = U, D, VT
    factors = property(_get_factors, _set_factors)

    def linear_map(self, x):
        if self.rankone:
            return self.factors[1][0,0] * np.dot(self.factors[0], np.dot(self.factors[2], x))
        else:
            return np.dot(self.factors[0], np.dot(np.diag(self.factors[1]), np.dot(self.factors[2], x)))

    def adjoint_map(self, x):
        if self.rankone:
            return self.factors[1][0,0] * np.dot(self.factors[2].T, np.dot(self.factors[0].T, x))
        else:
            return np.dot(self.factors[2].T, np.dot(np.diag(self.factors[1]), np.dot(self.factors[0].T, x)))

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
                          initial_rank=None,
                          warm_start=None,
                          min_singular=1e-16,
                          tol=1e-5,
                          debug=False,
                          stopping_rule=None,
                          padding=5,
                          update_rank=lambda R: 2*R,
                          max_rank=None):

    """
    Compute the SVD of a matrix using partial_svd.

    If no initial rank is given, it assumes a rank of size min(n,p) / 10.

    Iteratively calls :func:`partial_svd` until the `singular_values` are small
    enough.

    Parameters
    ----------
    transform : [linear_transform, ndarray]
        Linear_transform whose SVD is computed. If an
        ndarray, it is first cast with :func:`astransform()`
    initial_rank : None or int, optional
        A guess at the rank of the matrix.
    warm_start : np.ndarray(np.float), optional
        A guess at the left singular vectors of the matrix.
        For fat matrices, these should be right singular vectors,
        while for tall matrices these should be left singular vectors.
    min_singular : np.float, optional
        Stop when the singular value has this relative tolerance.
    tol: np.float, optional
        Tolerance at which the norm of the singular values are deemed
        to have converged.
    debug: bool, optional
        Print debugging statements.
    stopping_rule : None or callable, optional
        Function of the singular values ``D``, returning True | False,  used to
        determine whether to stop.  Continue while ``stopping_rule(D) ==
        False``, or when `stopping_rule` is None.
    padding : int, optional
        How many more singular vectors are found.
    update_rank : callable, optional
        A rule to update rank, defaults to doubling the rank.
    max_rank : None or int, optional
        Largest rank considered. Defaults to 2 * min(transform.output_shape[0], 
        transform.input_shape[0]). An exception is raised if algorithm exceeds
        given value.

    Returns
    -------
    U, D, VT, Ufull : np.ndarray(np.float)
        An SVD of the transform. Ufull is the full set of left singular vectors
        found.

    Examples
    --------

    >>> import regreg.api as rr
    >>> np.random.seed(0)
    >>> X = np.random.standard_normal((100, 200))
    >>> U, D, VT = rr.compute_iterative_svd(X)[:3]
    >>> np.linalg.norm(np.dot(U.T, U) - np.identity(100)) < 1.e-6
    True

    >>> np.linalg.norm(np.dot(VT, VT.T) - np.identity(100)) < 1.e-6
    True
    """

    transform = astransform(transform)

    n = transform.output_shape[0]
    p = transform.input_shape[0]
    
    if max_rank is None:
        max_rank = 2 * min(n, p)

    need_to_transpose = False
    if n < p:
        transform = adjoint(transform)
        need_to_transpose = True

    if initial_rank is None:
        rank = np.round(np.min([n,p]) * 0.1) + 1
    else:
        rank = np.max([initial_rank,1])

    # for warm start
    if warm_start is not None:
        U = warm_start
    else:
        U = None

    min_so_far = 1.
    D = [min_so_far]
    while D[-1] >= min_singular * np.max(D):
        if debug:
            print("Trying rank", rank)
        U, D, VT = partial_svd(transform, rank=rank, 
                               padding=padding, tol=tol, 
                               warm_start=U,
                               return_full=True, debug=debug)
        if debug:
            print("Singular values", D)
        if len(D) < rank:
            break

        if stopping_rule is not None and stopping_rule(np.fabs(D)):
            break

        rank = update_rank(rank)
        if rank > max_rank:
            raise ValueError('maximum rank exceeded')

        if np.max(D) < 1.e-14:
            break

    ind = np.where(D >= min_singular)[0]
    if not need_to_transpose:
        if U.ndim == 2:
            return U[:,ind], D[ind],  VT[ind,:]
        else:
            return U.reshape((-1,1)), D[ind], VT.reshape((1,-1))
    else:
        if U.ndim == 2:
            return VT[ind,:].T, D[ind],  U[:,ind].T
        else:
            return VT.reshape((-1,1)), D[ind], U.reshape((1,-1))


def partial_svd(transform,
                rank=1,
                padding=5,
                max_its = 1000,
                tol = 1e-8,
                warm_start=None,
                return_full=False,
                debug=False,
                stopping_rule=None):

    """
    Compute partial SVD of the linear transform `transform`

    Uses the Mazumder/Hastie algorithm in (TODO: CITE)

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

    warm_start : np.ndarray(np.float), optional
        A guess at the singular vectors of the matrix. 
        For fat matrices, these should be right singular vectors,
        while for tall matrices these should be left singular vectors.

    return_full: bool, optional
        Return a singular values / vectors from padding?

    debug: bool, optional
        Print debugging statements.

    stopping_rule : callable
        Function of the singular values used to determine whether to stop.

    Returns
    -------
    U, D, VT, Ufull : np.ndarray(np.float)
        An SVD up to `rank` of the transform.
        Ufull is the full set of left singular vectors found.

    Examples
    --------

    >>> from regreg.affine.factored_matrix import partial_svd
    >>> np.random.seed(0)
    >>> X = np.random.standard_normal((100, 200))
    >>> U, D, VT = partial_svd(X, rank=10)[:3]
    >>> assert(np.linalg.norm(np.dot(U.T, U) - np.identity(10)) < 1.e-4)
    >>> assert(np.linalg.norm(np.dot(VT, VT.T) - np.identity(10)) < 1.e-4)
    >>> U_np, D_np, VT_np = np.linalg.svd(X, full_matrices=False)
    >>> assert(np.linalg.norm(U - np.dot(U, np.dot(U_np.T[:10], U_np[:,:10]))) < 1.e-4)
    >>> assert(np.linalg.norm(VT - np.dot(VT, np.dot(VT_np[:10].T, VT_np[:10]))) < 1.e-4)

    """

    transform = astransform(transform)

    n = np.product(transform.output_shape)
    p = np.product(transform.input_shape)

    rank = np.int(np.min([rank,p]))
    q = np.min([rank + padding, p])
    if warm_start is not None:
        if warm_start.shape == (n,q):
            U = warm_start
        elif len(warm_start.shape) == 1:
            U = np.hstack([warm_start.reshape((warm_start.shape[0],1)), np.random.standard_normal((n,q-1))])            
        else:
            U = np.hstack([warm_start, np.random.standard_normal((n,q-warm_start.shape[1]))])            
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
            print(itercount, singular_rel_change,
                  np.sum(np.fabs(singular_values)>1e-12),
                  np.fabs(singular_values[np.arange(np.min([5,len(singular_values)]))]))
        V, _ = np.linalg.qr(transform.adjoint_map(U))
        X_V = transform.linear_map(V)
        U, R = np.linalg.qr(X_V)
        singular_values = np.diagonal(R)[change_ind]
        singular_rel_change = np.linalg.norm(singular_values - old_singular_values)/np.linalg.norm(singular_values)
        old_singular_values[:] = singular_values
        itercount += 1

        if stopping_rule is not None and stopping_rule(np.fabs(singular_values)):
            break

    singular_values = np.diagonal(R)[ind]

    nonzero = np.where(np.fabs(singular_values) > 1e-12)[0]
    if len(nonzero):
        return U[:,ind[nonzero]] * np.sign(singular_values[nonzero]), np.fabs(singular_values[nonzero]),  V[:,ind[nonzero]].T
    else:
        return U[:,ind[0]], np.zeros((1,1)),  V[:,ind[0]].T

@objective_doc_templater()
class nuclear_norm(nuclear_norm_atom):

    """
    The nuclear norm
    """
    prox_tol = 1.0e-5
    svd_tol = 1.e-5
    objective_vars = nuclear_norm_atom.objective_vars.copy()

    def __init__(self, shape, lagrange=None, bound=None,
                 offset=None, quadratic=None, initial=None,
                 initial_rank=10,
                 warm_start=None):

        nuclear_norm_atom.__init__(self,
                                   shape,
                                   lagrange=lagrange,
                                   bound=bound,
                                   offset=offset,
                                   quadratic=quadratic,
                                   initial=0)

        self.initial_rank = initial_rank
        # the warm start should be of whichever side has largest
        # dimension
        m = max(self.shape)
        if warm_start is None:
            self.warm_start = np.random.standard_normal((m, self.initial_rank))
        else:
            self.warm_start = warm_start
            if warm_start.shape != (m, self.initial_rank):
                raise ValueError('expecting warm_start to have shape %s' % (m, self.initial_rank))

    @doc_template_user
    def seminorm(self, X, check_feasibility=False,
                 lagrange=None):
        raise NotImplementedError('too expensive to compute')

    @doc_template_user
    def constraint(self, X, bound=None):
        raise NotImplementedError('too expensive to compute')

    @doc_template_user
    def lagrange_prox(self, X,  lipschitz=1, lagrange=None):
        lagrange = svd_atom.lagrange_prox(self, X, lipschitz, lagrange)

        def soft_threshold_rule(L):
            return lambda D: np.fabs(D).min() <= L

        svt_rule = soft_threshold_rule(lagrange)
        if self.warm_start is not None:
            padding = 0
        else:
            padding = 5

        U, D, VT = compute_iterative_svd(X, initial_rank=self.initial_rank, 
                                         warm_start=self.warm_start,
                                         stopping_rule=svt_rule, 
                                         tol=self.svd_tol,
                                         padding=padding)

        if self.shape[0] < self.shape[1]:
            self.warm_start = VT.T
        else:
            self.warm_start = U

        self.initial_rank = U.shape[1]

        D_proj = D - lagrange
        keep = D_proj > 0

        return factored_matrix((U[:,keep], D_proj[keep], VT[keep,:]))

    @doc_template_user
    def bound_prox(self, X, bound=None):
        bound = svd_atom.bound_prox(self, X, bound)

        def bound_rule(B):
            return lambda D: (np.fabs(D) - np.fabs(D).min()).sum() > B

        if self.warm_start is not None:
            padding = 0
        else:
            padding = 5

        U, D, VT = compute_iterative_svd(X, initial_rank=self.initial_rank,
                                         warm_start=self.warm_start,
                                         stopping_rule=bound_rule(bound),
                                         tol=self.svd_tol,
                                         padding=padding)

        if self.shape[0] < self.shape[1]:
            self.warm_start = VT.T
        else:
            self.warm_start = U

        self.initial_rank = U.shape[1]

        l1atom = l1norm(D.shape, bound=bound)
        D_projected = l1atom.bound_prox(D)
        keep = D_projected > 0
        return factored_matrix((U[:,keep], D_projected[keep], VT[keep,:]))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = operator_norm
            atom = cls(self.shape,
                       bound=self.lagrange, 
                       lagrange=self.bound,
                       quadratic=outq,
                       offset=offset,
                       initial_rank=self.initial_rank,
                       warm_start=self.warm_start)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate



@objective_doc_templater()
class operator_norm(operator_norm_atom):
    prox_tol = 1.0e-5
    svd_tol = 1.e-4
    objective_vars = operator_norm_atom.objective_vars.copy()

    def __init__(self, shape, lagrange=None, bound=None,
                 offset=None, quadratic=None, initial=None,
                 initial_rank=10, warm_start=None):

        operator_norm_atom.__init__(self,
                                    shape,
                                    lagrange=lagrange,
                                    bound=bound,
                                    offset=offset,
                                    quadratic=quadratic,
                                    initial=0)

        self.initial_rank = initial_rank
        if warm_start is None:
            self.warm_start = np.random.standard_normal((self.shape[0], self.initial_rank))
        else:
            self.warm_start = warm_start
            if warm_start.shape != (self.shape[0], self.initial_rank):
                raise ValueError('expecting warm_start to have shape %s' % (self.shape[0], self.initial_rank))

        self._nuclear_atom = nuclear_norm(shape,
                                          lagrange=lagrange,
                                          bound=bound,
                                          initial=initial,
                                          initial_rank=initial_rank,
                                          warm_start=self.warm_start)

    @doc_template_user
    def seminorm(self, X, check_feasibility=False,
                 lagrange=None):
        raise NotImplementedError('too expensive to compute')

    @doc_template_user
    def constraint(self, X, bound=None):
        raise NotImplementedError('too expensive to compute')

    @doc_template_user
    def lagrange_prox(self, X,  lipschitz=1, lagrange=None):
        lagrange = svd_atom.lagrange_prox(self, X, lipschitz, lagrange)

        bound = lagrange / lipschitz
        dual_proj = self._nuclear_atom.bound_prox(X, bound=bound)
        return affine_sum([X,dual_proj],[1.,-1.])

    @doc_template_user
    def bound_prox(self, X,  bound=None):
        bound = svd_atom.bound_prox(self, X, bound)

        lagrange = bound
        dual_lagrange = self._nuclear_atom.lagrange_prox(X, lagrange=lagrange)
        return affine_sum([X,dual_lagrange],[1.,-1.])

    @property
    @doc_template_user
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = nuclear_norm
            atom = cls(self.shape,
                       bound=self.lagrange, 
                       lagrange=self.bound,
                       quadratic=outq,
                       offset=offset,
                       initial_rank=self.initial_rank,
                       warm_start=self.warm_start)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
