"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
from md5 import md5
from copy import copy

import numpy as np

from ..atoms import atom, _work_out_conjugate
from .seminorms import conjugate_seminorm_pairs, seminorm, l1norm, supnorm
from .cones import cone, conjugate_cone_pairs
from .projl1_cython import projl1, projl1_epigraph
from .piecewise_linear import find_solution_piecewise_linear_c

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

@objective_doc_templater()
class svd_atom(seminorm):
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['var'] = 'X'
    objective_vars['shape'] = r'n \times p'

    @doc_template_provider
    def lagrange_prox(self, X, lipschitz=1, lagrange=None):
        r""" Return unique minimizer

        .. math::

           %(var)s^{\lambda}(U) =
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|U-%(var)s\|^2_F
            + \lambda   %(objective)s 

        Above, :math:`\lambda` is the Lagrange parameter.

        If the argument `lagrange` is None and the atom is in lagrange mode,
        self.lagrange is used as the lagrange parameter, else an exception is
        raised.

        The class atom's lagrange_prox just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        if lagrange is None:
            lagrange = self.lagrange
        if lagrange is None:
            raise ValueError('either atom must be in Lagrange mode or a keyword "lagrange" argument must be supplied')
        return lagrange

    @doc_template_provider
    def bound_prox(self, X, bound=None):
        r"""
        Return unique minimizer

        .. math::

           %(var)s^{\lambda}(U) \in 
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{1}{2}
           \|U-%(var)s\|^2_F 
           \text{s.t.} \   %(objective)s \leq \delta

        Above, :math:`\delta` is the bound parameter.

        If the argument `bound` is None and the atom is in bound mode,
        self.bound is used as the bound parameter, else an exception is raised.

        The class atom's bound_prox just returns the appropriate bound
        parameter for use by the subclasses.
        """
        if bound is None:
            bound = self.bound
        if bound is None:
            raise ValueError('either atom must be in bound mode or a keyword "bound" argument must be supplied')
        return bound


@objective_doc_templater()
class nuclear_norm(svd_atom):

    """
    The nuclear norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_*"""

    @doc_template_user
    def seminorm(self, X, check_feasibility=False,
                 lagrange=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        lagrange = seminorm.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        return lagrange * np.fabs(D).sum()

    @doc_template_user
    def constraint(self, X, bound=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        bound = seminorm.constraint(self, X, bound=bound)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        inbox = np.fabs(D).sum() <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, X,  lipschitz=1, lagrange=None):
        lagrange = svd_atom.lagrange_prox(self, X, lipschitz, lagrange)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        l1atom = l1norm(np.product(self.shape), lagrange=lagrange)
        D_soft_thresholded = l1atom.lagrange_prox(D, lipschitz=lipschitz, lagrange=lagrange)
        keepD = D_soft_thresholded > 0
        print 'subgrad', l1atom.check_subgradient(l1atom, D)
        print 'now', D, D_soft_thresholded
        print 'shape', U.shape, D.shape, V.shape
        print 'norm', np.linalg.norm(X - np.dot(U, D[:,np.newaxis] * V))
        U_new, D_new, V_new = U[:,keepD], D_soft_thresholded[keepD], V[keepD]
        return np.dot(U, D_soft_thresholded[:,np.newaxis] * V)

    @doc_template_user
    def bound_prox(self, X, bound=None):
        bound = svd_atom.bound_prox(self, X, bound)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        l1atom = l1norm(np.product(self.shape), bound=bound)
        D_projected = l1atom.bound_prox(D)
        keepD = D_projected > 0
        U_new, D_new, V_new = U[:,keepD], D_projected[keepD], V[keepD]
        return np.dot(U_new, D_new[:,np.newaxis] * V_new)
        return np.dot(U, D_projected[:,np.newaxis] * V)

@objective_doc_templater()
class operator_norm(svd_atom):

    """
    The operator norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_{\text{op}}"""

    @doc_template_user
    def seminorm(self, X, lagrange=None, check_feasibility=False):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        lagrange = seminorm.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        return lagrange * np.max(D)

    @doc_template_user
    def constraint(self, X, bound=None):
        # This will compute an svd of X
        # if the md5 hash of X doesn't match.
        bound = seminorm.constraint(self, X, bound=bound)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        inbox = np.max(D) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, X,  lipschitz=1, lagrange=None):
        lagrange = svd_atom.lagrange_prox(self, X, lipschitz, lagrange)
        U, D, V = np.linalg.svd(X, full_matrices=False)
        supatom = supnorm(np.product(self.shape), lagrange=lagrange)
        D_new = supatom.lagrange_prox(D, lipschitz=lipschitz, lagrange=lagrange)
        return np.dot(U, D_new[:,np.newaxis] * V)

    @doc_template_user
    def bound_prox(self, X, bound=None):
        bound = svd_atom.bound_prox(self, X, bound)
        U, D, V = np.linalg.svd(X, full_matrices=False)
        supatom = supnorm(np.product(self.shape), bound=bound)
        D_new = supatom.bound_prox(D)
        return np.dot(U, D_new[:,np.newaxis] * V)

@objective_doc_templater()
class svd_cone(cone):

    def __copy__(self):
        return self.__class__(copy(self.matrix_shape),
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    

    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.matrix_shape),
                 str(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.matrix_shape),
                 str(self.offset),
                 str(self.quadratic))

    def __init__(self, input_shape,
                 offset=None,
                 quadratic=None,
                 initial=None):

        shape = (np.product(input_shape)+1,)
        cone.__init__(self, shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.matrix_shape = input_shape

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            new_atom = cls(self.matrix_shape,
                       offset=offset,
                       quadratic=outq)
        else:
            new_atom = smooth_conjugate(self)
        self._conjugate = new_atom
        self._conjugate._conjugate = self
        return self._conjugate

@objective_doc_templater()
class nuclear_norm_epigraph(svd_cone):

    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = np.fabs(D[1:]).sum() <= (1 + self.tol) * norm
        if incone:
            return 0
        return np.inf

    def cone_prox(self, normX,  lipschitz=1):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)
        newD = np.zeros(D.shape[0]+1)
        newD[-1] = norm
        newD[:-1] = D
        newD = projl1_epigraph(newD)
        result = np.zeros_like(normX)
        result[-1] = newD[-1] 
        newX = np.dot(U, newD[:-1][:,np.newaxis] * V)
        result[:-1] = newX.reshape(-1)
        return result

@objective_doc_templater()
class nuclear_norm_epigraph_polar(svd_cone):
    
    def constraint(self, normX):
        """
        The non-negative constraint of x.
        """
        
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = D.max() <= (1 + self.tol) * (-norm)
        if incone:
            return 0
        return np.inf

    def cone_prox(self, normX,  lipschitz=1):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        newD = np.zeros(D.shape[0]+1)
        newD[-1] = norm
        newD[:-1] = D
        newD = newD - projl1_epigraph(newD)
        result = np.zeros_like(normX)
        result[-1] = newD[-1]
        newX = np.dot(U, newD[:-1][:,np.newaxis] * V)
        result[:-1] = newX.reshape(-1)
        return result

@objective_doc_templater()
class operator_norm_epigraph(svd_cone):
    
    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = D.max() <= (1 + self.tol) * norm
        if incone:
            return 0
        return np.inf

    def cone_prox(self, normX,  lipschitz=1):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        newD = np.zeros(D.shape[0]+1)
        newD[-1] = norm
        newD[:-1] = D
        newD = newD + projl1_epigraph(-newD)
        result = np.zeros_like(normX)
        result[-1] = newD[-1]
        newX = np.dot(U, newD[:-1][:,np.newaxis] * V)
        result[:-1] = newX.reshape(-1)
        return result

@objective_doc_templater()
class operator_norm_epigraph_polar(svd_cone):
    
    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = D.sum() <= (1 + self.tol) * (-norm)
        if incone:
            return 0
        return np.inf

    def cone_prox(self, normX,  lipschitz=1):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)
        newD = np.zeros(D.shape[0]+1)
        newD[-1] = norm
        newD[:-1] = D
        newD = -projl1_epigraph(-newD)
        result = np.zeros_like(normX)
        result[-1] = newD[-1]
        newX = np.dot(U, newD[:-1][:,np.newaxis] * V)
        result[:-1] = newX.reshape(-1)
        return result


conjugate_svd_pairs = {}
conjugate_svd_pairs[nuclear_norm] = operator_norm
conjugate_svd_pairs[operator_norm] = nuclear_norm

conjugate_seminorm_pairs[nuclear_norm] = operator_norm
conjugate_seminorm_pairs[operator_norm] = nuclear_norm

conjugate_cone_pairs[nuclear_norm_epigraph] = nuclear_norm_epigraph_polar
conjugate_cone_pairs[nuclear_norm_epigraph_polar] = nuclear_norm_epigraph
conjugate_cone_pairs[operator_norm_epigraph] = operator_norm_epigraph_polar
conjugate_cone_pairs[operator_norm_epigraph_polar] = operator_norm_epigraph
