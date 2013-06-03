"""
This module contains the implementation operator and nuclear norms, used in 
matrix completion problems and other low-rank factorization
problems.

"""
from copy import copy

import numpy as np

from ..atoms import atom, _work_out_conjugate
from .seminorms import conjugate_seminorm_pairs, seminorm, l1norm, supnorm
from .cones import cone, conjugate_cone_pairs
from .projl1_cython import projl1, projl1_epigraph
from .piecewise_linear import find_solution_piecewise_linear_c

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

# for doctests
from ..identity_quadratic import identity_quadratic

@objective_doc_templater()
class svd_atom(seminorm):

    objective_vars = seminorm.objective_vars.copy()
    objective_vars['var'] = 'B'
    objective_vars['normklass'] = 'nuclear_norm'
    objective_vars['dualnormklass'] = 'operator_norm'
    objective_vars['initargs'] = '(5,4)'
    objective_vars['shape'] = r'{n \times p}'


@objective_doc_templater()
class nuclear_norm(svd_atom):

    """
    The nuclear norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_*"""
    objective_vars = svd_atom.objective_vars.copy()

    @doc_template_user
    def seminorm(self, X, check_feasibility=False,
                 lagrange=None):
        lagrange = seminorm.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        return lagrange * np.fabs(D).sum()

    @doc_template_user
    def constraint(self, X, bound=None):
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class operator_norm(svd_atom):

    """
    The operator norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_{\text{op}}"""
    objective_vars = svd_atom.objective_vars.copy()
    objective_vars['normklass'] = 'operator_norm'
    objective_vars['dualnormklass'] = 'nuclear_norm'

    @doc_template_user
    def seminorm(self, X, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, X, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        U, D, V = np.linalg.svd(X, full_matrices=0)
        return lagrange * np.max(D)

    @doc_template_user
    def constraint(self, X, bound=None):
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class svd_cone(cone):

    objective_vars = svd_atom.objective_vars.copy()
    objective_vars['coneklass'] = 'nuclear_norm_epigraph'
    objective_vars['dualconeklass'] = 'nuclear_norm_epigraph_polar'
    objective_vars['shape'] = r'{n \times p + 1}'

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

    @doc_template_user
    def get_conjugate(self):
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
    conjugate = property(get_conjugate)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class nuclear_norm_epigraph(svd_cone):

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_* \leq %(var)s[-1])"""
    objective_vars = svd_cone.objective_vars.copy()
    objective_vars['coneklass'] = 'nuclear_norm_epigraph'
    objective_vars['dualconeklass'] = 'nuclear_norm_epigraph_polar'

    @doc_template_user
    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = np.fabs(D[1:]).sum() <= (1 + self.tol) * norm
        if incone:
            return 0
        return np.inf

    @doc_template_user
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return svd_cone.get_conjugate(self)

@objective_doc_templater()
class nuclear_norm_epigraph_polar(svd_cone):
    
    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{op} \leq -%(var)s[-1])"""
    objective_vars = svd_cone.objective_vars.copy()
    objective_vars['coneklass'] = 'nuclear_norm_epigraph_polar'
    objective_vars['dualconeklass'] = 'nuclear_norm_epigraph'

    @doc_template_user
    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = D.max() <= (1 + self.tol) * (-norm)
        if incone:
            return 0
        return np.inf

    @doc_template_user
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return svd_cone.get_conjugate(self)

@objective_doc_templater()
class operator_norm_epigraph(svd_cone):
    
    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{op} \leq %(var)s[-1])"""
    objective_vars = svd_cone.objective_vars.copy()
    objective_vars['coneklass'] = 'operator_norm_epigraph'
    objective_vars['dualconeklass'] = 'operator_norm_epigraph_polar'

    @doc_template_user
    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = D.max() <= (1 + self.tol) * norm
        if incone:
            return 0
        return np.inf

    @doc_template_user
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return svd_cone.get_conjugate(self)

@objective_doc_templater()
class operator_norm_epigraph_polar(svd_cone):
    
    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{*} \leq -%(var)s[-1])"""
    objective_vars = svd_cone.objective_vars.copy()
    objective_vars['coneklass'] = 'operator_norm_epigraph_polar'
    objective_vars['dualconeklass'] = 'operator_norm_epigraph'

    @doc_template_user
    def constraint(self, normX):
        norm = normX[-1]
        X = normX[:-1].reshape(self.matrix_shape)
        U, D, V = np.linalg.svd(X, full_matrices=False)

        incone = D.sum() <= (1 + self.tol) * (-norm)
        if incone:
            return 0
        return np.inf

    @doc_template_user
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return svd_cone.get_conjugate(self)

conjugate_svd_pairs = {}
conjugate_svd_pairs[nuclear_norm] = operator_norm
conjugate_svd_pairs[operator_norm] = nuclear_norm

conjugate_seminorm_pairs[nuclear_norm] = operator_norm
conjugate_seminorm_pairs[operator_norm] = nuclear_norm

conjugate_cone_pairs[nuclear_norm_epigraph] = nuclear_norm_epigraph_polar
conjugate_cone_pairs[nuclear_norm_epigraph_polar] = nuclear_norm_epigraph
conjugate_cone_pairs[operator_norm_epigraph] = operator_norm_epigraph_polar
conjugate_cone_pairs[operator_norm_epigraph_polar] = operator_norm_epigraph
