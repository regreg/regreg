from copy import copy
import warnings

import numpy as np

from ..problems.composite import composite, nonsmooth, smooth_conjugate
from ..affine import linear_transform, identity as identity_transform, selector
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate
from ..smooth import affine_smooth

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

from .seminorms import seminorm
from .cones import cone

from .mixed_lasso_cython import (mixed_lasso_bound_prox,
                                 mixed_lasso_lagrange_prox,
                                 mixed_lasso_epigraph,
                                 mixed_lasso_dual_bound_prox)

@objective_doc_templater()
class group_lasso(seminorm):

    """
    The group lasso seminorm.
    """

    objective_template = r"""\sum_g \|%(var)s[g]\|_2"""

    tol = 1.0e-05

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        shape = np.asarray(groups).shape
        seminorm.__init__(self, shape, offset=offset,
                          quadratic=quadratic,
                          initial=initial,
                          lagrange=lagrange,
                          bound=bound)

        self.weights = weights
        self.groups = np.asarray(groups)
        self._groups = np.zeros(shape, np.int)

        sg = sorted(np.unique(groups))
        self._weight_array = np.ones(len(sg))
        
        for i, g in enumerate(sorted(np.unique(groups))):
            self._groups[(groups == g)] = i
            self._weight_array[i] = self.weights.get(g, np.sqrt((groups == g).sum()))
            self.weights[g] = self._weight_array[i]

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.shape == other.shape and 
                    np.all(self.groups == other.groups)
                    and np.all(self.weights == other.weights)
                    and self.lagrange == other.lagrange)
        return False

    def __copy__(self):
        return self.__class__(copy(self.groups),
                              weights=self.weights,
                              offset=self.offset,
                              lagrange=self.lagrange,
                              bound=self.bound,
                              quadratic=self.quadratic,
                              initial=self.coefs)
    
    def __repr__(self):
        if self.lagrange is not None:
            if self.quadratic.iszero:
                return "%s(%s, lagrange=%s, weights=%s, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.groups),
                     self.lagrange,
                     repr(self.weights),
                     repr(self.offset))
            else:
                return "%s(%s, lagrange=%s, weights=%s, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.groups),
                     self.lagrange, 
                     repr(self.weights),
                     repr(self.offset),
                     repr(self.quadratic))

        if self.bound is not None:
            if self.quadratic.iszero:
                return "%s(%s, bound=%s, weights=%s, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.groups),
                     self.bound,
                     repr(self.weights),
                     repr(self.offset))
            else:
                return "%s(%s, bound=%s, weights=%s, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.groups),
                     self.bound,
                     repr(self.weights),
                     repr(self.offset),
                     repr(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = conjugate_seminorm_pairs[self.__class__]
            atom = cls(self.groups,
                       weights=self.weights,
                       bound=self.lagrange, 
                       lagrange=self.bound,
                       quadratic=outq,
                       offset=offset)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @doc_template_user
    def seminorm(self, arg, lagrange=1., check_feasibility=False):
        value = 0
        ngroups = self._weight_array.shape[0]
        for i in range(ngroups):
            group = arg[self._groups == i]
            value += self._weight_array[i] * np.linalg.norm(group)
        return value * lagrange

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg, np.float)
        return mixed_lasso_lagrange_prox(arg, float(lagrange),
                                         float(lipschitz),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         self._groups,
                                         self._weight_array)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg, np.float)
        return mixed_lasso_bound_prox(arg, float(bound),
                                      np.array([], np.int),
                                      np.array([], np.int),
                                      np.array([], np.int),
                                      np.array([], np.int),
                                      self._groups,
                                      self._weight_array)

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inball = self.seminorm(arg, lagrange=1) <= bound
        if inball:
            return 0
        else:
            return np.inf


@objective_doc_templater()
class group_lasso_dual(group_lasso):

    """
    The dual of the group lasso seminorm.
    """

    objective_template = r"""\max_g \|%(var)s[g]\|_2"""

    tol = 1.0e-05

    @doc_template_user
    def seminorm(self, arg, lagrange=1, check_feasibility=False):
        value = 0
        ngroups = self._weight_array.shape[0]
        for i in range(ngroups):
            group = arg[self._groups == i]
            value = max(value, np.linalg.norm(group) / self._weight_array[i])
        return value * lagrange

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        x = np.asarray(arg, np.float)
        return mixed_lasso_dual_bound_prox(arg, float(bound),
                                           np.array([], np.int),
                                           np.array([], np.int),
                                           np.array([], np.int),
                                           np.array([], np.int),
                                           self._groups,
                                           self._weight_array)

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        r = mixed_lasso_bound_prox(arg, lagrange / lipschitz,
                                   np.array([], np.int),
                                   np.array([], np.int),
                                   np.array([], np.int),
                                   np.array([], np.int),
                                   self._groups,
                                   self._weight_array)
        return x - r

# the conjugate method needs to be slightly modified
@objective_doc_templater()
class group_lasso_cone(cone):
    
    """
    A superclass for the group LASSO and group LASSO dual epigraph cones.
    """
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, weights=%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.groups),
                 repr(self.weights),
                 repr(self.offset))
        else:
            return "%s(%s, weights=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.groups),
                 repr(self.weights),
                 repr(self.offset),
                 repr(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            new_atom = cls(self.groups,
                       self.weights,
                       offset=offset,
                       quadratic=outq)
        else:
            new_atom = smooth_conjugate(self)
        self._conjugate = new_atom
        self._conjugate._conjugate = self
        return self._conjugate

    @property
    def weights(self):
        return self.snorm.weights

    @property
    def groups(self):
        return self.snorm.groups

@objective_doc_templater()
class group_lasso_epigraph(group_lasso_cone):

    """
    The epigraph of the group lasso seminorm.
    """

    objective_template = (r"""I^{\infty}\left(\sum_g \|%(var)s[g]\|_2 """
                          + r"""\leq %(var)s[-1]\right)""")

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        shape = groups.shape[0]+1
        cone.__init__(self, shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    @doc_template_user
    def constraint(self, x):
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x,  lipschitz=1):
        return mixed_lasso_epigraph(x,
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    self.snorm._groups,
                                    self.snorm._weight_array)

@objective_doc_templater()
class group_lasso_epigraph_polar(group_lasso_cone):

    """
    The polar of the epigraph of the group lasso seminorm.
    """

    objective_template = (r"""I^{\infty}(\max_g \|%(var)s[g]\|_2 \leq """
                          + r"""-%(var)s[-1]\)""")

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        shape = groups.shape[0]+1
        cone.__init__(self, shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    @doc_template_user
    def constraint(self, x):
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x,  lipschitz=1):
        return mixed_lasso_epigraph(x,
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    np.array([], np.int),
                                    self.snorm._groups,
                                    self.snorm._weight_array) - x


@objective_doc_templater()
class group_lasso_dual_epigraph(group_lasso_cone):

    """
    The group LASSO conjugate epigraph constraint.
    """

    objective_template = (r"""I^{\infty}(%(var)s: \max_g """ + 
                          r"""\|%(var)s[1:][g]\|_2 \leq %(var)s[0])""")

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        shape = groups.shape[0]+1
        cone.__init__(self, shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso_dual(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    @doc_template_user
    def constraint(self, x):
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x,  lipschitz=1):
        return x + mixed_lasso_epigraph(-x,
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         np.array([], np.int),
                                         self.snorm._groups,
                                         self.snorm._weight_array)

@objective_doc_templater()
class group_lasso_dual_epigraph_polar(group_lasso_cone):

    """
    The polar of the group LASSO dual epigraph constraint.
    """

    objective_template = (r"""I^{\infty}(%(var)s: \sum_g \|%(var)s[g]\|_2 \leq """
                          + r"""-%(var)s[-1]\}}""")

    def __init__(self, groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        shape = groups.shape[0]+1
        cone.__init__(self, shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        self.snorm = group_lasso_dual(groups,
                 weights=weights,
                 offset=offset,
                 lagrange=1,
                 bound=None,
                 quadratic=None,
                 initial=None)

    @doc_template_user
    def constraint(self, x):
        incone = self.snorm.seminorm(x[1:], lagrange=1) <= 1 + self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x,  lipschitz=1):
        return - mixed_lasso_epigraph(-x,
                                       np.array([], np.int),
                                       np.array([], np.int),
                                       np.array([], np.int),
                                       np.array([], np.int),
                                       self.snorm._groups,
                                       self.snorm._weight_array)

conjugate_seminorm_pairs = {}
conjugate_seminorm_pairs[group_lasso_dual] = group_lasso
conjugate_seminorm_pairs[group_lasso] = group_lasso_dual

conjugate_cone_pairs = {}
conjugate_cone_pairs[group_lasso_epigraph] = group_lasso_epigraph_polar
conjugate_cone_pairs[group_lasso_epigraph_polar] = group_lasso_epigraph
conjugate_cone_pairs[group_lasso_dual_epigraph_polar] = group_lasso_dual_epigraph
conjugate_cone_pairs[group_lasso_dual_epigraph] = group_lasso_dual_epigraph_polar
