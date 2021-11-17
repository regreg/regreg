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
                                 mixed_lasso_dual_bound_prox,
                                 seminorm_mixed_lasso,
                                 seminorm_mixed_lasso_dual)

@objective_doc_templater()
class group_lasso(seminorm):

    """
    The group lasso seminorm.
    """

    objective_template = r"""\sum_g w_g \|%(var)s[g]\|_2"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'group_lasso'
    objective_vars['dualnormklass'] = 'group_lasso_dual'
    objective_vars['initargs'] = '[1, 1, 2, 2, 2]'

    tol = 1.0e-05

    def __init__(self, 
                 groups,
                 weights={},
                 offset=None,
                 lagrange=None,
                 bound=None,
                 quadratic=None,
                 initial=None):

        self.groups = np.asarray(groups).copy()
        shape = self.groups.shape
        seminorm.__init__(self, shape, offset=offset,
                          quadratic=quadratic,
                          initial=initial,
                          lagrange=lagrange,
                          bound=bound)

        self.weights = copy(weights)
        self._group_array = np.zeros(shape, np.intp)
        self._group_dict = {}
        self._group_inv_dict = {}

        sg = self._sorted_groupids = sorted(np.unique(self.groups))
        self._weight_array = np.ones(len(sg))
        
        for i, g in enumerate(sg):
            group = self.groups == g
            self._group_array[group] = i
            self._group_dict[g] = i
            self._group_inv_dict[i] = g
            self._weight_array[i] = self.weights.get(g, np.sqrt(group.sum()))
            self.weights[g] = self._weight_array[i]

        # buffers for cython computations

        self._l1_cython = np.array([], np.intp)
        self._unpenalized_cython = np.array([], np.intp)
        self._positive_part_cython = np.array([], np.intp)
        self._nonnegative_cython = np.array([], np.intp)
        self._norms_cython = np.zeros_like(self._weight_array)
        self._factors_cython = np.zeros_like(self._weight_array)
        self._projection_cython = np.zeros(self.shape)

    def set_weight(self, group, weight):
        """
        Set a group's weight after initialization.
        """
        idx = self._sorted_groupids.index(group)
        self._weight_array[idx] = weight
        self.weights[group] = weight
        
    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.shape == other.shape and 
                    np.all(self.groups == other.groups)
                    and (self.weights == other.weights)
                    and self.lagrange == other.lagrange)
        return False

    def __copy__(self):
        return self.__class__(copy(self.groups),
                              weights=copy(self.weights),
                              offset=copy(self.offset),
                              lagrange=self.lagrange,
                              bound=self.bound,
                              quadratic=copy(self.quadratic),
                              initial=copy(self.coefs))
    
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

    @doc_template_user
    def get_conjugate(self):

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
    conjugate = property(get_conjugate)

    def terms(self, arg):
        """
        Return the args that are summed
        in computing the seminorm.

        >>> import regreg.api as rr
        >>> groups = [1,1,2,2,2]
        >>> penalty = rr.group_lasso(groups, lagrange=1.)
        >>> arg = [2,4,5,3,4]
        >>> list(penalty.terms(arg)) # doctest: +ELLIPSIS
        [6.3245..., 12.2474...]
        >>> penalty.seminorm(arg) # doctest: +ELLIPSIS
        18.5720...
        >>> np.sqrt((2**2 + 4**2)*2), np.sqrt((5**2 + 3**2 + 4**2) * 3.) # doctest: +ELLIPSIS
        (6.3245..., 12.2474...)
        >>> np.sqrt((2**2 + 4**2)*2) + np.sqrt((5**2 + 3**2 + 4**2) * 3.) # doctest: +ELLIPSIS
        18.5720...
        
        """
        arg = np.asarray(arg)
        norms = []
        for g in self._sorted_groupids:
            group = self.groups == g
            norms.append(np.linalg.norm(arg[group]) * self.weights[g])
        return np.array(norms)

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                     lagrange=lagrange,
                                     check_feasibility=check_feasibility)
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self._norms_cython[:] = 0

        return lagrange * seminorm_mixed_lasso(arg, 
                                               self._l1_cython,
                                               self._unpenalized_cython,
                                               self._positive_part_cython,
                                               self._nonnegative_cython,
                                               self._group_array,
                                               self._weight_array,
                                               self._norms_cython,
                                               False)

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self._norms_cython[:] = 0
        self._factors_cython[:] = 0
        self._projection_cython[:] = 0

        return mixed_lasso_lagrange_prox(arg, 
                                         float(lagrange),
                                         float(lipschitz),
                                         self._l1_cython,
                                         self._unpenalized_cython,
                                         self._positive_part_cython,
                                         self._nonnegative_cython,
                                         self._group_array,
                                         self._weight_array,
                                         self._norms_cython,
                                         self._factors_cython,
                                         self._projection_cython)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self._norms_cython[:] = 0
        self._factors_cython[:] = 0
        self._projection_cython[:] = 0

        return mixed_lasso_bound_prox(arg, float(bound),
                                      self._l1_cython,
                                      self._unpenalized_cython,
                                      self._positive_part_cython,
                                      self._nonnegative_cython,
                                      self._group_array,
                                      self._weight_array,
                                      self._norms_cython,
                                      self._factors_cython,
                                      self._projection_cython)

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inball = self.seminorm(arg, lagrange=1) <= bound
        if inball:
            return 0
        else:
            return np.inf

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
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class group_lasso_dual(group_lasso):

    """
    The dual of the group lasso seminorm.
    """

    objective_template = r"""\max_g \|%(var)s[g]\|_2 / w_g"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'group_lasso_dual'
    objective_vars['dualnormklass'] = 'group_lasso'

    tol = 1.0e-05

    def terms(self, arg, check_feasibility=True):
        """
        Return the args that are maximized
        in computing the seminorm.

        If `check_feasibility` then return `np.inf`
        for groups whose norms aren't 0 with a weight of 0.
        Otherwise, return 0 for these entries

        >>> import regreg.api as rr
        >>> groups = [1,1,2,2,2]
        >>> penalty = rr.group_lasso_dual(groups, lagrange=1.)
        >>> arg = [2,4,5,3,4]
        >>> list(penalty.terms(arg)) # doctest: +ELLIPSIS
        [3.1622..., 4.0824...]
        >>> np.sqrt((2**2 + 4**2)/2), np.sqrt((5**2 + 3**2 + 4**2) / 3.) # doctest: +ELLIPSIS
        (3.1622..., 4.0824...)
        >>> penalty.seminorm(arg) # doctest: +ELLIPSIS
        4.0824...

        """
        arg = np.asarray(arg)
        norms = []
        for g in self._sorted_groupids:
            group = self.groups == g
            w = self.weights[g]
            if w > 0:
                norms.append(np.linalg.norm(arg[group]) / w)
            else:
                if check_feasibility and np.linalg.norm(arg[group]) > 0:
                    norms.append(np.inf)  
                else:
                    norms.append(0)
        return np.array(norms)

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, lagrange=lagrange,
                                     check_feasibility=check_feasibility)
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self._norms_cython[:] = 0

        return lagrange * seminorm_mixed_lasso_dual(arg, 
                                                    self._l1_cython,
                                                    self._unpenalized_cython,
                                                    self._positive_part_cython,
                                                    self._nonnegative_cython,
                                                    self._group_array,
                                                    self._weight_array,
                                                    self._norms_cython,
                                                    False)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self._norms_cython[:] = 0
        self._factors_cython[:] = 0
        self._projection_cython[:] = 0

        return mixed_lasso_dual_bound_prox(arg, float(bound),
                                           self._l1_cython,
                                           self._unpenalized_cython,
                                           self._positive_part_cython,
                                           self._nonnegative_cython,
                                           self._group_array,
                                           self._weight_array,
                                           self._norms_cython,
                                           self._factors_cython,
                                           self._projection_cython)

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self._norms_cython[:] = 0
        self._factors_cython[:] = 0
        self._projection_cython[:] = 0

        r = mixed_lasso_bound_prox(arg, lagrange / lipschitz,
                                   self._l1_cython,
                                   self._unpenalized_cython,
                                   self._positive_part_cython,
                                   self._nonnegative_cython,
                                   self._group_array,
                                   self._weight_array,
                                   self._norms_cython,
                                   self._factors_cython,
                                   self._projection_cython)
        return arg - r


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
        return group_lasso.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class group_lasso_cone(cone):
    
    """
    A superclass for the group LASSO and group LASSO dual epigraph cones.
    """

    seminorm_class = group_lasso

    def __init__(self, 
                 groups,
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):

        groups = np.asarray(groups)
        shape = groups.shape[0]+1
        cone.__init__(self, shape, offset=offset,
                      quadratic=quadratic,
                      initial=initial)
        cls = self.__class__
        self.snorm = cls.seminorm_class(groups,
                                        weights=weights,
                                        offset=offset,
                                        lagrange=1,
                                        bound=None,
                                        quadratic=None,
                                        initial=None)

    def __copy__(self):
        return self.__class__(copy(self.groups),
                              weights=copy(self.weights),
                              offset=copy(self.offset),
                              initial=copy(self.coefs),
                              quadratic=copy(self.quadratic))
    

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

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            new_atom = cls(self.groups,
                           weights=copy(self.weights),
                           offset=offset,
                           quadratic=outq)
        else:
            new_atom = smooth_conjugate(self)
        self._conjugate = new_atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @property
    def weights(self):
        return self.snorm.weights

    @property
    def groups(self):
        return self.snorm.groups

    @doc_template_user
    def constraint(self, arg):
        incone = self.snorm.seminorm(arg[:-1], lagrange=1) <= (1 + self.tol) * arg[-1]
        if incone:
            return 0
        return np.inf


@objective_doc_templater()
class group_lasso_epigraph(group_lasso_cone):

    """
    The epigraph of the group lasso seminorm.
    """

    objective_template = (r"""I^{\infty}\left(\sum_g \|%(var)s[g]\|_2 """
                          + r"""\leq %(var)s[-1]\right)""")
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'group_lasso_epigraph'
    objective_vars['dualconeklass'] = 'group_lasso_epigraph_polar'
    objective_vars['initargs'] = '[1,1,2,2,2]'

    seminorm_class = group_lasso

    @doc_template_user
    def cone_prox(self, arg, lipschitz=1):
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self.snorm._norms_cython[:] = 0
        self.snorm._factors_cython[:] = 0
        self.snorm._projection_cython[:] = 0

        return mixed_lasso_epigraph(arg,
                                    self.snorm._l1_cython,
                                    self.snorm._unpenalized_cython,
                                    self.snorm._positive_part_cython,
                                    self.snorm._nonnegative_cython,
                                    self.snorm._group_array,
                                    self.snorm._weight_array,
                                    self.snorm._norms_cython,
                                    self.snorm._factors_cython,
                                    self.snorm._projection_cython)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return group_lasso_cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class group_lasso_epigraph_polar(group_lasso_cone):

    """
    The polar of the epigraph of the group lasso seminorm.
    """

    objective_template = (r"""I^{\infty}(\max_g \|%(var)s[g]\|_2 \leq """
                          + r"""-%(var)s[-1]\)""")
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'group_lasso_epigraph_polar'
    objective_vars['dualconeklass'] = 'group_lasso_epigraph'
    objective_vars['initargs'] = '[1,1,2,2,2]'

    seminorm_class = group_lasso_dual

    @doc_template_user
    def cone_prox(self, arg, lipschitz=1):
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self.snorm._norms_cython[:] = 0
        self.snorm._factors_cython[:] = 0
        self.snorm._projection_cython[:] = 0

        return arg - mixed_lasso_epigraph(arg,
                                          self.snorm._l1_cython,
                                          self.snorm._unpenalized_cython,
                                          self.snorm._positive_part_cython,
                                          self.snorm._nonnegative_cython,
                                          self.snorm._group_array,
                                          self.snorm._weight_array,
                                          self.snorm._norms_cython,
                                          self.snorm._factors_cython,
                                          self.snorm._projection_cython)

    @doc_template_user
    def constraint(self, arg):
        incone = self.snorm.seminorm(arg[:-1], lagrange=1) <= (1 + self.tol) * (-arg[-1])
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return group_lasso_cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class group_lasso_dual_epigraph(group_lasso_cone):

    """
    The group LASSO conjugate epigraph constraint.
    """

    objective_template = (r"""I^{\infty}(%(var)s: \max_g """ + 
                          r"""\|%(var)s[1:][g]\|_2 \leq %(var)s[0])""")

    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'group_lasso_dual_epigraph'
    objective_vars['dualconeklass'] = 'group_lasso_dual_epigraph_polar'
    objective_vars['initargs'] = '[1,1,2,2,2]'

    seminorm_class = group_lasso_dual

    @doc_template_user
    def cone_prox(self, arg, lipschitz=1):
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self.snorm._norms_cython[:] = 0
        self.snorm._factors_cython[:] = 0
        self.snorm._projection_cython[:] = 0

        return arg + mixed_lasso_epigraph(-arg,
                                           self.snorm._l1_cython,
                                           self.snorm._unpenalized_cython,
                                           self.snorm._positive_part_cython,
                                           self.snorm._nonnegative_cython,
                                           self.snorm._group_array,
                                           self.snorm._weight_array,
                                           self.snorm._norms_cython,
                                           self.snorm._factors_cython,
                                           self.snorm._projection_cython)

    @doc_template_user
    def constraint(self, arg):
        incone = self.snorm.seminorm(arg[:-1], lagrange=1) <= (1 + self.tol) * (arg[-1])
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return group_lasso_cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class group_lasso_dual_epigraph_polar(group_lasso_cone):

    """
    The polar of the group LASSO dual epigraph constraint.
    """

    objective_template = (r"""I^{\infty}(%(var)s: \sum_g \|%(var)s[g]\|_2 \leq """
                          + r"""-%(var)s[-1]\}}""")

    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'group_lasso_dual_epigraph_polar'
    objective_vars['dualconeklass'] = 'group_lasso_dual_epigraph'
    objective_vars['initargs'] = '[1,1,2,2,2]'


    seminorm_class = group_lasso

    @doc_template_user
    def cone_prox(self, arg,  lipschitz=1):
        arg = np.asarray(arg, np.float)

        # flush buffers for computations

        self.snorm._norms_cython[:] = 0
        self.snorm._factors_cython[:] = 0
        self.snorm._projection_cython[:] = 0

        return -mixed_lasso_epigraph(-arg,
                                      self.snorm._l1_cython,
                                      self.snorm._unpenalized_cython,
                                      self.snorm._positive_part_cython,
                                      self.snorm._nonnegative_cython,
                                      self.snorm._group_array,
                                      self.snorm._weight_array,
                                      self.snorm._norms_cython,
                                      self.snorm._factors_cython,
                                      self.snorm._projection_cython)

    @doc_template_user
    def constraint(self, arg):
        incone = self.snorm.seminorm(arg[:-1], lagrange=1) <= (1 + self.tol) * (-arg[-1])
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return group_lasso_cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

conjugate_seminorm_pairs = {}
conjugate_seminorm_pairs[group_lasso_dual] = group_lasso
conjugate_seminorm_pairs[group_lasso] = group_lasso_dual

conjugate_cone_pairs = {}
conjugate_cone_pairs[group_lasso_epigraph] = group_lasso_epigraph_polar
conjugate_cone_pairs[group_lasso_epigraph_polar] = group_lasso_epigraph
conjugate_cone_pairs[group_lasso_dual_epigraph_polar] = group_lasso_dual_epigraph
conjugate_cone_pairs[group_lasso_dual_epigraph] = group_lasso_dual_epigraph_polar

# for paths

def strong_set(glasso, 
               lagrange_cur, 
               lagrange_new, 
               grad,
               slope_estimate=1):

    """
    Guess at active groups at 
    lagrange_new based on gradient
    at lagrange_cur.
    """

    dual = glasso.conjugate
    terms = dual.terms(grad)
    candidates = np.nonzero(terms >= ((slope_estimate + 1) * lagrange_new - 
                                      slope_estimate * lagrange_cur))[0]
    return [glasso._group_inv_dict[i] for i in candidates]

def check_KKT(glasso, 
              grad, 
              solution, 
              lagrange, 
              tol=1.e-2):

    """
    Check whether (grad, solution) satisfy
    KKT conditions at a given tolerance.

    Assumes glasso is group_lasso in lagrange form
    so that glasso.lagrange is not None
    """

    failing = False
    norm_soln = np.linalg.norm(solutionn)
    active = glasso.terms(solution) > tol * norm_soln
    inactive_groups = np.nonzero(~active)[0]
    active_groups = np.nozero(active)[0]
    for g in active_groups:
        group = glasso.groups == g
        subgrad_g = -grad[group]
        soln_g = solution[group]
        if np.linalg.norm(subgrad_g) < glasso.lagrange * glasso._weight_array[g] * (1 - tol):
            failing = True
        if np.linalg.norm(subgrad_g / np.linalg.norm(subgrad_g) - soln_g / np.linalg.norm(soln_g)) > tol:
            failing = True
    for g in inactive_groups:
        group = glasso.groups == g
        subgrad_g = -grad[group]
        if np.linalg.norm(subgrad_g) > glasso.lagrange * glasso._weight_array[g] * (1 + tol):
            failing = True
    return failing
