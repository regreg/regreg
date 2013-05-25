from copy import copy
import warnings

import numpy as np

from ..problems.composite import composite, nonsmooth, smooth_conjugate
from ..affine import linear_transform, identity as identity_transform, selector
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate, atom
from ..smooth import affine_smooth

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

# Constants used below

UNPENALIZED = -1
L1_PENALTY = -2
POSITIVE_PART = -3
NONNEGATIVE = -4

reserved = [UNPENALIZED, L1_PENALTY, POSITIVE_PART,
            NONNEGATIVE]

try:
    from .mixed_lasso_cython import (mixed_lasso_lagrange_prox, 
                                     mixed_lasso_conjugate_bound_prox,
                                     mixed_lasso_bound_prox, 
                                     seminorm_mixed_lasso,
                                     seminorm_mixed_lasso_conjugate,
                                     strong_set_mixed_lasso,
                                     check_KKT_mixed_lasso)
except ImportError:
    raise ImportError('need cython module mixed_lasso_cython')

@objective_doc_templater()
class mixed_lasso(atom):

    objective_template = 'needs a template'

    """
    A class that defines the API for cone constraints.
    """
    tol = 1.0e-05

    def __init__(self, penalty_structure, lagrange, 
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):
        shape = np.asarray(penalty_structure).shape
        nonsmooth.__init__(self, shape, offset,
                           quadratic, initial)

        self.weights = weights
        self.lagrange = lagrange
        self.penalty_structure = penalty_structure
        self._groups = -np.ones(self.shape, np.int)
        groups = set(np.unique(self.penalty_structure)).difference(
            set(reserved))
        self._weight_array = np.zeros(len(groups))

        self._l1_penalty = np.nonzero(self.penalty_structure == L1_PENALTY)[0]
        self._positive_part = np.nonzero(self.penalty_structure == POSITIVE_PART)[0]
        self._unpenalized = np.nonzero(self.penalty_structure == UNPENALIZED)[0]
        self._nonnegative = np.nonzero(self.penalty_structure == NONNEGATIVE)[0]

        for idx, label in enumerate(groups):
            g = self.penalty_structure == label
            self._groups[g] = idx
            self._weight_array[idx] = self.weights.get(label, np.sqrt(g.sum()))

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.shape == other.shape and 
                    np.all(self.penalty_structure == other.penalty_structure)
                    and np.all(self.weights == other.weights)
                    and self.lagrange == other.lagrange)
        return False

    def __copy__(self):
        return self.__class__(copy(self.penalty_structure),
                              self.lagrange,
                              weights=self.weights,
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, weights=%s, offset=%s)" % \
                (self.__class__.__name__,
                 self.lagrange,
                 `self.penalty_structure`,
                 `self.weights`,
                 str(self.offset))
        else:
            return "%s(%s, %s, weights=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 self.lagrange, 
                 `self.penalty_structure`,
                 `self.weights`,
                 str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = mixed_lasso_conjugate
            atom = cls(self.penalty_structure,
                       self.lagrange,
                       weights=self.weights,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @property
    def dual(self):
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    
    @doc_template_provider
    def constraint(self, x, bound=None):
        r"""
        Verify :math:`\cdot %(objective)s \leq \lambda`, where :math:`\lambda`
        is bound, :math:`\alpha` is self.offset (if any).

        If True, returns 0, else returns np.inf.

        The class atom's constraint just returns the appropriate bound
        parameter for use by the subclasses.
        """
        if bound is None:
            raise ValueError('bound must be suppled')
        x_offset = self.apply_offset(x)
        return self.seminorm(x_offset) <= bound

    def nonsmooth_objective(self, x, check_feasibility=False):
        x_offset = self.apply_offset(x)
        v = self.seminorm(x_offset, check_feasibility=check_feasibility)
        v += self.quadratic.objective(x, 'func')
        return v

    def seminorm(self, x, check_feasibility=False):
        x_offset = self.apply_offset(x)
        v = seminorm_mixed_lasso(x_offset,
                                 self._l1_penalty,
                                 self._unpenalized,
                                 self._positive_part,
                                 self._nonnegative,
                                 self._groups, 
                                 self._weight_array,
                                 int(check_feasibility))
        return v * self.lagrange

    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """

        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        eta = mixed_lasso_lagrange_prox(prox_arg, self.lagrange, totalq.coef, 
                               self._l1_penalty,
                               self._unpenalized,
                               self._positive_part,
                               self._nonnegative,
                               self._groups, 
                               self._weight_array)

        if offset is None:
            return eta
        else:
            return eta - offset


@objective_doc_templater()
class mixed_lasso_conjugate(mixed_lasso):

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    """
    Conjugate of the group lasso seminorm (in bound form only for now).
    """
    tol = 1.0e-05

    def __init__(self, penalty_structure, bound, 
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):

        mixed_lasso.__init__(self, penalty_structure, bound, 
                             weights=weights,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)
        del(self.lagrange)
        self.bound = bound

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.shape == other.shape and 
                    np.all(self.penalty_structure == other.penalty_structure)
                    and np.all(self.weights == other.weights)
                    and self.bound == other.bound)
        return False

    def __copy__(self):
        return self.__class__(copy(self.penalty_structure),
                              self.bound,
                              weights=self.weights,
                              offset=copy(self.offset),
                              initial=self.coefs,
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, weights=%s, offset=%s)" % \
                (self.__class__.__name__,
                 self.bound,
                 `self.penalty_structure`,
                 `self.weights`,
                 str(self.offset))
        else:
            return "%s(%s, %s, weights=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 self.bound, 
                 `self.penalty_structure`,
                 `self.weights`,
                 str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = mixed_lasso
            atom = cls(self.penalty_structure,
                       self.bound,
                       weights=self.weights,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @property
    def dual(self):
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    
    @doc_template_user
    def constraint(self, x, bound=None):
        if bound is None:
            raise ValueError('bound must be suppled')
        x_offset = self.apply_offset(x)
        return self.seminorm(x_offset) <= bound

    def nonsmooth_objective(self, x, check_feasibility=False):
        x_offset = self.apply_offset(x)
        if check_feasibility:
            v = self.constraint(x_offset, self.bound)
        v += self.quadratic.objective(x, 'func')
        return v

    def seminorm(self, x, lagrange=1, check_feasibility=False):
        x_offset = self.apply_offset(x)
        v = seminorm_mixed_lasso_conjugate(x_offset,
                                 self._l1_penalty,
                                 self._unpenalized,
                                 self._positive_part,
                                 self._nonnegative,          
                                 self._groups, 
                                 self._weight_array)
        return v 

    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator. If the atom is in
        Bound mode, this has the form

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v+\alpha) + \langle v, \eta \rangle

        where :math:`\alpha` is the offset of self.affine_transform and
        :math:`\eta` is self.linear_term.

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \text{s.t.} \   h(v+\alpha) \leq \lambda

        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        eta = mixed_lasso_conjugate_bound_prox(prox_arg, self.bound, 
                                  self._l1_penalty,
                                  self._unpenalized,
                                  self._positive_part,
                                  self._groups, 
                                  self._weight_array)

        if offset is None:
            return eta
        else:
            return eta - offset

def strong_set(glasso, lagrange_cur, lagrange_new, grad,
               slope_estimate=1):

    p = grad.shape[0]
    value = strong_set_mixed_lasso(grad, 
                                   lagrange_new,
                                   lagrange_cur,
                                   slope_estimate,
                                   glasso._l1_penalty, 
                                   glasso._unpenalized,
                                   glasso._positive_part,
                                   glasso._nonnegative,
                                   glasso._groups,
                                   glasso._weight_array)
    value = value.astype(np.bool)
    return value, selector(value, (p,))

def check_KKT(glasso, grad, solution, lagrange, tol=1.e-2):

    failing = check_KKT_mixed_lasso(grad, 
                                    solution, 
                                    lagrange,
                                    glasso._l1_penalty, 
                                    glasso._unpenalized,
                                    glasso._positive_part, 
                                    glasso._nonnegative,
                                    glasso._groups,
                                    glasso._weight_array,
                                    tol=tol)
    return failing > 0
