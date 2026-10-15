r"""
Mixed LASSO penalties.

A mixed LASSO penalty combines, on different coordinates of the same
vector, a group LASSO penalty, an :math:`\ell_1` penalty, a positive
part penalty, a nonnegativity constraint or no penalty at all. The
treatment of each coordinate is specified by a `penalty_structure` array
of labels: the reserved labels `UNPENALIZED`, `L1_PENALTY`,
`POSITIVE_PART` and `NONNEGATIVE` (all negative integers) select the
corresponding treatment, and any other label is the name of a group.
"""
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
                                     mixed_lasso_dual_bound_prox,
                                     mixed_lasso_bound_prox,
                                     seminorm_mixed_lasso,
                                     seminorm_mixed_lasso_dual,
                                     strong_set_mixed_lasso,
                                     mixed_lasso_epigraph,
                                     check_KKT_mixed_lasso)
except ImportError:
    raise ImportError('need cython module mixed_lasso_cython')

@objective_doc_templater()
class mixed_lasso(atom):

    r"""
    Mixed LASSO penalty in Lagrange form.

    The penalty is

    .. math::

       \lambda \left(\sum_g w_g \|\beta_g\|_2 + \sum_{j \in L} |\beta_j|
       + \sum_{j \in P} \max(\beta_j, 0) \right)

    where :math:`g` ranges over the groups, :math:`L` is the set of
    coordinates labelled `L1_PENALTY` and :math:`P` the set of coordinates
    labelled `POSITIVE_PART`. Coordinates labelled `UNPENALIZED` are not
    penalized and coordinates labelled `NONNEGATIVE` are constrained to be
    nonnegative. The proximal map also sets negative `POSITIVE_PART`
    coordinates to zero.

    Parameters
    ----------
    penalty_structure : np.ndarray
        One label per coordinate. Labels in `reserved` select the
        special treatments described above; every other label defines
        a group.
    lagrange : float
        Lagrange parameter :math:`\lambda`.
    weights : dict, optional
        Maps group labels to weights :math:`w_g`. Groups not in
        `weights` get weight :math:`\sqrt{|g|}`.
    offset : np.ndarray, optional
        Offset subtracted from the argument before the penalty is
        evaluated.
    quadratic : regreg.identity_quadratic.identity_quadratic, optional
        Quadratic term added to the atom.
    initial : np.ndarray, optional
        Initial value for the coefficients.
    """

    objective_template = 'needs a template'

    tol = 1.0e-05

    def __init__(self, 
                 penalty_structure, 
                 lagrange, 
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):
        shape = np.asarray(penalty_structure).shape
        nonsmooth.__init__(self, shape, offset,
                           quadratic, initial)

        self.weights = weights
        self.lagrange = lagrange
        self.penalty_structure = np.asarray(penalty_structure)
        self._groups = -np.ones(self.shape, np.intp)
        groups = set(np.unique(self.penalty_structure)).difference(
            set(reserved))
        self._weight_array = np.zeros(len(groups))

        self._l1_penalty = np.nonzero(self.penalty_structure == L1_PENALTY)[0].astype(np.intp)
        self._positive_part = np.nonzero(self.penalty_structure == POSITIVE_PART)[0].astype(np.intp) 
        self._unpenalized = np.nonzero(self.penalty_structure == UNPENALIZED)[0].astype(np.intp) 
        self._nonnegative = np.nonzero(self.penalty_structure == NONNEGATIVE)[0].astype(np.intp)

        for idx, label in enumerate(groups):
            g = self.penalty_structure == label
            self._groups[g] = idx
            self._weight_array[idx] = self.weights.get(label, np.sqrt(g.sum()))

        # buffers for cython computations

        self._norms_cython = np.zeros_like(self._weight_array)
        self._factors_cython = np.zeros_like(self._weight_array)
        self._projection_cython = np.zeros(self.shape)

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
            return "%s(%r, %s, weights=%r, offset=%s)" % \
                (self.__class__.__name__,
                 self.penalty_structure,
                 self.lagrange,
                 self.weights,
                 str(self.offset))
        else:
            return "%s(%s, %r, weights=%r, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 self.penalty_structure,
                 self.lagrange,
                 self.weights,
                 str(self.offset),
                 str(self.quadratic))

    @property
    def conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = mixed_lasso_dual
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
        """
        Check whether the seminorm of `x` is at most `bound`.

        As with `seminorm`, the offset is not applied here:
        `nonsmooth_objective` subtracts it before calling this method.

        Parameters
        ----------
        x : np.ndarray
            Argument of the constraint.
        bound : float
            Bound on the seminorm. Must be supplied; a `ValueError`
            is raised if it is None.

        Returns
        -------
        value : float
            0 if the seminorm of `x` is at most `bound`, `np.inf`
            otherwise.
        """
        if bound is None:
            raise ValueError('bound must be suppled')
        if self.seminorm(x, check_feasibility=True) <= bound * (1 + self.tol):
            return 0
        return np.inf

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        The nonsmooth objective of the atom, including its quadratic term.

        Parameters
        ----------
        x : np.ndarray
            Argument of the objective.
        check_feasibility : bool, optional
            If True, return `np.inf` when `x` violates the
            nonnegativity constraints.

        Returns
        -------
        value : float
            Value of the penalty plus `self.quadratic` at `x`.
        """
        x_offset = self.apply_offset(x)
        v = self.seminorm(x_offset, check_feasibility=check_feasibility)
        v += self.quadratic.objective(x, 'func')
        return v

    def seminorm(self, x, check_feasibility=False):
        """
        The mixed LASSO penalty, multiplied by `self.lagrange`.

        Parameters
        ----------
        x : np.ndarray
            Argument of the penalty. The offset is not applied here:
            `nonsmooth_objective` subtracts it before calling this method.
        check_feasibility : bool, optional
            If True, return `np.inf` when the `NONNEGATIVE` or
            `POSITIVE_PART` coordinates are not (strictly) positive.

        Returns
        -------
        value : float
            Value of the penalty.
        """
        # flush buffers for computations

        self._norms_cython[:] = 0

        v = seminorm_mixed_lasso(x,
                                 self._l1_penalty,
                                 self._unpenalized,
                                 self._positive_part,
                                 self._nonnegative,
                                 self._groups,
                                 self._weight_array,
                                 self._norms_cython,
                                 int(check_feasibility))

        return v * self.lagrange

    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator.

        The atom is always in Lagrange mode, so the proximal map is

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \lambda h(v-\alpha) + \langle v, \eta \rangle

        where :math:`h` is the mixed LASSO penalty, :math:`\alpha` is
        `self.offset`, and :math:`L`, :math:`x` and :math:`\eta` are the
        coefficient, center and linear term of the sum of `proxq` and
        `self.quadratic`.

        Parameters
        ----------
        proxq : regreg.identity_quadratic.identity_quadratic
            A quadratic added to the atom before minimizing.
        prox_control : dict, optional
            Ignored.

        Returns
        -------
        Z : np.ndarray
            The proximal map.
        """

        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        # flush buffers for computations

        self._norms_cython[:] = 0
        self._factors_cython[:] = 0
        self._projection_cython[:] = 0

        eta = mixed_lasso_lagrange_prox(prox_arg, 
                                        self.lagrange, 
                                        totalq.coef, 
                                        self._l1_penalty,
                                        self._unpenalized,
                                        self._positive_part,
                                        self._nonnegative,
                                        self._groups,
                                        self._weight_array,
                                        self._norms_cython,
                                        self._factors_cython,
                                        self._projection_cython)

        if offset is None:
            return eta
        else:
            return eta + offset


@objective_doc_templater()
class mixed_lasso_dual(mixed_lasso):

    r"""
    Conjugate of the mixed LASSO penalty, in bound form only.

    This is the indicator of the set where the dual seminorm

    .. math::

       h^*(u) = \max\left(\max_g \frac{\|u_g\|_2}{w_g},
       \max_{j \in L} |u_j|, \max_{j \in P} \max(u_j, 0)\right)

    is at most `bound` (see `mixed_lasso` for notation).

    Parameters
    ----------
    penalty_structure : np.ndarray
        One label per coordinate, as in `mixed_lasso`.
    bound : float
        Bound on the dual seminorm.
    weights : dict, optional
        Maps group labels to weights :math:`w_g`. Groups not in
        `weights` get weight :math:`\sqrt{|g|}`.
    offset : np.ndarray, optional
        Offset subtracted from the argument before the constraint is
        evaluated.
    quadratic : regreg.identity_quadratic.identity_quadratic, optional
        Quadratic term added to the atom.
    initial : np.ndarray, optional
        Initial value for the coefficients.
    """

    _doc_dict = {'linear':r' + \langle \eta, x \rangle',
                 'constant':r' + \tau',
                 'objective': '',
                 'shape':'p',
                 'var':r'x'}

    tol = 1.0e-05

    def __init__(self, 
                 penalty_structure, 
                 bound, 
                 weights={},
                 offset=None,
                 quadratic=None,
                 initial=None):

        mixed_lasso.__init__(self, 
                             penalty_structure, 
                             bound, 
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
            return "%s(%s, %r, weights=%r, offset=%s)" % \
                (self.__class__.__name__,
                 self.bound,
                 self.penalty_structure,
                 self.weights,
                 str(self.offset))
        else:
            return "%s(%s, %r, weights=%r, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 self.bound, 
                 self.penalty_structure,
                 self.weights,
                 str(self.offset),
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
        if self.seminorm(x, check_feasibility=True) <= bound * (1 + self.tol):
            return 0
        return np.inf

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        The nonsmooth objective of the atom, including its quadratic term.

        Parameters
        ----------
        x : np.ndarray
            Argument of the objective.
        check_feasibility : bool, optional
            If True, include the result of `self.constraint` (0 if
            feasible, `np.inf` otherwise); otherwise the constraint term
            is taken to be 0.

        Returns
        -------
        value : float
            Value of the objective at `x`.
        """
        x_offset = self.apply_offset(x)
        if check_feasibility:
            v = self.constraint(x_offset, self.bound)
        else:
            v = 0
        v += self.quadratic.objective(x, 'func')
        return v

    def seminorm(self, x, lagrange=1, check_feasibility=False):
        """
        The dual seminorm of the mixed LASSO penalty.

        Parameters
        ----------
        x : np.ndarray
            Argument of the dual seminorm. The offset is not applied
            here: `nonsmooth_objective` subtracts it before calling
            `constraint`.
        lagrange : float, optional
            Ignored.
        check_feasibility : bool, optional
            If True, return `np.inf` when `x` is outside the dual set: a
            nonzero unpenalized coordinate or group of weight 0, a negative
            positive-part coordinate, or a positive nonnegative one.

        Returns
        -------
        value : float
            Value of the dual seminorm.
        """
        # flush buffers for computations

        self._norms_cython[:] = 0

        v = seminorm_mixed_lasso_dual(np.asarray(x, float), 
                                      self._l1_penalty,
                                      self._unpenalized,
                                      self._positive_part,
                                      self._nonnegative,
                                      self._groups,
                                      self._weight_array,
                                      self._norms_cython,
                                      check_feasibility=int(check_feasibility))
        return v 

    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator.

        The atom is always in bound mode, so the proximal map is

        .. math::

           v^{\delta}(x) = \text{argmin}_{v \in \mathbb{R}^p} \frac{L}{2}
           \|x-v\|^2_2 + \langle v, \eta \rangle \ \text{s.t.} \
           h^*(v-\alpha) \leq \delta

        where :math:`h^*` is the dual seminorm, :math:`\delta` is
        `self.bound`, :math:`\alpha` is `self.offset`, and :math:`L`,
        :math:`x` and :math:`\eta` are the coefficient, center and linear
        term of the sum of `proxq` and `self.quadratic`.

        Parameters
        ----------
        proxq : regreg.identity_quadratic.identity_quadratic
            A quadratic added to the atom before minimizing.
        prox_control : dict, optional
            Ignored.

        Returns
        -------
        Z : np.ndarray
            The proximal map.
        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        # flush buffers for computations

        self._norms_cython[:] = 0
        self._factors_cython[:] = 0
        self._projection_cython[:] = 0

        eta = mixed_lasso_dual_bound_prox(prox_arg, 
                                          self.bound, 
                                          self._l1_penalty,
                                          self._unpenalized,
                                          self._positive_part,
                                          self._nonnegative,
                                          self._groups,
                                          self._weight_array,
                                          self._norms_cython,
                                          self._factors_cython,
                                          self._projection_cython)

        if offset is None:
            return eta
        else:
            return eta + offset

def strong_set(glasso, 
               lagrange_cur, 
               lagrange_new, 
               grad,
               slope_estimate=1):

    """
    Guess the active variables at a new value of the Lagrange parameter.

    Uses the strong rules, based on the gradient at the current value of
    the Lagrange parameter, to guess which variables are active at
    `lagrange_new`.

    Parameters
    ----------
    glasso : mixed_lasso
        The penalty.
    lagrange_cur : float
        Current value of the Lagrange parameter.
    lagrange_new : float
        New value of the Lagrange parameter.
    grad : np.ndarray
        Gradient of the smooth part of the objective at the solution
        for `lagrange_cur`.
    slope_estimate : float, optional
        Estimate of the slope used by the strong rules.

    Returns
    -------
    value : np.ndarray
        Boolean array, True for the variables guessed to be active.
    selector : regreg.affine.selector
        Selector of the guessed active variables.
    """

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
    value = value.astype(bool)
    return value, selector(value, (p,))

def check_KKT(glasso, 
              grad, 
              solution, 
              lagrange, 
              tol=1.e-2):

    """
    Check whether (grad, solution) satisfy the KKT conditions.

    Parameters
    ----------
    glasso : mixed_lasso
        The penalty.
    grad : np.ndarray
        Gradient of the smooth part of the objective at `solution`.
    solution : np.ndarray
        Candidate solution.
    lagrange : float
        Lagrange parameter.
    tol : float, optional
        Relative tolerance for the checks.

    Returns
    -------
    failing : np.ndarray
        Boolean array, True for the coordinates where the KKT
        conditions fail.
    """

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

conjugate_mixed_lasso_pairs = {}
for n1, n2 in [(mixed_lasso, mixed_lasso_dual)]:
    conjugate_mixed_lasso_pairs[n1] = n2
    conjugate_mixed_lasso_pairs[n2] = n1


