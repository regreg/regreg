"""
Implementation of the SLOPE penalty and its proximal operator.

See Bogdan, van den Berg, Sabatti, Su and Candes, "SLOPE -- adaptive
variable selection via convex optimization",
https://statweb.stanford.edu/~candes/papers/SLOPE.pdf
"""
from warnings import warn
from copy import copy
import numpy as np
from scipy import sparse

try:
     from sklearn.isotonic import isotonic_regression as isotonic_regression_sk
     have_sklearn_iso = True
except ImportError:
     warn('unable to import isotonic regression from sklearn, using a pure python implementation')
     have_sklearn_iso = False
from ._isotonic_py import pav as isotonic_regression_py

from .seminorms import seminorm

from ..atoms import _work_out_conjugate
from ..problems.composite import smooth_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

@objective_doc_templater()
class slope(seminorm):

    r"""
    The SLOPE penalty.

    The sorted :math:`\ell_1` norm :math:`\sum_j w_j |\beta|_{(j)}`,
    where :math:`|\beta|_{(1)} \geq |\beta|_{(2)} \geq \dots` are the
    sorted absolute values of :math:`\beta`.

    Parameters
    ----------
    weights : np.ndarray
        Positive, non-increasing weights :math:`w_j`. The shape of the
        atom is the shape of `weights`.
    lagrange : float, optional
        Lagrange parameter (exactly one of `lagrange` and `bound` must be
        specified).
    bound : float, optional
        Bound parameter.
    offset : np.ndarray, optional
        Offset subtracted from the argument before the penalty is
        evaluated.
    quadratic : identity_quadratic, optional
        Quadratic term added to the penalty.
    initial : np.ndarray, optional
        Initial value of the coefficients.

    Notes
    -----
    Only the Lagrange form has a proximal map implemented: `bound_prox`
    raises ``NotImplementedError``.
    """

    use_sklearn = have_sklearn_iso

    objective_template = r"""\sum_j \lambda_j |%(var)s_{(j)}|"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'slope'
    objective_vars['dualnormklass'] = 'slope_conjugate'
    objective_vars['initargs'] = '[3., 2., 1.]'

    def __init__(self, weights, 
                 lagrange=None, 
                 bound=None, 
                 offset=None, 
                 quadratic=None,
                 initial=None):

         weights = np.array(weights, float)
         if not np.allclose(-weights, np.sort(-weights)):
              raise ValueError('weights should be non-increasing')
         if not np.all(weights > 0):
              raise ValueError('weights must be positive')
         
         self.weights = weights
         self._dummy = np.arange(self.weights.shape[0])
         
         seminorm.__init__(self, self.weights.shape,
                           lagrange=lagrange,
                           bound=bound,
                           quadratic=quadratic,
                           initial=initial,
                           offset=offset)

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
         lagrange = seminorm.seminorm(self, x, 
                                      check_feasibility=check_feasibility, 
                                      lagrange=lagrange)
         xsort = np.sort(np.fabs(x))[::-1]
         return lagrange * np.fabs(xsort * self.weights).sum()

    @doc_template_user
    def constraint(self, x, bound=None):
         bound = seminorm.constraint(self, x, bound=bound)
         inbox = self.seminorm(x, lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
         if inbox:
              return 0
         else:
              return np.inf

    @doc_template_user
    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
         lagrange = seminorm.lagrange_prox(self, x, lipschitz, lagrange)
         return _basic_proximal_map(x, self.weights * lagrange / lipschitz, use_sklearn=self.use_sklearn)
    
    @doc_template_user
    def bound_prox(self, x, bound=None):
         raise NotImplementedError

    def __copy__(self):
         return self.__class__(self.weights.copy(),
                               quadratic=self.quadratic,
                               initial=self.coefs,
                               bound=copy(self.bound),
                               lagrange=copy(self.lagrange),
                               offset=copy(self.offset))
    
    def __repr__(self):
         if self.lagrange is not None:
              if self.quadratic.iszero:
                   return "%s(%s, lagrange=%f, offset=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.lagrange,
                        str(self.offset))
              else:
                   return "%s(%s, lagrange=%f, offset=%s, quadratic=%s)" % \
                       (self.__class__.__name__, 
                        str(self.weights),
                        self.lagrange,
                        str(self.offset),
                        self.quadratic)
         else:
              if self.quadratic.iszero:
                   return "%s(%s, bound=%f, offset=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.bound,
                        str(self.offset))
              else:
                   return "%s(%s, bound=%f, offset=%s, quadratic=%s)" % \
                       (self.__class__.__name__,
                        str(self.weights),
                        self.bound,
                        str(self.offset),
                        self.quadratic)
              

    @doc_template_user
    def get_conjugate(self):
         if self.quadratic.coef == 0:
              offset, outq = _work_out_conjugate(self.offset, self.quadratic)
              if self.bound is None:
                   cls = conjugate_slope_pairs[self.__class__]
                   atom = cls(self.weights, 
                              bound=self.lagrange, 
                              lagrange=None,
                              offset=offset,
                              quadratic=outq)
              else:
                   cls = conjugate_slope_pairs[self.__class__]
                   atom = cls(self.weights,
                              lagrange=self.bound, 
                              bound=None,
                              offset=offset,
                              quadratic=outq)
         else:
              atom = smooth_conjugate(self)

         self._conjugate = atom
         self._conjugate._conjugate = self
         return self._conjugate

@objective_doc_templater()
class slope_conjugate(slope):

     r"""
     The dual norm of the SLOPE penalty.

     For weights :math:`w`, this is
     :math:`\max_k \left(\sum_{j \leq k} |\beta|_{(j)}\right) /
     \left(\sum_{j \leq k} w_j\right)`.

     Notes
     -----
     Only the bound form has a proximal map implemented, computed from
     the SLOPE proximal map via the Moreau decomposition:
     `lagrange_prox` raises ``NotImplementedError``.
     """

     objective_template = r"""{\cal D}^{SLOPE}(%(var)s)"""
     objective_vars = slope.objective_vars.copy()
     objective_vars['normklass'] = 'slope_conjugate'
     objective_vars['dualnormklass'] = 'slope'

     @doc_template_user
     def seminorm(self, x, lagrange=None, check_feasibility=False):
          lagrange = seminorm.seminorm(self, x, 
                                       check_feasibility=check_feasibility, 
                                       lagrange=lagrange)
          xsort = np.sort(np.fabs(x))[::-1]
          xsort_cumsum = np.cumsum(xsort)
          w_cumsum = np.cumsum(self.weights)
          return lagrange * np.fabs(xsort_cumsum / w_cumsum).max()

     @doc_template_user
     def constraint(self, x, bound=None):
          bound = seminorm.constraint(self, x, bound=bound)
          inbox = self.seminorm(x, lagrange=1,
                                check_feasibility=True) <= bound * (1+self.tol)
          if inbox:
               return 0
          else:
               return np.inf

     @doc_template_user
     def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
          raise NotImplementedError

     @doc_template_user
     def bound_prox(self, x, bound=None):
          bound = seminorm.bound_prox(self, x, bound)
          
          # the proximal map is evaluated
          # by working out the SLOPE proximal
          # map and computing the residual
          
          _slope_prox = _basic_proximal_map(x, self.weights * bound, use_sklearn=self.use_sklearn)
          return x - _slope_prox

def _basic_proximal_map(center, weights, use_sklearn=have_sklearn_iso):
     """
     Compute the SLOPE proximal map.

     Uses the algorithm described in (2.3) of the SLOPE paper, though
     sklearn's isotonic regression has the ordering reversed.

     Parameters
     ----------
     center : np.ndarray
         Center of the proximal map.
     weights : np.ndarray
         Non-increasing, nonnegative weights (already scaled by the
         Lagrange parameter and Lipschitz constant).
     use_sklearn : bool, optional
         If True, use `sklearn.isotonic.isotonic_regression`; otherwise
         use the pure Python implementation in `_isotonic_py`.

     Returns
     -------
     prox : np.ndarray
         The proximal map of `center`.
     """
     
     # the proximal map sorts the absolute values,
     # runs isotonic regression with an offset
     # reassigns the signs
     
     _arg = np.argsort(np.fabs(center))
     shifted_center = np.array(np.fabs(center)[_arg] - weights[::-1]).astype(float)
     if use_sklearn:
          _prox_val = np.clip(isotonic_regression_sk(shifted_center), 0, np.inf)
     else:
          _prox_val = np.clip(isotonic_regression_py(shifted_center), 0, np.inf)
     _return_val = np.zeros_like(_prox_val)
     _return_val[_arg] = _prox_val
     _return_val *= np.sign(center)
     return _return_val

conjugate_slope_pairs = {}
for n1, n2 in [(slope, slope_conjugate)]:
     conjugate_slope_pairs[n1] = n2
     conjugate_slope_pairs[n2] = n1
