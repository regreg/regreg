"""
Implementation of the sparse group LASSO atom.

This penalty is defined by groups and individual feature weights and is
$$
\beta \mapsto \sum_j \lambda_j |\beta_j| + \sum_g \alpha_g \|\beta_g\|_2
$$

"""

from warnings import warn
from copy import copy
import numpy as np
from scipy import sparse

from .seminorms import seminorm
from .group_lasso import group_lasso
from .weighted_atoms import l1norm as weighted_l1norm
from ..atoms import _work_out_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)
from ..problems.composite import smooth_conjugate

@objective_doc_templater()
class sparse_group_lasso(group_lasso, seminorm):

    """
    Sparse group LASSO
    """

    objective_template = (r""" \left[ \sum_j \alpha_j |%(var)s_j|""" + 
                          r""" + \sum_g \lambda_g \|%(var)s[g]\|_2 \right]""")
    objective_vars = group_lasso.objective_vars.copy()
    objective_vars['normklass'] = 'sparse_group_lasso'
    objective_vars['dualnormklass'] = 'sparse_group_lasso_dual'
    objective_vars['initargs'] = '[1, 1, 2, 2, 2], [0, 0.5, 0.2, 0.2, 0.2]'

    def __init__(self, 
                 groups,
                 lasso_weights, 
                 weights={},
                 lagrange=None, 
                 bound=None, 
                 offset=None, 
                 quadratic=None,
                 initial=None):

         group_lasso.__init__(self, 
                              groups, 
                              weights=weights,
                              lagrange=lagrange,
                              bound=bound,
                              offset=offset,
                              quadratic=quadratic,
                              initial=initial)

         self.lasso_weights = lasso_weights
         self._weighted_l1norm = weighted_l1norm(lasso_weights, 
                                                 lagrange=lagrange,
                                                 bound=bound)
         self._weighted_supnorm = self._weighted_l1norm.conjugate

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):

        lagrange = seminorm.seminorm(self, x, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)

        val = group_lasso.seminorm(self, 
                                   x,
                                   lagrange=lagrange,
                                   check_feasibility=check_feasibility)

        val += self._weighted_l1norm.seminorm(x,
                                              lagrange=lagrange,
                                              check_feasibility=check_feasibility)

        return val

    @doc_template_user
    def constraint(self, x, bound=None):
         bound = seminorm.constraint(self, x, bound=bound)
         inbox = self.seminorm(x, 
                               lagrange=1,
                               check_feasibility=True) <= bound * (1+self.tol)
         if inbox:
             return 0
         else:
             return np.inf

    @doc_template_user
    def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
         lagrange = seminorm.lagrange_prox(self, x, lipschitz, lagrange)

         # this gives residual from projection onto the 
         # weighted supnorm ball

         initial_proj = self._weighted_l1norm.lagrange_prox(x, 
                                                            lipschitz, 
                                                            lagrange)

         # each group component is then appropriately
         # group lasso soft-thresholded

         final_proj = group_lasso.lagrange_prox(self, 
                                                initial_proj,
                                                lipschitz,
                                                lagrange)

         return final_proj

    @doc_template_user
    def bound_prox(self, x, bound=None):
        raise NotImplementedError

    def __copy__(self):
         return self.__class__(self.groups.copy(),
                               self.lasso_weights.copy(),
                               weights=copy(self.weights),
                               quadratic=self.quadratic,
                               initial=self.coefs,
                               bound=copy(self.bound),
                               lagrange=copy(self.lagrange),
                               offset=copy(self.offset))
    def __repr__(self):
        if self.lagrange is not None:
            if not self.quadratic.iszero:
                return "%s(%s, %s, weights=%s, lagrange=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     str(self.groups),
                     str(self.lasso_weights),
                     str(self.weights),
                     self.lagrange,
                     str(self.offset))
            else:
                return "%s(%s, %s, weights=%s, lagrange=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__, 
                     str(self.groups),
                     str(self.lasso_weights),
                     str(self.weights),
                     self.lagrange,
                     str(self.offset),
                     self.quadratic)
        else:
            if not self.quadratic.iszero:
                return "%s(%s, %s, weights=%s, bound=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     str(self.groups),
                     str(self.lasso_weights),
                     str(self.weights),
                     self.bound,
                     str(self.offset))
            else:
                return "%s(%s, %s, weights=%s, bound=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     str(self.groups),
                     str(self.lasso_weights),
                     str(self.weights),
                     self.bound,
                     str(self.offset),
                     self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            if self.bound is None:
                cls = conjugate_sparse_group_lasso_pairs[self.__class__]
                atom = cls(self.groups,
                           self.lasso_weights,
                           weights=self.weights, 
                           bound=self.lagrange, 
                           lagrange=None,
                           offset=offset,
                           quadratic=outq)
            else:
                cls = conjugate_sparse_group_lasso_pairs[self.__class__]
                atom = cls(self.groups,
                           self.lasso_weights,
                           weights=self.weights, 
                           lagrange=self.bound, 
                           bound=None,
                           offset=offset,
                           quadratic=outq)
        else:
            atom = smooth_conjugate(self)
            
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

@objective_doc_templater()
class sparse_group_lasso_dual(sparse_group_lasso):

     r"""
     The dual of the slope penalty:math:`\ell_{\infty}` norm
     """

     objective_template = r"""{\cal D}^{group sparse}(%(var)s)"""
     objective_vars = seminorm.objective_vars.copy()

     @property
     def base_conjugate(self):
         """
         For a sparse group LASSO dual in bound form,
         an atom that agrees with its primal Lagrange form.
         
         Note: this conjugate does not take into account the atom's quadratic
         and offset terms. 
         """
         if not hasattr(self, "_conjugate_atom"):
              self._base_conjugate_atom = sparse_group_lasso(  
                                             self.groups,
                                             self.lasso_weights,
                                             weights=self.weights,
                                             lagrange=self.bound)
         return self._base_conjugate_atom

     @doc_template_user
     def seminorm(self, x, lagrange=None, check_feasibility=False):
         lagrange = seminorm.seminorm(self, x, 
                                      check_feasibility=check_feasibility, 
                                      lagrange=lagrange)
         return _gauge_function_dual(self.base_conjugate, x) * lagrange

     @doc_template_user
     def constraint(self, x, bound=None):
         bound = seminorm.constraint(self, x, bound=bound)
         return inside_set(self.base_conjugate, x)

     @doc_template_user
     def lagrange_prox(self, x,  lipschitz=1, lagrange=None):
         raise NotImplementedError

     @doc_template_user
     def bound_prox(self, x, bound=None):
          bound = seminorm.bound_prox(self, x, bound)
          resid = self.base_conjugate.lagrange_prox(x, lagrange=bound)
          return x - resid

def _gauge_function_dual(atom,
                         point,
                         tol=1.e-6,
                         max_iter=50): 

     """
     Work out dual norm of sparse group LASSO by binary search.

     NOTE: will have problems if the atom has infinite feature weights
     """

     point = np.asarray(point)

     # find upper and lower bounds

     lower, upper = 1., 1.
     point_inside = inside_set(atom, point)
     
     iter = 0
     if point_inside:
          # gauge is upper bounded by 1
          # find a lower bound

          while True:
               lower = lower / 2
               candidate = point / lower

               if not inside_set(atom, candidate):
                    break
               else:
                    upper = lower

               iter += 1
               if iter == max_iter:
                    return 0
     else:
          # gauge is lower bounded by 1
          # find an upper bound

          while True:
               upper *= 2
               candidate = point / upper

               if inside_set(atom, candidate):
                    break
               else:
                    lower = upper

               iter += 1
               if iter == max_iter:
                    return np.inf

     # binary search

     assert (not inside_set(atom, point / lower))
     assert inside_set(atom, point / upper)

     while (upper - lower) > tol * 0.5 * (upper + lower):
         mid = 0.5 * (upper + lower)
         candidate = point / mid
         if inside_set(atom, candidate):
             upper = mid
         else:
             lower = mid
     return mid

def inside_set(atom, point):
    """
    Is the point in the dual ball?

    If the atom is the primal (necessarily in lagrange form), we check
    whether point is in the dual ball (i.e.
    the one determining the norm.

    If the atom is a dual (necessarily in bound form), we just check
    if we project onto the same point.
    """

    if atom.__class__ == sparse_group_lasso:
        proj_point = point - atom.lagrange_prox(point,
                                                lipschitz=1,
                                                lagrange=1)
    elif atom.__class__ == sparse_group_lasso_dual:
        proj_point = atom.bound_prox(point)
    else:
        raise ValueError('must be a sparse group lasso class')

    if np.linalg.norm(proj_point - point) > max(np.linalg.norm(point), 1) * 1.e-7:
        return False
    return True

conjugate_sparse_group_lasso_pairs = {}
for n1, n2 in [(sparse_group_lasso, sparse_group_lasso_dual)]:
    conjugate_sparse_group_lasso_pairs[n1] = n2
    conjugate_sparse_group_lasso_pairs[n2] = n1
