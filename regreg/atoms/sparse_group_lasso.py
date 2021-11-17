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

         self.lasso_weights = np.asarray(lasso_weights)
         if self.lasso_weights.shape != self.groups.shape:
             self.lasso_weights = self.lasso_weights * np.ones(self.shape)
         self._weighted_l1norm = weighted_l1norm(self.lasso_weights, 
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

    def terms(self, arg):
        """
        Return the args that are summed
        in computing the seminorm.

        >>> import regreg.api as rr
        >>> groups = [1,1,2,2,2]
        >>> l1weights = [1,1,1,1,1]
        >>> penalty = rr.sparse_group_lasso(groups, l1weights, lagrange=1.)
        >>> arg = [2,4,5,3,4]
        >>> list(penalty.terms(arg)) # doctest: +ELLIPSIS
        [12.3245..., 24.2474...]
        >>> penalty.seminorm(arg) # doctest: +ELLIPSIS
        36.5720...
        >>> np.sqrt((2**2 + 4**2)*2) + 6, np.sqrt((5**2 + 3**2 + 4**2) * 3.) + 12 # doctest: +ELLIPSIS
        (12.3245..., 24.2474...)
        >>> np.sqrt((2**2 + 4**2)*2) + np.sqrt((5**2 + 3**2 + 4**2) * 3.) + np.sum(np.fabs(arg)) # doctest: +ELLIPSIS
        36.5720...
        
        """
        arg = np.asarray(arg)
        norms = []
        for g in np.unique(self.groups):
            group = self.groups == g
            arg_g = arg[group]
            term = np.linalg.norm(arg_g) * self.weights[g]
            term += np.fabs(arg_g * self.lasso_weights[group]).sum()
            norms.append(term)
        return np.array(norms)

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
         return gauge_function_dual(self.base_conjugate, x) * lagrange

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

     def terms(self, arg):
         """
         Return the args that are maximized
         in computing the seminorm.
         
         >>> import regreg.api as rr
         >>> groups = [1,1,2,2,2]
         >>> l1weights = [1,1,1,1,1]
         >>> penalty = rr.sparse_group_lasso_dual(groups, l1weights, lagrange=1.)
         >>> arg = [2,4,5,3,4]
         >>> list(penalty.terms(arg)) # doctest: +ELLIPSIS
         [1.666..., 2.0833...]

         """
         arg = np.asarray(arg)
         norms = []
         for g in np.unique(self.groups):
             group = self.groups == g
             arg_g = arg[group]
             term = _gauge_function_dual_strong(arg_g,
                                                self.lasso_weights[group],
                                                self.weights[g])[0]
             norms.append(term)
         return np.array(norms)

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

def gauge_function_dual(atom,
                        point,
                        tol=1.e-10,
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

# bookkeeping for automatically computing conjugates

conjugate_sparse_group_lasso_pairs = {}
for n1, n2 in [(sparse_group_lasso, sparse_group_lasso_dual)]:
    conjugate_sparse_group_lasso_pairs[n1] = n2
    conjugate_sparse_group_lasso_pairs[n2] = n1

# for terms of strong rules

def _inside_set_strong(point, bound, lasso_weights, group_weight, tol=1.e-5):

    # soft_thresh = np.sign(point) * np.maximum(np.fabs(point) - bound * lasso_weights, 0)
    # sign doesn't matter for testing inside the dual ball
    soft_thresh = np.maximum(np.fabs(point) - bound * lasso_weights, 0)
    norm_soft = np.linalg.norm(soft_thresh)
    if norm_soft <= bound * (group_weight + tol):
        return True
    return False

def _gauge_function_dual_strong(point,
                                lasso_weights, 
                                group_weight,
                                tol=1.e-6,
                                max_iter=50): 

     """
     Work out dual norm of sparse group LASSO by binary search.

     NOTE: will have problems if the atom has infinite feature weights
     """

     point = np.asarray(point)

     # find upper and lower bounds

     lower, upper = 1., 1.
     point_inside = _inside_set_strong(point,
                                       lower, 
                                       lasso_weights,
                                       group_weight)
     
     iter = 0
     if point_inside:
          # gauge is upper bounded by 1
          # find a lower bound

          while True:
               lower = lower / 2
               if not _inside_set_strong(point,
                                         lower,
                                         lasso_weights,
                                         group_weight):
                    break
               else:
                    upper = lower

               iter += 1
               if iter == max_iter:
                    return 0, None, None
     else:
          # gauge is lower bounded by 1
          # find an upper bound

          while True:
               upper *= 2
               if _inside_set_strong(point,
                                     upper, 
                                     lasso_weights,
                                     group_weight):
                    break
               else:
                    lower = upper

               iter += 1
               if iter == max_iter:
                    return np.inf, None, None

     # binary search

     assert (not _inside_set_strong(point,
                                    lower,
                                    lasso_weights,
                                    group_weight))
     assert _inside_set_strong(point,
                               upper,
                               lasso_weights,
                               group_weight)

     while (upper - lower) > tol * 0.5 * (upper + lower):
         mid = 0.5 * (upper + lower)
         if _inside_set_strong(point,
                               mid,
                               lasso_weights,
                               group_weight):
             upper = mid
         else:
             lower = mid

     soft_thresh = np.maximum(np.fabs(point) - mid * lasso_weights, 0) * np.sign(point)
     l1_subgrad = point - soft_thresh # a point in the appropriate cube
     l2_subgrad = soft_thresh / np.linalg.norm(soft_thresh) * group_weight * mid
     return mid, l1_subgrad, l2_subgrad

