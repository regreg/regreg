"""
This module contains the implementation of block norms, i.e.
l1/l*, linf/l* norms. These are used in multiresponse LASSOs.

"""
from __future__ import print_function, division, absolute_import

import warnings
from copy import copy

import numpy as np

from . import seminorms
from ..identity_quadratic import identity_quadratic
from ..problems.composite import smooth_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)
from ..atoms import _work_out_conjugate
from .block_norms import l1_l2
from .sparse_group_lasso import _gauge_function_dual_strong, _inside_set_strong

# for the docstring, we need l1norm
l1norm = seminorms.l1norm

@objective_doc_templater()
class sparse_group_block(l1_l2):

    objective_template = r"""w_1\|%(var)s\|_{1,1} + w_1\|%(var)s\|_{1,2}"""
    objective_vars = l1_l2.objective_vars.copy()
    objective_vars['var'] = 'B'
    objective_vars['normklass'] = 'sparse_group_block'
    objective_vars['dualnormklass'] = 'sparse_group_block_dual'
    objective_vars['initargs'] = '(5, 4), 1, 2'
    objective_vars['shape'] = r'n \times p'

    def __init__(self, 
                 shape,
                 l1_weight,
                 l2_weight,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):

        l1_l2.__init__(self,
                       shape,
                       lagrange=lagrange,
                       bound=bound,
                       offset=offset,
                       quadratic=quadratic,
                       initial=initial)

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    @doc_template_user
    def lagrange_prox(self, arg, lipschitz=1, lagrange=None):
        arg = arg.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        return _lagrange_prox(arg, 
                              lagrange * self.l1_weight / lipschitz,
                              lagrange * self.l2_weight / lipschitz)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        raise NotImplementedError('sparse_group_block bound form not implemented')

    @doc_template_user
    def constraint(self, x):
        x = x.reshape(self.shape)
        l1_norms = np.fabs(x).sum()
        l2_norms = np.sqrt(np.sum(x**2), 1).sum()
        norm_sum = self.l1_weight * l1_norms + self.l2_weight * l2_norms
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                               check_feasibility=check_feasibility)
        l1_norms = np.fabs(x).sum()
        l2_norms = np.sqrt(np.sum(x**2, 1)).sum()
        return lagrange * (self.l1_weight * l1_norms +
                           self.l2_weight * l2_norms)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = sparse_group_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate

            atom = cls(self.shape, 
                       self.l1_weight,
                       self.l2_weight,
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound,
                       quadratic=outq)

        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    def __copy__(self):
         return self.__class__(self.shape,
                               self.l1_weight,
                               self.l2_weight,
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
        terms = (np.fabs(arg).sum(1) * self.l1_weight + 
                 np.sqrt((arg**2).sum(1)) * self.l1_weight)
        return terms

class sparse_group_block_dual(sparse_group_block):

    objective_template = r"""\|%(var)s\|_{w_1,w_2,\text{block}}"""
    objective_vars = l1_l2.objective_vars.copy()
    objective_vars['var'] = 'B'
    objective_vars['normklass'] = 'sparse_group_block_dual'
    objective_vars['dualnormklass'] = 'sparse_group_block'
    objective_vars['initargs'] = '(5, 4), 1, 2'
    objective_vars['shape'] = r'n \times p'

    def __init__(self, 
                 shape,
                 l1_weight,
                 l2_weight,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):

        l1_l2.__init__(self,
                       shape,
                       lagrange=lagrange,
                       bound=bound,
                       offset=offset,
                       quadratic=quadratic,
                       initial=initial)

        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    @doc_template_user
    def lagrange_prox(self, arg, lipschitz=1, lagrange=None):
        raise NotImplementedError('sparse_group_block Lagrange form not implemented')

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        arg = arg.reshape(self.shape)
        bound = seminorms.seminorm.bound_prox(self, arg, bound)
        _prox = _lagrange_prox(arg, 
                               bound * self.l1_weight,
                               bound * self.l2_weight)
        return arg - _prox

    @doc_template_user
    def constraint(self, x):
        x = x.reshape(self.shape)
        dual_norm = _gauge_function_dual(x,
                                         self.l1_weight,
                                         self.l2_weight)
        if dual_norm <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                               check_feasibility=check_feasibility)
        return lagrange * _gauge_function_dual(x,
                                               self.l1_weight,
                                               self.l2_weight)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = sparse_group_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate

            atom = cls(self.shape, 
                       self.l1_weight,
                       self.l2_weight,
                       offset=offset,
                       lagrange=conj_atom.lagrange,
                       bound=conj_atom.bound,
                       quadratic=outq)

        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    def terms(self, arg):
         """
         Return the args that are maximized
         in computing the seminorm.
         
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
         return np.array([_gauge_function_dual_strong(arg[i],
                                                      self.l1_weight,
                                                      self.l2_weight)[0] for i in range(arg.shape[0])])

# fast Lagrange prox

def _lagrange_prox(arg, l1_weight, l2_weight):
    soft_thresh = np.sign(arg) * np.maximum(np.fabs(arg) - l1_weight, 0)
    norms = np.sqrt(np.sum(soft_thresh**2, 1))
    norm_factors = np.maximum(norms - l2_weight, 0) / (norms + (norms == 0))
    return soft_thresh * norm_factors[:, None]
    
# for computing dual norm

def _inside_set(point, bound, l1_weight, l2_weight):

    prox_point = _lagrange_prox(point,
                                bound * l1_weight,
                                bound * l2_weight)
    if np.linalg.norm(prox_point) > max(np.linalg.norm(point), 1) * 1.e-10:
        return False
    return True

def _gauge_function_dual(point,
                         l1_weight,
                         l2_weight,
                         tol=1.e-6,
                         max_iter=50): 

     """
     Work out dual norm of sparse group LASSO by binary search.

     NOTE: will have problems if the atom has infinite feature weights
     """

     point = np.asarray(point)

     # find upper and lower bounds

     lower, upper = 1., 1.
     point_inside = _inside_set(point,
                                lower, 
                                l1_weight,
                                l2_weight)
     
     iter = 0
     if point_inside:
          # gauge is upper bounded by 1
          # find a lower bound

          while True:
               lower = lower / 2
               if not _inside_set(point,
                                  lower,
                                  l1_weight,
                                  l2_weight):
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
               if _inside_set(point,
                              upper,
                              l1_weight,
                              l2_weight):
                    break
               else:
                    lower = upper

               iter += 1
               if iter == max_iter:
                    return np.inf

     # binary search

     assert (not _inside_set(point,
                             lower,
                             l1_weight,
                             l2_weight))

     assert _inside_set(point,
                        upper,
                        l1_weight,
                        l2_weight)

     while (upper - lower) > tol * 0.5 * (upper + lower):
         mid = 0.5 * (upper + lower)
         if _inside_set(point,
                        mid,
                        l1_weight,
                        l2_weight):
             upper = mid
         else:
             lower = mid
     return mid

sparse_group_block_pairs = {}
for n1, n2 in [(sparse_group_block, sparse_group_block_dual)]:
    sparse_group_block_pairs[n1] = n2
    sparse_group_block_pairs[n2] = n1

