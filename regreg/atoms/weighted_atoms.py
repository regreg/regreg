import numpy as np
from scipy import sparse
from copy import copy
import warnings

from .seminorms import seminorm as unweighted_seminorm

from ..problems.composite import composite, nonsmooth, smooth_conjugate
from ..affine import (linear_transform, identity as identity_transform, 
                     affine_transform, selector)
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)
from .piecewise_linear import find_solution_piecewise_linear

class seminorm(unweighted_seminorm):

    """
    A class that defines the API for support functions.
    """
    tol = 1.0e-05

    def __init__(self, weights, lagrange=None, bound=None, 
                 offset=None, 
                 quadratic=None,
                 initial=None):

        self.weights = np.asarray(weights, float)
        unweighted_seminorm.__init__(self, self.weights.shape,
                                     lagrange=lagrange,
                                     bound=bound,
                                     quadratic=quadratic,
                                     initial=initial,
                                     offset=offset)

        self.invweights = np.zeros_like(self.weights)
        zero_weight = self.weights == 0
        self.invweights[~zero_weight] = 1. / self.weights[~zero_weight]
        self.invweights[zero_weight] = np.inf
        
        if not np.all(self.weights >= 0):
            raise ValueError('weights should be nonnegative')

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            weights_equal = np.all(np.equal(self.weights, other.weights))
            if self.bound is not None:
                return self.bound == other.bound and weights_equal
            return self.lagrange == other.lagrange and weights_equal
        return False

    def __copy__(self):
        return self.__class__(self.weights.copy(),
                              quadratic=self.quadratic,
                              initial=self.coefs,
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset))
    
    def __repr__(self):
        if self.lagrange is not None:
            if not self.quadratic.iszero:
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
            if not self.quadratic.iszero:
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
                cls = conjugate_weighted_pairs[self.__class__]
                atom = cls(self.invweights, 
                           bound=self.lagrange, 
                           lagrange=None,
                           offset=offset,
                           quadratic=outq)
            else:
                cls = conjugate_weighted_pairs[self.__class__]
                atom = cls(self.invweights,
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
    
    def form_transform(self, subsample=False):
        '''
        By subsampling we can get rid of some variables that have 0 weights.
        '''
        if not hasattr(self, '_linear_transform'):
            if self.weights is not None:
                test = self.weights == 0
                if test.sum() and subsample:
                    self._linear_transform = selector(~test, self.shape)
                else:
                    self._linear_transform = identity_transform(self.shape)
            else:
                self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    linear_transform = property(form_transform)


@objective_doc_templater()
class l1norm(seminorm):

    """
    The l1 norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|W%(var)s\|_1"""

    @doc_template_user
    def seminorm(self,
                 arg,
                 lagrange=None,
                 check_feasibility=False):
        lagrange = seminorm.seminorm(self,
                                     arg, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (arg != 0)
            if check_zero.sum():
                return np.inf
        arg = np.asarray(arg)
        return lagrange * np.fabs(arg[finite] * self.weights[finite]).sum()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = self.seminorm(arg,
                              lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        return np.sign(arg) * np.maximum(np.fabs(arg)-lagrange * self.weights
                                       / lipschitz, 0)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        _isinf = np.isinf(self.weights)
        _is0 = self.weights == 0
        _keep = ~(_isinf + _is0)
        _lagrange = find_solution_piecewise_linear(bound,
                                                   0,
                                                   np.fabs(arg[_keep]),
                                                   self.weights[_keep])
        cut = self.weights * (_lagrange + _isinf)
        value = np.sign(arg) * np.maximum(np.fabs(arg) - cut, 0)
        return value
    
    def terms(self, arg):
        """
        Return the args that are summed
        in computing the seminorm.

        """
        arg = np.asarray(arg)
        return np.fabs(arg) * self.weights

@objective_doc_templater()
class supnorm(seminorm):

    r"""
    The :math:`\ell_{\infty}` norm
    """

    objective_template = r"""\|W%(var)s\|_{\infty}"""

    @doc_template_user
    def seminorm(self,
                 arg,
                 lagrange=None,
                 check_feasibility=False):
        lagrange = seminorm.seminorm(self,
                                     arg, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (arg != 0)
            if check_zero.sum():
                return np.inf
        return lagrange * np.fabs(arg[finite] * self.weights[finite]).max()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = self.seminorm(arg, lagrange=1,
                              check_feasibility=True) <= bound * (1+self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        weights = self.invweights
        _isinf = np.isinf(weights)
        _is0 = weights == 0
        _keep = ~(_isinf + _is0)
        _lagrange = find_solution_piecewise_linear(lagrange / lipschitz,
                                                   0,
                                                   np.fabs(arg[_keep]),
                                                   weights[_keep])
        resid = np.sign(arg) * np.maximum(np.fabs(arg)-_lagrange * weights, 0)

        return arg - resid

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        return np.clip(arg, -bound * self.invweights, bound * self.invweights)

    def terms(self, arg, check_feasibility=True):
        """
        Return the args that are maximized
        in computing the seminorm.

        """
        arg = np.asarray(arg)
        v = np.fabs(arg) / self.weights
        if check_feasibility:
            v[self.weights == 0] = np.where(np.fabs(arg[self.weights == 0]) != 0, np.inf, 0)
        return v

conjugate_weighted_pairs = {}
for n1, n2 in [(l1norm,supnorm)]:
    conjugate_weighted_pairs[n1] = n2
    conjugate_weighted_pairs[n2] = n1
