"""
This module contains the implementation of block norms, i.e.
l1/l*, linf/l* norms. These are used in multiresponse LASSOs.

"""
from __future__ import print_function, division, absolute_import

import warnings

import numpy as np

from . import seminorms
from ..identity_quadratic import identity_quadratic
from ..problems.composite import smooth_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)
from ..atoms import _work_out_conjugate
from .piecewise_linear import find_solution_piecewise_linear_c

# for the docstring, we need l1norm
l1norm = seminorms.l1norm

@objective_doc_templater()
class block_sum(seminorms.seminorm):

    objective_template = r"""\|%(var)s\|_{1,\|\cdot\|}"""
    objective_vars = seminorms.seminorm.objective_vars.copy()
    objective_vars['var'] = 'B'
    objective_vars['normklass'] = 'block_sum'
    objective_vars['dualnormklass'] = 'block_max'
    objective_vars['initargs'] = 'rr.l1norm, (5, 4)'
    objective_vars['shape'] = r'n \times p'

    def __init__(self, atom_cls, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):

        seminorms.seminorm.__init__(self,
                            shape,
                            quadratic=quadratic,
                            offset=offset,
                            initial=initial,
                            lagrange=lagrange,
                            bound=bound)

        self.atom = atom_cls(shape[1:], lagrange=lagrange,
                             bound=bound,
                             offset=None,
                             quadratic=quadratic)

    @doc_template_user
    @doc_template_provider
    def seminorms(self, x, lagrange=None, check_feasibility=False):
        """
        Compute all seminorms in the block norm.
        """
        value = np.empty(self.shape[0])
        for i in range(self.shape[0]):
            value[i] = self.atom.seminorm(x[i], lagrange=lagrange,
                                          check_feasibility=False)
        return value

    @doc_template_user
    def seminorm(self, x, check_feasibility=False,
                 lagrange=None):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.sum( \
            self.seminorms(x, check_feasibility=check_feasibility,
                           lagrange=1.))

    @doc_template_user
    def constraint(self, x):
        # XXX should we check feasibility here?
        x = x.reshape(self.shape)
        v = np.sum(self.seminorms(x, check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, x, lipschitz, lagrange)
        v = np.empty(x.shape)
        for i in xrange(self.shape[0]):
            v[i] = self.atom.lagrange_prox(x[i], lipschitz=lipschitz,
                                           lagrange=lagrange)
        return v

    @doc_template_user
    def bound_prox(self, x, bound=None):
        x = x.reshape(self.shape)
        warnings.warn('bound_prox of block_sum requires a little thought -- should be like l1prox')
        return 0 * x

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = conjugate_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate
            atom_cls = conj_atom.__class__

            atom = cls(atom_cls, 
                       self.shape, 
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorms.seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)


@objective_doc_templater()
class block_max(block_sum):

    objective_template = r"""\|%(var)s\|_{\infty,\| \cdot\|}"""
    objective_vars = block_sum.objective_vars.copy()
    objective_vars['normklass'] = 'block_max'
    objective_vars['dualnormklass'] = 'block_sum'

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        return lagrange * np.max(self.seminorms(x,  
                                                lagrange=1.,
                                                check_feasibility=check_feasibility))

    @doc_template_user
    def constraint(self, x, bound=None):
        x = x.reshape(self.shape)
        bound = seminorms.seminorm.constraint(self, x, bound=bound)
        # XXX should we check feasibility here?
        v = np.max(self.seminorms(x, lagrange=1., check_feasibility=False))
        if v <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def lagrange_prox(self, x, lipschitz=1, lagrange=None):
        raise ValueError('lagrange_prox of block_max requires a little thought -- should be like l1prox')

    @doc_template_user
    def bound_prox(self, x, bound=None):
        x = x.reshape(self.shape)
        bound = seminorms.seminorm.bound_prox(self, x,
                                      bound=bound)
        v = np.empty(x.shape)
        for i in xrange(self.shape[0]):
            v[i] = self.atom.bound_prox(x[i], 
                                        bound=bound)
        return v

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorms.seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return block_sum.get_conjugate(self)


@objective_doc_templater()
class linf_l2(block_max):

    objective_template = r"""\|%(var)s\|_{\infty,2}"""
    objective_vars = block_sum.objective_vars.copy()
    objective_vars['normklass'] = 'linf_l2'
    objective_vars['dualnormklass'] = 'l1_l2'
    objective_vars['initargs'] = '(5,4)'

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_max.__init__(self, seminorms.l2norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    @doc_template_user
    def constraint(self, arg):
        arg = arg.reshape(self.shape)
        norm_max = np.sqrt((arg**2).sum(1)).max()
        if norm_max <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        arg = arg.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, arg, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_max = np.sqrt((arg**2).sum(1)).max()
        return lagrange * norm_max

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        arg = arg.reshape(self.shape)
        norm = np.sqrt((arg**2).sum(1))
        bound = seminorms.seminorm.bound_prox(self, arg,
                                              bound=bound)
        v = arg.copy()
        v[norm >= bound] *= bound / norm[norm >= bound][:,np.newaxis]
        return v

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        arg = arg.reshape(self.shape)
        norm = np.sqrt((arg**2).sum(1))
        lagrange = seminorms.seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        cut = find_solution_piecewise_linear_c(lagrange / lipschitz, 0, norm)
        if cut < np.inf:
            proj_factor = (norm - cut) * (norm > cut)
        else:
            proj_factor = arg
        factor = (norm - proj_factor) / norm
        v = arg.copy()
        v *= factor[:,np.newaxis]
        return v

    @doc_template_user
    def get_conjugate(self):

        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = conjugate_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate

            atom = cls(self.shape, 
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

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorms.seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)


@objective_doc_templater()
class linf_linf(linf_l2):

    objective_template = r"""\|%(var)s\|_{\infty,\infty}"""
    objective_vars = linf_l2.objective_vars.copy()
    objective_vars['normklass'] = 'linf_linf'
    objective_vars['dualnormklass'] = 'l1_l1'

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_max.__init__(self, seminorms.supnorm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    @doc_template_user
    def constraint(self, arg):
        arg = arg.reshape(self.shape)
        norm_max = np.fabs(arg).max()
        if norm_max <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        arg = arg.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, arg, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_max = np.fabs(arg).max()
        return lagrange * norm_max

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        arg = arg.reshape(self.shape)
        bound = seminorms.seminorm.bound_prox(self, arg,
                                              bound=bound)
        return np.clip(arg, -bound, bound).reshape(self.shape)

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorms.seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg, np.float).reshape(-1)
        absarg = np.fabs(arg)
        cut = find_solution_piecewise_linear_c(lagrange / lipschitz, 0, absarg)
        if cut < np.inf:
            proj = np.sign(arg) * (absarg - cut) * (absarg > cut)
        else:
            proj = arg
        return (arg - proj).reshape(self.shape)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorms.seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return linf_l2.get_conjugate(self)


@objective_doc_templater()
class l1_l2(block_sum):

    objective_template = r"""\|%(var)s\|_{1,2}"""
    objective_vars = linf_l2.objective_vars.copy()
    objective_vars['normklass'] = 'l1_l2'
    objective_vars['dualnormklass'] = 'linf_l2'

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_sum.__init__(self, seminorms.l2norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    @doc_template_user
    def lagrange_prox(self, arg, lipschitz=1, lagrange=None):
        arg = arg.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        norm = np.sqrt((arg**2).sum(1))
        mult = np.maximum(norm - lagrange / lipschitz, 0) / norm
        return arg * mult[:, np.newaxis]

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        arg = arg.reshape(self.shape)
        norm = np.sqrt((arg**2).sum(1))
        bound = seminorms.seminorm.bound_prox(self, arg,
                                              bound=bound)
        cut = find_solution_piecewise_linear_c(bound, 0, norm)
        if cut < np.inf:
            proj_factor = (norm - cut) * (norm > cut)
        else:
            proj_factor = arg
        factor = proj_factor / norm
        v = arg.copy()
        v *= factor[:,np.newaxis]
        return v

    @doc_template_user
    def constraint(self, x):
        x = x.reshape(self.shape)
        norm_sum = np.sqrt((x**2).sum(1)).sum()
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def seminorm(self, x, lagrange=None, check_feasibility=False):
        x = x.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, x, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_sum = np.sum(np.sqrt((x**2).sum(1)))
        return lagrange * norm_sum

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorms.seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)

            cls = conjugate_block_pairs[self.__class__]
            conj_atom = self.atom.conjugate

            atom = cls(self.shape, 
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

@objective_doc_templater()
class l1_l1(l1_l2):

    objective_template = r"""\|%(var)s\|_{1,1}"""
    objective_vars = l1_l2.objective_vars.copy()
    objective_vars['normklass'] = 'l1_l1'
    objective_vars['dualnormklass'] = 'linf_linf'

    def __init__(self, shape,
                 lagrange=None,
                 bound=None,
                 offset=None,
                 quadratic=None,
                 initial=None):
        block_sum.__init__(self, seminorms.l1norm,
                           shape,
                           lagrange=lagrange,
                           bound=bound,
                           offset=offset,
                           quadratic=quadratic,
                           initial=initial)

    @doc_template_user
    def lagrange_prox(self, arg, lipschitz=1, lagrange=None):
        arg = arg.reshape(self.shape)
        lagrange = seminorms.seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        norm = np.fabs(arg)
        return (np.maximum(norm - lagrange / lipschitz, 0) * np.sign(arg)).reshape(self.shape)

    @doc_template_user
    def constraint(self, arg):
        arg = arg.reshape(self.shape)
        norm_sum = np.fabs(arg).sum()
        if norm_sum <= self.bound * (1 + self.tol):
            return 0
        return np.inf

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        arg = arg.reshape(self.shape)
        lagrange = seminorms.seminorm.seminorm(self, arg, lagrange=lagrange,
                                 check_feasibility=check_feasibility)
        norm_sum = np.fabs(arg).sum()
        return lagrange * norm_sum

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorms.seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg, np.float).reshape(-1)
        absarg = np.fabs(arg)
        cut = find_solution_piecewise_linear_c(bound, 0, absarg)
        if cut < np.inf:
            value = np.sign(arg) * (absarg - cut) * (absarg > cut)
        return value.reshape(self.shape)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorms.seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorms.seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorms.seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorms.seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorms.seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_dual(self):
        return seminorms.seminorm.dual(self)

    @doc_template_user
    def get_conjugate(self):
        return l1_l2.get_conjugate(self)

conjugate_block_pairs = {}
for n1, n2 in [(block_max, block_sum),
               (l1_l2, linf_l2),
               (l1_l1, linf_linf)
               ]:
    conjugate_block_pairs[n1] = n2
    conjugate_block_pairs[n2] = n1
