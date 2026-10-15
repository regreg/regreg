"""
Base classes for problems in composite (smooth plus nonsmooth) form.
"""

from numpy import zeros, asarray, any as npany
from copy import copy

# local imports

from ..identity_quadratic import identity_quadratic as sq
from ..algorithms import FISTA

from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

@objective_doc_templater()
class composite(object):
    """
    A generic way to specify a problem in composite form.

    A composite objective is the sum of a smooth part
    (`smooth_objective`), a nonsmooth part (`nonsmooth_objective`) and a
    quadratic (`quadratic`).

    Parameters
    ----------
    shape : int or tuple
        Shape of the argument of the objective.
    offset : ndarray, optional
        Offset subtracted from the argument before evaluation.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic part of the objective.
    initial : ndarray, optional
        Initial value for ``coefs``. Defaults to zeros.
    """

    objective_template = r"""f(%(var)s)"""
    objective_vars = {'var': r'\beta', 'shape':'p', 'offset':r'\alpha'}

    def __init__(self, shape, offset=None,
                 quadratic=None, initial=None):

        self.offset = offset

        if type(shape) == type(1):
            self.shape = (shape,)
        else:
            self.shape = shape

        if quadratic is not None:
            self.quadratic = quadratic
        else:
            self.quadratic = sq(0,0,0,0)

        if initial is None:
            self.coefs = zeros(self.shape)
        else:
            self.coefs = copy(initial)

    def set_offset(self, value):
        """
        Set the offset; an offset that is all zeros is ignored.
        """
        if value is not None:
            value = asarray(value)
            if npany(value != 0):
                self._offset = value
        else:
            self._offset = None

    def get_offset(self):
        """
        Get the offset (None if there is no offset).
        """
        if not hasattr(self, "_offset"):
            self._offset = None
        return self._offset
    offset = property(get_offset, set_offset)

    def latexify(self, var=None, idx=''):
        """
        Return a LaTeX representation of the objective.

        Parameters
        ----------
        var : str, optional
            LaTeX name for the argument of the objective. Defaults to
            ``self.objective_vars['var']``.
        idx : str, optional
            Subscript to attach to the parameters.

        Returns
        -------
        latex : str
            LaTeX string.
        """
        template_dict = copy(self.objective_vars)
        template_dict['idx'] = idx
        if var is not None:
            template_dict['var'] = var
        if hasattr(self, 'offset') and self.offset is not None and npany(self.offset != 0):
            template_dict['var'] = (r'%(var)s - %(offset)s_{%(idx)s}' % template_dict)

        obj = self.objective_template % template_dict
        template_dict['obj'] = obj

        if not self.quadratic.iszero:
            return ' + '.join([obj, self.quadratic.latexify(var=var, idx=idx)])
        return obj

    def _repr_latex_(self):
        return r'$$' + self.latexify() + r'$$'

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        The nonsmooth part of the objective; here just the quadratic.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate.
        check_feasibility : bool, optional
            Check whether `x` is feasible (unused here).

        Returns
        -------
        value : float
            Value of the nonsmooth part.
        """
        return self.quadratic.objective(x, 'func')

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        The smooth part of the objective and/or its gradient.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate.
        mode : str, optional
            One of ``['func', 'grad', 'both']``.
        check_feasibility : bool, optional
            Check whether `x` is feasible?

        Returns
        -------
        value : float or ndarray or tuple
            Function value, gradient, or both.
        '''
        raise NotImplementedError

    def objective(self, x, check_feasibility=False):
        """
        Total objective: smooth plus nonsmooth part.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate.
        check_feasibility : bool, optional
            Check whether `x` is feasible?

        Returns
        -------
        value : float
            Value of the objective.
        """
        return self.smooth_objective(x,mode='func', check_feasibility=check_feasibility) + self.nonsmooth_objective(x, check_feasibility=check_feasibility)

    @doc_template_provider
    def proximal_optimum(self, quadratic):
        r"""
        Compute a proximal step, returning the minimizer and the optimal value.

        Parameters
        ----------
        quadratic : `regreg.identity_quadratic.identity_quadratic`
            Quadratic term of the proximal problem.

        Returns
        -------
        argmin : ndarray
            Minimizer of the proximal problem.
        optimal_value : float
            Value of `quadratic` plus the nonsmooth objective (and
            ``self.quadratic``) at `argmin`.
        """
        argmin = self.proximal(quadratic)
        if self.quadratic is None:
            return argmin, quadratic.objective(argmin) + self.nonsmooth_objective(argmin) 
        else: 
            return argmin, quadratic.objective(argmin) + self.nonsmooth_objective(argmin) + self.quadratic.objective(argmin, 'func') 

    def proximal_step(self, quadratic, prox_control=None):
        """
        Compute the proximal optimization.

        Parameters
        ----------
        quadratic : `regreg.identity_quadratic.identity_quadratic`
            Quadratic term of the proximal problem.
        prox_control : dict, optional
            If not None, then a dictionary of parameters for the prox
            procedure.

        Returns
        -------
        argmin : ndarray
            Minimizer of the proximal problem.
        """
        # This seems like a null op -- if all proximals accept optional prox_control
        if prox_control is None:
            return self.proximal(quadratic)
        else:
            return self.proximal(quadratic, prox_control=prox_control)

    def apply_offset(self, x):
        """
        If ``self.offset`` is not None, return ``x-self.offset``, else
        return `x`.
        """
        if self.offset is not None:
            return x - self.offset
        return x

    def set_quadratic(self, quadratic):
        """
        Set the quadratic part of the composite.
        """
        self._quadratic = quadratic

    def get_quadratic(self):
        """
        Get the quadratic part of the composite.
        """
        if not hasattr(self, "_quadratic"):
            self._quadratic = sq(None, None, None, None)
        return self._quadratic
    quadratic = property(get_quadratic, set_quadratic, doc='Quadratic part of the object, instance of `regreg.identity_quadratic.identity_quadratic`.')

    def smoothed(self, smoothing_quadratic):
        '''
        Smooth the composite by adding a quadratic to its conjugate.

        Parameters
        ----------
        smoothing_quadratic : `regreg.identity_quadratic.identity_quadratic`
            Quadratic added to the conjugate; its coefficient must be
            nonzero.

        Returns
        -------
        smoothed_atom : object
            The conjugate of the smoothed conjugate, a smooth approximation
            to this composite.
        '''
        conjugate_atom = copy(self.conjugate)
        sq = smoothing_quadratic
        if sq.coef in [None, 0]:
            raise ValueError('quadratic term of smoothing_quadratic must be non 0')
        total_q = sq

        if conjugate_atom.quadratic is not None:
            total_q = sq + conjugate_atom.quadratic
        conjugate_atom.set_quadratic(total_q)
        smoothed_atom = conjugate_atom.conjugate
        return smoothed_atom

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        """
        Solve the problem. Must be implemented by subclasses.
        """
        raise NotImplementedError('subclasses must implement their own solve methods')

@objective_doc_templater()
class nonsmooth(composite):
    """
    A composite subclass that explicitly returns 0 as smooth_objective.
    """

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        The zero function (and its zero gradient).
        '''

        if mode == 'both':
            return 0., zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return zeros(x.shape)
        raise ValueError("Mode not specified correctly")

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        """
        Minimize the objective plus `quadratic` with a single proximal step.

        Parameters
        ----------
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic added to the objective.
        return_optimum : bool, optional
            Also return the optimal value?
        fit_args : dict
            Ignored.

        Returns
        -------
        optimal_value : float
            Only returned if `return_optimum` is True.
        coefs : ndarray
            The minimizer.
        """
        if quadratic is None:
            quadratic = sq(0, 0, 0, 0)
        self.coefs = self.proximal(quadratic)
        if return_optimum:
            return self.objective(self.coefs) + quadratic.objective(self.coefs, 'func'), self.coefs
        else:
            return self.coefs

@objective_doc_templater()
class smooth(composite):

    """
    A composite subclass with no nonsmooth part.

    The nonsmooth part is just the quadratic, so the proximal map
    is that of a quadratic.
    """

    objective_vars = copy(composite.objective_vars)
    objective_vars['coef'] = 'C'

    def get_lipschitz(self):
        """
        Lipschitz constant of the gradient, including the quadratic.
        """
        if hasattr(self, '_lipschitz'):
            return self._lipschitz + self.quadratic.coef
        return self.quadratic.coef

    def set_lipschitz(self, value):
        """
        Set the Lipschitz constant of the gradient (excluding the quadratic).
        """
        if value < 0:
            raise ValueError('Lipschitz constant must be non-negative')
        self._lipschitz = value
    lipschitz = property(get_lipschitz, set_lipschitz)

    def smooth_objective(self, x, mode='func', check_feasibility=False):
        """
        Evaluate ``self._smooth_objective``; see `composite.smooth_objective`.
        """
        return self._smooth_objective(x, mode=mode, check_feasibility=check_feasibility)

    def proximal(self, quadratic):
        """
        Minimizer of ``self.quadratic`` plus `quadratic`.
        """
        totalq = self.quadratic + quadratic
        return -totalq.linear_term / totalq.coef

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        """
        Minimize the objective plus `quadratic` using `FISTA`.

        Parameters
        ----------
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic added to the objective.
        return_optimum : bool, optional
            Also return the optimal value?
        fit_args : dict
            Keyword arguments passed to `regreg.algorithms.FISTA.fit`.

        Returns
        -------
        optimal_value : float
            Only returned if `return_optimum` is True.
        coefs : ndarray
            The minimizer.
        """
        if quadratic is None:
            quadratic = sq(0, 0, 0, 0)
        oldq, self.quadratic = self.quadratic, self.quadratic + quadratic
        self.solver = FISTA(self)
        self.solver.fit(**fit_args)
        self.quadratic = oldq

        if return_optimum:
            # the optimal value of the problem solved, which includes
            # `quadratic` (as in `nonsmooth.solve`)
            return (self.objective(self.coefs) +
                    quadratic.objective(self.coefs, 'func')), self.coefs
        else:
            return self.coefs

class smooth_conjugate(smooth):

    """
    Smooth conjugate of an atom.

    Given an atom, compute the conjugate of this atom plus an
    identity_quadratic, which will be a smooth version of the conjugate of
    the atom.

    Parameters
    ----------
    atom : `regreg.atoms.atom`
        The atom to conjugate.
    smoothing_quadratic : `identity_quadratic`, optional
        Quadratic added to the atom. The total quadratic coefficient (the
        atom's plus this one) must be nonzero.

    Notes
    -----
    Should we have an argument "collapse" that makes a copy?
    """

    def __init__(self, atom, smoothing_quadratic=None):
        # this holds a pointer to the original atom,
        # but will be replaced later

        self.atom = atom
        if smoothing_quadratic is None:
            smoothing_quadratic = sq(0,0,0,0)
        self.smoothing_quadratic = smoothing_quadratic
        total_quadratic = self.atom.quadratic + self.smoothing_quadratic

        if total_quadratic.coef in [0,None]:
            raise ValueError('the atom must have non-zero quadratic term to compute ensure smooth conjugate')

        self.lipschitz = 1. / total_quadratic.coef

        self.shape = atom.shape
        self.coefs = zeros(self.shape)

        # A smooth conjugate is the conjugate of some $f$ with an identity quadratic added to it, or
        # $$
        # h(u) = \sup_x \left( u^Tx - \frac{\kappa}{2} \|x\|^2_2 - \beta^Tx-c-f(x) \right).
        # $$
        # Suppose we add a quadratic to $h$ to get
        # $$
        # \tilde{h}(u) = \frac{r}{2} \|u\|^2_2 + u^T\gamma + a + h(u)$$
        # and take the conjugate again:
        # $$
        # \begin{aligned}
        # g(v) &= \sup_{u} u^Tv - \tilde{h}(u) \\
        # &= \sup_u u^Tv -  \frac{r}{2} \|u\|^2_2 - u^T\gamma-a - h(u) \\
        # &=  \sup_u u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a  - \sup_x \left( u^Tx - \frac{\kappa}{2} \|x\|^2_2 - \beta^Tx-c-f(x)  \right)\\
        # &= \sup_u u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a + \inf_x \left(  \frac{\kappa}{2} \|x\|^2_2  +\beta^Tx + c +f(x) - u^Tx  \right)\\
        # &= \sup_u \inf_x u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x) - u^Tx \\
        # &=  \inf_x \sup_u u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2  + \beta^Tx + c +f(x) - u^Tx \\
        # &=  \inf_x \sup_u \left(u^Tv - \frac{r}{2} \|u\|^2_2 - u^T\gamma- u^Tx\right)-a +  \frac{\kappa}{2} \|x\|^2_2 +   \beta^Tx + c +f(x)  \\
        # &=  \inf_x \frac{1}{2r} \|x+\gamma-v\|^2_2 -a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x)  \\
        # &= c-a + \frac{1}{2r} \|\gamma-v\|^2_2 - \sup_x \left((v/r)^Tx - \frac{1}{2}\left(\frac{1}{r} + \kappa\right) \|x\|^2_2 - x^T(\beta+\gamma/r) - f(x) \right) \\
        # \end{aligned}
        # $$

        # This says that for $r > 0$ the conjugate of a smooth conjugate with a quadratic added to it is a quadratic plus a modified smooth conjugate evaluated at $v/r$.

        # What if $r=0$? Well,
        # then 
        # $$
        # \begin{aligned}
        # g(v) &= \sup_{u} u^Tv - \tilde{h}(u) \\
        # &= \sup_u u^Tv  - u^T\gamma-a - h(u) \\
        # &=  \sup_u u^Tv  - u^T\gamma-a  - \sup_x \left( u^Tx - \frac{\kappa}{2} \|x\|^2_2 - \beta^Tx-c-f(x)  \right)\\
        # &= \sup_u u^Tv - u^T\gamma-a + \inf_x \left(  \frac{\kappa}{2} \|x\|^2_2  +\beta^Tx + c +f(x) - u^Tx  \right)\\
        # &= \sup_u \inf_x u^Tv - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x) - u^Tx \\
        # &= \inf_x \sup_u u^Tv - u^T\gamma-a +  \frac{\kappa}{2} \|x\|^2_2 + \beta^Tx + c +f(x) - u^Tx \\
        # &=   \frac{\kappa}{2} \|v-\gamma\|^2_2 + \beta^T(v-\gamma) + c-a +f(v-\gamma) \\
        # \end{aligned}
        # $$
        # where, in the last line we have used the fact that the $\sup$ over $u$ in the second to last line is infinite unless $x=v-\gamma$.

    def get_conjugate(self):
        """
        The conjugate of the smooth conjugate, i.e. the atom with
        the quadratics added back in.
        """
        if self.quadratic.iszero:
            if self.smoothing_quadratic.iszero:
                return self.atom
            else:
                atom = copy(self.atom)
                atom.quadratic = atom.quadratic + self.smoothing_quadratic
                return atom
        else:
            # h is this smooth conjugate, the conjugate of F = atom plus
            # the total quadratic (collapsed: kappa/2 |x|^2 + beta^T x + c),
            # and q(u) = r/2 |u|^2 + gamma^T u + a is self.quadratic.
            # We want the conjugate g of h + q.
            q = self.quadratic.collapsed()
            r, gamma, a = q.coef, q.linear_term, q.constant_term
            if r == 0:
                # g(v) = F(v - gamma) - a: shift the atom by gamma and
                # evaluate the total quadratic at v - gamma
                total = (self.atom.quadratic + self.smoothing_quadratic).collapsed()
                beta = total.linear_term
                atom = copy(self.atom)
                if atom.offset is None:
                    atom.offset = gamma
                else:
                    atom.offset = atom.offset + gamma
                atom.quadratic = sq(total.coef,
                                    gamma,
                                    beta,
                                    total.constant_term - asarray(beta * gamma).sum() - a)
                return atom
            else:
                # g(v) = |v - gamma|^2 / (2r) - a - H(v / r), where H is the
                # smooth conjugate with smoothing quadratic
                # smoothing_quadratic + 1/(2r) |x|^2 + (gamma/r)^T x
                H = smooth_conjugate(self.atom,
                                     smoothing_quadratic=self.smoothing_quadratic +
                                     sq(1. / r, 0, gamma / r, 0))
                return _smooth_conjugate_conjugate(H, r, gamma, a)
    conjugate = property(get_conjugate)

    def latexify(self, var=None, idx=''):
        """
        Return a LaTeX representation of the smooth conjugate.

        Parameters
        ----------
        var : str, optional
            LaTeX name for the argument.
        idx : str, optional
            Subscript to attach to the parameters.

        Returns
        -------
        latex : str
            LaTeX string.
        """
        template_dict = copy(self.atom.objective_vars)
        template_dict['idx'] = idx
        if var is not None:
            template_dict['var'] = var

        template_dict['f'] = self.atom.latexify(var='u', idx=idx)
        quad_latex = self.smoothing_quadratic.latexify(var='u', idx=idx)
        if quad_latex:
            template_dict['f'] = ' + '.join([template_dict['f'],
                                             quad_latex])

        objective = r' \sup_{u \in \mathbb{R}^{%(shape)s} } \left[ \langle %(var)s, u \rangle - \left(%(f)s \right) \right]' % template_dict
        return objective

    def __repr__(self):
        return 'smooth_conjugate(%s,%s)' % (repr(self.atom), repr(self.smoothing_quadratic + self.atom.quadratic))

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate the smooth conjugate and/or its gradient.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate.
        mode : str, optional
            If ``'both'``, return both function value and gradient;
            if ``'grad'``, return only the gradient;
            if ``'func'``, return only the function value.
        check_feasibility : bool, optional
            Unused.

        Returns
        -------
        value : float or ndarray or tuple
            Function value, gradient, or both.
        """

        q = self.smoothing_quadratic + sq(0,0,-x,0) 

        if mode == 'both':
            optimal_value, argmin = self.atom.solve(quadratic=q, return_optimum=True)
            objective = -optimal_value
            # retain a reference
            self.grad = argmin
            return objective, self.grad
        elif mode == 'grad':
            argmin = self.atom.solve(quadratic=q)
            # retain a reference
            self.grad = argmin
            return self.grad
        elif mode == 'func':
            optimal_value, argmin = self.atom.solve(quadratic=q, return_optimum=True)
            objective = -optimal_value
            # retain a reference
            self.grad = argmin
            return objective
        else:
            raise ValueError("mode incorrectly specified")

    def proximal(self, proxq, prox_control=None):
        """
        Not defined for a smooth conjugate; raises ValueError.
        """
        raise ValueError('no proximal defined')

class _smooth_conjugate_conjugate(smooth):

    r"""
    Conjugate of a `smooth_conjugate` plus a quadratic with positive
    coefficient :math:`r`:

    .. math::

       g(v) = \frac{1}{2r} \|v - \gamma\|^2_2 - a - H(v / r)

    where :math:`H` is a modified smooth conjugate (see
    `smooth_conjugate.get_conjugate`).

    Parameters
    ----------
    H : `smooth_conjugate`
        The modified smooth conjugate.
    r : float
        Coefficient of the quadratic, must be positive.
    gamma : ndarray or float
        Linear term of the quadratic.
    a : float
        Constant term of the quadratic.
    """

    def __init__(self, H, r, gamma, a):
        smooth.__init__(self, H.shape)
        self.H, self.r, self.gamma, self.a = H, r, gamma, a
        self.lipschitz = 1. / r

    def smooth_objective(self, v, mode='both', check_feasibility=False):
        """
        Evaluate the objective, its gradient, or both.

        Parameters
        ----------
        v : ndarray
            Where to evaluate the objective.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the value ('func'), the gradient ('grad')
            or both ('both').
        check_feasibility : bool, optional
            Ignored: the objective is finite everywhere.

        Returns
        -------
        f : float
            Value at `v`; returned when `mode` is 'func' or 'both'.
        g : ndarray
            Gradient at `v`; returned when `mode` is 'grad' or 'both'.
        """
        r, d = self.r, v - self.gamma
        if mode == 'func':
            return (d**2).sum() / (2 * r) - self.a - self.H.smooth_objective(v / r, 'func')
        elif mode == 'grad':
            return d / r - self.H.smooth_objective(v / r, 'grad') / r
        elif mode == 'both':
            Hv, Hg = self.H.smooth_objective(v / r, 'both')
            return (d**2).sum() / (2 * r) - self.a - Hv, d / r - Hg / r
        else:
            raise ValueError("mode incorrectly specified")
