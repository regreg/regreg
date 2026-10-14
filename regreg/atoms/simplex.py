"""
The simplex constraint and its conjugate, the maximum function.
"""
import numpy as np
from copy import copy

from ..atoms import atom, _work_out_conjugate, affine_atom
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import doc_template_user

from .piecewise_linear import find_solution_piecewise_linear_uncon

class simplex_constraint(atom):

    r"""
    The simplex constraint.

    The indicator of the probability simplex
    :math:`\{\beta : \beta_i \geq 0, \sum_i \beta_i = 1\}`. Its
    conjugate is the `maximum` function.
    """

    tol = 1.0e-05

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'simplex_constraint'
    objective_vars['dualklass'] = 'maximum'
    objective_vars['initargs'] = '(4,)'

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.shape == other.shape
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_simplex_pairs[self.__class__]
            atom = cls(self.shape, 
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @classmethod
    @doc_template_user
    def linear(cls, linear_operator, diag=False,
               linear_term=None, offset=None):
        l = linear_transform(linear_operator, diag=diag)
        atom = cls(l.input_shape, 
                   linear_term=linear_term, offset=offset)
        return affine_atom(atom, l)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        r"""
        Projection onto the simplex.

        Computes the proximal map of the simplex constraint, i.e. the
        Euclidean projection onto the simplex of the center implied by
        `proxq` (combined with `self.quadratic` and `self.offset`).

        Parameters
        ----------
        proxq : identity_quadratic
            A quadratic added to the atom before minimizing.
        prox_control : None or dict, optional
            Ignored.

        Returns
        -------
        Z : np.ndarray
            The proximal map of the implied center of `proxq`.
        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        eta = self._basic_prox(prox_arg, totalq.coef)
        if offset is None:
            return eta
        else:
            return eta + offset

    def _basic_prox(self, prox_arg, lipschitz):
        # lipschitz is ignored because it is a constraint
        value = find_solution_piecewise_linear_uncon(1, prox_arg)
        return (prox_arg - value).clip(min=0)        

    @doc_template_user
    def constraint(self, arg):
        proj = self._basic_prox(arg, 1)
        in_simplex = (np.linalg.norm(arg - proj)
                      / max([np.linalg.norm(arg) ,1]) < self.tol)
        if in_simplex:
            return 0
        return np.inf

    @doc_template_user
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Value of the simplex constraint.

        Always 0 (plus the value of `self.quadratic`) unless
        `check_feasibility` is True.

        Parameters
        ----------
        arg : np.ndarray
            Argument of the constraint.
        check_feasibility : bool, optional
            If True, then return ``np.inf`` if `arg` is not in the
            simplex.

        Returns
        -------
        value : float
            The value of the constraint at `arg`.
        """
        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)
        if check_feasibility:
            value = self.constraint(x_offset)
        else:
            value = 0
        value += self.quadratic.objective(arg, 'func')
        return value

class maximum(atom):

    """
    The conjugate of the simplex constraint: ``x.max()``.
    """

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'maximum'
    objective_vars['dualklass'] = 'simplex_constraint'
    objective_vars['initargs'] = '(4,)'

    tol = 1.0e-05

    objective_vars = atom.objective_vars.copy()
    objective_vars['klass'] = 'simplex_constraint'
    objective_vars['dualklass'] = 'maximum'
    objective_vars['initargs'] = '(4,)'

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.shape == other.shape
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_simplex_pairs[self.__class__]
            atom = cls(self.shape, 
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @classmethod
    @doc_template_user
    def linear(cls, linear_operator, diag=False,
               linear_term=None, offset=None):
        l = linear_transform(linear_operator, diag=diag)
        atom = cls(l.input_shape, 
                   linear_term=linear_term, offset=offset)
        return affine_atom(atom, l)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        r"""
        Proximal map of the maximum function.

        By the Moreau decomposition, this is the residual after
        projecting the implied center of `proxq` onto the simplex
        scaled by ``1 / L``, where :math:`L` is the coefficient of the
        combined quadratic.

        Parameters
        ----------
        proxq : identity_quadratic
            A quadratic added to the atom before minimizing.
        prox_control : None or dict, optional
            Ignored.

        Returns
        -------
        Z : np.ndarray
            The proximal map of the implied center of `proxq`.
        """
        offset, totalq = (self.quadratic + proxq).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef
        eta = self._basic_prox(prox_arg, totalq.coef)
        if offset is None:
            return eta
        else:
            return eta + offset

    def _basic_prox(self, prox_arg, lipschitz):
        value = find_solution_piecewise_linear_uncon(1 / lipschitz, prox_arg)
        proj_simplex = (prox_arg - value).clip(min=0)
        return prox_arg - proj_simplex
    
    @doc_template_user
    def constraint(self, x):
        return 0 # no constraints for maximum function

    @doc_template_user
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Value of the maximum function.

        Returns the maximum entry of `arg` (after applying the offset)
        plus the value of `self.quadratic`.

        Parameters
        ----------
        arg : np.ndarray
            Argument of the maximum function.
        check_feasibility : bool, optional
            Ignored: the maximum function is finite everywhere.

        Returns
        -------
        value : float
            The value of the maximum function at `arg`.
        """

        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)
        value = x_offset.max()
        value += self.quadratic.objective(arg, 'func')
        return value

conjugate_simplex_pairs = {}
for n1, n2 in [(simplex_constraint, maximum)]:
    conjugate_simplex_pairs[n1] = n2
    conjugate_simplex_pairs[n2] = n1
