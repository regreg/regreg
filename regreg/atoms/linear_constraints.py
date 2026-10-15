import numpy as np
from scipy import sparse
from copy import copy
import warnings

from ..problems.composite import composite, nonsmooth, smooth_conjugate
from ..affine import linear_transform
from .cones import cone, affine_cone
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import doc_template_user

@objective_doc_templater()
class linear_constraint(cone):

    r"""
    Base class for linear (subspace) constraints.

    This class allows specification of linear constraints
    of the form :math:`x \in \text{row}(L)` by specifying
    an orthonormal basis for the rowspace of :math:`L`.

    If the constraint is of the form :math:`Ax=0`, then
    this linear constraint can also be created using the
    `linear` classmethod of the `zero_constraint` cone in
    `regreg.atoms.cones`.

    Parameters
    ----------
    shape : int or tuple
        Shape of the argument of the constraint.
    basis : np.ndarray
        Matrix whose rows are an orthonormal basis for the subspace.
    offset : np.ndarray, optional
        Offset subtracted from the argument before the constraint is
        evaluated.
    initial : np.ndarray, optional
        Initial value of the coefficients.
    quadratic : identity_quadratic, optional
        Quadratic term added to the constraint.
    """
    tol = 1.0e-05

    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'projection'
    objective_vars['dualconeklass'] = 'projection_complement'
    objective_vars['initargs'] = '(4,), [[1,0,0,0],[0,1,0,0]]'

    #XXX should basis by a linear operator instead?
    def __init__(self, shape, basis,
                 offset=None,
                 initial=None, 
                 quadratic=None):

        cone.__init__(self,
                      shape,
                      offset=offset,
                      initial=initial,
                      quadratic=quadratic)
        self.basis = np.asarray(basis)
        

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return (self.shape == other.shape
                    and np.allclose(other.basis, self.basis))
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              self.basis.copy(),
                              offset=copy(self.offset),
                              quadratic=self.quadratic)
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, %s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.basis),
                 repr(self.shape), 
                 repr(self.offset))
        else:
            return "%s(%s, %s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.basis),
                 repr(self.shape), 
                 repr(self.offset),
                 self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_cone_pairs[self.__class__]
            atom = cls(self.shape, 
                       self.basis,
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate

    @classmethod
    @doc_template_user
    def linear(cls, linear_operator, basis, diag=False,
               offset=None, quadratic=None):
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if offset is None:
            offset = 0
        cone = cls(l.output_shape, basis,
                   offset=-offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)

@objective_doc_templater()
class projection(linear_constraint):

    """
    An atom representing a linear constraint.

    The constraint restricts the argument to the span of the rows of
    `basis`, a matrix that is assumed to be a set of row vectors
    spanning the space.

    Notes
    -----
    It is assumed (without checking) that the rows of `basis`
    are orthonormal, so that projecting `x` onto their span
    is simply ``np.dot(basis.T, np.dot(basis, x))``.
    """

    objective_vars = linear_constraint.objective_vars.copy()
    objective_vars['coneklass'] = 'projection'
    objective_vars['dualconeklass'] = 'projection_complement'

    @doc_template_user
    def constraint(self, x):
        projx = self.cone_prox(x)
        incone = np.linalg.norm(x-projx) / max([np.linalg.norm(x),1]) < self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x,  lipschitz=1):
        coefs = np.dot(self.basis, x)
        return np.dot(coefs, self.basis)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        return cone.proximal(self, proxq, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return linear_constraint.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.get_dual(self)

@objective_doc_templater()
class projection_complement(linear_constraint):

    """
    An atom representing a linear constraint.

    The orthogonal complement of `projection`: the argument is
    constrained to the orthogonal complement of the span of the rows of
    `basis`, which are assumed to be orthonormal.
    """

    objective_vars = linear_constraint.objective_vars.copy()
    objective_vars['coneklass'] = 'projection_complement'
    objective_vars['dualconeklass'] = 'projection'

    @doc_template_user
    def constraint(self, x):
        projx = self.cone_prox(x)
        incone = np.linalg.norm(projx) / max([np.linalg.norm(x),1]) < self.tol
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x,  lipschitz=1):
        coefs = np.dot(self.basis, x)
        return x - np.dot(coefs, self.basis)

    @doc_template_user
    def proximal(self, proxq, prox_control=None):
        return cone.proximal(self, proxq, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return linear_constraint.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.get_dual(self)

conjugate_cone_pairs = {}
for n1, n2 in [(projection, projection_complement),
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1
