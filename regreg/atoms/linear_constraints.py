import numpy as np
from scipy import sparse
from copy import copy
import warnings

from ..problems.composite import composite, nonsmooth
from .cones import cone, affine_cone
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import doc_template_user

@objective_doc_templater()
class linear_constraint(cone):

    """
    This class allows specifications of linear constraints
    of the form :math:`x \in \text{row}(L)` by specifying
    an orthonormal basis for the rowspace of :math:`L`.

    If the constraint is of the form :math:`Ax=0`, then
    this linear constraint can be created using the
    *linear* classmethod of the *zero* cone in *regreg.cones*.
    
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
    conjugate = property(get_conjugate)

    @classmethod
    @doc_template_user
    def linear(cls, linear_operator, basis, diag=False,
               linear_term=None, offset=None):
        l = linear_transform(linear_operator, diag=diag)
        cone = cls(l.input_shape, basis,
                   linear_term=linear_term, offset=offset)
        return affine_cone(cone, l)

@objective_doc_templater()
class projection(linear_constraint):

    objective_vars = linear_constraint.objective_vars.copy()
    objective_vars['coneklass'] = 'projection'
    objective_vars['dualconeklass'] = 'projection_complement'

    """
    An atom representing a linear constraint.
    It is specified via a matrix that is assumed
    to be an set of row vectors spanning the space.

    Notes
    =====

    It is assumed (without checking) that the rows of basis
    are orthonormal, so that projecting *x* onto their span
    is simply np.dot(basis.T, np.dot(basis, x))
    """

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
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class projection_complement(linear_constraint):

    """
    An atom representing a linear constraint.
    The orthogonal complement of projection, it is specified
    with an orthonormal basis for the complement
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
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

conjugate_cone_pairs = {}
for n1, n2 in [(projection, projection_complement),
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1
