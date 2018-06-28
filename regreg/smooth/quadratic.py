import numpy as np
import warnings
try:
    from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded
except ImportError:
    warnings.warn('cannot import some cholesky solvers from scipy')

from ..affine import affine_transform, astransform, composition
from ..smooth import smooth_atom
from ..problems.composite import smooth_conjugate
from ..atoms.cones import zero
from ..atoms import _work_out_conjugate
from ..identity_quadratic import identity_quadratic

class quadratic_loss(smooth_atom):
    """
    Half of the square of the l2 norm

    Parameters
    ----------

    shape : tuple
       Shape of argument to `smooth_objective`

    Q: ndarray 
       positive definite matrix (optional),
       defaults to identity. If `Qdiag` then
       `Q` is one-dimensional.

    Qdiag : bool
       Is the quadratic form diagonal?

    coef : float (optional)
       Scalar multiple to be applied (must be nonnegative)

    offset : ndarray (optional)
       Vector to be subtracted before evaluating `smooth_objective`. 

    quadratic : `identity_quadratic` (optional)
       Instance of `identity_quadratic` to be added to overall
       objective.

    initial : ndarray (optional)
       Initial value for coefficients.

    """

    objective_vars = smooth_atom.objective_vars.copy()
    objective_vars['Q'] = 'Q'
    objective_template = r"""\frac{%(coef)s}{2} \cdot %(var)s^T %(Q)s %(var)s"""

    def __init__(self, 
                 shape, 
                 Q=None, 
                 Qdiag=False,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None):
        smooth_atom.__init__(self,
                             shape,
                             coef=coef,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)
        
        self.Q = Q
        self.Qdiag = Qdiag
        if self.Q is not None:
            self.Q_transform = affine_transform(Q, None, Qdiag)

    @staticmethod
    def fromarray(Q, 
                  offset=None,
                  quadratic=None,
                  initial=None):
        return quadratic_loss((Q.shape[0],), 
                         Q=Q,
                         offset=offset,
                         quadratic=quadratic,
                         initial=initial)

    @staticmethod
    def squared_transform(transform, 
                          offset=None,
                          quadratic=None,
                          initial=None):
        transform = astransform(transform)
        Q = composition(transform.T, transform)
        return quadratic_loss(Q.input_shape, 
                              Q=Q,
                              offset=offset,
                              quadratic=quadratic,
                              initial=initial)

    @staticmethod
    def diagonal(D, 
                 offset=None,
                 quadratic=None,
                 initial=None):
        D = np.asarray(D)
        return quadratic_loss((D.shape[0],), 
                         Q=D,
                         Qdiag=True,
                         offset=offset,
                         quadratic=quadratic,
                         initial=initial)


    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, coef=%s, Q=%s, Qdiag=%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.coef),
                 repr(self.Q),
                 repr(self.Qdiag),
                 str(self.offset))
        else:
            return "%s(%s, coef=%s, Q=%s, Qdiag=%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.coef),
                 repr(self.Q),
                 repr(self.Qdiag),
                 str(self.offset),
                 self.quadratic)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        x = self.apply_offset(x)
        if self.Q is None:
            if mode == 'both':
                f, g  = self.scale(np.linalg.norm(x)**2) / 2., self.scale(x)
                return f, g
            elif mode == 'grad':
                f, g = None, self.scale(x)
                return g
            elif mode == 'func':
                f, g = self.scale(np.linalg.norm(x)**2) / 2., None
                return f
            else:
                raise ValueError("mode incorrectly specified")
        else:
            if mode == 'both':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) / 2., self.scale(self.Q_transform.linear_map(x))
                return f, g
            elif mode == 'grad':
                f, g = None, self.scale(self.Q_transform.linear_map(x))
                return g
            elif mode == 'func':
                f, g = self.scale(np.sum(x * self.Q_transform.linear_map(x))) / 2., None
                return f
            else:
                raise ValueError("mode incorrectly specified")

    def get_conjugate(self, factor=False):

        if self.Q is None:
            as_quad = (identity_quadratic(self.coef, self.offset, 0, 0) + self.quadratic).collapsed()
            return smooth_conjugate(zero(self.shape, quadratic=as_quad))
        else:
            #XXX this needs to be tested
            sq = self.quadratic.collapsed()
            if self.offset is not None:
                sq.linear_term -= self.scale(self.Q_transform.linear_map(self.offset))
            if self.Q_transform.diagD:
                return quadratic_loss(self.shape,
                                      Q=1./(self.coef*self.Q_transform.linear_operator + sq.coef),
                                      offset=offset,
                                      quadratic=outq, 
                                      coef=1.,
                                      Qdiag=True)
            elif factor:
                return quadratic_loss(self.shape,
                                      Q=cholesky(self.coef * self.Q + sq.coef * np.identity(self.shape)),
                                      Qdiag=False,
                                      offset=offset,
                                      quadratic=outq,
                                      coef=1.)
            else:
                raise ValueError('factor is False, so no factorization was done')

class cholesky(object):

    '''

    Given :math:`Q > 0`, returns a linear transform
    that is multiplication by :math:`Q^{-1}` by
    first computing the Cholesky decomposition of :math:`Q`.

    Parameters
    ----------

    Q: array
       positive definite matrix 

    '''

    def __init__(self, Q, cholesky=None, banded=False):
        self.input_shape = Q.shape[0]
        self.output_shape = Q.shape[0]
        self.affine_offset = None
        self._Q = Q
        self.banded = banded
        if cholesky is None:
            if not self.banded:
                self._cholesky = cho_factor(Q)
            else:
                self._cholesky = cholesky_banded(Q)
        else:
            self._cholesky = cholesky

    def linear_map(self, x):
        if not self.banded:
            return cho_solve(self._cholesky, x)
        else:
            return cho_solve_banded(self._cholesky, x)

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        return self.linear_map(x)

def squared_error(X, Y, coef=1):
    r"""
    Least squares with design $X$

    .. math::

       \frac{C}{2} \|X\beta-Y\|^2_2

    Parameters
    ----------

    X : affine_transform
        Design matrix

    Y : np.array

    """
    atom = quadratic_loss.affine(X, -Y, coef=coef)
    atom.atom.objective_vars['offset'] = 'Y'
    atom.atom.objective_template = r"""\frac{%(coef)s}{2}\left\|%(var)s\right\|^2_2"""
    return atom

def signal_approximator(signal, coef=1):
    """
    Least squares with design $I$

    .. math::

       \frac{C}{2} \|\beta-Y\|^2_2

    """
    atom = quadratic_loss.shift(signal, coef=coef)    
    atom.objective_vars['offset'] = 'Y'
    atom.objective_template = r"""\frac{%(coef)s}{2}\left\|%(var)s\right\|^2_2"""
    return atom


