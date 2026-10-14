"""Quadratic losses, such as squared error and the signal approximator."""

import numpy as np
import warnings
try:
    from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded
except ImportError:
    warnings.warn('cannot import some cholesky solvers from scipy')

from ..affine import affine_transform, astransform, composition, scaler
from ..smooth import smooth_atom
from ..problems.composite import smooth_conjugate
from ..atoms.cones import zero
from ..atoms import _work_out_conjugate
from ..identity_quadratic import identity_quadratic

class quadratic_loss(smooth_atom):
    r"""Half of a (possibly weighted) squared :math:`\ell_2` norm.

    Represents

    .. math::

       \beta \mapsto \frac{C}{2} \beta^T Q \beta

    with :math:`Q = I` if `Q` is None.

    Parameters
    ----------
    shape : int or tuple
        Shape of the argument to `smooth_objective`.
    Q : ndarray or `regreg.affine.affine_transform`, optional
        Positive definite matrix, defaults to the identity. If `Qdiag`
        then `Q` is one-dimensional and holds the diagonal.
    Qdiag : bool, optional
        Is the quadratic form diagonal?
    coef : float, optional
        Scalar multiple :math:`C` to be applied (must be nonnegative).
    offset : ndarray, optional
        Vector to be subtracted from the argument before evaluating
        `smooth_objective`.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term to be added to the overall objective.
    initial : ndarray, optional
        Initial value for the coefficients.
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
            if Qdiag:
                self.Q_transform = scaler(Q)
            else:
                self.Q_transform = affine_transform(Q, None, False)
    @staticmethod
    def fromarray(Q, 
                  offset=None,
                  quadratic=None,
                  initial=None):
        """Create a quadratic loss from a (dense) matrix.

        Parameters
        ----------
        Q : ndarray
            Positive definite matrix of shape ``(p, p)``.
        offset : ndarray, optional
            Vector to be subtracted from the argument before evaluating the
            loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial value for the coefficients.

        Returns
        -------
        loss : `quadratic_loss`
        """
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
        r"""Create the quadratic loss of a linear transform.

        The loss is :math:`\beta \mapsto \frac{1}{2}\|X\beta\|^2_2`, with
        :math:`Q = X^T X` represented as a composition of transforms.

        Parameters
        ----------
        transform : ndarray or `regreg.affine.affine_transform`
            The linear transform :math:`X`.
        offset : ndarray, optional
            Vector to be subtracted from the argument before evaluating the
            loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial value for the coefficients.

        Returns
        -------
        loss : `quadratic_loss`
        """
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
        """Create a quadratic loss with diagonal quadratic form.

        Parameters
        ----------
        D : ndarray
            Diagonal of the (positive definite) quadratic form.
        offset : ndarray, optional
            Vector to be subtracted from the argument before evaluating the
            loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial value for the coefficients.

        Returns
        -------
        loss : `quadratic_loss`
        """
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
        """Evaluate the smooth objective, its gradient, or both.

        Parameters
        ----------
        x : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `x` is not in the domain.

        Returns
        -------
        f : float
            Objective value at `x`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `x`; returned when `mode` is 'grad'
            or 'both'.
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
        """Return the convex conjugate of the loss.

        Parameters
        ----------
        factor : bool, optional
            For a non-diagonal `Q`, whether to compute a Cholesky
            factorization to represent the conjugate. Requires `Q` to be
            an ndarray.

        Returns
        -------
        conjugate : `regreg.smooth.smooth_atom`

        Raises
        ------
        ValueError
            If `Q` is not diagonal and `factor` is False, or if `factor`
            is True and `Q` is not an ndarray.
        """

        if self.Q is None:
            as_quad = (identity_quadratic(self.coef, self.offset, 0, 0) + self.quadratic).collapsed()
            return smooth_conjugate(zero(self.shape, quadratic=as_quad))
        else:
            # Including its quadratic term, the loss is
            #   f(x) = C/2 (x-a)^T Q (x-a) + kappa/2 ||x||^2 + l^T x + c
            #        = 1/2 x^T M x - b^T x + k
            # with M = C Q + kappa I, b = C Q a - l and k = C/2 a^T Q a + c,
            # so its conjugate is f^*(u) = 1/2 (u + b)^T M^{-1} (u + b) - k.
            sq = self.quadratic.collapsed()
            if self.offset is None:
                Qa = np.zeros(self.shape)
                k = sq.constant_term
            else:
                Qa = self.Q_transform.linear_map(self.offset)
                k = self.coef * np.sum(self.offset * Qa) / 2. + sq.constant_term
            b = self.coef * Qa - sq.linear_term

            if self.Qdiag:
                Minv = 1. / (self.coef * np.asarray(self.Q) + sq.coef)
            elif factor:
                if not isinstance(self.Q, np.ndarray):
                    raise ValueError('factor=True requires Q to be an ndarray')
                Minv = cholesky(self.coef * self.Q + sq.coef * np.identity(self.shape[0]))
            else:
                raise ValueError('factor is False, so no factorization was done')

            return quadratic_loss(self.shape,
                                  Q=Minv,
                                  Qdiag=self.Qdiag,
                                  offset=-b,
                                  quadratic=identity_quadratic(0, 0, 0, -k),
                                  coef=1.)

class cholesky(object):

    """Linear transform given by multiplication by :math:`Q^{-1}`.

    Given :math:`Q > 0`, represents multiplication by :math:`Q^{-1}` by
    first computing the Cholesky decomposition of :math:`Q`. The
    transform is symmetric, so `adjoint_map` equals `linear_map`.

    Parameters
    ----------
    Q : ndarray
        Positive definite matrix (in banded storage if `banded`).
    cholesky : optional
        A precomputed Cholesky factorization, as returned by
        `scipy.linalg.cho_factor` (or `scipy.linalg.cholesky_banded` if
        `banded`).
    banded : bool, optional
        Whether `Q` is stored in banded form.
    """

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
        """Return :math:`Q^{-1}x`."""
        if not self.banded:
            return cho_solve(self._cholesky, x)
        else:
            return cho_solve_banded(self._cholesky, x)

    def affine_map(self, x):
        """Return :math:`Q^{-1}x` (the transform has no offset)."""
        return self.linear_map(x)

    def adjoint_map(self, x):
        """Return :math:`Q^{-1}x` (the transform is symmetric)."""
        return self.linear_map(x)

def squared_error(X, Y, coef=1):
    r"""Least squares loss with design :math:`X`.

    .. math::

       \beta \mapsto \frac{C}{2} \|X\beta - Y\|^2_2

    Parameters
    ----------
    X : ndarray or `regreg.affine.affine_transform`
        Design matrix.
    Y : ndarray
        Response vector.
    coef : float, optional
        Scalar multiple :math:`C` (must be nonnegative).

    Returns
    -------
    loss : `regreg.smooth.affine_smooth`
    """
    atom = quadratic_loss.affine(X, -Y, coef=coef)
    atom.atom.objective_vars = dict(atom.atom.objective_vars, offset='Y')
    atom.atom.objective_template = r"""\frac{%(coef)s}{2}\left\|%(var)s\right\|^2_2"""
    return atom

def signal_approximator(signal, coef=1):
    r"""Least squares loss with identity design.

    .. math::

       \beta \mapsto \frac{C}{2} \|\beta - Y\|^2_2

    Parameters
    ----------
    signal : ndarray
        The signal :math:`Y` to be approximated.
    coef : float, optional
        Scalar multiple :math:`C` (must be nonnegative).

    Returns
    -------
    loss : `quadratic_loss`
    """
    atom = quadratic_loss.shift(signal, coef=coef)    
    atom.objective_vars = dict(atom.objective_vars, offset='Y')
    atom.objective_template = r"""\frac{%(coef)s}{2}\left\|%(var)s\right\|^2_2"""
    return atom


