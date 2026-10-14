"""Commonly used losses that are not in `regreg.smooth.glm`.

The only current example is the Huberized SVM.
"""

import numpy as np

from . import smooth_atom
from ..affine import astransform
from ..identity_quadratic import identity_quadratic
from ..atoms.seminorms import positive_part

class huberized_svm(smooth_atom):
    r"""Huberized SVM loss for a design matrix and binary labels.

    The labels are encoded as :math:`y_i \in \{-1, 1\}` and the loss is a
    smoothed version of the hinge loss

    .. math::

       \beta \mapsto \sum_i \left(1 - y_i x_i^T \beta\right)_+,

    obtained by (Nesterov) smoothing of the positive part: the quadratic
    :math:`\frac{\epsilon}{2} \|u\|^2_2` is subtracted inside the
    conjugate representation, where :math:`\epsilon` is
    `smoothing_parameter`.

    Parameters
    ----------
    X : ndarray or `regreg.affine.affine_transform`
        Design matrix.
    labels : ndarray
        Binary labels (any two distinct values; the larger is coded as 1).
    smoothing_parameter : float
        Smoothing parameter for the Huberized hinge loss.
    coef : float, optional
        Nonnegative scaling to be put in front of the loss.
    offset : ndarray, optional
        Offset to be subtracted in parameter space before evaluating
        the loss.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term to be added to the objective.
    initial : ndarray, optional
        Initial guess at the coefficients.
    """

    objective_template = r"""\ell^{\text{Huber SVM}}\left(%(var)s\right)"""

    def __init__(self, 
                 X, 
                 labels,
                 smoothing_parameter,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None):

        shape = astransform(X).input_shape
        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)
        
        if len(np.unique(labels)) > 2:
            raise ValueError('labels should be binary')
        # force labels to be \pm 1
        Y = 2 * (labels == np.max(labels)) - 1
        atom = positive_part.affine(-Y[:, None] * X, np.ones_like(Y), lagrange=1.)
        Q = identity_quadratic(smoothing_parameter, 0, 0, 0)
        self.smoothed_atom = atom.smoothed(Q)
                                         
    def smooth_objective(self, beta, mode='both', check_feasibility=False):
        """Evaluate the smooth objective, its gradient, or both.

        Parameters
        ----------
        beta : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `beta` is not in the domain.

        Returns
        -------
        f : float
            Objective value at `beta`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `beta`; returned when `mode` is 'grad'
            or 'both'.
        """

        value = self.smoothed_atom.smooth_objective(self.apply_offset(beta),
                                                    mode=mode,
                                                    check_feasibility=check_feasibility)
        if mode == 'both':
            return self.scale(value[0]), self.scale(value[1])
        return self.scale(value)

