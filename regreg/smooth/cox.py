"""Cox proportional hazards partial likelihood loss."""

from copy import copy

import numpy as np
from scipy.stats import rankdata

from . import smooth_atom, affine_smooth
from .cox_utils import (cox_objective,
                        cox_gradient,
                        cox_hessian)

class cox_loglike(smooth_atom):

    r"""Negative log partial likelihood of the Cox proportional hazards model.

    A saturated loss in the natural parameter (linear predictor)
    :math:`\eta`, with ties handled by Breslow's method:

    .. math::

       \ell(\eta) = -\sum_{i: \delta_i = 1} w_i \left[\eta_i -
           \log \left(\sum_{j: T_j \geq T_i} w_j e^{\eta_j}\right)\right]

    where :math:`T_i` are the event times, :math:`\delta_i` the censoring
    indicators and :math:`w_i` the case weights. The computations are done
    in C (see `regreg.smooth.cox_utils`).

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    event_times : ndarray
        Observed times for the Cox proportional hazards model.
    censoring : ndarray
        Censoring indicator for the Cox proportional hazards model:
        1 indicates the observation is a failure, 0 a censored
        observation.
    coef : float, optional
        Nonnegative scaling to be put in front of the loss.
    offset : ndarray, optional
        Offset to be subtracted from the natural parameter before
        evaluating the loss.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term to be added to the objective.
    initial : ndarray, optional
        Initial guess at the coefficients.
    case_weights : ndarray, optional
        Non-negative case weights. Defaults to all ones.
    """

    objective_template = r"""\ell^{\text{Cox}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 event_times,
                 censoring,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None,
                 case_weights=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        self.data = np.asarray([event_times, censoring]).T

        self._ordering = np.argsort(self.event_times).astype(np.intp)
        self._rankmax = (rankdata(self.event_times, method='max') - 1).astype(np.intp)
        self._rankmin = (rankdata(self.event_times, method='min') - 1).astype(np.intp)

        if case_weights is not None:
            case_weights = np.asarray(case_weights).astype(float)
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
        else:
            self.case_weights = np.ones(self.event_times.shape, float)

        # buffers to store results used by C code
        self._G = np.zeros(self.event_times.shape, float) # gradient 
        self._exp_buffer = np.zeros(self.event_times.shape, float) # exp(eta)
        self._exp_accum = np.zeros(self.event_times.shape, float) # accum of exp(eta)
        self._expZ_accum = np.zeros(self.event_times.shape, float) # accum of Z*exp(eta)
        self._outer_1st = np.zeros(self.event_times.shape, float) # for log(W)
        self._outer_2nd = np.zeros(self.event_times.shape, float) # used in Hessian

    def smooth_objective(self, 
                         natural_param, 
                         mode='both', 
                         check_feasibility=False,
                         case_weights=None):
        """Evaluate the smooth objective, its gradient, or both.

        Parameters
        ----------
        natural_param : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `natural_param` is not in the domain.
        case_weights : ndarray, optional
            Non-negative case weights, multiplied with any case weights
            stored on the loss.

        Returns
        -------
        f : float
            Objective value at `natural_param`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `natural_param`; returned when `mode` is 'grad'
            or 'both'.
        """
        
        eta = natural_param # shorthand

        eta = self.apply_offset(eta)

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        censoring = self.censoring

        if mode in ['both', 'grad']:
            G = cox_gradient(self._G,
                             eta,
                             self._exp_buffer,
                             self._exp_accum,
                             self._outer_1st,
                             cw,
                             censoring,
                             self._ordering,
                             self._rankmin,
                             self._rankmax,
                             eta.shape[0])

        if mode in ['both', 'func']:
            F = cox_objective(eta,
                              self._exp_buffer,
                              self._exp_accum,
                              self._outer_1st,
                              cw,
                              censoring,
                              self._ordering,
                              self._rankmin,
                              self._rankmax,
                              eta.shape[0])

        if mode == 'both':
            return self.scale(F), self.scale(G)
        elif mode == 'grad':
            return self.scale(G)
        elif mode == 'func':
            return self.scale(F)
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian_mult(self, 
                     natural_param, 
                     right_vector,
                     case_weights=None):
        """Multiply the Hessian of the loss by a vector.

        Parameters
        ----------
        natural_param : ndarray
            Parameters at which the Hessian is evaluated.
        right_vector : ndarray
            Vector multiplied on the right of the Hessian.
        case_weights : ndarray, optional
            Non-negative case weights.

        Returns
        -------
        hess_mult : ndarray
            The product of the Hessian at `natural_param` with
            `right_vector`.
        """

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        eta = natural_param # shorthand
        eta = self.apply_offset(eta)
        censoring = self.censoring

        H = np.zeros(eta.shape, float)

        return cox_hessian(H,
                           eta,
                           right_vector,
                           self._exp_buffer,
                           self._exp_accum,
                           self._expZ_accum,
                           self._outer_1st,
                           self._outer_2nd,
                           cw,
                           censoring,
                           self._ordering,
                           self._rankmin,
                           self._rankmax,
                           eta.shape[0])

    def get_data(self):
        """Return the data, with columns ``(event_times, censoring)``."""
        return np.array([self.event_times, self.censoring]).T

    def set_data(self, data):
        """Set the data.

        Parameters
        ----------
        data : ndarray
            Array of shape ``(n, 2)`` with columns
            ``(event_times, censoring)``.
        """
        event_times, censoring = np.asarray(data).T
        self.event_times, self.censoring = (np.asarray(event_times),
                                            np.asarray(censoring).astype(np.intp))

    data = property(get_data, set_data)

    def __copy__(self):
        event_times, censoring = self.event_times, self.censoring
        return cox_loglike(self.shape,
                           copy(event_times),
                           copy(censoring),
                           coef=self.coef,
                           offset=copy(self.offset),
                           quadratic=copy(self.quadratic),
                           initial=copy(self.coefs),
                           case_weights=copy(self.case_weights))

    def subsample(self, case_idx):
        """Create a saturated loss using a subsample of the data.

        Makes a copy of the loss and multiplies its case weights by the
        indicator of `case_idx`.

        Parameters
        ----------
        case_idx : index
            Indices of ``np.arange(n)`` to keep.

        Returns
        -------
        subsample_loss : `regreg.smooth.smooth_atom`
            Copy of the loss with zero case weight for all cases not in
            `case_idx`.
        """
        loss_cp = copy(self)
        if loss_cp.case_weights is None:
            case_weights = loss_cp.case_weights = np.ones(self.shape[0])
        else:
            case_weights = loss_cp.case_weights

        idx_bool = np.zeros_like(case_weights, np.bool)
        idx_bool[case_idx] = 1

        case_weights *= idx_bool
        return loss_cp

    def latexify(self, var=None, idx=''):
        # a trick to get latex representation looking right
        # coxph should be written similar to logistic
        # composed with a linear transform (TODO)
        """Return a LaTeX representation of the loss.

        Parameters
        ----------
        var : str, optional
            LaTeX name of the argument.
        idx : str, optional
            Index appended as a subscript.

        Returns
        -------
        latex : str
        """
        return smooth_atom.latexify(self, var=var, idx=idx)
