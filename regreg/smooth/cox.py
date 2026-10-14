"""Cox proportional hazards partial likelihood loss."""

from copy import copy

import numpy as np
from coxdev import CoxDeviance

from . import smooth_atom, combine_case_weights, affine_smooth

class cox_loglike(smooth_atom):

    r"""Negative log partial likelihood of the Cox proportional hazards model.

    A saturated loss in the natural parameter (linear predictor)
    :math:`\eta`. With Breslow's method for ties it is

    .. math::

       \ell(\eta) = -\sum_{i: \delta_i = 1} w_i \left[\eta_i -
           \log \left(\sum_{j: T_j \geq T_i} w_j e^{\eta_j}\right)\right]

    where :math:`T_i` are the event times, :math:`\delta_i` the censoring
    indicators and :math:`w_i` the case weights. The objective, gradient
    and Hessian are computed by `coxdev`.

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
    tie_breaking : {'breslow', 'efron'}, optional
        Method for handling tied event times.
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
                 case_weights=None,
                 tie_breaking='breslow'):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        self.tie_breaking = tie_breaking
        self.data = np.asarray([event_times, censoring]).T

        if case_weights is not None:
            case_weights = np.asarray(case_weights).astype(float)
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
        else:
            self.case_weights = np.ones(self.event_times.shape, float)

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

        eta = self.apply_offset(natural_param)

        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        # coxdev returns the deviance 2 * (loglik_sat - loglik) and its
        # gradient, so the negative log partial likelihood is
        # deviance / 2 - loglik_sat, with gradient gradient / 2
        result = self._coxdev(eta, cw)

        if mode == 'both':
            return (self.scale(result.deviance / 2 - result.loglik_sat),
                    self.scale(result.gradient / 2))
        elif mode == 'grad':
            return self.scale(result.gradient / 2)
        elif mode == 'func':
            return self.scale(result.deviance / 2 - result.loglik_sat)
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

        eta = self.apply_offset(natural_param)

        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        # the information matrix of coxdev is the Hessian of the
        # negative log partial likelihood (no factor of 2)
        info = self._coxdev.information(eta, cw)
        return self.scale(info @ right_vector)

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
        self.event_times, self.censoring = (np.asarray(event_times, float),
                                            np.asarray(censoring).astype(np.intp))
        self._coxdev = CoxDeviance(event=self.event_times,
                                   status=self.censoring,
                                   tie_breaking=self.tie_breaking)

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
                           case_weights=copy(self.case_weights),
                           tie_breaking=self.tie_breaking)

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

        idx_bool = np.zeros_like(case_weights, bool)
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
