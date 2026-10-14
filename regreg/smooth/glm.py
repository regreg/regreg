r"""Generalized linear model losses.

These are losses of the form :math:`\beta \mapsto \ell(X\beta)` for
a saturated loss :math:`\ell`. The `glm` class composes a saturated
loss (e.g. `gaussian_loglike`, `poisson_loglike`, `huber_loss`, or the
binary and Cox losses imported from `regreg.smooth.binary` and
`regreg.smooth.cox`) with a design matrix, and has constructors for the
common models.
"""

import warnings
from copy import copy

import numpy as np
from scipy import sparse
from scipy.stats import norm as normal_dbn

from . import smooth_atom, combine_case_weights, affine_smooth
from ..affine import (astransform, 
                      linear_transform, 
                      affine_transform,
                      scaler)
from ..identity_quadratic import identity_quadratic
from ..atoms.seminorms import l1norm
from .cox import cox_loglike
from .binary import (logistic_loglike,
                     probit_loglike,
                     cloglog_loglike,
                     huber_svm)

class glm(smooth_atom):

    r"""A general linear model, usually a negative log-likelihood.

    The loss is :math:`\beta \mapsto \ell(X\beta)` for a response
    :math:`Y` whose mean is modelled through :math:`X\beta`, where
    :math:`\ell` is a saturated loss.

    Usual examples are Gaussian (least squares regression), logistic
    regression and Poisson log-linear regression. The Huber loss is also
    implemented as an example.
    """

    def __init__(self, 
                 X, 
                 Y, 
                 loss, 
                 quadratic=None, 
                 initial=None,
                 offset=None,
                 case_weights=None):

        r"""Initialize the general linear model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            The design matrix.
        Y : ndarray
            The response.
        loss : `regreg.smooth.smooth_atom`
            The saturated loss, taking arguments the same size as `Y`. For
            Gaussian regression the loss is just the map
            :math:`\mu \mapsto \|\mu - Y\|^2_2/2`.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term added to the objective.
        initial : ndarray, optional
            An initial guess at the minimizer.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        case_weights : ndarray, optional
            Non-negative case weights. Defaults to all ones.
        """

        self.saturated_loss = loss
        self.data = X, Y
        self.affine_atom = affine_smooth(loss, X)
        if case_weights is None:
            case_weights = np.ones(X.shape[0])
        self.case_weights = case_weights
        
        if self.case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != loss.shape[:1]:
                raise ValueError('case_weights should have same shape as loss.shape[:1]')

        smooth_atom.__init__(self,
                             X.shape[1],
                             coef=1.,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)

    def smooth_objective(self, 
                         beta, 
                         mode='func', 
                         check_feasibility=False):
        r"""Evaluate the loss :math:`\ell(X\beta)`, its gradient, or both.

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
        beta = self.apply_offset(beta)
        linear_pred = self.affine_atom.affine_transform.dot(beta)
        value = self.saturated_loss.smooth_objective(linear_pred, 
                                                     mode=mode, 
                                                     check_feasibility=check_feasibility,
                                                     case_weights=self.case_weights)

        if mode == 'func':
            return self.scale(value)
        elif mode == 'grad':
            return self.scale(self.affine_atom.affine_transform.adjoint_map(value))
        else:
            return self.scale(value[0]), self.scale(self.affine_atom.affine_transform.adjoint_map(value[1]))

    def get_data(self):
        """Return the data ``(X, Y)``."""
        return self._X, self.saturated_loss.data

    def set_data(self, data):
        """Set the data.

        Parameters
        ----------
        data : tuple
            The pair ``(X, Y)`` of design and response.
        """
        X, Y = data
        self._transform = astransform(X)
        self._X = X
        self._is_transform = id(self._X) == id(self._transform) # i.e. astransform was a nullop
        self.saturated_loss.data = Y

    data = property(get_data, set_data, doc="Data for the general linear model.")

    def linear_predictor(self, beta):
        r"""Compute the linear predictor :math:`X\beta`.

        Parameters
        ----------
        beta : ndarray
            Parameters.

        Returns
        -------
        linpred : ndarray
            The linear predictor.
        """
        # can have an `R`-type offset by using affine_map here
        return self._transform.linear_map(beta)

    def objective(self, beta):
        r"""Compute the loss :math:`\ell(X\beta)`.

        Parameters
        ----------
        beta : ndarray
            Parameters.

        Returns
        -------
        objective : float
            Value of the loss at :math:`\beta`.
        """
        return self.smooth_objective(beta, 'func')

    def gradient(self, beta):
        r"""Compute the gradient of the loss :math:`\nabla \ell(X\beta)`.

        Parameters
        ----------
        beta : ndarray
            Parameters.

        Returns
        -------
        grad : ndarray
            Gradient of the loss at :math:`\beta`.
        """

        return self.smooth_objective(beta, 'grad')

    def hessian(self, beta):
        r"""Compute the Hessian of the loss :math:`\nabla^2 \ell(X\beta)`.

        Requires the design to be an ndarray or a `scipy.sparse` matrix, and
        the saturated loss to have a `hessian` or `hessian_mult` method.

        Parameters
        ----------
        beta : ndarray
            Parameters.

        Returns
        -------
        hess : ndarray
            Hessian of the loss at :math:`\beta`, i.e. :math:`X^T W X`.
        """

        linpred = self.linear_predictor(self.apply_offset(beta))
        X = self.data[0]
        if self._is_transform:
            raise ValueError('refusing to form Hessian for arbitrary affine_transform, use an ndarray or scipy.sparse')

        if not hasattr(self.saturated_loss, 'hessian'):
            if not hasattr(self.saturated_loss, 'hessian_mult'):
                raise ValueError('loss has no hessian or hessian_mult method')
            right_mult = np.zeros(X.shape)
            for j in range(X.shape[1]):
                right_mult[:,j] = self.saturated_loss.hessian_mult(linpred, 
                                                                   X[:,j], 
                                                                   case_weights=self.case_weights)
        else:
            W = self.saturated_loss.hessian(linpred, 
                                            case_weights=self.case_weights)
            right_mult = W[:,None] * X
        if not sparse.issparse(X): # assuming it is an ndarray
            return X.T.dot(right_mult)
        else:
            return X.T * right_mult

    def latexify(self, var=None, idx=''):
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
        return self.affine_atom.latexify(var=var, idx=idx)

    def __copy__(self):
        klass = self.__class__
        X, Y = self.data
        return klass(copy(X), 
                     copy(Y), 
                     copy(self.saturated_loss), 
                     quadratic=copy(self.quadratic),
                     initial=copy(self.coefs),
                     offset=copy(self.offset),
                     case_weights=self.case_weights.copy())

    def subsample(self, idx):
        """Create a loss using a subsample of the data.

        Makes a copy of the loss and multiplies its case weights by the
        indicator of `idx`.

        Parameters
        ----------
        idx : index
            Indices of ``np.arange(n)`` to keep.

        Returns
        -------
        subsample_loss : `glm`
            Copy of the loss with zero case weight for all cases not in
            `idx`.
        """

        subsample_loss = copy(self)
        n = subsample_loss.saturated_loss.shape[0]

        idx_bool = np.zeros(n, bool)
        idx_bool[idx] = 1

        subsample_loss.case_weights *= idx_bool

        return subsample_loss
        
    @classmethod
    def gaussian(klass,
                 X, 
                 response,
                 case_weights=None,
                 coef=1., 
                 saturated_offset=None,
                 offset=None,
                 quadratic=None, 
                 initial=None):
        """Create a loss for a Gaussian regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        response : ndarray
            Response vector.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = gaussian_loglike(response.shape,
                                response,
                                coef=coef,
                                offset=saturated_offset)

        return klass(X, 
                     response, 
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def logistic(klass, 
                 X, 
                 successes, 
                 trials=None,
                 case_weights=None,
                 coef=1., 
                 offset=None,
                 saturated_offset=None,
                 quadratic=None, 
                 initial=None):
        """Create a loss for a logistic regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        successes : ndarray
            Number of successes for each case (non-negative).
        trials : ndarray, optional
            Number of trials for each case. If None, the response must be
            binary and ``np.ones_like(successes)`` is used.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = logistic_loglike(successes.shape,
                                successes,
                                coef=coef,
                                offset=saturated_offset,
                                trials=trials)

        return klass(X, 
                     (successes, loss.trials),
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def probit(klass, 
               X, 
               successes, 
               trials=None,
               case_weights=None,
               coef=1., 
               offset=None,
               saturated_offset=None,
               quadratic=None, 
               initial=None):
        """Create a loss for a probit regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        successes : ndarray
            Number of successes for each case (non-negative).
        trials : ndarray, optional
            Number of trials for each case. If None, the response must be
            binary and ``np.ones_like(successes)`` is used.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = probit_loglike(successes.shape,
                              successes,
                              coef=coef,
                              offset=saturated_offset,
                              trials=trials)

        return klass(X, 
                     (successes, loss.trials),
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def cloglog(klass, 
                X, 
                successes, 
                trials=None,
                case_weights=None,
                coef=1., 
                offset=None,
                saturated_offset=None,
                quadratic=None, 
                initial=None):
        """Create a loss for a complementary log-log regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        successes : ndarray
            Number of successes for each case (non-negative).
        trials : ndarray, optional
            Number of trials for each case. If None, the response must be
            binary and ``np.ones_like(successes)`` is used.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = cloglog_loglike(successes.shape,
                               successes,
                               coef=coef,
                               offset=saturated_offset,
                               trials=trials)

        return klass(X, 
                     (successes, loss.trials),
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def poisson(klass,
                X, 
                counts,
                case_weights=None,
                coef=1., 
                saturated_offset=None,
                offset=None,
                quadratic=None, 
                initial=None):
        """Create a loss for a Poisson regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        counts : ndarray
            Response vector of non-negative integer counts.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = poisson_loglike(counts.shape,
                               counts,
                               offset=saturated_offset,
                               coef=coef)

        return klass(X, counts, loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def huber(klass,
              X, 
              response,
              smoothing_parameter,
              case_weights=None,
              coef=1., 
              saturated_offset=None,
              offset=None,
              quadratic=None, 
              initial=None):
        """Create a loss for a regression model using the Huber loss.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        response : ndarray
            Response vector.
        smoothing_parameter : float
            Smoothing parameter for the Huber loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = huber_loss(response.shape,
                          response,
                          smoothing_parameter,
                          offset=saturated_offset,
                          coef=coef)

        return klass(X, 
                     response, 
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def huber_svm(klass,
                  X, 
                  successes,
                  smoothing_parameter,
                  case_weights=None,
                  coef=1., 
                  offset=None,
                  saturated_offset=None,
                  quadratic=None, 
                  initial=None):
        """Create a loss for a binary model using the Huber SVM loss.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        successes : ndarray
            Binary (0/1) response vector.
        smoothing_parameter : float
            Smoothing parameter for the Huberized hinge loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = huber_svm(successes.shape,
                         successes,
                         smoothing_parameter,
                         offset=saturated_offset,
                         coef=coef)

        return klass(X, 
                     successes,
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def cox(klass, 
            X, 
            event_times,
            censoring,
            case_weights=None,
            coef=1., 
            saturated_offset=None,
            offset=None,
            quadratic=None, 
            initial=None,
            tie_breaking='breslow'):
        """Create a loss for a Cox proportional hazards regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        event_times : ndarray
            Observed times for the Cox proportional hazards model.
        censoring : ndarray
            Censoring indicator for the Cox proportional hazards model:
            1 indicates the observation is a failure, 0 a censored
            observation.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        saturated_offset : ndarray, optional
            Offset to be subtracted in the saturated (natural) parameter
            space before evaluating the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.
        tie_breaking : {'breslow', 'efron'}, optional
            Method for handling tied event times.

        Returns
        -------
        glm_obj : `regreg.smooth.glm.glm`
            General linear model loss.
        """

        loss = cox_loglike(event_times.shape,
                           event_times,
                           censoring,
                           offset=saturated_offset,
                           coef=coef,
                           tie_breaking=tie_breaking)

        return klass(X, 
                     np.array([event_times, censoring]).T,
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)


class gaussian_loglike(smooth_atom):

    r"""The Gaussian loss for observations :math:`y`.

    .. math::

       \mu \mapsto \frac{1}{2} \sum_i w_i (y_i - \mu_i)^2

    where :math:`w_i` are the case weights.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    response : ndarray
        Response vector :math:`y`.
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
        Non-negative case weights.
    """

    objective_template = r"""\ell^{\text{Gauss}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape,
                 response, 
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

        if sparse.issparse(response):
            self.response = response.toarray().flatten()
        else:
            self.response = np.asarray(response)

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.response.shape:
                raise ValueError('case_weights should have same shape as response')
        else:
            self.case_weights = None

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
        
        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        natural_param = self.apply_offset(natural_param)
        resid = natural_param - self.response 

        if mode == 'both':
            f, g = self.scale(np.sum(cw*resid**2)) / 2., self.scale(cw*resid)
            return f, g
        elif mode == 'grad':
            return self.scale(cw*resid) 
        elif mode == 'func':
            return self.scale(np.sum(cw*resid**2)) / 2.
        else:
            raise ValueError("mode incorrectly specified")
            
    # Begin loss API

    def hessian(self, 
                natural_param, 
                case_weights=None):
        """Compute the diagonal of the Hessian of the loss.

        Parameters
        ----------
        natural_param : ndarray
            Parameters at which the Hessian is evaluated.
        case_weights : ndarray, optional
            Non-negative case weights.

        Returns
        -------
        hess : ndarray
            1D array holding the diagonal of the Hessian evaluated at
            `natural_param`.
        """
        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)
        return self.scale(np.ones_like(natural_param) * cw)

    def get_data(self):
        """Return the response."""
        return self.response

    def set_data(self, data):
        """Set the response."""
        self.response = data

    data = property(get_data, set_data)

    def __copy__(self):
        return gaussian_loglike(self.shape,
                                copy(self.response),
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

        idx_bool = np.zeros_like(case_weights, bool)
        idx_bool[case_idx] = 1

        case_weights *= idx_bool
        return loss_cp

    # End loss API

    def mean_function(self, eta):
        """Mean function (inverse link), the identity."""
        return eta

class poisson_loglike(smooth_atom):

    r"""The Poisson negative log-likelihood with log link.

    .. math::

       \eta \mapsto \sum_i w_i \left(e^{\eta_i} - y_i \eta_i\right)

    where :math:`y_i` are the counts and :math:`w_i` the case weights. A
    constant term is added through `quadratic`.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    counts : ndarray
        Non-negative integer counts.
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
        Non-negative case weights.
    """

    objective_template = r"""\ell^{\text{Pois}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 counts, 
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

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray().flatten()
        else:
            self.counts = counts

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        # saturated log-likelihood is sum(y * log(y) - y), with 0 * log(0) = 0
        _counts = np.asarray(self.counts, float)
        _mask = _counts > 0
        loss_terms = -_counts
        loss_terms[_mask] += _counts[_mask] * np.log(_counts[_mask])

        loss_constant = - coef * loss_terms.sum()

        devq = identity_quadratic(0,0,0,-loss_constant)
        self.quadratic += devq

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
        else:
            self.case_weights = None

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

        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        x = natural_param # shorthand

        x = self.apply_offset(x)
        exp_x = np.exp(x)
        
        if mode == 'both':
            f, g = - self.scale(-np.sum(cw * exp_x) + np.dot(cw * self.counts,x)), - self.scale(cw * (self.counts - exp_x))
            return f, g
        elif mode == 'grad':
            f, g = None, - self.scale(cw * (self.counts - exp_x))
            return g
        elif mode == 'func':
            f, g =  - self.scale(-np.sum(cw * exp_x) + np.dot(cw * self.counts,x)), None
            return f
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian(self, natural_param, case_weights=None):
        """Compute the diagonal of the Hessian of the loss.

        Parameters
        ----------
        natural_param : ndarray
            Parameters at which the Hessian is evaluated.
        case_weights : ndarray, optional
            Non-negative case weights.

        Returns
        -------
        hess : ndarray
            1D array holding the diagonal of the Hessian evaluated at
            `natural_param`.
        """
        x = natural_param # shorthand
        x = self.apply_offset(x)

        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        return self.scale(cw * np.exp(x))
            
    def get_data(self):
        """Return the counts."""
        return self.counts

    def set_data(self, data):
        """Set the counts."""
        self.counts = data

    data = property(get_data, set_data)

    def __copy__(self):
        counts = self.data
        return poisson_loglike(self.shape,
                               copy(counts),
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

        idx_bool = np.zeros_like(case_weights, bool)
        idx_bool[case_idx] = 1

        case_weights *= idx_bool
        return loss_cp

    # End loss API

    def mean_function(self, eta):
        r"""Mean function (inverse link), :math:`\eta \mapsto e^{\eta}`."""
        return np.exp(eta)

class huber_loss(smooth_atom):
    r"""The Huber loss of the residuals.

    .. math::

       \mu \mapsto \sum_i w_i \, h_{\epsilon}(\mu_i - y_i), \qquad
       h_{\epsilon}(r) = \begin{cases}
           r^2 / (2 \epsilon) & |r| \leq \epsilon \\
           |r| - \epsilon / 2 & |r| > \epsilon
       \end{cases}

    where :math:`\epsilon` is `smoothing_parameter` and :math:`w_i` the
    case weights.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    response : ndarray
        Response vector :math:`y`.
    smoothing_parameter : float
        Smoothing parameter for the Huber loss.
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
        Non-negative case weights.
    """

    objective_template = r"""\ell^{\text{Huber}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 response, 
                 smoothing_parameter,
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

        self.smoothing_parameter = smoothing_parameter
        atom = l1norm(shape, lagrange=1.)
        Q = identity_quadratic(smoothing_parameter, 0, 0, 0)
        self.smoothed_atom = atom.smoothed(Q)
                                         
        if sparse.issparse(response):
            self.response = response.toarray().flatten()
        else:
            self.response = np.asarray(response)

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.response.shape:
                raise ValueError('case_weights should have same shape as response')
        else:
            self.case_weights = None

    def smooth_objective(self, 
                         param, 
                         mode='both', 
                         check_feasibility=False,
                         case_weights=None):
        """Evaluate the smooth objective, its gradient, or both.

        Parameters
        ----------
        param : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `param` is not in the domain.
        case_weights : ndarray, optional
            Non-negative case weights, multiplied with any case weights
            stored on the loss.

        Returns
        -------
        f : float
            Objective value at `param`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `param`; returned when `mode` is 'grad'
            or 'both'.
        """
        
        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  param.shape)

        param = self.apply_offset(param)
        resid = param - self.response 

        f, g = _huber_loss(resid, smoothing_parameter=self.smoothing_parameter)

        if mode == 'func':
            return self.scale((f * cw).sum())
        elif mode == 'grad':
            return self.scale(g * cw)
        elif mode == 'both':
            return self.scale((f * cw).sum()), self.scale(g * cw)
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian(self, param, case_weights=None):
        """Hessian of the loss (not implemented).

        The Huber loss is not twice differentiable, so this always
        raises ``NotImplementedError``.

        Parameters
        ----------
        param : ndarray
            Parameters at which the Hessian would be evaluated.
        case_weights : ndarray, optional
            Non-negative case weights.

        Raises
        ------
        NotImplementedError
        """

        # it is piecewise C^2 though... maybe use this?
        raise NotImplementedError('Huber loss is not twice differentiable')

    def get_data(self):
        """Return the response."""
        return self.response

    def set_data(self, data):
        """Set the response."""
        self.response = data

    data = property(get_data, set_data)

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

    def __copy__(self):
        response = self.data
        return huber_loss(self.shape,
                          copy(response),
                          self.smoothing_parameter,
                          coef=self.coef,
                          offset=copy(self.offset),
                          quadratic=copy(self.quadratic),
                          initial=copy(self.coefs),
                          case_weights=copy(self.case_weights))

    # End loss API

def _huber_loss(arg, smoothing_parameter):
    # returns vector whose sum is total loss as well as gradient vector
    eps = smoothing_parameter
    proj_arg = np.sign(arg) * np.minimum(np.abs(arg) / eps, 1) # the maximizer is the gradient
                                                               # by convex conjugacy
    return arg * proj_arg - eps * proj_arg**2 / 2, proj_arg 

class stacked_loglike(smooth_atom):

    """Stack ``K = len(losses)`` saturated losses, summed over the losses.

    The natural parameter is the concatenation of the natural parameters
    of the individual losses. Roughly speaking, this is a model of `K`
    independent measurements per individual.

    Parameters
    ----------
    losses : sequence
        Saturated losses to be stacked.
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
        Non-negative case weights.
    """

    objective_template = r"""\ell^{\text{stack}}\left(%(var)s\right)"""

    def __init__(self, 
                 losses,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None,
                 case_weights=None):

        shape = (np.sum([l.shape[0] for l in losses]),)
        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        responses = [l.data for l in losses]
        shapes = [r.shape for r in responses]
        dims = np.array([len(s) for s in shapes])
        if np.all(dims == 1):
            self.data = np.hstack(responses)
        elif np.all(dims == 2):
            self.data = np.vstack(responses)
        else:
            raise ValueError('expecting either 1 or dimensional data for saturated losses')

        self._slices = []
        idx = 0
        for l in losses:
            self._slices.append(slice(idx, idx + l.shape[0], 1))
            idx += l.shape[0]

        self._losses = losses
        self._gradient = np.zeros(self.shape)

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.shape[:1]:
                raise ValueError('case_weights should have same shape as response')
        else:
            self.case_weights = None

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
            Non-negative case weights. Currently ignored: each stacked loss
            uses its own case weights.

        Returns
        -------
        f : float
            Objective value at `natural_param`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `natural_param`; returned when `mode` is 'grad'
            or 'both'.
        """
        
        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape[:1])

        linpred = natural_param # shorthand

        linpred = self.apply_offset(linpred)
        if mode == 'grad':
            for d, slice in enumerate(self._slices):
                self._gradient[slice] = self._losses[d].smooth_objective(linpred[slice], 'grad')
            return self.scale(self._gradient)
        elif mode == 'func':
            value = 0
            for d, slice in enumerate(self._slices):
                value += self._losses[d].smooth_objective(linpred[slice], 'func')
            return self.scale(value)
        elif mode == 'both':
            value = 0
            for d, slice in enumerate(self._slices):
                f, g = self._losses[d].smooth_objective(linpred[slice], 'both')
                self._gradient[slice] = g
                value += f
            return self.scale(value), self.scale(self._gradient)
        else:
            raise ValueError("mode incorrectly specified")

    def get_data(self):
        """Return the stacked responses."""
        return self._data

    def set_data(self, data):
        """Set the stacked responses."""
        self._data = data

    data = property(get_data, set_data)

    def __copy__(self):
        return stacked_loglike(copy(self._losses),
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

        idx_bool = np.zeros_like(case_weights, bool)
        idx_bool[case_idx] = 1

        case_weights *= idx_bool
        return loss_cp

    @classmethod
    def gaussian(klass,
                 responses,
                 case_weights=None,
                 coef=1., 
                 offset=None,
                 quadratic=None, 
                 initial=None):
        """Create a stacked loss of Gaussian losses.

        Parameters
        ----------
        responses : sequence of ndarray
            Response vectors, one per stacked loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [gaussian_loglike(response.shape,
                                   response,
                                   coef=coef)
                  for response in responses]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def logistic(klass, 
                 successes, 
                 trials=None,
                 case_weights=None,
                 coef=1., 
                 offset=None,
                 quadratic=None, 
                 initial=None):
        """Create a stacked loss of logistic losses.

        Parameters
        ----------
        successes : sequence of ndarray
            Numbers of successes, one array per stacked loss.
        trials : sequence of ndarray
            Numbers of trials, one array per stacked loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [logistic_loglike(successes[i],
                                   trials[i],
                                   coef=coef)
                  for i in range(len(successes))]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def probit(klass, 
               successes, 
               trials=None,
               case_weights=None,
               coef=1., 
               offset=None,
               quadratic=None, 
               initial=None):
        """Create a stacked loss of probit losses.

        Parameters
        ----------
        successes : sequence of ndarray
            Numbers of successes, one array per stacked loss.
        trials : sequence of ndarray
            Numbers of trials, one array per stacked loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [probit_loglike(successes[i],
                                 trials[i],
                                 coef=coef)
                  for i in range(len(successes))]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def cloglog(klass, 
                successes, 
                trials=None,
                case_weights=None,
                coef=1., 
                offset=None,
                quadratic=None, 
                initial=None):
        """Create a stacked loss of complementary log-log losses.

        Parameters
        ----------
        successes : sequence of ndarray
            Numbers of successes, one array per stacked loss.
        trials : sequence of ndarray
            Numbers of trials, one array per stacked loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [cloglog_loglike(successes[i],
                                  trials[i],
                                  coef=coef)
                  for i in range(len(successes))]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def poisson(klass,
                counts,
                case_weights=None,
                coef=1., 
                offset=None,
                quadratic=None, 
                initial=None):
        """Create a stacked loss of Poisson losses.

        Parameters
        ----------
        counts : sequence of ndarray
            Non-negative integer counts, one array per stacked loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [logistic_loglike(successes[i],
                                   trials[i],
                                   coef=coef)
                  for i in range(len(successes))]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def huber(klass,
              X, 
              responses,
              smoothing_parameter,
              case_weights=None,
              coef=1., 
              offset=None,
              quadratic=None, 
              initial=None):
        """Create a stacked loss of Huber losses.

        Parameters
        ----------
        X : object
            Unused.
        responses : sequence of ndarray
            Response vectors, one per stacked loss.
        smoothing_parameter : float
            Smoothing parameter for the Huber loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [huber_loss(response.shape,
                             response,
                             smoothing_parameter) 
                  for response in responses]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def huber_svm(klass,
                  X, 
                  successes,
                  smoothing_parameter,
                  case_weights=None,
                  coef=1., 
                  offset=None,
                  quadratic=None, 
                  initial=None):
        """Create a stacked loss of Huber SVM losses.

        Parameters
        ----------
        X : object
            Unused.
        successes : sequence of ndarray
            Binary (0/1) responses, one array per stacked loss.
        smoothing_parameter : float
            Smoothing parameter for the Huberized hinge loss.
        case_weights : ndarray, optional
            Non-negative case weights.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [huber_svm(response.shape,
                            response,
                            smoothing_parameter) 
                  for response in successes]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def cox(klass, 
            X, 
            event_times,
            censoring,
            coef=1., 
            offset=None,
            quadratic=None, 
            initial=None,
            case_weights=None):
        """Create a stacked loss of Cox proportional hazards losses.

        Parameters
        ----------
        X : object
            Unused.
        event_times : sequence of ndarray
            Observed times, one array per stacked loss.
        censoring : sequence of ndarray
            Censoring indicators (1 for a failure, 0 for censored), one
            array per stacked loss.
        coef : float, optional
            Nonnegative scaling to be put in front of the loss.
        offset : ndarray, optional
            Offset to be subtracted in parameter space before evaluating
            the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        initial : ndarray, optional
            Initial guess at the coefficients.
        case_weights : ndarray, optional
            Non-negative case weights.

        Returns
        -------
        stacked_loss : `stacked_loglike`
            Stacked saturated loss.
        """

        losses = [cox_loglike(event_times[i].shape,
                              event_times[i],
                              censoring[i],
                              coef=coef)
                  for i in range(len(event_times))]

        return klass(losses,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)


# Deprecated

def logistic_loss(X, Y, trials=None, coef=1.):
    """Construct a logistic loss for successes `Y` and design `X`.

    This function is deprecated, use `regreg.smooth.glm.glm.logistic`
    instead.

    Parameters
    ----------
    X : ndarray or `regreg.affine.affine_transform`
        Design matrix.
    Y : ndarray
        Number of successes.
    trials : ndarray, optional
        Number of trials. If None, `Y` must be binary.
    coef : float, optional
        Nonnegative scaling to be put in front of the loss.

    Returns
    -------
    glm_obj : `regreg.smooth.glm.glm`
        General linear model loss.
    """
    warnings.warn('"logistic_loss" is deprecated use "regreg.smooth.glm.logistic" instead')
    return glm.logistic(X, 
                        Y,
                        trials=trials,
                        coef=coef)

