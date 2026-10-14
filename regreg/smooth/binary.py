"""Losses for binary (binomial) responses.

Defines the logistic, probit and complementary log-log negative
log-likelihoods, and the Huberized SVM loss.
"""

from copy import copy

import numpy as np
from scipy import sparse
from scipy.stats import norm as normal_dbn

from ..identity_quadratic import identity_quadratic
from . import smooth_atom, combine_case_weights

class logistic_loglike(smooth_atom):

    r"""Logistic loss for binomial responses.

    The negative log-likelihood of a binomial response with
    logit link, as a function of the natural parameter (linear
    predictor) :math:`\eta`:

    .. math::

       \ell(\eta) = -\sum_i w_i \left[ y_i \log \pi_i +
           (n_i - y_i) \log(1 - \pi_i) \right],
       \qquad \pi_i = \frac{e^{\eta_i}}{1 + e^{\eta_i}}

    where :math:`y_i` are the successes, :math:`n_i` the trials and
    :math:`w_i` the case weights. The log-likelihood of the saturated
    model (times `coef`) is stored as a constant term in `quadratic`, so
    that with unit case weights the full objective is `coef` times half
    the deviance.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    successes : ndarray
        Number of successes for each case (non-negative).
    trials : ndarray, optional
        Number of trials for each case. If None, the response must be
        binary and ``np.ones_like(successes)`` is used.
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

    objective_template = r"""\ell^{\text{logit}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 successes, 
                 trials=None, 
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

        self.data = (successes, trials)

        saturated = self.successes / self.trials

        _mask = (saturated != 0) * (saturated != 1)
        loss_terms = (np.log(saturated[_mask]) * self.successes[_mask] +
                      np.log(1 - saturated[_mask]) *
                      ((self.trials - self.successes)[_mask]))
        loss_constant = -coef * loss_terms.sum()

        devq = identity_quadratic(0,0,0,-loss_constant)
        self.quadratic += devq

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.successes.shape:
                raise ValueError('case_weights should have same shape as successes')
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

        #Check for overflow in np.exp (can occur during initial backtracking steps)
        x = self.apply_offset(x)
        if np.max(x) > 1e2:
            overflow = True
            not_overflow_ind = np.where(x <= 1e2)[0]
            exp_x = np.exp(x[not_overflow_ind])
        else:
            overflow = False
            exp_x = np.exp(x)
            
        if mode == 'both':
            ratio = self.trials * 1.
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
                ratio *= exp_x/(1.+exp_x)
                
            f, g = -self.scale((np.dot(cw * self.successes, x) - 
                                np.sum(cw * self.trials * log_exp_x))), - self.scale(cw * (self.successes - ratio))
            return f, g
        elif mode == 'grad':
            ratio = self.trials * 1.
            if overflow:
                ratio[not_overflow_ind] *= exp_x/(1.+exp_x)
            else:
                ratio *= exp_x/(1.+exp_x)
            f, g = None, - self.scale(cw * (self.successes - ratio))
            return g

        elif mode == 'func':
            if overflow:
                log_exp_x = x * 1.
                log_exp_x[not_overflow_ind] = np.log(1.+exp_x)
            else:
                log_exp_x = np.log(1.+exp_x)
            f, g = (- self.scale(np.dot(cw * self.successes, x) - 
                                 np.sum(cw * self.trials * log_exp_x)), None)
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

        # p * (1 - p) is symmetric in x, so use exp(-|x|) to avoid overflow
        exp_x = np.exp(-np.fabs(x))

        return self.scale(exp_x / (1 + exp_x)**2 * self.trials * cw)

    def get_data(self):
        """Return the vector of successes."""
        return self.successes

    def set_data(self, data):
        """Set the response.

        Parameters
        ----------
        data : ndarray or tuple
            Either `successes`, or a tuple ``(successes, trials)``. If `trials`
            is not given, the existing trials are reused if present, otherwise
            the response must be binary and the trials are set to one.
        """
        if type(data) == type((3,)):
            successes, trials = data
        else:
            successes = data
            trials = None

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten() * 1.
        else:
            self.successes = np.asarray(successes).astype(float)

        if trials is None and hasattr(self, 'trials') and self.trials is not None:
            trials = self.trials

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones_like(self.successes)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials * 1.

    data = property(get_data, set_data)

    def __copy__(self):
        successes, trials = self.data, self.trials
        return logistic_loglike(self.shape,
                                copy(successes),
                                trials=copy(trials),
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

    def mean_function(self, eta, trials=None):
        r"""Compute the mean function (inverse link).

        The success probability is :math:`e^{\eta} / (1 + e^{\eta})`.

        Parameters
        ----------
        eta : ndarray
            Natural parameter (linear predictor).
        trials : ndarray, optional
            Ignored.

        Returns
        -------
        mean : ndarray
            Success probabilities.
        """
        _exp_eta = np.exp(eta)
        return _exp_eta / (1. + _exp_eta)

class probit_loglike(smooth_atom):

    r"""Probit loss for binomial responses.

    The negative log-likelihood of a binomial response with
    probit link, as a function of the natural parameter (linear
    predictor) :math:`\eta`:

    .. math::

       \ell(\eta) = -\sum_i w_i \left[ y_i \log \pi_i +
           (n_i - y_i) \log(1 - \pi_i) \right],
       \qquad \pi_i = \Phi(\eta_i)

    where :math:`y_i` are the successes, :math:`n_i` the trials and
    :math:`w_i` the case weights. The log-likelihood of the saturated
    model (times `coef`) is stored as a constant term in `quadratic`, so
    that with unit case weights the full objective is `coef` times half
    the deviance.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    successes : ndarray
        Number of successes for each case (non-negative).
    trials : ndarray, optional
        Number of trials for each case. If None, the response must be
        binary and ``np.ones_like(successes)`` is used.
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

    objective_template = r"""\ell^{\text{probit}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 successes, 
                 trials=None, 
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

        self.data = (successes, trials)

        saturated = self.successes / self.trials

        _mask = (saturated != 0) * (saturated != 1)
        loss_terms = (np.log(saturated[_mask]) * self.successes[_mask] +
                      np.log(1 - saturated[_mask]) *
                      ((self.trials - self.successes)[_mask]))
        loss_constant = -coef * loss_terms.sum()

        devq = identity_quadratic(0,0,0,-loss_constant)
        self.quadratic += devq

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.successes.shape:
                raise ValueError('case_weights should have same shape as successes')
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


        prob = normal_dbn.cdf(x)
        prob[x < -20] = normal_dbn.cdf(-20)   # to prevent overflow
        prob_c = normal_dbn.sf(x) # to prevent overflow
        prob_c[x > 20] = normal_dbn.sf(20)

        if mode == 'both':

            lprob = np.log(prob)
            lprob_c = np.log(prob_c)
            pdf = normal_dbn.pdf(x)

            f, g = (-self.scale(np.sum(cw * (self.successes * lprob + 
                                            (self.trials - self.successes) * lprob_c))),
                     -self.scale(cw * pdf * (self.successes / prob - 
                                             (self.trials - self.successes) / prob_c)))
            return f, g
        elif mode == 'grad':
            pdf = normal_dbn.pdf(x)
            g = - self.scale(cw * pdf * (self.successes / prob - 
                                         (self.trials - self.successes) / prob_c)) 
            return g

        elif mode == 'func':

            lprob = np.log(prob)
            lprob_c = np.log(prob_c)

            f = -self.scale(np.sum(cw * (self.successes * lprob + 
                                         (self.trials - self.successes) * lprob_c)))
            return f
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian(self, natural_param, case_weights=None):
        r"""Compute the diagonal of the Hessian of the loss.

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

        Notes
        -----
        This is the expected (Fisher information) Hessian
        :math:`w_i n_i \pi'(\eta_i)^2 / (\pi_i (1 - \pi_i))`, which differs
        from the observed Hessian for this non-canonical link.
        """

        x = natural_param # shorthand
        x = self.apply_offset(x)

        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        pi = self.mean_function(x)
        V = pi * (1 - pi)
        deriv_term = normal_dbn.pdf(x)**2
        return self.scale(deriv_term * self.trials * cw / V)

    def get_data(self):
        """Return the vector of successes."""
        return self.successes

    def set_data(self, data):
        """Set the response.

        Parameters
        ----------
        data : ndarray or tuple
            Either `successes`, or a tuple ``(successes, trials)``. If `trials`
            is not given, the existing trials are reused if present, otherwise
            the response must be binary and the trials are set to one.
        """
        if type(data) == type((3,)):
            successes, trials = data
        else:
            successes = data
            trials = None

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten() * 1.
        else:
            self.successes = np.asarray(successes).astype(float)

        if trials is None and hasattr(self, 'trials') and self.trials is not None:
            trials = self.trials

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones_like(self.successes)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials * 1.

    data = property(get_data, set_data)

    def __copy__(self):
        successes, trials = self.data, self.trials
        return probit_loglike(self.shape,
                              copy(successes),
                              trials=copy(trials),
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

    def mean_function(self, eta, trials=None):
        r"""Compute the mean function (inverse link).

        The success probability is the standard normal CDF :math:`\Phi(\eta)`.

        Parameters
        ----------
        eta : ndarray
            Natural parameter (linear predictor).
        trials : ndarray, optional
            Ignored.

        Returns
        -------
        mean : ndarray
            Success probabilities.
        """
        return normal_dbn.cdf(eta)

class cloglog_loglike(smooth_atom):

    r"""Complementary log-log loss for binomial responses.

    The negative log-likelihood of a binomial response with
    complementary log-log link, as a function of the natural parameter (linear
    predictor) :math:`\eta`:

    .. math::

       \ell(\eta) = -\sum_i w_i \left[ y_i \log \pi_i +
           (n_i - y_i) \log(1 - \pi_i) \right],
       \qquad \pi_i = 1 - \exp(-e^{\eta_i})

    where :math:`y_i` are the successes, :math:`n_i` the trials and
    :math:`w_i` the case weights. The log-likelihood of the saturated
    model (times `coef`) is stored as a constant term in `quadratic`, so
    that with unit case weights the full objective is `coef` times half
    the deviance.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    successes : ndarray
        Number of successes for each case (non-negative).
    trials : ndarray, optional
        Number of trials for each case. If None, the response must be
        binary and ``np.ones_like(successes)`` is used.
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

    objective_template = r"""\ell^{\text{cloglog}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 successes, 
                 trials=None, 
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

        self.data = (successes, trials)

        saturated = self.successes / self.trials

        _mask = (saturated != 0) * (saturated != 1)
        loss_terms = (np.log(saturated[_mask]) * self.successes[_mask] +
                      np.log(1 - saturated[_mask]) *
                      ((self.trials - self.successes)[_mask]))
        loss_constant = -coef * loss_terms.sum()

        devq = identity_quadratic(0,0,0,-loss_constant)
        self.quadratic += devq

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.successes.shape:
                raise ValueError('case_weights should have same shape as successes')
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

        cutoff = 5
        x_ = np.clip(x, -cutoff, cutoff) # to prevent overflow
        prob_ = 1 - np.exp(-np.exp(x_))
        prob_c_ = np.exp(-np.exp(x_)) 
        prob = prob_.copy()
        prob_c = prob_c_.copy()
        if mode == 'both':

            lprob = np.log(prob)
            lprob_c = np.log(prob_c)
            lpdf = x - np.exp(x)
            pdf = np.exp(lpdf)

            f, g = (-self.scale(np.sum(cw * (self.successes * lprob + 
                                            (self.trials - self.successes) * lprob_c))),
                     -self.scale(cw * pdf * (self.successes / prob - 
                                             (self.trials - self.successes) / prob_c)))
            return f, g
        elif mode == 'grad':
            lpdf = x - np.exp(x)
            pdf = np.exp(lpdf)

            g = - self.scale(cw * pdf * (self.successes / prob - 
                                         (self.trials - self.successes) / prob_c)) 
            return g

        elif mode == 'func':

            lprob = np.log(prob)
            lprob_c = np.log(prob_c)

            f = -self.scale(np.sum(cw * (self.successes * lprob + 
                                         (self.trials - self.successes) * lprob_c)))
            return f
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian(self, natural_param, case_weights=None):
        r"""Compute the diagonal of the Hessian of the loss.

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

        Notes
        -----
        This is the expected (Fisher information) Hessian
        :math:`w_i n_i \pi'(\eta_i)^2 / (\pi_i (1 - \pi_i))`, which differs
        from the observed Hessian for this non-canonical link.
        """

        x = natural_param # shorthand
        x = self.apply_offset(x)

        cw = combine_case_weights(case_weights,
                                  self.case_weights,
                                  natural_param.shape)

        pi = self.mean_function(x)
        V = pi * (1 - pi)
        lpdf = x - np.exp(x)
        pdf = np.exp(lpdf)
        deriv_term = pdf**2
        return self.scale(deriv_term * self.trials * cw / V)

    def get_data(self):
        """Return the vector of successes."""
        return self.successes

    def set_data(self, data):
        """Set the response.

        Parameters
        ----------
        data : ndarray or tuple
            Either `successes`, or a tuple ``(successes, trials)``. If `trials`
            is not given, the existing trials are reused if present, otherwise
            the response must be binary and the trials are set to one.
        """
        if type(data) == type((3,)):
            successes, trials = data
        else:
            successes = data
            trials = None

        if sparse.issparse(successes):
            #Convert sparse success vector to an array
            self.successes = successes.toarray().flatten() * 1.
        else:
            self.successes = np.asarray(successes).astype(float)

        if trials is None and hasattr(self, 'trials') and self.trials is not None:
            trials = self.trials

        if trials is None:
            if not set([0,1]).issuperset(np.unique(self.successes)):
                raise ValueError("Number of successes is not binary - must specify number of trials")
            self.trials = np.ones_like(self.successes)
        else:
            if np.min(trials-self.successes) < 0:
                raise ValueError("Number of successes greater than number of trials")
            if np.min(self.successes) < 0:
                raise ValueError("Response coded as negative number - should be non-negative number of successes")
            self.trials = trials * 1.

    data = property(get_data, set_data)

    def __copy__(self):
        successes, trials = self.data, self.trials
        return cloglog_loglike(self.shape,
                               copy(successes),
                               trials=copy(trials),
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

    def mean_function(self, eta, trials=None):
        r"""Compute the mean function (inverse link).

        The success probability is :math:`1 - \exp(-e^{\eta})`, with
        :math:`\eta` clipped to :math:`[-5, 5]`.

        Parameters
        ----------
        eta : ndarray
            Natural parameter (linear predictor).
        trials : ndarray, optional
            Ignored.

        Returns
        -------
        mean : ndarray
            Success probabilities.
        """
        eta = np.clip(eta, -5, 5)
        return 1 - np.exp(-np.exp(eta))

class huber_svm(smooth_atom):
    r"""Huberized hinge (SVM) loss for binary responses.

    The binary response :math:`y \in \{0, 1\}` is encoded as
    :math:`2y - 1 \in \{-1, 1\}` and a smoothed version of the hinge loss
    :math:`(1 - (2y - 1)\eta)_+` is evaluated, the smoothing being
    controlled by `smoothing_parameter`.

    Parameters
    ----------
    shape : int or tuple
        Shape of the natural parameter.
    successes : ndarray
        Binary (0/1) responses.
    smoothing_parameter : float
        Smoothing parameter for the Huberized hinge loss.
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

    objective_template = r"""\ell^{\text{Huber,SVM}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 successes, 
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
                                         
        if sparse.issparse(successes):
            self.successes = successes.toarray().flatten()
        else:
            self.successes = np.asarray(successes)

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.successes.shape:
                raise ValueError('case_weights should have same shape as successes')
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
        huber_arg = (2 * self.successes - 1) * param # first encode binary as \pm 1
                                                     # then multiply by linpred

        f, g = _huber_svm(huber_arg, smoothing_parameter=self.smoothing_parameter)
        g *= (2 * self.successes - 1) # chain rule for (2 * y - 1) in encoding of y to binary
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

        The Huber SVM loss is not twice differentiable, so this always
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
        raise NotImplementedError('Huber SVM loss is not twice differentiable')

    def get_data(self):
        """Return the vector of successes."""
        return self.successes

    def set_data(self, data):
        """Set the vector of (binary) successes."""
        self.successes = data

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
        successes = self.data
        return huber_svm(self.shape,
                         copy(successes),
                         self.smoothing_parameter,
                         coef=self.coef,
                         offset=copy(self.offset),
                         quadratic=copy(self.quadratic),
                         initial=copy(self.coefs),
                         case_weights=copy(self.case_weights))
    
    # End loss API

def _huber_svm(z, smoothing_parameter):
    eps = smoothing_parameter
    arg = (1 - z) / eps # function of (2 * y - 1) * eta when y is binary
    proj_arg = (0 < arg) * (arg < 1) * arg + (arg >= 1) 
    return (1 - z) * proj_arg - eps * proj_arg**2 / 2, -proj_arg

