from copy import copy

import numpy as np
from scipy import sparse
from scipy.stats import norm as normal_dbn

from ..identity_quadratic import identity_quadratic
from . import smooth_atom

class logistic_loglike(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
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
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        natural_param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `natural_param` is not
            in the domain.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `natural_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

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
        """
        Hessian of the loss.

        Parameters
        ----------

        natural_param : ndarray
            Parameters where Hessian will be evaluated.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `natural_param`.
        """

        x = natural_param # shorthand

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        if np.max(x) > 1e2:
            overflow = True
            not_overflow_ind = np.where(x <= 1e2)[0]
            exp_x = np.zeros_like(x)
            exp_x[not_overflow_ind] = np.exp(x[not_overflow_ind])
            exp_x[~not_overflow_ind] = np.exp(100)
        else:
            overflow = False
            exp_x = np.exp(x)

        return self.scale(exp_x / (1 + exp_x)**2 * self.trials * cw)

    def get_data(self):
        return self.successes

    def set_data(self, data):
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
        """
        Create a saturated loss using a subsample of the data.
        Makes a copy of the loss and 
        multiplies case_weights by the indicator for
        `idx`.

        Parameters
        ----------

        idx : index
            Indices of np.arange(n) to keep.

        Returns
        -------

        subsample_loss : `smooth_atom`
            Loss after discarding all
            cases not in `idx.

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

    # End loss API

    def mean_function(self, eta, trials=None):
        _exp_eta = np.exp(eta)
        return _exp_eta / (1. + _exp_eta)

class probit_loglike(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
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
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        natural_param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `natural_param` is not
            in the domain.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `natural_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        x = natural_param # shorthand


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
        """
        Hessian of the loss.

        Parameters
        ----------

        natural_param : ndarray
            Parameters where Hessian will be evaluated.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `natural_param`.
        """

        x = natural_param # shorthand

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        pi = self.mean_function(x)
        V = pi * (1 - pi)
        deriv_term = normal_dbn.pdf(x)**2
        return self.scale(deriv_term * self.trials * cw / V)

    def get_data(self):
        return self.successes

    def set_data(self, data):
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
        """
        Create a saturated loss using a subsample of the data.
        Makes a copy of the loss and 
        multiplies case_weights by the indicator for
        `idx`.

        Parameters
        ----------

        idx : index
            Indices of np.arange(n) to keep.

        Returns
        -------

        subsample_loss : `smooth_atom`
            Loss after discarding all
            cases not in `idx.

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

    # End loss API

    def mean_function(self, eta, trials=None):
        return normal_dbn.cdf(eta)

class cloglog_loglike(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
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
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        natural_param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `natural_param` is not
            in the domain.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `natural_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        x = natural_param # shorthand

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
        """
        Hessian of the loss.

        Parameters
        ----------

        natural_param : ndarray
            Parameters where Hessian will be evaluated.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `natural_param`.
        """

        x = natural_param # shorthand

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        pi = self.mean_function(x)
        V = pi * (1 - pi)
        lpdf = x - np.exp(x)
        pdf = np.exp(lpdf)
        deriv_term = pdf**2
        return self.scale(deriv_term * self.trials * cw / V)

    def get_data(self):
        return self.successes

    def set_data(self, data):
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
        """
        Create a saturated loss using a subsample of the data.
        Makes a copy of the loss and 
        multiplies case_weights by the indicator for
        `idx`.

        Parameters
        ----------

        idx : index
            Indices of np.arange(n) to keep.

        Returns
        -------

        subsample_loss : `smooth_atom`
            Loss after discarding all
            cases not in `idx.

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

    # End loss API

    def mean_function(self, eta, trials=None):
        eta = np.clip(eta, -5, 5)
        return 1 - np.exp(-np.exp(eta))

class huber_svm(smooth_atom):

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
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `param` is not
            in the domain.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        if case_weights is None:
            case_weights = np.ones_like(param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

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
        """
        Hessian of the loss.

        Parameters
        ----------

        param : ndarray
            Parameters where Hessian will be evaluated.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `natural_param`.
        """

        # it is piecwise C^2 though... maybe use this?
        raise NotImplementedError('Huber SVM loss is not twice differentiable')

    def get_data(self):
        return self.successes

    def set_data(self, data):
        self.successes = data

    data = property(get_data, set_data)

    def subsample(self, case_idx):
        """
        Create a saturated loss using a subsample of the data.
        Makes a copy of the loss and 
        multiplies case_weights by the indicator for
        `idx`.

        Parameters
        ----------

        idx : index
            Indices of np.arange(n) to keep.

        Returns
        -------

        subsample_loss : `smooth_atom`
            Loss after discarding all
            cases not in `idx.

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

    def __copy__(self):
        successes = self.data
        return huber_svm(self.shape,
                         copy(successes),
                         self.smoothing_parameter,
                         coef=self.coef,
                         offset=copy(self.offset),
                         quadratic=copy(self.quadratic),
                         initial=copy(self.coefs))
    
    # End loss API

def _huber_svm(z, smoothing_parameter):
    eps = smoothing_parameter
    arg = (1 - z) / eps # function of (2 * y - 1) * eta when y is binary
    proj_arg = (0 < arg) * (arg < 1) * arg + (arg >= 1) 
    return arg * proj_arg - eps * proj_arg**2 / 2, -proj_arg

