import warnings
from copy import copy

import numpy as np
from scipy import sparse

from . import smooth_atom, affine_smooth
from ..affine import (astransform, 
                      linear_transform, 
                      affine_transform,
                      scaler)
from ..identity_quadratic import identity_quadratic
from ..atoms.seminorms import l1norm
from .cox import cox_loglike

class glm(smooth_atom):

    """
    A general linear model, usually a log-likelihood
    for response $Y$ whose mean is modelled through
    $X\beta$. 

    Usual examples are Gaussian (least squares regression),
    logistic regression and Poisson log-linear regression.
    
    Huber loss is also implemented as an example.

    """

    def __init__(self, 
                 X, 
                 Y, 
                 loss, 
                 quadratic=None, 
                 initial=None,
                 offset=None,
                 case_weights=None):

        """

        Parameters
        ----------

        X : ndarray
            The design matrix.

        Y : ndarray
            The response.

        loss : `regreg.smooth.smooth_atom`
            The loss function that takes arguments the same
            size as `Y`. So, for Gaussian regression 
            the loss is just the map $\mu \mapsto \|\mu - Y\|^2_2/2$.

        quadratic : `regreg.identity_quadratic.identity_quadratic`
            Optional quadratic part added to objective.

        initial : ndarray
            An initial guess at the minimizer.

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        case_weights : ndarray
            Non-negative case weights

        """

        self.saturated_loss = loss
        self.data = X, Y
        self.affine_atom = affine_smooth(loss, X)
        if case_weights is None:
            case_weights = np.ones(X.shape[0])
        self.case_weights = case_weights
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
        """

        Parameters
        ----------

        beta : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `beta` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `beta`, else if `mode` is 'grad' returns the gradient
        else returns both.
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
        return self._X, self.saturated_loss.data

    def set_data(self, data):
        X, Y = data
        self._transform = astransform(X)
        self._X = X
        self._is_transform = id(self._X) == id(self._transform) # i.e. astransform was a nullop
        self.saturated_loss.data = Y

    data = property(get_data, set_data, doc="Data for the general linear model.")

    def linear_predictor(self, beta):
        """

        Compute $X\beta$.

        Parameters
        ----------

        beta : ndarray
            Parameters.

        Returns
        -------

        linpred : ndarray

        """
        # can have an `R`-type offset by using affine_map here
        return self._transform.linear_map(beta)

    def objective(self, beta):
        """
        Compute the loss $\ell(X\beta)$.

        Parameters
        ----------

        beta : ndarray
            Parameters.

        Returns
        -------

        objective : float
            Value of the loss at $\beta$.

        """
        return self.smooth_objective(beta, 'func')

    def gradient(self, beta):
        """

        Compute the gradient of the loss $ \nabla \ell(X\beta)$.

        Parameters
        ----------

        beta : ndarray
            Parameters.

        Returns
        -------

        grad : ndarray
            Gradient of the loss at $\beta$.
        """

        return self.smooth_objective(beta, 'grad')

    def hessian(self, beta):
        """

        Compute the Hessian of the loss $ \nabla^2 \ell(X\beta)$.


        Parameters
        ----------

        beta : ndarray
            Parameters.

        Returns
        -------

        hess : ndarray
            Hessian of the loss at $\beta$, if defined.

        """

        linpred = self.linear_predictor(beta)
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
        """
        Create a loss using a subsample of the data.
        Makes a copy of the loss and 
        multiplies case_weights by the indicator for
        `idx`.

        Parameters
        ----------

        idx : index
            Indices of np.arange(n) to keep.

        Returns
        -------

        subsample_loss : `glm`
            Loss after discarding all
            cases not in `idx.

        """

        subsample_loss = copy(self)
        n = subsample_loss.saturated_loss.shape[0]

        idx_bool = np.zeros(n, np.bool)
        idx_bool[idx] = 1

        subsample_loss.case_weights *= idx_bool

        return subsample_loss
        
        
    @classmethod
    def gaussian(klass,
                 X, 
                 response,
                 case_weights=None,
                 coef=1., 
                 offset=None,
                 quadratic=None, 
                 initial=None):
        """
        Create a loss for a Gaussian regression model.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        response : ndarray
            Response vector.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        glm_obj : `regreg.glm.glm`
            General linear model loss.

        """

        loss = gaussian_loglike(response.shape,
                                response,
                                coef=coef)

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
                 quadratic=None, 
                 initial=None):
        """
        Create a loss for a logistic regression model.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        successes : ndarray
            Responses (should be non-negative integers).

        trials : ndarray (optional)
            Number of trials for each success. If `None`,
            defaults to `np.ones_like(successes)`.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        glm_obj : `regreg.glm.glm`
            General linear model loss.

        """

        loss = logistic_loglike(successes.shape,
                                successes,
                                coef=coef,
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
                offset=None,
                quadratic=None, 
                initial=None):
        """
        Create a loss for a Poisson regression model.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        counts : ndarray
            Response vector. Should be non-negative integers.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        glm_obj : `regreg.glm.glm`
            General linear model loss.

        """

        loss = poisson_loglike(counts.shape,
                               counts,
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
              offset=None,
              quadratic=None, 
              initial=None):
        """
        Create a loss for a regression model using
        Huber loss.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        response : ndarray
            Response vector. 

        smoothing_parameter : float
            Smoothing parameter for Huber loss.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        glm_obj : `regreg.glm.glm`
            General linear model loss.

        """

        loss = huber_loss(response.shape,
                          response,
                          smoothing_parameter,
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
                  quadratic=None, 
                  initial=None):
        """
        Create a loss for a binary regression model using
        Huber SVM loss.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        successes : ndarray
            Response vector. 

        smoothing_parameter : float
            Smoothing parameter for Huber loss.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        glm_obj : `regreg.glm.glm`
            General linear model loss.

        """

        loss = huber_svm(successes.shape,
                         successes,
                         smoothing_parameter,
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
            offset=None,
            quadratic=None, 
            initial=None):
        """
        Create a loss for a logistic regression model.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        event_times : ndarray
            Observed times for Cox proportional hazard model.

        censoring : ndarray 
            Censoring indicator for Cox proportional hazard model
            - 1 indicates observation is a failure, 0 a censored observation.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        glm_obj : `regreg.glm.glm`
            General linear model loss.

        """

        loss = cox_loglike(event_times.shape,
                           event_times,
                           censoring,
                           coef=coef)

        return klass(X, 
                     np.array([event_times, censoring]).T,
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)


class gaussian_loglike(smooth_atom):

    """
    The Gaussian loss for observations $y$:

    .. math::
    
       \mu \mapsto \frac{1}{2} \|y-\mu\|^2_2

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
        """
        Hessian of the loss.

        Parameters
        ----------

        natural_param : ndarray
            Parameters where Hessian will be evaluated.

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `natural_param`.

        case_weights : ndarray
            Non-negative case weights

        """
        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights
        return self.scale(np.ones_like(natural_param) * cw)

    def get_data(self):
        return self.response

    def set_data(self, data):
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

    def mean_function(self, eta):
        return eta

class logistic_loglike(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{\text{logit}}\left(%(var)s\right)"""
    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

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
            self.successes = np.asarray(successes).astype(np.float)

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

class poisson_loglike(smooth_atom):

    """
    A class for combining the Poisson log-likelihood with a general seminorm
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

        saturated = counts
        loss_terms = - coef * ((counts - 1) * np.log(counts))
        loss_terms[counts == 0] = 0

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

        return self.scale(cw * np.exp(x))
            
    def get_data(self):
        return self.counts

    def set_data(self, data):
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

    def mean_function(self, eta):
        return np.exp(eta)

class huber_loss(smooth_atom):

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
        raise NotImplementedError('Huber loss is not twice differentiable')

    def get_data(self):
        return self.response

    def set_data(self, data):
        self.response = data

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
        response = self.data
        return huber_loss(self.shape,
                          copy(response),
                          self.smoothing_parameter,
                          coef=self.coef,
                          offset=copy(self.offset),
                          quadratic=copy(self.quadratic),
                          initial=copy(self.coefs))

    # End loss API

def _huber_loss(arg, smoothing_parameter):
    # returns vector whose sum is total loss as well as gradient vector
    eps = smoothing_parameter
    proj_arg = np.sign(arg) * np.minimum(np.abs(arg) / eps, 1) # the maximizer is the gradient
                                                               # by convex conjugacy
    return arg * proj_arg - eps * proj_arg**2 / 2, proj_arg 

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
        atom = l1norm(shape, lagrange=1.)
        Q = identity_quadratic(smoothing_parameter, 0, 0, 0)
        self.smoothed_atom = atom.smoothed(Q)
                                         
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

class stacked_loglike(smooth_atom):

    """
    A class for stacking `K=len(losses)` saturated losses with common
    shapes (roughly speaking) summed over losses.

    Roughly speaking a model of `K` independent measurements per individual.
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

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `natural_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        if case_weights is None:
            case_weights = np.ones(natural_param.shape[:1])
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

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
        return self._data

    def set_data(self, data):
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

    @classmethod
    def gaussian(klass,
                 responses,
                 case_weights=None,
                 coef=1., 
                 offset=None,
                 quadratic=None, 
                 initial=None):
        """
        Create a loss for a Gaussian regression model.

        Parameters
        ----------

        responses : ndarray
            Response vectors.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        mglm_obj : `regreg.mglm.mglm`
            General linear model loss.

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
        """
        Create a loss for a logistic regression model.

        Parameters
        ----------

        successes : ndarray
            Responses (should be non-negative integers).

        trials : ndarray (optional)
            Number of trials for each success. If `None`,
            defaults to `np.ones_like(successes)`.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        mglm_obj : `regreg.mglm.mglm`
            General linear model loss.

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
    def poisson(klass,
                counts,
                case_weights=None,
                coef=1., 
                offset=None,
                quadratic=None, 
                initial=None):
        """
        Create a loss for a Poisson regression model.

        Parameters
        ----------

        counts : ndarray
            Response vector. Should be non-negative integers.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        mglm_obj : `regreg.mglm.mglm`
            General linear model loss.

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
        """
        Create a loss for a regression model using
        Huber loss.

        Parameters
        ----------

        response : ndarray
            Response vector. 

        smoothing_parameter : float
            Smoothing parameter for Huber loss.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        mglm_obj : `regreg.mglm.mglm`
            General linear model loss.

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
        """
        Create a loss for a regression model using
        Huber loss.

        Parameters
        ----------

        response : ndarray
            Response vector. 

        smoothing_parameter : float
            Smoothing parameter for Huber loss.

        case_weights : ndarray
            Non-negative case weights

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        mglm_obj : `regreg.mglm.mglm`
            General linear model loss.

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
        """
        Create a loss for a Cox regression model.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        event_times : ndarray
            Observed times for Cox proportional hazard model.

        censoring : ndarray 
            Censoring indicator for Cox proportional hazard model
            - 1 indicates observation is a failure, 0 a censored observation.

        offset : ndarray (optional)
            Offset to be applied in parameter space before 
            evaluating loss.

        quadratic : `regreg.identity_quadratic.identity_quadratic` (optional)
            Optional quadratic to be added to objective.

        initial : ndarray
            Initial guess at coefficients.
           
        Returns
        -------

        mglm_obj : `regreg.mglm.mglm`
            General linear model loss.

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
    '''
    Construct a logistic loss function for successes Y and
    affine transform X.

    Parameters
    ----------

    X : [affine_transform, ndarray]
        Design matrix

    Y : ndarray

    '''
    warnings.warn('"logistic_loss" is deprecated use "regreg.smooth.glm.logistic" instead')
    return glm.logistic(X, 
                        Y,
                        trials=trials,
                        coef=coef)
