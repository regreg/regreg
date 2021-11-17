import warnings
from copy import copy

import numpy as np
from scipy import sparse

from . import smooth_atom, affine_smooth
from ..affine import (astransform, 
                      linear_transform, 
                      affine_transform)
from ..identity_quadratic import identity_quadratic
from ..atoms.seminorms import l1norm
from .cox import cox_loglike

from .glm import (glm,
                  gaussian_loglike,
                  logistic_loglike,
                  poisson_loglike,
                  cox_loglike,
                  huber_loss,
                  huber_svm)

class mglm(glm):

    """
    A multiresponse general linear model, usually a log-likelihood
    for response $Y$ whose mean is modelled through
    $X\beta$. 

    Usual examples are multivariate Gaussian (least squares regression),
    and multinomial regression.

    """

    def __init__(self, 
                 X, 
                 loss, 
                 quadratic=None, 
                 initial=None,
                 offset=None,
                 case_weights=None):

        """

        Parameters
        ----------

        X : ndarray((n,p))
            The design matrix.

        loss : `regreg.smooth.smooth_atom`
            The loss function that takes arguments the same
            size as `Y`. So, for Gaussian regression 
            the loss is just the map $\mu \mapsto \|\mu - Y\|^2_F/2$
            the Frobenius loss. Should have shape (p,q).

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
        self.data = X, loss.data
        self.affine_atom = affine_smooth(loss, X)
        if case_weights is None:
            case_weights = np.ones(X.shape[0])
        self.case_weights = case_weights
        smooth_atom.__init__(self,
                             (X.shape[1], loss.shape[1]),
                             coef=1.,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial)

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

        subsample_loss : `mglm`
            Loss after discarding all
            cases not in `idx.

        """

        subsample_loss = copy(self)
        n = subsample_loss.saturated_loss.shape[0]

        idx_bool = np.zeros(n, np.bool)
        idx_bool[idx] = 1

        subsample_loss.case_weights *= idx_bool

        return subsample_loss
        
      
    def hessian(self, beta):
        """

        Compute the Hessian of the loss $ \nabla \ell(X\beta)$.
        NOT IMPLEMENTED.

        Parameters
        ----------

        beta : ndarray
            Parameters.
        """

        raise NotImplementedError

    def latexify(self, var=None, idx=''):
        return self.affine_atom.latexify(var=var, idx=idx)

    def __copy__(self):
        klass = self.__class__
        X, _ = self.data
        return klass(copy(X), 
                     copy(self.saturated_loss), 
                     quadratic=copy(self.quadratic),
                     initial=copy(self.coefs),
                     offset=copy(self.offset),
                     case_weights=copy(self.case_weights))
       
    @classmethod
    def multinomial(klass, 
                    X, 
                    successes, 
                    case_weights=None,
                    coef=1., 
                    saturated_offset=None,
                    offset=None,
                    quadratic=None, 
                    initial=None,
                    baseline=False):
        """
        Create a loss for a multinomial regression model.

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        successes : ndarray
            Responses (should be non-negative integers).

        case_weights : ndarray
            Non-negative case weights

        saturated_offset : ndarray (optional)
            Offset to be applied in saturated parameter space before 
            evaluating loss.

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

        if not baseline:
            loss = multinomial_loglike(successes.shape,
                                       successes,
                                       coef=coef,
                                       offset=saturated_offset)
        else:
            loss = multinomial_baseline_loglike((successes.shape[0],
                                                 successes.shape[1]-1),
                                                successes,
                                                coef=coef,
                                                offset=saturated_offset)

        return klass(X, 
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

    @classmethod
    def stacked(klass, 
                X, 
                losses,
                case_weights=None,
                coef=1., 
                offset=None,
                saturated_offset=None,
                quadratic=None, 
                initial=None):
        """
        Create a stacked loss. 

        Parameters
        ----------

        X : [ndarray, `regreg.affine.affine_transform`]
            Design matrix

        losses : sequence
            Sequence of saturated losses.

        coef : float
            Scaling to be put in front of loss.

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

        loss = stacked_common_loglike(losses,
                                      successes,
                                      coef=coef,
                                      offset=saturated_offset)

        return klass(X, 
                     loss,
                     offset=offset,
                     quadratic=quadratic,
                     initial=initial,
                     case_weights=case_weights)

class stacked_common_loglike(smooth_atom):

    """
    A class for stacking `K=len(losses)` saturated losses with common
    shapes (roughly speaking) summed over losses.

    Roughly speaking a model of `K` independent measurements per individual.
    """

    objective_template = r"""\ell^{\text{logit}}\left(%(var)s\right)"""

    def __init__(self, 
                 losses,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None,
                 case_weights=None):

        shape = losses[0].shape + (len(losses),)
        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        assert(np.all([l.shape == losses[0].shape for l in losses]))
        self.data = np.array([l.data for l in losses]).T

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

        x = natural_param # shorthand

        x = self.apply_offset(x)
        if mode == 'grad':
            for i in range(len(self._losses)):
                self._gradient[:,i] = self._losses[i].smooth_objective(x[:,i], 
                                                                       'grad', 
                                                                       case_weights=cw) 
            return self.scale(self._gradient)
        elif mode == 'func':
            value = 0
            for i in range(len(self._losses)):
                value += self._losses[i].smooth_objective(x[:,i], 
                                                          'func', 
                                                          case_weights=cw)
            return self.scale(value)
        elif mode == 'both':
            value = 0
            for i in range(len(self._losses)):
                f, g = self._losses[i].smooth_objective(x[:,i], 
                                                        'both', 
                                                        case_weights=cw)
                self._gradient[:,i] = g
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
        return stacked_common_loglike(copy(self._losses),
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

        successes : ndarray
            Response vectors. 

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

class multinomial_loglike(smooth_atom):

    overflow_clip = 50

    """
    An overparametrized multinomial regression for nominal responses (e.g. Agresti, pg 267)
    """

    objective_template = r"""\ell^{M}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 counts, 
                 coef=1., 
                 offset=None,
                 initial=None,
                 quadratic=None,
                 case_weights=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray()
        else:
            self.counts = counts

        self.J = self.counts.shape[1]

        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        self.trials = np.sum(self.counts, axis=1)

        saturated = self.counts / (1. * self.trials[:,np.newaxis])
        loss_terms = np.log(saturated) * self.counts
        loss_terms[np.isnan(loss_terms)] = 0
        loss_constant = - coef * loss_terms.sum()

        devq = identity_quadratic(0,0,0,-loss_constant)
        self.quadratic += devq

        self.data = self.counts, self.trials

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.shape[:1]:
                raise ValueError('case_weights should have same shape as response')
        else:
            self.case_weights = None

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    data = property(get_data, set_data)

    def smooth_objective(self, 
                         natural_param, 
                         mode='both', 
                         check_feasibility=False,
                         case_weights=None):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        eta = self.apply_offset(natural_param)  # shorthand
        eta = np.clip(eta, 
                      -self.overflow_clip, 
                      self.overflow_clip) # to prevent overflow
        exp_eta = np.exp(eta)

        if case_weights is None:
            cw = np.ones(natural_param.shape[0])
        else:
            cw = case_weights

        if mode == 'both':
            ratio = ((self.trials/np.sum(exp_eta, axis=1)) * exp_eta.T).T
            f, g = (- self.scale(np.sum(cw[:,None] * self.counts * eta) -  np.dot(cw * self.trials, np.log(np.sum(exp_eta, axis=1)))),
                      - self.scale(cw[:,None] * (self.counts - ratio)))
            return f, g
        elif mode == 'grad':
            ratio = ((self.trials/np.sum(exp_eta, axis=1)) * exp_eta.T).T
            f, g = None, - self.scale(cw[:,None] * (self.counts - ratio))
            return g
        elif mode == 'func':
            f, g = - self.scale(np.sum(cw[:,None] * self.counts * eta) -  np.dot(cw * self.trials, np.log(np.sum(exp_eta, axis=1)))), None
            return f
        else:
            raise ValueError("mode incorrectly specified")

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

class multinomial_baseline_loglike(smooth_atom):

    overflow_clip = 50

    """
    A class for baseline-category logistic regression for nominal responses (e.g. Agresti, pg 267)
    """

    objective_template = r"""\ell^{M}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 counts, 
                 coef=1., 
                 offset=None,
                 initial=None,
                 quadratic=None,
                 case_weights=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        if sparse.issparse(counts):
            #Convert sparse success vector to an array
            self.counts = counts.toarray()
        else:
            self.counts = counts

        self.J = self.counts.shape[1]
        #Select the counts for the first J-1 categories
        self.firstcounts = self.counts[:,range(self.J-1)]

        if not np.allclose(np.round(self.counts),self.counts):
            raise ValueError("Counts vector is not integer valued")
        if np.min(self.counts) < 0:
            raise ValueError("Counts vector is not non-negative")

        self.trials = np.sum(self.counts, axis=1)

        if shape[1] != self.J - 1:
            raise ValueError("Primal shape is incorrect - should only have coefficients for first J-1 categories")

        saturated = self.counts / (1. * self.trials[:,np.newaxis])
        loss_terms = np.log(saturated) * self.counts
        loss_terms[np.isnan(loss_terms)] = 0
        loss_constant = - coef * loss_terms.sum()

        devq = identity_quadratic(0,0,0,-loss_constant)
        self.quadratic += devq

        self.data = self.counts, self.trials

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.shape[:1]:
                raise ValueError('case_weights should have same shape as response')
        else:
            self.case_weights = None

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    data = property(get_data, set_data)

    def smooth_objective(self, 
                         natural_parameter,
                         mode='both', 
                         check_feasibility=False,
                         case_weights=None):
        """
        Evaluate a smooth function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """
        eta = self.apply_offset(natural_parameter) # shorthand
        eta = np.clip(eta, 
                      -self.overflow_clip, 
                      self.overflow_clip) # to prevent overflow
        exp_eta = np.exp(eta)

        if case_weights is None:
            cw = np.ones(eta.shape[0])
        else:
            cw = case_weights

        #TODO: Using transposes to scale the rows of a 2d array - should we use an affine_transform to do this?
        #JT: should be able to do this with np.newaxis

        if mode == 'both':
            ratio = ((self.trials/(1. + np.sum(exp_eta, axis=1))) * exp_eta.T).T
            f, g = (- self.scale(np.sum(cw[:,None] * self.firstcounts * eta) -  np.dot(cw * self.trials, np.log(1. + np.sum(exp_eta, axis=1)))), 
                      - self.scale(cw[:,None] * (self.firstcounts - ratio)))
            return f, g
        elif mode == 'grad':
            ratio = ((self.trials/(1. + np.sum(exp_eta, axis=1))) * exp_eta.T).T
            f, g = None, - self.scale(cw[:,None] * (self.firstcounts - ratio))
            return g
        elif mode == 'func':
            f, g = - self.scale(np.sum(cw[:,None] * self.firstcounts * eta) -  np.dot(cw * self.trials, np.log(1. + np.sum(exp_eta, axis=1)))), None
            return f
        else:
            raise ValueError("mode incorrectly specified")

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

