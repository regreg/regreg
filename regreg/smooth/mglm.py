r"""Multiresponse generalized linear models.

Losses of the form :math:`B \mapsto \ell(XB)` where the coefficients
:math:`B` form a matrix with one column per response (or category),
e.g. multinomial regression.
"""

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

    r"""A multiresponse general linear model.

    Usually a negative log-likelihood, the loss is
    :math:`B \mapsto \ell(XB)` for a response :math:`Y` whose mean is
    modelled through :math:`XB`, where :math:`B` has shape ``(p, q)``.

    Usual examples are multivariate Gaussian (least squares regression)
    and multinomial regression.
    """

    def __init__(self, 
                 X, 
                 loss, 
                 quadratic=None, 
                 initial=None,
                 offset=None,
                 case_weights=None):

        r"""Initialize the multiresponse general linear model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            The design matrix, of shape ``(n, p)``.
        loss : `regreg.smooth.smooth_atom`
            The saturated loss, of shape ``(n, q)``, taking arguments the same
            size as `Y`. For Gaussian regression the loss is just the
            Frobenius loss :math:`\mu \mapsto \|\mu - Y\|^2_F/2`.
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
        """Create a loss using a subsample of the data.

        Makes a copy of the loss and multiplies its case weights by the
        indicator of `idx`.

        Parameters
        ----------
        idx : index
            Indices of ``np.arange(n)`` to keep.

        Returns
        -------
        subsample_loss : `mglm`
            Copy of the loss with zero case weight for all cases not in
            `idx`.
        """

        subsample_loss = copy(self)
        n = subsample_loss.saturated_loss.shape[0]

        idx_bool = np.zeros(n, np.bool)
        idx_bool[idx] = 1

        subsample_loss.case_weights *= idx_bool

        return subsample_loss
        
      
    def hessian(self, beta):
        r"""Compute the Hessian of the loss :math:`\nabla^2 \ell(XB)`.

        Not implemented for multiresponse models.

        Parameters
        ----------
        beta : ndarray
            Parameters.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError

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
        """Create a loss for a multinomial regression model.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        successes : ndarray
            Responses of shape ``(n, J)``: non-negative counts for each of the
            `J` categories.
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
        baseline : bool, optional
            If True, use the baseline-category parametrization
            (`multinomial_baseline_loglike`) with coefficients for the first
            ``J - 1`` categories only; otherwise use the overparametrized
            `multinomial_loglike`.

        Returns
        -------
        mglm_obj : `regreg.smooth.mglm.mglm`
            Multiresponse general linear model loss.
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
        """Create a multiresponse model from a stack of saturated losses.

        Parameters
        ----------
        X : ndarray or `regreg.affine.affine_transform`
            Design matrix.
        losses : sequence
            Sequence of saturated losses with common shape.
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
        mglm_obj : `regreg.smooth.mglm.mglm`
            Multiresponse general linear model loss.
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

    """Stack ``K = len(losses)`` saturated losses with common shapes.

    The natural parameter has one column per loss, and the objective is
    the sum of the losses evaluated on the corresponding columns. Roughly
    speaking, this is a model of `K` independent measurements per
    individual.

    Parameters
    ----------
    losses : sequence
        Saturated losses, all with the same shape.
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
        """Return the stacked responses."""
        return self._data

    def set_data(self, data):
        """Set the stacked responses."""
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
        stacked_loss : `stacked_common_loglike`
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
        stacked_loss : `stacked_common_loglike`
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
        stacked_loss : `stacked_common_loglike`
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
              response,
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
        response : sequence of ndarray
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
        stacked_loss : `stacked_common_loglike`
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
        stacked_loss : `stacked_common_loglike`
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
        stacked_loss : `stacked_common_loglike`
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

class multinomial_loglike(smooth_atom):
    r"""An overparametrized multinomial loss for nominal responses.

    See e.g. Agresti, *Categorical Data Analysis*, p. 267. With natural
    parameter :math:`\eta` of shape ``(n, J)``, the loss is

    .. math::

       \eta \mapsto -\sum_i w_i \left[\sum_j y_{ij} \eta_{ij} -
           n_i \log \sum_j e^{\eta_{ij}}\right]

    where :math:`y_{ij}` are the counts, :math:`n_i = \sum_j y_{ij}` and
    :math:`w_i` the case weights. The counts need not be integers. The
    log-likelihood of the saturated model is stored as a constant term in
    `quadratic`.

    Parameters
    ----------
    shape : tuple
        Shape ``(n, J)`` of the natural parameter.
    counts : ndarray
        Non-negative counts of shape ``(n, J)``.
    coef : float, optional
        Nonnegative scaling to be put in front of the loss.
    offset : ndarray, optional
        Offset to be subtracted from the natural parameter before
        evaluating the loss.
    initial : ndarray, optional
        Initial guess at the coefficients.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term to be added to the objective.
    case_weights : ndarray, optional
        Non-negative case weights.
    """

    overflow_clip = 50

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
        """Return the data ``(counts, trials)``."""
        return self._data

    def set_data(self, data):
        """Set the data ``(counts, trials)``."""
        self._data = data

    data = property(get_data, set_data)

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
            Non-negative case weights. If None, unit weights are used (case
            weights stored on the loss are not applied).

        Returns
        -------
        f : float
            Objective value at `natural_param`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `natural_param`; returned when `mode` is 'grad'
            or 'both'.
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

class multinomial_baseline_loglike(smooth_atom):
    r"""Baseline-category logistic loss for nominal responses.

    See e.g. Agresti, *Categorical Data Analysis*, p. 267. The last of the
    `J` categories is the baseline, so the natural parameter :math:`\eta`
    has shape ``(n, J - 1)`` and the loss is

    .. math::

       \eta \mapsto -\sum_i w_i \left[\sum_{j < J} y_{ij} \eta_{ij} -
           n_i \log \left(1 + \sum_{j < J} e^{\eta_{ij}}\right)\right]

    where :math:`y_{ij}` are the (integer) counts, :math:`n_i = \sum_j
    y_{ij}` and :math:`w_i` the case weights. The log-likelihood of the
    saturated model is stored as a constant term in `quadratic`.

    Parameters
    ----------
    shape : tuple
        Shape ``(n, J - 1)`` of the natural parameter.
    counts : ndarray
        Non-negative integer counts of shape ``(n, J)``.
    coef : float, optional
        Nonnegative scaling to be put in front of the loss.
    offset : ndarray, optional
        Offset to be subtracted from the natural parameter before
        evaluating the loss.
    initial : ndarray, optional
        Initial guess at the coefficients.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term to be added to the objective.
    case_weights : ndarray, optional
        Non-negative case weights.
    """

    overflow_clip = 50

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
        """Return the data ``(counts, trials)``."""
        return self._data

    def set_data(self, data):
        """Set the data ``(counts, trials)``."""
        self._data = data

    data = property(get_data, set_data)

    def smooth_objective(self, 
                         natural_parameter,
                         mode='both', 
                         check_feasibility=False,
                         case_weights=None):
        """Evaluate the smooth objective, its gradient, or both.

        Parameters
        ----------
        natural_parameter : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `natural_parameter` is not in the domain.
        case_weights : ndarray, optional
            Non-negative case weights. If None, unit weights are used (case
            weights stored on the loss are not applied).

        Returns
        -------
        f : float
            Objective value at `natural_parameter`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `natural_parameter`; returned when `mode` is 'grad'
            or 'both'.
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

