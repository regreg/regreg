"""Mixin classes for building sklearn estimators from regreg problems.

If `sklearn` is importable, this module defines mixin classes that pair
regression or classification losses with an atom so that, for
example, cross-validation error can be evaluated.
"""

import numpy as np

from regreg.smooth.glm import (glm, 
                               gaussian_loglike,
                               huber_loss,
                               logistic_loglike,
                               poisson_loglike,
                               cox_loglike)

from regreg.api import (simple_problem,
                        identity_quadratic)

have_sklearn = True
try:
    from sklearn.base import (BaseEstimator, 
                              RegressorMixin,
                              ClassifierMixin)
except ImportError:
    have_sklearn = False



def _null_success_probability(successes, trials=None, case_weights=None):
    """(Weighted) overall success probability, the fitted mean of the null model."""
    successes = np.asarray(successes, float)
    w = np.ones_like(successes) if case_weights is None else np.asarray(case_weights, float)
    t = np.ones_like(successes) if trials is None else np.asarray(trials, float)
    return np.sum(w * successes) / np.sum(w * t)

def _binary_accuracy(predictions, successes, trials=None, case_weights=None,
                     threshold=0):
    """(Weighted) proportion of cases whose label is predicted correctly.

    The predicted label is ``predictions > threshold``, where
    `predictions` is the linear predictor and `threshold` the value at
    which the fitted probability is 1/2. The observed label is whether the
    proportion of successes exceeds 1/2.
    """
    successes = np.asarray(successes, float)
    observed = successes if trials is None else successes / np.asarray(trials, float)
    correct = (np.asarray(predictions) > threshold) == (observed > 0.5)
    return np.average(correct, weights=case_weights)

class base_mixin(object):
    """Base mixin pairing a loss with an atom used as a regularizer."""

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 offset=False,
                 coef=1., 
                 quadratic=None,
                 score_method='deviance',
                 solve_args={},
                 initial=None):

        """Initialize the estimator.

        Parameters
        ----------
        atom_constructor : callable
            Atom constructor that is to be used as a regularizer. The
            final atom will be constructed as
            ``atom_constructor(**atom_params)``.
        atom_params : dict
            Dictionary of arguments to construct the atom.
        case_weights : bool, optional
            Will `y` include a case weight column when fitting?
        offset : bool, optional
            Will `y` include an offset column when fitting?
        coef : float, optional
            Scaling to be put in front of the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the problem.
        score_method : str, optional
            Which score to use as default `score`. One of
            ``['deviance', 'mean_deviance', 'R2']``.
        solve_args : dict, optional
            Keyword arguments passed to `simple_problem.solve`.
        initial : ndarray, optional
            Warm start for the coefficients.

        Notes
        -----
        If `case_weights` or `offset` is True, then the `fit` method
        should pass a 2-d array for `y` rather than just a response. If
        only one of these is True, then the quantity is assumed to be the
        last column. If both are True, then the second to last column
        holds the case weights and the last column holds the offset.

        The offset, if present, is *subtracted* from the linear predictor
        before evaluating the loss -- this sign is different than the
        behavior in e.g. `R` where it is *added* instead.
        """

        self.atom_constructor = atom_constructor
        self.atom_params = atom_params
        self.case_weights = case_weights
        self.coef = coef
        self.offset = offset
        self.score_method = score_method
        self.solve_args = solve_args
        self.initial = initial
        self.quadratic = quadratic

    def fit(self, X, y):
        """Fit a regularized regression estimator.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n, p)``.
        y : ndarray
            Response, of shape ``(n,)``, or a 2-d array that also holds case
            weights and/or offset columns (see the Notes of `__init__`).

        Returns
        -------
        self
        """

        self._loglike = loglike = self._loglike_factory(X, y)
        atom_ = self._construct_atom()
        problem = simple_problem(loglike, atom_)
        if self.initial is not None:
            problem.coefs[:] = self.initial
        self._coefs = problem.solve(quadratic=self.quadratic, **self.solve_args)
        return self

    def predict(self, X):
        r"""Predict new responses in the regression setting.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n, p)``.

        Returns
        -------
        yhat : ndarray
            Linear predictor :math:`X\hat{\beta}` of shape ``(n,)``.
        """
        return X.dot(self._coefs)

    def score(self, X, y, sample_weight=None):
        """Score the fitted model on data `X`, `y`.

        The score is computed according to `score_method`. Deviance-type
        scores are negated so that larger is better, as sklearn's model
        selection tools maximize scores.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`, possibly with case weight and offset
            columns as in `fit`.
        sample_weight : array-like, shape (n_samples,), optional
            Sample weights, multiplied into the case weights if `y`
            includes them, and used as the case weights otherwise.

        Returns
        -------
        score : float
        """

        response, case_weights, offset = self._check_y_arg(y)
        predictions = X.dot(self._coefs)
        if offset is not None:
            predictions -= offset   # this is how offsets are incorporated
                                    # in regreg
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if case_weights is not None:
                case_weights = case_weights * sample_weight
            else:
                case_weights = sample_weight
        return self._saturated_score(predictions, response, case_weights=case_weights)

    def _construct_atom(self):
        return self.atom_constructor(**self.atom_params)

    def _loglike_factory(self, X, y):
        raise NotImplementedError

    def _check_y_arg(self, y):
        """Split the `y` argument of `fit` into its components.

        The components are the response, case weights and offset.

        Returns
        -------
        response : ndarray
        case_weights : ndarray or None
        offset : ndarray or None
        """
        response_idx, case_weights_idx, offset_idx = {
            (False, False): (None, None, None),
            (False, True): (0, None, -1),
            (True, False): (0, -1, None),
            (True, True): (0, -2, -1)}[self.case_weights,
                                     self.offset]

        if response_idx is None:
            response = y
            case_weights = offset = None
        else:
            neg_index = 0

            if case_weights_idx is not None:
                case_weights = y[:,case_weights_idx]
                neg_index -= 1
            else:
                case_weights = None

            if offset_idx is not None:
                offset = y[:,offset_idx]
                neg_index -= 1
            else:
                offset = None

            if neg_index < 0:
                response = np.squeeze(y[:,:neg_index])

        return response, case_weights, offset

    def _saturated_score(self,
                         predictions,
                         response,
                         case_weights=None):
        raise NotImplementedError

class lagrange_mixin(base_mixin):

    """A regularized estimator where the atom is in Lagrange form.

    Allows for an elastic net type ridge term. The atom constructor can
    use the gradient of the loss at 0 (or at the null solution) to
    construct the appropriate atom -- commonly the Lagrange parameter is
    taken to be some multiple in [0, 1] of the dual norm of this vector.
    """

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 offset=False,
                 enet_alpha=0.,
                 coef=1., 
                 score_method='deviance',
                 unpenalized=False,
                 quadratic=None,
                 solve_args={},
                 initial=None):

        """Initialize the estimator.

        Parameters
        ----------
        atom_constructor : callable
            Atom constructor that is to be used as a regularizer. The final
            atom will be constructed as
            ``atom_constructor(score, **atom_params)`` where `score` is the
            gradient of the loss at the null solution.
        atom_params : dict
            Dictionary of arguments to construct the atom.
        case_weights : bool, optional
            Will `y` include a case weight column when fitting?
        offset : bool, optional
            Will `y` include an offset column when fitting?
        enet_alpha : float, optional
            Weighting in [0, 1] between the ridge term and the regularizer:
            the atom's Lagrange parameter is multiplied by
            ``1 - enet_alpha`` and a ridge term with coefficient
            ``enet_alpha * lagrange`` is added.
        coef : float, optional
            Scaling to be put in front of the loss.
        score_method : str, optional
            Which score to use as default `score`. One of
            ``['deviance', 'mean_deviance', 'R2']``.
        unpenalized : bool, optional
            Whether any features are unpenalized. If so, the null solution
            (with ``lagrange=np.inf``) requires solving a problem, whose
            gradient is then used to construct the atom.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the problem. Its gradient is
            included in the gradient used to construct the atom.
        solve_args : dict, optional
            Keyword arguments passed to `simple_problem.solve`.
        initial : ndarray, optional
            Warm start for the coefficients.

        Notes
        -----
        If `case_weights` or `offset` is True, then the `fit` method
        should pass a 2-d array for `y` rather than just a response. If
        only one of these is True, then the quantity is assumed to be the
        last column. If both are True, then the second to last column
        holds the case weights and the last column holds the offset.

        The offset, if present, is *subtracted* from the linear predictor
        before evaluating the loss -- this sign is different than the
        behavior in e.g. `R` where it is *added* instead.
        """

        self.atom_constructor = atom_constructor
        self.atom_params = atom_params
        self.case_weights = case_weights
        self.coef = coef
        self.offset = offset
        self.score_method = score_method
        self.enet_alpha = enet_alpha
        self.unpenalized = unpenalized
        self.quadratic = quadratic
        self.solve_args = solve_args
        self.initial = initial
        
    def fit(self, X, y):
        """Fit a regularized regression estimator.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n, p)``.
        y : ndarray
            Response, of shape ``(n,)``, or a 2-d array that also holds case
            weights and/or offset columns (see the Notes of `__init__`).

        Returns
        -------
        self
        """

        self._loglike = loglike = self._loglike_factory(X, y)

        # with unpenalized parameters possible,
        # this may be best found by solving a problem with an atom with lagrange=np.inf
        # this could get expensive though

        null_soln = np.zeros(loglike.shape)
        null_grad = self._null_grad(loglike, null_soln)
        atom_ = self._construct_atom(null_grad)
        if self.unpenalized:
            null_grad = self._fit_null_soln(loglike, atom_)
        atom_ = self._construct_atom(null_grad)
        problem = simple_problem(loglike, atom_)
        if self.initial is not None:
            problem.coefs[:] = self.initial
        self._coefs = problem.solve(quadratic=self.quadratic, **self.solve_args)

        return self

    def _null_grad(self, loglike, null_soln):
        # gradient of the smooth part of the objective
        grad = loglike.smooth_objective(null_soln, 'grad')
        if getattr(self, 'quadratic', None) is not None:
            grad = grad + self.quadratic.objective(null_soln, 'grad')
        return grad

    def _fit_null_soln(self, loglike, atom):
        # a large finite value, as in regreg.paths: with np.inf the objective
        # is inf * 0 = nan at any point where the penalty is 0
        atom.lagrange = 1e12
        problem = simple_problem(loglike, atom)
        null_soln = problem.solve(quadratic=getattr(self, 'quadratic', None))
        return self._null_grad(loglike, null_soln)

    def _construct_atom(self, score):
        atom_ = self.atom_constructor(score, **self.atom_params)
        if self.enet_alpha != 0:
            # reweight penalty and add elastic net
            ridge_term = identity_quadratic(self.enet_alpha * atom_.lagrange, 0, 0, 0)
            atom_.lagrange *= (1 - self.enet_alpha)
            # add to, rather than replace, any quadratic the atom carries
            if atom_.quadratic is not None:
                ridge_term = atom_.quadratic + ridge_term
            atom_.quadratic = ridge_term
        return atom_
            
class classifier_mixin(base_mixin):
    """Mixin for classifiers.

    Predictions are labels rather than linear predictors.
    """

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 coef=1., 
                 quadratic=None,
                 score_method='accuracy',
                 solve_args={},
                 initial=None):

        """Initialize the estimator.

        Parameters
        ----------
        atom_constructor : callable
            Atom constructor that is to be used as a regularizer. The
            final atom will be constructed as
            ``atom_constructor(**atom_params)``.
        atom_params : dict
            Dictionary of arguments to construct the atom.
        case_weights : bool, optional
            Will `y` include a case weight column when fitting?
        coef : float, optional
            Scaling to be put in front of the loss.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the problem.
        score_method : str, optional
            Which score to use as default `score`. One of ``['accuracy']``.
        solve_args : dict, optional
            Keyword arguments passed to `simple_problem.solve`.
        initial : ndarray, optional
            Warm start for the coefficients.

        Notes
        -----
        If `case_weights` is True, then the `fit` method should pass at least
        a 2-d array for `y` rather than just a response. The last column will
        be taken to be the case weights.
        """

        base_mixin.__init__(self,
                            atom_constructor, 
                            atom_params, 
                            case_weights=case_weights,
                            coef=coef,
                            quadratic=quadratic,
                            score_method=score_method,
                            solve_args=solve_args,
                            initial=initial)

    def predict(self, X):
        """Predict new labels in the classification setting.

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n, p)``.

        Returns
        -------
        labels : ndarray
            Boolean labels, True where the linear predictor is positive.
        """
        return X.dot(self._coefs) > 0

    def predict_proba(self, X):
        """Predict new probabilities in the classification setting (abstract).

        Parameters
        ----------
        X : ndarray
            Feature matrix of shape ``(n, p)``.

        Returns
        -------
        probs : ndarray
            Predicted probabilities, of shape ``(n,)``.
        """
        raise NotImplementedError

class classifier_lagrange_mixin(classifier_mixin, lagrange_mixin):
    """Classifier mixin with the atom in Lagrange form."""

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 enet_alpha=0.,
                 coef=1., 
                 score_method='accuracy',
                 unpenalized=False,
                 quadratic=None,
                 solve_args={},
                 initial=None):

        """Initialize the estimator.

        Parameters
        ----------
        atom_constructor : callable
            Atom constructor that is to be used as a regularizer. The final
            atom will be constructed as
            ``atom_constructor(score, **atom_params)`` where `score` is the
            gradient of the loss at the null solution.
        atom_params : dict
            Dictionary of arguments to construct the atom.
        case_weights : bool, optional
            Will `y` include a case weight column when fitting?
        enet_alpha : float, optional
            Weighting in [0, 1] between the ridge term and the regularizer:
            the atom's Lagrange parameter is multiplied by
            ``1 - enet_alpha`` and a ridge term with coefficient
            ``enet_alpha * lagrange`` is added.
        coef : float, optional
            Scaling to be put in front of the loss.
        score_method : str, optional
            Which score to use as default `score`. One of ``['accuracy']``.
        unpenalized : bool, optional
            Whether any features are unpenalized; see `lagrange_mixin`.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the problem.
        solve_args : dict, optional
            Keyword arguments passed to `simple_problem.solve`.
        initial : ndarray, optional
            Warm start for the coefficients.

        Notes
        -----
        If `case_weights` is True, then the `fit` method should pass at least
        a 2-d array for `y` rather than just a response. The last column will
        be taken to be the case weights.
        """

        lagrange_mixin.__init__(self, 
                                atom_constructor, 
                                atom_params, 
                                case_weights=case_weights,
                                offset=False,
                                enet_alpha=enet_alpha,
                                coef=coef, 
                                score_method=score_method,
                                unpenalized=unpenalized,
                                quadratic=quadratic,
                                solve_args=solve_args,
                                initial=initial)
                       

class survival_mixin(base_mixin):
    """Mixin for survival models."""

    pass

class survival_mixin_lagrange(survival_mixin, lagrange_mixin):
    """Survival mixin with the atom in Lagrange form."""

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 offset=False,
                 enet_alpha=0.,
                 coef=1., 
                 score_method='deviance',
                 unpenalized=False,
                 quadratic=None,
                 solve_args={},
                 initial=None):

        """Initialize the estimator.

        Parameters
        ----------
        atom_constructor : callable
            Atom constructor that is to be used as a regularizer. The final
            atom will be constructed as
            ``atom_constructor(score, **atom_params)`` where `score` is the
            gradient of the loss at the null solution.
        atom_params : dict
            Dictionary of arguments to construct the atom.
        case_weights : bool, optional
            Will `y` include a case weight column when fitting?
        offset : bool, optional
            Will `y` include an offset column when fitting?
        enet_alpha : float, optional
            Weighting in [0, 1] between the ridge term and the regularizer:
            the atom's Lagrange parameter is multiplied by
            ``1 - enet_alpha`` and a ridge term with coefficient
            ``enet_alpha * lagrange`` is added.
        coef : float, optional
            Scaling to be put in front of the loss.
        score_method : str, optional
            Which score to use as default `score`. One of
            ``['deviance', 'mean_deviance', 'R2', 'C-index']``.
        unpenalized : bool, optional
            Whether any features are unpenalized; see `lagrange_mixin`.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the problem.
        solve_args : dict, optional
            Keyword arguments passed to `simple_problem.solve`.
        initial : ndarray, optional
            Warm start for the coefficients.

        Notes
        -----
        If `case_weights` or `offset` is True, then the `fit` method
        should pass a 2-d array for `y` rather than just a response. If
        only one of these is True, then the quantity is assumed to be the
        last column. If both are True, then the second to last column
        holds the case weights and the last column holds the offset.

        The offset, if present, is *subtracted* from the linear predictor
        before evaluating the loss -- this sign is different than the
        behavior in e.g. `R` where it is *added* instead.
        """

        lagrange_mixin.__init__(self, 
                                atom_constructor, 
                                atom_params, 
                                case_weights=case_weights,
                                offset=offset,
                                enet_alpha=enet_alpha,
                                coef=coef, 
                                score_method=score_method,
                                unpenalized=unpenalized,
                                quadratic=quadratic,
                                solve_args=solve_args,
                                initial=initial)
                       

### Final mixins

if have_sklearn:

    class sklearn_regression_lagrange(lagrange_mixin, BaseEstimator, RegressorMixin): 
        """sklearn regressor with a regularizer in Lagrange form."""

        pass

    class sklearn_regression(base_mixin, BaseEstimator, RegressorMixin): 
        """sklearn regressor with an arbitrary atom as regularizer."""

        pass

    class sklearn_classifier(classifier_mixin, BaseEstimator, ClassifierMixin):
        """sklearn classifier with an arbitrary atom as regularizer."""

        pass

    class sklearn_classifier_lagrange(classifier_lagrange_mixin, BaseEstimator, ClassifierMixin):
        """sklearn classifier with a regularizer in Lagrange form."""

        pass

    class sklearn_survival(survival_mixin, BaseEstimator, RegressorMixin):
        """sklearn survival estimator with an arbitrary atom as regularizer."""

        pass

    class sklearn_survival_lagrange(survival_mixin_lagrange, BaseEstimator, RegressorMixin):
        """sklearn survival estimator with a regularizer in Lagrange form."""

        pass

