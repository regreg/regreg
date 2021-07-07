"""

If `sklearn` is importable, this module defines mixin classes that pair
regression or classification losses with an atom so that cross-validation
error can be evaluated.

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


class base_mixin(object):

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

        """

        Parameters
        ----------

        atom_constructor : callable

            Atom constructor that is to to be used as a regularizer.
            Final atom will be constructed as `atom_constructor(**atom_params)`.

        atom_params : dict
            Dictionary of arguments to construct atom.

        offset : bool
            While `y` include an offset column when fitting?

        case_weights : bool
            Will `y` include a case weight column when fitting?

        coef : float
            Scaling to be put in front of loss.

        score_method : str
            Which score to use as default `score`?
            One of ['deviance', 'mean_deviance', 'R2']

        solve_args : dict
            Keyword arguments passed to `simple_problem.solve`

        initial : ndarray
            Warmstart.

        Notes
        -----

        If `case_weights` or `offset` is True, then the `fit`
        method should pass a 2-d array for `y` rather than just a
        response. If only one of these is not None, then the
        quantity is assumed to be the last column.  If both are
        not None, then the 2nd to last are to be the case weights
        and the last is to be the offset.

        The `offset` argument, if not None
        is *subtracted* from the linear predictor
        before evaluating the loss -- this sign is different
        than behavior in e.g. `R` where it is *added* instead.

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
        """
        Fit a regularized regression estimator.
        
            Parameters
        ----------

        X : np.ndarray((n, p))
            Feature matrix.

        y : np.ndarray(n)
            Response vector.

        Returns
        -------

        self

        """

        self._loglike = loglike = self._loglike_factory(X, y)
        atom_ = self._construct_atom()
        problem = simple_problem(loglike, atom_)
        if self.initial is not None:
            problem.coefs[:] = self.initial
        self._coefs = problem.solve(**self.solve_args)
        return self

    def predict(self, X):
        """
        Predict new response in regression setting.

        Parameters
        ----------

        X : np.ndarray((n, p))
            Feature matrix.

        Returns
        -------

        yhat : np.ndarray(n)
            Predictions from regression model.

        """
        return X.dot(self._coefs)

    def score(self, X, y, sample_weight=None):
        '''

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
        '''

        response, case_weights, offset = self._check_y_arg(y)
        predictions = X.dot(self._coefs)
        if offset is not None:
            predictions -= offset   # this is how offsets are incorporated
                                    # in regreg
        if sample_weight is not None:
            if case_weights is not None:
                case_weights *= sample_weight
        return self._saturated_score(predictions, response, case_weights=case_weights)

    def _construct_atom(self):
        return self.atom_constructor(**self.atom_params)

    def _loglike_factory(self, X, y):
        raise NotImplementedErorr

    def _check_y_arg(self, y):
        """
        Find response, case weights and 
        offset (if applicable) from
        `y` argument to fit.
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

    """
    A regularized estimator where atom constructed
    is assumed to be in Lagrange form. Allows 
    for ElasticNet type term.

    The atom constructor can use the gradient
    of the loss at 0 in order to construct
    the appropriate atom -- commonly the 
    Lagrange parameter is taken to be some multiple
    in [0,1] of the dual norm of this vector.
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
                 solve_args={},
                 initial=None):

        """

        Parameters
        ----------

        atom_constructor : callable

            Atom constructor that is to to be used as a regularizer.
            Final atom will be constructed as `atom_constructor(**atom_params)`.

        atom_params : dict
            Dictionary of arguments to construct atom.

        offset : bool
            While `y` include an offset column when fitting?

        case_weights : bool
            Will `y` include a case weight column when fitting?

        enet_alpha : float
            Weighting in [0, 1] between ridge and regularizer.

        coef : float
            Scaling to be put in front of loss.

        score_method : str
            Which score to use as default `score`?
            One of ['deviance', 'mean_deviance', 'R2']

        unpenalized : bool
            If any features are unpenalized,
            the null solution with
            `lagrange=np.inf` requires solving a 
            problem. 

        solve_args : dict
            Keyword arguments passed to `simple_problem.solve`

        Notes
        -----

        If `case_weights` or `offset` is True, then the `fit`
        method should pass a 2-d array for `y` rather than just a
        response. If only one of these is not None, then the
        quantity is assumed to be the last column.  If both are
        not None, then the 2nd to last are to be the case weights
        and the last is to be the offset.

        The `offset` argument, if not None
        is *subtracted* from the linear predictor
        before evaluating the loss -- this sign is different
        than behavior in e.g. `R` where it is *added* instead.

        """

        self.atom_constructor = atom_constructor
        self.atom_params = atom_params
        self.case_weights = case_weights
        self.coef = coef
        self.offset = offset
        self.score_method = score_method
        self.enet_alpha = enet_alpha
        self.unpenalized = unpenalized
        self.solve_args = solve_args
        self.initial = initial
        
    def fit(self, X, y):
        """
        Fit a regularized regression estimator.

        Parameters
        ----------

        X : np.ndarray((n, p))
            Feature matrix.

        y : np.ndarray(n)
            Response vector.

        Returns
        -------

        self

        """

        self._loglike = loglike = self._loglike_factory(X, y)

        # with unpenalized parameters possible,
        # this may be best found by solving a problem with an atom with lagrange=np.inf
        # this could get expensive though

        null_grad = loglike.smooth_objective(np.zeros(loglike.shape), 'grad')
        atom_ = self._construct_atom(null_grad)
        if self.unpenalized:
            null_grad = self._fit_null_soln(loglike, atom_)
        atom_ = self._construct_atom(null_grad)
        problem = simple_problem(loglike, atom_)
        if self.initial is not None:
            problem.coefs[:] = self.initial
        self._coefs = problem.solve(**self.solve_args)

        return self

    def _fit_null_soln(self, loglike, atom):
        old_lagrange = atom.lagrange
        atom.lagrange = np.inf
        problem = simple_problem(loglike, atom)
        null_soln = problem.solve()
        null_grad = loglike.smooth_objective(null_soln, 'grad')
        return null_grad

    def _construct_atom(self, score):
        atom_ = self.atom_constructor(score, **self.atom_params)
        if self.enet_alpha != 0:
            # reweight penalty and add elastic net
            ridge_term = identity_quadratic(self.enet_alpha * atom_.lagrange, 0, 0, 0)
            atom_.lagrange *= (1 - self.enet_alpha)
            atom_.quadratic = ridge_term
        return atom_
            
class classifier_mixin(base_mixin):

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 coef=1., 
                 quadratic=None,
                 score_method='accuracy',
                 initial=None):

        """

        Parameters
        ----------

        atom_constructor : callable

            Atom constructor that is to to be used as a regularizer.
            Final atom will be constructed as `atom_constructor(**atom_params)`.

        atom_params : dict
            Dictionary of arguments to construct atom.

        case_weights : bool
            Will `y` include a case weight column when fitting?

        coef : float
            Scaling to be put in front of loss.

        score_method : str
            Which score to use as default `score`?
            One of ['accuracy']

        Notes
        -----

        If `case_weights` is True, then the `fit`
        method should pass at least 2-d array for `y` rather than just a
        response. The last column will be taken to be
        `case_weights`.

        """

        base_mixin.__init__(self,
                            atom_constructor, 
                            atom_params, 
                            case_weights=case_weights,
                            coef=coef,
                            quadratic=quadratic,
                            score_method=score_method,
                            initial=initial)

    def predict(self, X):
        """
        Predict new labels in setting.

        Parameters
        ----------

        X : np.ndarray((n, p))
            Feature matrix.

        Returns
        -------

        labels : np.ndarray(n)
            Predictions from classification model.

        """
        return X.dot(self._coefs) > 0

    def predict_proba(self, X):
        """
        Predict new probabilities in classification setting.

        Parameters
        ----------

        X : np.ndarray((n, p))
            Feature matrix.

        Returns
        -------

        probs : np.ndarray(n)
            Predictions from classification model.

        """
        raise NotImplementedError

class classifier_lagrange_mixin(classifier_mixin, lagrange_mixin):

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 enet_alpha=0.,
                 coef=1., 
                 score_method='accuracy',
                 initial=None):

        """

        Parameters
        ----------

        atom_constructor : callable

            Atom constructor that is to to be used as a regularizer.
            Final atom will be constructed as `atom_constructor(**atom_params)`.

        atom_params : dict
            Dictionary of arguments to construct atom.

        case_weights : bool
            Will `y` include a case weight column when fitting?

        enet_alpha : float
            Weighting in [0, 1] between ridge and regularizer.

        coef : float
            Scaling to be put in front of loss.

        score_method : str
            Which score to use as default `score`?
            One of ['accuracy']

        Notes
        -----

        If `case_weights` is True, then the `fit`
        method should pass at least 2-d array for `y` rather than just a
        response. The last column will be taken to be
        `case_weights`.

        """

        lagrange_mixin(self, 
                       atom_constructor, 
                       atom_params, 
                       case_weights=case_weights,
                       offset=offset,
                       enet_alpha=enet_alpha,
                       coef=coef, 
                       score_method=score_method,
                       initial=initial)
                       

class survival_mixin(base_mixin):
    pass

class survival_mixin_lagrange(survival_mixin, lagrange_mixin):

    def __init__(self, 
                 atom_constructor, 
                 atom_params, 
                 case_weights=False,
                 offset=False,
                 enet_alpha=0.,
                 coef=1., 
                 score_method='deviance',
                 initial=None):

        """

        Parameters
        ----------

        atom_constructor : callable

            Atom constructor that is to to be used as a regularizer.
            Final atom will be constructed as `atom_constructor(**atom_params)`.

        atom_params : dict
            Dictionary of arguments to construct atom.

        offset : bool
            While `y` include an offset column when fitting?

        case_weights : bool
            Will `y` include a case weight column when fitting?

        enet_alpha : float
            Weighting in [0, 1] between ridge and regularizer.

        coef : float
            Scaling to be put in front of loss.

        score_method : str
            Which score to use as default `score`?
            One of ['deviance', 'mean_deviance', 'R2', 'C-index']

        Notes
        -----

        If `case_weights` or `offset` is True, then the `fit`
        method should pass a 2-d array for `y` rather than just a
        response. If only one of these is not None, then the
        quantity is assumed to be the last column.  If both are
        not None, then the 2nd to last are to be the case weights
        and the last is to be the offset.

        The `offset` argument, if not None
        is *subtracted* from the linear predictor
        before evaluating the loss -- this sign is different
        than behavior in e.g. `R` where it is *added* instead.

        """

        lagrange_mixin.__init__(self, 
                                atom_constructor, 
                                atom_params, 
                                case_weights=case_weights,
                                offset=offset,
                                enet_alpha=enet_alpha,
                                coef=coef, 
                                score_method=score_method,
                                initial=initial)
                       

### Final mixins

if have_sklearn:

    class sklearn_regression_lagrange(lagrange_mixin, BaseEstimator, RegressorMixin): 
        pass

    class sklearn_regression(base_mixin, BaseEstimator, RegressorMixin): 
        pass

    class sklearn_classifier(classifier_mixin, BaseEstimator, ClassifierMixin):
        pass

    class sklearn_classifier_lagrange(classifier_lagrange_mixin, BaseEstimator, ClassifierMixin):
        pass

    class sklearn_survival(survival_mixin, BaseEstimator, RegressorMixin):
        pass

    class sklearn_survival_lagrange(classifier_mixin, BaseEstimator, RegressorMixin):
        pass

