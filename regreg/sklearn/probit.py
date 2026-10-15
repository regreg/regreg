"""sklearn estimators for probit regression and classification."""

import numpy as np
from scipy.stats import norm as normal_dbn

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange,
                       sklearn_classifier,
                       sklearn_classifier_lagrange,
                       _null_success_probability,
                       _binary_accuracy)
except:
    have_sklearn = False
from ..smooth.glm import (glm, 
                          probit_loglike)

if have_sklearn:
    class sklearn_probit(sklearn_regression):

        """Regularized probit regression for sklearn.

        Any atom can be used as a regularizer.

        The response passed to `fit` is either a vector of binary successes
        or a 2-d array with columns ``(successes, trials)``.
        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            if response.ndim == 2:
                successes = response[:,0]
                trials = response[:,1]
            else:
                successes = response
                trials = None

            return glm.probit(X, 
                              successes,
                              trials=trials,
                              case_weights=case_weights_,
                              coef=self.coef,
                              saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            if response.ndim == 2:
                successes = response[:,0]
                trials = response[:,1]
            else:
                successes = response
                trials = None

            loss = lambda yhat: probit_loglike(successes.shape,
                                               successes,
                                               trials=trials,
                                               case_weights=case_weights).smooth_objective(yhat, 'func')

            # factor of 2 to form proper deviance (default is negative log-likelihood,
            # while deviance is 2 * negative log-likelihood
            # negative sign is to align with sklearn's maximizing a score with grid search

            if self.score_method == 'deviance':
                return -2 * loss(predictions)
            elif self.score_method == 'mean_deviance':
                return -2 * loss(predictions) / predictions.shape[0]
            elif self.score_method == 'R2':
                SSE = 2 * loss(predictions)
                # the null model fits the (weighted) overall success probability
                pi_0 = _null_success_probability(successes, trials, case_weights)
                eta_0 = normal_dbn.ppf(pi_0)
                SST = 2 * loss(eta_0 * np.ones(successes.shape[0]))
                return 1 - SSE / SST
            elif self.score_method == 'accuracy':
                return _binary_accuracy(predictions, successes, trials,
                                        case_weights, threshold=0)
            else:
                return np.nan

    class sklearn_probit_lagrange(sklearn_regression_lagrange, sklearn_probit):
        """Lagrange-form probit regression for sklearn."""

        pass

    class sklearn_probit_classifier(sklearn_classifier):

        """Regularized probit classifier for sklearn.

        Any atom can be used as a regularizer.
        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            if response.ndim == 2:
                successes = response[:,0]
                trials = response[:,1]
            else:
                successes = response
                trials = None

            return glm.probit(X, 
                              successes,
                              trials=trials,
                              case_weights=case_weights_,
                              coef=self.coef,
                              saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            if response.ndim == 2:
                successes = response[:,0]
                trials = response[:,1]
            else:
                successes = response
                trials = None

            if self.score_method == 'accuracy':
                return _binary_accuracy(predictions, successes, trials,
                                        case_weights, threshold=0)
            return np.nan

        def predict_proba(self, X):
            """Predict success probabilities in the classification setting.

            The probabilities are the standard normal CDF
            applied to the linear predictor.

            Parameters
            ----------
            X : ndarray
                Feature matrix of shape ``(n, p)``.

            Returns
            -------
            probs : ndarray
                Predicted probabilities, of shape ``(n,)``.
            """
            linpred = X.dot(self._coefs)
            return normal_dbn.cdf(linpred)

    class sklearn_probit_classifier_lagrange(sklearn_classifier_lagrange, sklearn_probit_classifier):
        """Lagrange-form probit classifier for sklearn."""

        pass

