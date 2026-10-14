"""sklearn estimators for complementary log-log regression models."""

import numpy as np
from scipy.stats import gumbel_r

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange,
                       sklearn_classifier,
                       sklearn_classifier_lagrange)
except ImportError:
    have_sklearn = False

from ..smooth.glm import (glm, 
                          cloglog_loglike)

if have_sklearn:
    class sklearn_cloglog(sklearn_regression):

        """Regularized complementary log-log regression for sklearn.

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

            return glm.cloglog(X, 
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

            loss = lambda yhat: cloglog_loglike(successes.shape,
                                                successes,
                                                trials=trials,
                                                case_weights=case_weights).smooth_objective(yhat, 'func')

            # factor of 2 to form proper deviance (default is negative log-likelihood,
            # while deviance is 2 * negative log-likelihood
            # negative sign is to align with sklearn's maximizing a score with grid search

            if self.score_method == 'deviance':
                return -2 * loss(predictions)
            elif self.score_method == 'mean_deviance':
                return -2 * loss(predictions)
            elif self.score_method == 'R2':
                SSE = 2 * loss(predictions)
                pi_0 = response.mean()
                cloglog_0 = gumbel_r.ppf(pi_0)
                SST = 2 * loss(cloglog_0 * np.ones_like(response)) # X: correct for cloglog?
                return 1 - SSE / SST
            elif self.score_method == 'accuracy':
                labels = predictions > 0
                return np.mean(labels == response)
            else:
                return np.nan

    class sklearn_cloglog_lagrange(sklearn_regression_lagrange, sklearn_cloglog):
        """Lagrange-form complementary log-log regression for sklearn."""

        pass

    class sklearn_cloglog_classifier(sklearn_classifier):

        """Regularized complementary log-log classifier for sklearn.

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

            return glm.cloglog(X, 
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
                return np.mean(predictions == successes)
            return np.nan

        def predict_proba(self, X):
            r"""Predict success probabilities in the classification setting.

            The probabilities are :math:`1 - \exp(-e^{\eta})`
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
            return 1 - np.exp(-np.exp(linpred))

    class sklearn_cloglog_classifier_lagrange(sklearn_cloglog_classifier):
        """Complementary log-log classifier (intended Lagrange form).

        Currently this is identical to `sklearn_cloglog_classifier`.
        """

        pass

