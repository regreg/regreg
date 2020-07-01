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

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

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
        pass

    class sklearn_cloglog_classifier(sklearn_classifier):

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

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
            linpred = X.dot(self._coefs)
            return 1 - np.exp(-np.exp(linpred))

    class sklearn_cloglog_classifier_lagrange(sklearn_cloglog_classifier):
        pass

