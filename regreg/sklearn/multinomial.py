"""sklearn estimators for multinomial regression and classification."""

import numpy as np

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange,
                       sklearn_classifier,
                       sklearn_classifier_lagrange)
except ImportError:
    have_sklearn = False

from ..smooth.mglm import (mglm, 
                           multinomial_loglike)

def _multinomial_accuracy(predictions, successes, case_weights=None):
    """(Weighted) proportion of cases whose most frequent category is the
    one with the largest linear predictor."""
    correct = np.argmax(predictions, 1) == np.argmax(successes, 1)
    return np.average(correct, weights=case_weights)

if have_sklearn:
    class sklearn_multinomial(sklearn_regression):

        """Regularized multinomial regression for sklearn.

        Any atom can be used as a regularizer.

        The response passed to `fit` is an ``(n, J)`` array of counts for
        each of the `J` categories.
        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            successes = response

            return mglm.multinomial(X, 
                                    successes,
                                    case_weights=case_weights_,
                                    coef=self.coef,
                                    saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            successes = response

            loss = lambda yhat: multinomial_loglike(successes.shape,
                                                    successes,
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
                # the null model fits the (weighted) overall category proportions
                w = np.ones(successes.shape[0]) if case_weights is None else case_weights
                totals = (w[:, None] * successes).sum(0)
                multinom_0 = np.log(totals / totals.sum())
                SST = 2 * loss(np.ones((successes.shape[0], 1)) * multinom_0[None, :])
                return 1 - SSE / SST
            elif self.score_method == 'accuracy':
                return _multinomial_accuracy(predictions, successes, case_weights)
            else:
                return np.nan

    class sklearn_multinomial_lagrange(sklearn_regression_lagrange, sklearn_multinomial):
        """Lagrange-form multinomial regression for sklearn."""

        pass

    class sklearn_multinomial_classifier(sklearn_classifier):

        """Regularized multinomial classifier for sklearn.

        Any atom can be used as a regularizer.
        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            successes = response

            return mglm.multinomial(X, 
                                    successes,
                                    case_weights=case_weights_,
                                    coef=self.coef,
                                    saturated_offset=offset_ # should be None
                                    )

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            successes = response

            if self.score_method == 'accuracy':
                return _multinomial_accuracy(predictions, successes, case_weights)
            return np.nan

        def predict(self, X):
            """Predict labels in the classification setting.

            Parameters
            ----------
            X : ndarray
                Feature matrix of shape ``(n, p)``.

            Returns
            -------
            labels : ndarray
                Index of the category with the largest linear predictor, of
                shape ``(n,)``.
            """
            linpred = X.dot(self._coefs)
            return np.argmax(linpred, 1)

        def predict_proba(self, X):
            """Predict category probabilities in the classification setting.

            Parameters
            ----------
            X : ndarray
                Feature matrix of shape ``(n, p)``.

            Returns
            -------
            probs : ndarray
                Predicted probabilities (softmax of the linear predictor), of
                shape ``(n, J)``.
            """
            linpred = X.dot(self._coefs)
            exp_lin = np.exp(linpred - linpred.max(1)[:, None])
            return exp_lin / exp_lin.sum(1)[:, None]

    class sklearn_multinomial_classifier_lagrange(sklearn_classifier_lagrange, sklearn_multinomial_classifier):
        """Lagrange-form multinomial classifier for sklearn."""

        pass

