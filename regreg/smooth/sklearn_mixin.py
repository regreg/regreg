"""

If `sklearn` is importable, this module defines mixin classes that pair
regression or classification losses with an atom so that cross-validation
error can be evaluated.

"""

from regreg.smooth.glm import glm
from regreg.api import simple_problem

have_sklearn = True
try:
    from sklearn.base import (BaseEstimator, 
                              RegressorMixin,
                              ClassifierMixin)
except ImportError:
    have_sklearn = False

if have_sklearn:
    class sklearn_regression(BaseEstimator, RegressorMixin): 

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

        """

        def __init__(self, loglike_factory, atom):

            """

            Parameters
            ----------

            loglike_factory : callable
                Loss constructor with signature (X, y)

            atom : regreg.atoms.atom
                Atom to be used as a regularizer.
                Should be solvable with `simple_problem(loss, atom)`.

            """

            self.loglike_factory = loglike_factory
            self.atom = atom

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

            None

            """

            loglike = self.loglike_factory(X, y)
            problem = simple_problem(loglike, self.atom)
            self._coefs = problem.solve()

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
                Preictions from regression model.

            """
            return X.dot(self._coefs)

        @staticmethod
        def gaussian(atom):
            """
            Create a Gaussian loss mixin for sklearn.

            Parameters
            ----------

            atom : regreg.atoms.atom
                Atom to be used as a regularizer.
                Should be solvable with `simple_problem(loss, atom)`.

            Returns
            -------

            mixin : sklearn_regression

            """

            return sklearn_regression(glm.gaussian, atom)

        @staticmethod
        def huber(smoothing_parameter, atom):
            """
            Create a Huber loss mixin for sklearn.

            Parameters
            ----------

            atom : regreg.atoms.atom
                Atom to be used as a regularizer.
                Should be solvable with `simple_problem(loss, atom)`.

            Returns
            -------

            mixin : sklearn_regression

            """
            huber_factory = lambda X, y: glm.huber(X, y, smoothing_parameter)
            return sklearn_regression(huber_factory, atom)

    class sklearn_classifier(BaseEstimator, ClassifierMixin): 

        def __init__(self, loglike_factory, atom):

            """

            Parameters
            ----------

            loglike_factory : callable
                Loss constructor with signature (X, y)

            atom : regreg.atoms.atom
                Atom to be used as a regularizer.
                Should be solvable with `simple_problem(loss, atom)`.

            """

            self.loglike_factory = loglike_factory
            self.atom = atom

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

            None

            """
            loglike = self.loglike_factory(X, y)
            problem = simple_problem(loglike, self.atom)
            self._coefs = problem.solve()

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
                Preictions from regression model.

            """
            return X.dot(self._coefs) > 0

        @staticmethod
        def logistic(atom):
            """
            Create a logistic loss mixin for sklearn.

            Parameters
            ----------

            atom : regreg.atoms.atom
                Atom to be used as a regularizer.
                Should be solvable with `simple_problem(loss, atom)`.

            Returns
            -------

            mixin : sklearn_classifier

            """
            return sklearn_classifier(glm.logistic, atom)


