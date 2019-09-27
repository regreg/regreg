from sklearn.base import (BaseEstimator, 
                          RegressorMixin,
                          ClassifierMixin)

from regreg.smooth.glm import glm
from regreg.api import simple_problem

class sklearn_regression(BaseEstimator, RegressorMixin): 

    def __init__(self, loglike_factory, atom):
        self.loglike_factory = loglike_factory
        self.atom = atom

    def fit(self, X, y):
        loglike = self.loglike_factory(X, y)
        problem = simple_problem(loglike, self.atom)
        self._coefs = problem.solve()

    def predict(self, X):
        return X.dot(self._coefs)
    
    @staticmethod
    def gaussian(atom):
        return sklearn_regression(glm.gaussian, atom)

    @staticmethod
    def huber(atom):
        return sklearn_regression(glm.gaussian, atom)

class sklearn_classifier(BaseEstimator, ClassifierMixin): 

    def __init__(self, loglike_factory, atom):
        self.loglike_factory = loglike_factory
        self.atom = atom

    def fit(self, X, y):
        loglike = self.loglike_factory(X, y)
        problem = simple_problem(loglike, self.atom)
        self._coefs = problem.solve()

    def predict(self, X):
        return X.dot(self._coefs)
    
    @staticmethod
    def logistic(atom):
        return sklearn_classifier(glm.logistic, atom)


if __name__ == "__main__":

    import numpy as np
    from sklearn.model_selection import cross_val_score, cross_validate
    from regreg.api import l1norm

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    ybin = y > 0
    loss = glm.gaussian(X, y)
    pen = l1norm(20, lagrange=2 * np.sqrt(n))
    gaussian_lasso = sklearn_regression.gaussian(pen)
    logistic_lasso = sklearn_classifier.logistic(pen)
    print(cross_validate(gaussian_lasso, X, y, cv=3))
    print(cross_validate(logistic_lasso, X, y, cv=3))
