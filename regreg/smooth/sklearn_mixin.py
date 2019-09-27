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


