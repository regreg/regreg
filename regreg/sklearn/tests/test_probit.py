import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (probit,
                       probit_lagrange,
                       probit_classifier,
                       probit_classifier_lagrange)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm
from ...tests.decorators import set_seed_for_test

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_probit():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    y = y > 0

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    probit_lasso = probit(atom,
                          atom_args)

    probit_lasso.fit(X, y)
    print(cross_validate(probit_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(probit_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    probit_lasso_offset = probit(atom,
                                 atom_args,
                                 offset=True)
    y_offset = np.array([y, np.random.standard_normal(n)]).T
    probit_lasso_offset.fit(X, y_offset)

    probit_lasso_weights = probit(atom,
                                  atom_args,
                                  case_weights=True,
                                  score_method='accuracy')

    y_weights = np.array([y, np.ones_like(y)]).T
    probit_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(probit_lasso_weights._coefs,
                               probit_lasso._coefs)

    GridSearchCV(probit_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(probit_lasso_weights, params, cv=3).fit(X, y_weights)

    probit_lasso_both = probit(atom,
                               atom_args,
                               offset=True,
                               case_weights=True,
                               score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(probit_lasso_both, params, cv=3).fit(X, y_both)

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_probit_classifier():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    y = y > 0

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    probit_lasso = probit_classifier(atom,
                                     atom_args)

    probit_lasso.fit(X, y)
    print(cross_validate(probit_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(probit_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    probit_lasso_weights = probit_classifier(atom,
                                             atom_args,
                                             case_weights=True)

    y_weights = np.array([y, np.ones_like(y)]).T
    probit_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(probit_lasso_weights._coefs,
                               probit_lasso._coefs)

    GridSearchCV(probit_lasso_weights, params, cv=3).fit(X, y_weights)




