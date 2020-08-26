import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (cox,
                       cox_lagrange)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm
from ...tests.decorators import set_seed_for_test


@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_cox():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    Y = np.random.exponential(1, size=(n,))
    C = np.random.binomial(1, 0.5, size=(n,))
    response = np.array([Y, C]).T

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    cox_lasso = cox(atom,
                    atom_args)

    cox_lasso.fit(X, response)
    print(cross_validate(cox_lasso, X, response, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(cox_lasso, params, cv=3)
    lasso_cv.fit(X, response)

    cox_lasso_offset = cox(atom,
                           atom_args,
                           offset=True)
    response_offset = np.hstack([response, np.random.standard_normal((n, 1))])
    cox_lasso_offset.fit(X, response_offset)

    cox_lasso_weights = cox(atom,
                            atom_args,
                            case_weights=True,
                            score_method='C-index')

    response_weights = np.hstack([response, np.ones((n, 1))])
    cox_lasso_weights.fit(X, response_weights)

    np.testing.assert_allclose(cox_lasso_weights._coefs,
                               cox_lasso._coefs)

    GridSearchCV(cox_lasso_offset, params, cv=3).fit(X, response_offset)
    GridSearchCV(cox_lasso_weights, params, cv=3).fit(X, response_weights)

    cox_lasso_both = cox(atom,
                         atom_args,
                         offset=True,
                         case_weights=True,
                         score_method='R2')
    response_both = np.hstack([response, np.ones((n, 1)), response_offset[:,-1:]])
    GridSearchCV(cox_lasso_both, params, cv=3).fit(X, response_both)



