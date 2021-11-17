import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (multinomial,
                       multinomial_lagrange,
                       multinomial_classifier,
                       multinomial_classifier_lagrange)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1_l2
from ...tests.decorators import set_seed_for_test

@np.testing.dec.skipif(True or not have_sklearn, msg='multinomial not working on its own yet')
@set_seed_for_test()
def test_multinomial():

    n, p, q = 100, 20, 3
    X = np.random.standard_normal((n, p))
    Z = np.random.standard_normal((n, q))
    A = np.argmax(Z, 1)
    Y = np.zeros_like(Z)
    for i in range(n):
        Y[i,A[i]] = 1
    response = Y

    atom = l1_l2
    atom_args = {'shape':(p, q), 'bound':3}

    multinomial_lasso = multinomial(atom,
                                    atom_args)

    multinomial_lasso.fit(X, response)
    print(cross_validate(multinomial_lasso, X, response, cv=10))

    # grid search
    params = {'atom_params':[{'shape':(p, q),
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(multinomial_lasso, params, cv=3)
    lasso_cv.fit(X, response)

    multinomial_lasso_offset = multinomial(atom,
                                           atom_args,
                                           offset=True)
    response_offset = np.hstack([response, np.random.standard_normal((n, 1))])
    multinomial_lasso_offset.fit(X, response_offset)

    multinomial_lasso_weights = multinomial(atom,
                                            atom_args,
                                            case_weights=True)

    response_weights = np.hstack([response, np.ones((n, 1))])
    multinomial_lasso_weights.fit(X, response_weights)

    np.testing.assert_allclose(multinomial_lasso_weights._coefs,
                               multinomial_lasso._coefs)

    GridSearchCV(multinomial_lasso_offset, params, cv=3).fit(X, response_offset)
    GridSearchCV(multinomial_lasso_weights, params, cv=3).fit(X, response_weights)

    def atom_constructor(null_grad, **args):
        return group_lasso(**args)
    multinomial_lasso_lag = multinomial_lagrange(atom_constructor,
                                                 atom_args,
                                                 case_weights=True,
                                                 score_method='R2')
    GridSearchCV(multinomial_lasso_lag, params, cv=3).fit(X, response_weights)



