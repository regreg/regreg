import numpy as np, regreg.api as rr, regreg.affine as ra
import nose.tools as nt
import regreg.paths.group_lasso as group_lasso
import regreg.paths.lasso as lasso

from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_lasso_agreement(n=200,p=50):
    '''
    check to see if it agrees with lasso path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = np.arange(p)
    group_lasso1 = group_lasso.group_lasso_path.gaussian(X, 
                                                         Y, 
                                                         groups,
                                                         nstep=23)
    sol1 = group_lasso1.main(inner_tol=1.e-12)

    weights = np.ones(p)
    lasso2 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights,
                                       nstep=23)
    sol2 = lasso2.main(inner_tol=1.e-12)
    beta1 = sol1['beta']
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2)

@set_seed_for_test()
def test_path():
    '''
    run a basic path algorithm

    '''
    n, p = 200, 50
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(9):
        groups.extend([str(i)]*5)

    np.random.shuffle(groups)
    group_lasso1 = group_lasso.group_lasso_path.gaussian(X, 
                                                         Y, 
                                                         groups,
                                                         nstep=23)
    sol1 = group_lasso1.main(inner_tol=1.e-5)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_unpenalized(n=200, p=50):
    '''
    run a basic path algorithm with some unpenalized variables

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(9):
        groups.extend([str(i)]*5)

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['2'] = weights['3'] = 0
    weights['a'] = weights['b'] = 2

    group_lasso1 = group_lasso.group_lasso_path.gaussian(X, 
                                                         Y, 
                                                         groups,
                                                         weights=weights,
                                                         nstep=23)
    sol1 = group_lasso1.main(inner_tol=1.e-5)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_elastic_net(n=200, p=50):
    '''
    run a basic elastic net path algorithm

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.zeros(X.shape[1])
    enet[4:7] = 0

    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(9):
        groups.extend([str(i)]*5)

    group_lasso1 = group_lasso.group_lasso_path.gaussian(X, 
                                                         Y, 
                                                         groups,
                                                         nstep=23,
                                                         alpha=0.5,
                                                         elastic_net_param=enet)
    sol1 = group_lasso1.main(inner_tol=1.e-9)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_elastic_net_unpenalized(n=200, p=50):
    '''
    run a basic elastic net path algorithm with unpenalized

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(9):
        groups.extend([str(i)]*5)

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['2'] = weights['3'] = 0
    weights['a'] = weights['b'] = 2

    group_lasso1 = group_lasso.group_lasso_path.gaussian(X, 
                                                         Y, 
                                                         groups,
                                                         weights=weights,
                                                         nstep=23,
                                                         alpha=0.5,
                                                         elastic_net_param=enet)
    sol1 = group_lasso1.main(inner_tol=1.e-9)
    beta1 = sol1['beta']

