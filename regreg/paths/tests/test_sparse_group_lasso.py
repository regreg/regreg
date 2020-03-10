import numpy as np

from .. import lasso, sparse_group_lasso, strong_rules
from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_lasso_agreement1(n=200,p=50):
    '''
    check to see if it agrees with lasso path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = np.arange(p)
    l1weights = np.ones(p)
    weights = dict([(j,0) for j in np.arange(p)])
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      l1_weight=1,
                                                      weights=weights)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-12)

    weights = np.ones(p)
    lasso2 = lasso.gaussian(X, 
                            Y, 
                            weights)
    sol2 = strong_rules(lasso2, lagrange_sequence, inner_tol=1.e-15)
    beta1 = sol1['beta']
    beta2 = sol2['beta']

    assert(np.linalg.norm(beta1 - beta2) / np.linalg.norm(beta1) < 1.e-5)

@set_seed_for_test()
def test_path_subsample(n=200,p=50):
    '''
    compare a subsample path to the full path on subsampled data

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(9):
        groups.extend([str(i)]*5)

    np.random.shuffle(groups)
    l1weights = np.ones(p)
    weights = {}
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      l1_weight=1,
                                                      weights=weights)
    cases = range(100)
    sparse_group_lasso1 = sparse_group_lasso1.subsample(cases)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-10)
    beta1 = sol1['beta']

    sparse_group_lasso2 = sparse_group_lasso.gaussian(X[cases], 
                                                      Y[cases], 
                                                      groups,
                                                      l1weights,
                                                      l1_weight=1,
                                                      weights=weights)
    sol2 = strong_rules(sparse_group_lasso2, lagrange_sequence, inner_tol=1.e-10)
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2, rtol=1.e-3)

@set_seed_for_test()
def test_lasso_agreement2(n=200,p=50):
    '''
    check to see if it agrees with lasso path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = np.arange(p)
    l1weights = np.zeros(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      l1_weight=0)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-12)

    weights = np.ones(p)
    lasso2 = lasso.gaussian(X, 
                            Y, 
                            weights)
    sol2 = strong_rules(lasso2, lagrange_sequence, inner_tol=1.e-15)
    beta1 = sol1['beta']
    beta2 = sol2['beta']

    assert(np.linalg.norm(beta1 - beta2) / np.linalg.norm(beta1) < 1.e-5)

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
    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-12)
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
    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      weights=weights)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-12)
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

    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      alpha=0.5,
                                                      elastic_net_param=enet)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-12)
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

    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      weights=weights,
                                                      alpha=0.5,
                                                      elastic_net_param=enet)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, lagrange_sequence, inner_tol=1.e-12)
    beta1 = sol1['beta']

