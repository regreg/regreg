import numpy as np

from .. import lasso, sparse_group_lasso, strong_rules, warm_start
from ..basil import basil_inner_loop, basil
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
                                                      l1_alpha=1,
                                                      weights=weights)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution),
                        inner_tol=1.e-12)

    weights = np.ones(p)

    lasso2 = lasso.gaussian(X, 
                            Y, 
                            weights)

    lagrange_sequence2 = lasso.default_lagrange_sequence(lasso2.penalty,
                                                         lasso2.grad_solution,
                                                         nstep=23) # initialized at "null" model
    sol2 = strong_rules(lasso2, 
                        lagrange_sequence, 
                        (lasso2.solution, lasso2.grad_solution),
                        inner_tol=1.e-15)
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
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) > p:
            break
    groups = groups[:p]

    np.random.shuffle(groups)
    l1weights = np.ones(p)
    weights = {}
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      l1_alpha=1,
                                                      weights=weights)
    cases = range(100)
    sparse_group_lasso1 = sparse_group_lasso1.subsample(cases)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution), 
                        inner_tol=1.e-10)
    beta1 = sol1['beta']

    sparse_group_lasso2 = sparse_group_lasso.gaussian(X[cases], 
                                                      Y[cases], 
                                                      groups,
                                                      l1weights,
                                                      l1_alpha=1,
                                                      weights=weights)
    lagrange_sequence2 = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso2.penalty,
                                                                      sparse_group_lasso2.grad_solution,
                                                                      nstep=23) # initialized at "null" model

    sol2 = strong_rules(sparse_group_lasso2, 
                        lagrange_sequence, 
                        (sparse_group_lasso2.solution, sparse_group_lasso2.grad_solution),
                        inner_tol=1.e-10)
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
                                                      l1_alpha=0)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution),
                        inner_tol=1.e-12)

    weights = np.ones(p)
    lasso2 = lasso.gaussian(X, 
                            Y, 
                            weights)
    sol2 = strong_rules(lasso2, 
                        lagrange_sequence, 
                        (lasso2.solution, lasso2.grad_solution),
                        inner_tol=1.e-15)
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
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution),
                        inner_tol=1.e-12)
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
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution),
                        inner_tol=1.e-12)
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
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution),
                        inner_tol=1.e-12)
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
    sol1 = strong_rules(sparse_group_lasso1, 
                        lagrange_sequence, 
                        (sparse_group_lasso1.solution, sparse_group_lasso1.grad_solution),
                        inner_tol=1.e-12)
    beta1 = sol1['beta']


@set_seed_for_test()
def test_basil_inner_loop(n=600,p=200):
    '''
    test one run of the BASIL inner loop

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) > p:
            break
    groups = groups[:p]

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
                                                                     nstep=100) # initialized at "null" model
    sol1 = basil_inner_loop(sparse_group_lasso1, 
                            lagrange_sequence[:50], 
                            (sparse_group_lasso1.solution.copy(), sparse_group_lasso1.grad_solution.copy()),
                            inner_tol=1.e-14,
                            step_nvar=10)
    lagrange1, beta1, grad1, active1 = sol1
    print(np.array(beta1).shape, 'chunk of path')

    print(active1, 'active')

@set_seed_for_test()
def test_basil(n=200,p=100):
    '''
    run BASIL

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) > p:
            break
    groups = groups[:p]

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['a'] = weights['b'] = 2

    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      weights=weights)

    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     lagrange_proportion=0.5,
                                                                     nstep=20) # initialized at "null" model
    print(sparse_group_lasso1.penalty.conjugate.seminorm(sparse_group_lasso1.grad_solution, lagrange=1))
    print(lagrange_sequence, 'lagrange 1')
    sol1 = basil(sparse_group_lasso1, 
                 lagrange_sequence.copy(), 
                 (sparse_group_lasso1.solution.copy(), sparse_group_lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)
    
    print(lagrange_sequence, 'lagrange 2')

    sparse_group_lasso2 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      weights=weights)
    print(sparse_group_lasso2.penalty.conjugate.seminorm(sparse_group_lasso2.grad_solution, lagrange=1))
    print(lagrange_sequence, 'lagrange 3')

    sol2 = warm_start(sparse_group_lasso2, 
                      lagrange_sequence, 
                      (sparse_group_lasso2.solution.copy(), sparse_group_lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    print(lagrange_sequence, 'lagrange 4')
    print(sparse_group_lasso2.penalty.conjugate.seminorm(sparse_group_lasso2.grad_solution, lagrange=1))
    print(sparse_group_lasso1.penalty.conjugate.seminorm(sparse_group_lasso1.grad_solution, lagrange=1))
    print(np.linalg.norm(sparse_group_lasso1.grad_solution - sparse_group_lasso2.grad_solution))
    assert(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol2) < 1.e-4)

@set_seed_for_test()
def test_basil_unpenalized(n=200,p=100):
    '''
    run BASIL w/ unpenalized groups

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) > p:
            break
    groups = groups[:p]

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['a'] = weights['b'] = 2
    weights['2'] = weights['3'] = 0
    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      weights=weights)

    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution,
                                                                     nstep=100) # initialized at "null" model
    sol1 = basil(sparse_group_lasso1, 
                 lagrange_sequence, 
                 (sparse_group_lasso1.solution.copy(), sparse_group_lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)
    
    sparse_group_lasso2 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights,
                                                      weights=weights)
    sol2 = warm_start(sparse_group_lasso2, 
                      lagrange_sequence, 
                      (sparse_group_lasso2.solution.copy(), sparse_group_lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    assert(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol2) < 1.e-4)
