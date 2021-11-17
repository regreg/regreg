import numpy as np

from .. import lasso, group_lasso, strong_rules, warm_start
from ..basil import basil_inner_loop, basil
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
    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=23) # initialized at "null" model
    sol1 = strong_rules(group_lasso1, 
                        lagrange_sequence, 
                        (group_lasso1.solution, group_lasso1.grad_solution),
                        inner_tol=1.e-12)

    weights = np.ones(p)
    lasso2 = lasso.gaussian(X, 
                            Y, 
                            weights)
    sol2 = strong_rules(lasso2, 
                        lagrange_sequence, 
                        (lasso2.solution, lasso2.grad_solution),
                        inner_tol=1.e-12)
    beta1 = sol1['beta']
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2)

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

    cases = range(n//2)
    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(9):
        groups.extend([str(i)]*5)

    np.random.shuffle(groups)
    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups)
    group_lasso1 = group_lasso1.subsample(cases)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=23) # initialized at "null" model
    sol1 = strong_rules(group_lasso1, 
                        lagrange_sequence, 
                        (group_lasso1.solution, group_lasso1.grad_solution),
                        inner_tol=1.e-10)
    beta1 = sol1['beta']

    group_lasso2 = group_lasso.gaussian(X[cases],
                                        Y[cases],
                                        groups)
    sol2 = strong_rules(group_lasso2, 
                        lagrange_sequence, 
                        (group_lasso2.solution, group_lasso2.grad_solution),
                        inner_tol=1.e-10)
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2, rtol=1.e-3)



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
    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=23) # initialized at "null" model
    sol1 = strong_rules(group_lasso1, 
                        lagrange_sequence, 
                        (group_lasso1.solution, group_lasso1.grad_solution),
                        inner_tol=1.e-5)
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

    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=23) # initialized at "null" model
    sol1 = strong_rules(group_lasso1, 
                        lagrange_sequence, 
                        (group_lasso1.solution, group_lasso1.grad_solution), 
                        inner_tol=1.e-5)
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

    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        alpha=0.5,
                                        elastic_net_param=enet)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=23) # initialized at "null" model
    sol1 = strong_rules(group_lasso1, 
                        lagrange_sequence, 
                        (group_lasso1.solution, group_lasso1.grad_solution),
                        inner_tol=1.e-5)
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

    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights,
                                        alpha=0.5,
                                        elastic_net_param=enet)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=23) # initialized at "null" model
    sol1 = strong_rules(group_lasso1, 
                        lagrange_sequence, 
                        (group_lasso1.solution, group_lasso1.grad_solution), 
                        inner_tol=1.e-5)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_basil_inner_loop(n=1000,p=600):
    '''
    test one run of the BASIL inner loop

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = np.array([3,4,5]) * np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) >= p:
            break
    groups = groups[:p]

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['2'] = weights['3'] = 0
    weights['a'] = weights['b'] = 2

    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights,
                                        alpha=0.5,
                                        elastic_net_param=enet)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=100) # initialized at "null" model
    sol1 = basil_inner_loop(group_lasso1, 
                            lagrange_sequence[:50], 
                            (group_lasso1.solution.copy(), group_lasso1.grad_solution.copy()),
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
    betaX[:3] = np.array([3,4,5]) * np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) >= p:
            break
    groups = groups[:p]

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['a'] = weights['b'] = 2

    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights)
    print(group_lasso1.penalty)
    print(np.linalg.norm(group_lasso1.solution))
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=100) # initialized at "null" model
    sol1 = basil(group_lasso1, 
                 lagrange_sequence, 
                 (group_lasso1.solution.copy(), group_lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)

    group_lasso2 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights)
    print(group_lasso2.penalty)
    print(np.linalg.norm(group_lasso2.solution))
    sol2 = warm_start(group_lasso2, 
                      lagrange_sequence, 
                      (group_lasso2.solution.copy(), group_lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    assert(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol2) < 1.e-4)

@set_seed_for_test()
def test_basil_unpenalized(n=200,p=100):
    '''
    run BASIL w/ unpenalized variables

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = np.array([3,4,5]) * np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) >= p:
            break
    groups = groups[:p]

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['a'] = weights['b'] = 2
    weights['2'] = weights['3'] = 0
    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights,
                                        alpha=0.5,
                                        elastic_net_param=enet)

    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=100) # initialized at "null" model
    sol1 = basil(group_lasso1, 
                 lagrange_sequence, 
                 (group_lasso1.solution.copy(), group_lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)

    group_lasso2 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights,
                                        alpha=0.5,
                                        elastic_net_param=enet)
    sol2 = warm_start(group_lasso2, 
                      lagrange_sequence, 
                      (group_lasso2.solution.copy(), group_lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    assert(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol2) < 1.e-4)


@set_seed_for_test()
def test_basil_enet(n=200,p=100):
    '''
    run BASIL w/enet

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = np.array([3,4,5]) * np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    groups = ['a', 'a', 'a', 'b', 'b']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        i += 1
        if len(groups) >= p:
            break
    groups = groups[:p]

    weights = dict([(g,1) for g in np.unique(groups)])
    weights['a'] = weights['b'] = 2

    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights,
                                        alpha=0.5,
                                        elastic_net_param=enet)

    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=100) # initialized at "null" model
    sol1 = basil(group_lasso1, 
                 lagrange_sequence, 
                 (group_lasso1.solution.copy(), group_lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)

    group_lasso2 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups,
                                        weights=weights,
                                        alpha=0.5,
                                        elastic_net_param=enet)
    sol2 = warm_start(group_lasso2, 
                      lagrange_sequence, 
                      (group_lasso2.solution.copy(), group_lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    rel_errors = np.array([np.linalg.norm(sol1[i]-sol2[i]) / np.linalg.norm(sol1[i])
                           for i in range(1, sol1.shape[0])])

    assert(np.percentile(rel_errors, 90) < 1.e-4)


