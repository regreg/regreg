from copy import copy

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

def _unpenalized_problem(seed=0, n=100, alpha=1.):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 9))
    Y = X[:, :3].sum(1) + rng.standard_normal(n)
    groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    l1weights = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1.])
    weights = {0: 0., 1: 1., 2: 1.}
    enet = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1.])
    path = sparse_group_lasso.gaussian(X,
                                       Y,
                                       groups,
                                       l1weights,
                                       weights=weights,
                                       alpha=alpha,
                                       elastic_net_param=enet)
    return X, Y, path

def test_unpenalized_group_detected():
    # gh-81: the mask of penalized variables was all False, and a group
    # with zero L2 and L1 weights was not detected as unpenalized
    _, _, path = _unpenalized_problem()
    assert path.unpenalized[0] == [0]
    np.testing.assert_array_equal(path._penalized_vars, [False] * 3 + [True] * 6)
    # the null model fits the unpenalized group
    assert np.all(path.solution[:3] != 0)
    assert np.all(path.solution[3:] == 0)

def test_unpenalized_elastic_net_path():
    # gh-81: check the path against the problem it documents, in which the
    # ridge term of penalized variables is multiplied by the Lagrange parameter
    import regreg.api as rr
    from regreg.smooth.quadratic import quadratic_loss

    alpha = 0.5
    X, Y, path = _unpenalized_problem(alpha=alpha)
    # with an unpenalized group, the dual norm ignores it, so lambda_max is
    # finite
    lagrange_seq = sparse_group_lasso.default_lagrange_sequence(path.penalty,
                                                                path.grad_solution,
                                                                nstep=5,
                                                                alpha=alpha)
    assert lagrange_seq[0] < 100
    beta = strong_rules(path,
                        lagrange_seq,
                        (path.solution, path.grad_solution),
                        inner_tol=1.e-12)['beta']

    # the first row is the null solution passed in
    penalty = copy(path.penalty)
    for lagrange, b in zip(lagrange_seq[1:], beta[1:]):
        penalty.lagrange = alpha * lagrange
        ridge = (1 - alpha) * lagrange * path.elastic_net_param
        loss = rr.smooth_sum([path.full_loss(),
                              quadratic_loss(ridge.shape, ridge, Qdiag=True)])
        direct = rr.simple_problem(loss, penalty).solve(tol=1.e-14, min_its=500)
        np.testing.assert_allclose(b, direct, atol=1.e-6)

def test_check_KKT_unpenalized():
    # gh-81: the unpenalized branch referred to an undefined name
    from ..sparse_group_lasso import _check_KKT
    _, _, path = _unpenalized_problem()
    grad = np.zeros(9)
    active, _ = _check_KKT(path.penalty, grad, path.solution, 1.)
    assert active[0] == 0
    grad[:3] = 1.
    active, _ = _check_KKT(path.penalty, grad, path.solution, 1.)
    assert active[0] == 3 # UNPENALIZED

def test_subsample_l1_alpha():
    # subsample passed the already scaled weights back to the constructor,
    # applying l1_alpha twice
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    Y = rng.standard_normal(40)
    path = sparse_group_lasso.gaussian(X,
                                       Y,
                                       np.array([0, 0, 1, 1, 2, 2]),
                                       np.ones(6),
                                       l1_alpha=0.5)
    sub = path.subsample(np.arange(20))
    np.testing.assert_allclose(sub.penalty.lasso_weights, path.penalty.lasso_weights)
    assert sub.penalty.weights == path.penalty.weights
