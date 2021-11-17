import numpy as np

from .. import lasso, strong_rules, warm_start
from ..basil import basil_inner_loop, basil
from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_path(n=200,p=50):
    '''
    run a basic path algorithm

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    lasso1 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = strong_rules(lasso1, 
                        lagrange_sequence, 
                        (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                        inner_tol=1.e-5)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_warm_start(n=200,p=50):
    '''
    compare full problem with warm start to strong rules path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    lasso1 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = strong_rules(lasso1, 
                        lagrange_sequence, 
                        (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                        inner_tol=1.e-14)
    beta1 = sol1['beta']

    sol2 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-14)
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2, rtol=1.e-4)

@set_seed_for_test()
def test_warm_start_enet(n=200,p=50):
    '''
    compare full problem with warm start to strong rules path
    including elastic net

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    lasso1 = lasso.gaussian(X, 
                            Y, 
                            np.ones(X.shape[1]),
                            alpha=0.5,
                            elastic_net_param=enet)

    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23,
                                                        alpha=lasso1.alpha) # initialized at "null" model
    sol1 = strong_rules(lasso1, 
                        lagrange_sequence, 
                        (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                        inner_tol=1.e-14)
    beta1 = sol1['beta']

    sol2 = warm_start(lasso1, 
                      lagrange_sequence, 
                      (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                      inner_tol=1.e-14)
    beta2 = sol2['beta']

    assert(np.linalg.norm(beta1 - beta2)/np.linalg.norm(beta1) < 1.e-4)
    
   
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
    lasso1 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    lasso1_sub = lasso1.subsample(cases)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1_sub.penalty,
                                                        lasso1_sub.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = strong_rules(lasso1_sub, 
                        lagrange_sequence, 
                        (lasso1_sub.solution.copy(), lasso1_sub.grad_solution.copy()),
                        inner_tol=1.e-10)
    beta1 = sol1['beta']

    lasso2 = lasso.gaussian(X[cases], Y[cases], np.ones(X.shape[1]))
    sol2 = strong_rules(lasso2, 
                        lagrange_sequence, 
                        (lasso2.solution.copy(), lasso2.grad_solution.copy()),
                        inner_tol=1.e-10)
    beta2 = sol2['beta']

    yield np.testing.assert_allclose, beta1, beta2, 1.e-3

    # check that the subsample did not change the original
    # path object

    lasso3 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    beta3 = strong_rules(lasso1, 
                         lagrange_sequence, 
                         (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                         inner_tol=1.e-10)['beta']
    beta4 = strong_rules(lasso3, 
                         lagrange_sequence, 
                         (lasso3.solution.copy(), lasso3.grad_solution.copy()),
                         inner_tol=1.e-10)['beta']

    yield np.testing.assert_allclose, beta3, beta4, 1.e-3

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

    weights = np.ones(X.shape[1])
    weights[0:2] = 0
    lasso1 = lasso.gaussian(X, 
                            Y, 
                            weights)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = strong_rules(lasso1, 
                        lagrange_sequence, 
                        (lasso1.solution.copy(), lasso1.grad_solution.copy()),
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
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    weights = np.ones(X.shape[1])
    lasso1 = lasso.gaussian(X, 
                            Y, 
                            weights,
                            alpha=0.5,
                            elastic_net_param=np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23,
                                                        alpha=lasso1.alpha) # initialized at "null" model
    sol1 = strong_rules(lasso1,
                        lagrange_sequence, 
                        (lasso1.solution.copy(), lasso1.grad_solution.copy()),
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
    enet[0] = 0
    weights = np.ones(X.shape[1])
    weights[0] = 0

    lasso1 = lasso.gaussian(X, 
                            Y, 
                            weights,
                            alpha=0.5,
                            elastic_net_param=enet)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23,
                                                        alpha=lasso1.alpha) # initialized at "null" model
    sol1 = strong_rules(lasso1, 
                        lagrange_sequence, 
                        (lasso1.solution.copy(), lasso1.grad_solution.copy()),
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
    betaX[:3] = [3,4,5] / np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    lasso1 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=100) # initialized at "null" model
    sol1 = basil_inner_loop(lasso1, 
                            lagrange_sequence[:50], 
                            (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                            inner_tol=1.e-14,
                            step_nvar=10)
    lagrange1, beta1, grad1, active1 = sol1
    print(np.array(beta1).shape, 'chunk of path')
    print(active1, 'active')

@set_seed_for_test()
def test_basil(n=1000,p=600):
    '''
    test BASIL

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5] / np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    lasso1 = lasso.gaussian(X, 
                            Y, 
                            np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=100) # initialized at "null" model
    sol1 = basil(lasso1, 
                 lagrange_sequence, 
                 (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)

    lasso2 = lasso.gaussian(X, 
                            Y, 
                            np.ones(X.shape[1]))
    sol2 = warm_start(lasso2, 
                      lagrange_sequence, 
                      (lasso2.solution.copy(), lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    assert(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol2) < 1.e-4)

@set_seed_for_test()
def test_basil_unpenalized(n=500,p=200):
    '''
    test BASIL

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5] / np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    W = np.ones(X.shape[1])
    W[2:4] = 0
    lasso1 = lasso.gaussian(X, 
                            Y, 
                            W)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.4,
                                                        nstep=20) # initialized at "null" model
    sol1 = basil(lasso1, 
                 lagrange_sequence, 
                 (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)

    lasso2 = lasso.gaussian(X, 
                            Y, 
                            W)
    sol2 = warm_start(lasso2, 
                      lagrange_sequence, 
                      (lasso2.solution.copy(), lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    assert(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol2) < 1.e-4)

@set_seed_for_test()
def test_basil_enet(n=500,p=200):
    '''
    test BASIL w/enet

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5] / np.sqrt(n)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    enet = np.ones(X.shape[1])
    enet[4:8] = 0

    lasso1 = lasso.gaussian(X, 
                            Y, 
                            np.ones(X.shape[1]),
                            alpha=0.5,
                            elastic_net_param=enet)

    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        lagrange_proportion=0.5,
                                                        nstep=50,
                                                        alpha=lasso1.alpha) # initialized at "null" model
    sol1 = basil(lasso1, 
                 lagrange_sequence, 
                 (lasso1.solution.copy(), lasso1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)

    lasso2 = lasso.gaussian(X, 
                            Y, 
                            np.ones(X.shape[1]),
                            alpha=0.5,
                            elastic_net_param=enet)

    sol2 = warm_start(lasso2, 
                      lagrange_sequence, 
                      (lasso2.solution.copy(), lasso2.grad_solution.copy()),
                      inner_tol=1.e-14)['beta']

    lagrange_sequence2 = lasso.default_lagrange_sequence(lasso2.penalty,
                                                         lasso2.grad_solution,
                                                         lagrange_proportion=0.5,
                                                         nstep=50,
                                                         alpha=lasso2.alpha) # initialized at "null" model

    rel_errors = np.array([np.linalg.norm(sol1[i]-sol2[i]) / np.linalg.norm(sol1[i])
                           for i in range(1, sol1.shape[0])])

    assert(np.percentile(rel_errors, 90) < 1.e-4)

