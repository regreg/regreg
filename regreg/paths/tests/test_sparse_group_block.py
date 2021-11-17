import numpy as np

from .. import lasso, sparse_group_block, strong_rules, warm_start
from ..basil import basil_inner_loop, basil
from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_gaussian_multiresponse():
    '''
    run a basic path algorithm

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = np.dot(X, betaX) + np.random.standard_normal((n,q))

    sparse_group_block1 = sparse_group_block.multiresponse_gaussian(X, 
                                                                    Y, 
                                                                    1,
                                                                    np.sqrt(q))
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_block1, 
                        lagrange_sequence, 
                        (sparse_group_block1.solution, sparse_group_block1.grad_solution),
                        inner_tol=1.e-12)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_gaussian_blocks():
    '''
    run a basic path algorithm

    '''
    n1, n2, n3, p, q = 190, 200, 210, 50, 3
    X1 = np.random.standard_normal((n1,p))
    X2 = np.random.standard_normal((n2,p))
    X3 = np.random.standard_normal((n3,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = [np.dot(X, beta) + np.random.standard_normal(n) for
         X, beta, n in zip([X1, X2, X3],
                           betaX.T,
                           [n1, n2, n3])]
                                     

    sparse_group_block1 = sparse_group_block.stacked_gaussian([X1, X2, X3], 
                                                              Y, 
                                                              1,
                                                              np.sqrt(q))
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_block1, 
                        lagrange_sequence, 
                        (sparse_group_block1.solution, sparse_group_block1.grad_solution),
                        inner_tol=1.e-12)
    beta1 = sol1['beta']

    print(np.sqrt((beta1**2).sum(2).sum(1)))
    
@set_seed_for_test()
def test_multinomial():
    '''
    run a basic path algorithm with multinomial loss

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = np.array([3,4,5]) / np.sqrt(n)
    np.random.shuffle(betaX)
    eta = np.dot(X, betaX)
    prob = np.exp(eta) / (1 + np.exp(eta))
    Y = np.random.binomial(1, prob) # not really "multinomial", but it will do for testing

    sparse_group_block1 = sparse_group_block.multinomial(X, 
                                                         Y, 
                                                         1,
                                                         np.sqrt(q))
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_block1, 
                        lagrange_sequence, 
                        (sparse_group_block1.solution, sparse_group_block1.grad_solution),
                        inner_tol=1.e-12)
    beta1 = sol1['beta']

@set_seed_for_test()
def test_path_subsample(n=200,p=50):
    '''
    compare a subsample path to the full path on subsampled data

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = np.dot(X, betaX) + np.random.standard_normal((n,q))

    sparse_group_block1 = sparse_group_block.multiresponse_gaussian(X, 
                                                                    Y, 
                                                                    1,
                                                                    np.sqrt(q))
    cases = range(100)

    sparse_group_block1 = sparse_group_block1.subsample(cases)
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_block1, 
                        lagrange_sequence, 
                        (sparse_group_block1.solution, sparse_group_block1.grad_solution),
                        inner_tol=1.e-10)
    beta1 = sol1['beta']

    sparse_group_block2 = sparse_group_block.multiresponse_gaussian(X[cases], 
                                                                    Y[cases], 
                                                                    1,
                                                                    np.sqrt(q))
    sol2 = strong_rules(sparse_group_block2, 
                        lagrange_sequence, 
                        (sparse_group_block2.solution, sparse_group_block2.grad_solution),
                        inner_tol=1.e-10)
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2, rtol=1.e-3)

@set_seed_for_test()
def test_elastic_net(n=200, p=50):
    '''
    run a basic elastic net path algorithm

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = np.dot(X, betaX) + np.random.standard_normal((n,q))


    enet = np.zeros((X.shape[1], Y.shape[1]))
    enet[4:7] = 0

    sparse_group_block1 = sparse_group_block.multiresponse_gaussian(X, 
                                                                    Y, 
                                                                    1,
                                                                    np.sqrt(2),
                                                                    alpha=0.5,
                                                                    elastic_net_param=enet)
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = strong_rules(sparse_group_block1, 
                        lagrange_sequence, 
                        (sparse_group_block1.solution, sparse_group_block1.grad_solution),
                        inner_tol=1.e-12)
    beta1 = sol1['beta']


@set_seed_for_test()
def test_basil_inner_loop(n=200,p=50):
    '''
    test one run of the BASIL inner loop

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = np.dot(X, betaX) + np.random.standard_normal((n,q))

    enet = np.zeros((X.shape[1], Y.shape[1]))
    enet[4:7] = 0

    sparse_group_block1 = sparse_group_block.multiresponse_gaussian(X, 
                                                                    Y, 
                                                                    1,
                                                                    np.sqrt(2),
                                                                    alpha=0.5,
                                                                    elastic_net_param=enet)
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                              sparse_group_block1.grad_solution,
                                                              nstep=100) # initialized at "null" model
    sol1 = basil_inner_loop(sparse_group_block1, 
                            lagrange_sequence[:50], 
                            (sparse_group_block1.solution.copy(), sparse_group_block1.grad_solution.copy()),
                            inner_tol=1.e-14,
                            step_nvar=10)
    lagrange1, beta1, grad1, active1 = sol1
    print(np.array(beta1).shape, 'chunk of path')

    print(active1, 'active')

@set_seed_for_test()
def test_basil(n=300,p=100):
    '''
    run BASIL

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = np.dot(X, betaX) + np.random.standard_normal((n,q))

    sparse_group_block1 = sparse_group_block.multiresponse_gaussian(X, 
                                                                    Y, 
                                                                    1,
                                                                    np.sqrt(2))
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     lagrange_proportion=0.4,
                                                                     nstep=50) # initialized at "null" model
    sol1 = basil(sparse_group_block1, 
                 lagrange_sequence, 
                 (sparse_group_block1.solution.copy(), sparse_group_block1.grad_solution.copy()),
                 inner_tol=1.e-14,
                 step_nvar=10,
                 step_lagrange=20)
    
    sparse_group_block2 = sparse_group_block.multiresponse_gaussian(X, 
                                                                    Y, 
                                                                    1,
                                                                    np.sqrt(2))

    return sparse_group_block2
