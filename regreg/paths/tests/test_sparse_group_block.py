import numpy as np

from .. import lasso, sparse_group_block
from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_path():
    '''
    run a basic path algorithm

    '''
    n, p, q = 200, 50, 3
    X = np.random.standard_normal((n,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y = np.dot(X, betaX) + np.random.standard_normal((n,q))

    sparse_group_block1 = sparse_group_block.sparse_group_block_path.multiresponse_gaussian(X, 
                                                                                            Y, 
                                                                                            1,
                                                                                            np.sqrt(q))
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = sparse_group_block1.main(lagrange_sequence, inner_tol=1.e-12)
    beta1 = sol1['beta']

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

    sparse_group_block1 = sparse_group_block.sparse_group_block_path.multinomial(X, 
                                                                                 Y, 
                                                                                 1,
                                                                                 np.sqrt(q))
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = sparse_group_block1.main(lagrange_sequence, inner_tol=1.e-12)
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

    sparse_group_block1 = sparse_group_block.sparse_group_block_path.multiresponse_gaussian(X, 
                                                                                            Y, 
                                                                                            1,
                                                                                            np.sqrt(q))
    cases = range(100)
    sparse_group_block1 = sparse_group_block1.subsample(cases)
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = sparse_group_block1.main(lagrange_sequence, inner_tol=1.e-10)
    beta1 = sol1['beta']

    sparse_group_block2 = sparse_group_block.sparse_group_block_path.multiresponse_gaussian(X[cases], 
                                                                                            Y[cases], 
                                                                                            1,
                                                                                            np.sqrt(q))
    sol2 = sparse_group_block2.main(lagrange_sequence, inner_tol=1.e-10)
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

    sparse_group_block1 = sparse_group_block.sparse_group_block_path.multiresponse_gaussian(X, 
                                                                                            Y, 
                                                                                            1,
                                                                                            np.sqrt(2),
                                                                                            alpha=0.5,
                                                                                            elastic_net_param=enet)
    lagrange_sequence = sparse_group_block.default_lagrange_sequence(sparse_group_block1.penalty,
                                                                     sparse_group_block1.grad_solution,
                                                                     nstep=23) # initialized at "null" model
    sol1 = sparse_group_block1.main(lagrange_sequence, inner_tol=1.e-12)
    beta1 = sol1['beta']

