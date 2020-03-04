import numpy as np, regreg.api as rr, regreg.affine as ra
import regreg.paths.sparse_group_block as sparse_group_block

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

    sparse_group_block1 = sparse_group_block.sparse_group_block_path.gaussian(X, 
                                                                              Y, 
                                                                              1,
                                                                              np.sqrt(q),
                                                                              nstep=23)
    sol1 = sparse_group_block1.main(inner_tol=1.e-5)
    beta1 = sol1['beta']

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

    sparse_group_block1 = sparse_group_block.sparse_group_block_path.gaussian(X, 
                                                                              Y, 
                                                                              1,
                                                                              np.sqrt(2),
                                                                              nstep=23,
                                                                              alpha=0.5,
                                                                              elastic_net_param=enet)
    sol1 = sparse_group_block1.main(inner_tol=1.e-5)
    beta1 = sol1['beta']

