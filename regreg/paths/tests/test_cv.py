import numpy as np

from .. import (lasso, 
                group_lasso,
                sparse_group_lasso,
                sparse_group_block,
                cross_validate)

from ...tests.decorators import set_seed_for_test

@set_seed_for_test()
def test_lasso(n=1000, p=100, nstep=20):
    '''
    CV for a LASSO path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    lasso1 = lasso.gaussian(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution, # initialized at "null" model
                                                        nstep=nstep)

    print(cross_validate.cross_validate(lasso1,
                                        lagrange_sequence,
                                        inner_tol=1.e-5,
                                        cv=3))

@set_seed_for_test()
def test_group_lasso(n=1000, p=100, nstep=20):
    '''
    CV for a LASSO path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b', 'c']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        if len(groups) > p:
            groups = np.asarray(groups)
            groups = groups[:p]
            break
        i += 1
    np.random.shuffle(groups)
    group_lasso1 = group_lasso.gaussian(X, 
                                        Y, 
                                        groups)
    lagrange_sequence = group_lasso.default_lagrange_sequence(group_lasso1.penalty,
                                                              group_lasso1.grad_solution,
                                                              nstep=nstep) # initialized at "null" model

    cross_validate.cross_validate(group_lasso1,
                                  lagrange_sequence,
                                  inner_tol=1.e-5,
                                  cv=3)


@set_seed_for_test()
def test_sparse_group_lasso(n=1000, p=100, nstep=20):
    '''
    CV for a LASSO path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b', 'c']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        if len(groups) > p:
            groups = np.asarray(groups)
            groups = groups[:p]
            break
        i += 1
    np.random.shuffle(groups)
    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution, # initialized at "null" model
                                                                     nstep=nstep) 

    cross_validate.cross_validate(sparse_group_lasso1,
                                  lagrange_sequence,
                                  inner_tol=1.e-10,
                                  cv=3)



@set_seed_for_test()
def test_sparse_group_lasso(n=1000, p=100, nstep=20):
    '''
    CV for a LASSO path

    '''
    X = np.random.standard_normal((n,p))
    Y = np.random.standard_normal(n)
    betaX = np.zeros(p)
    betaX[:3] = [3,4,5]
    np.random.shuffle(betaX)
    Y += np.dot(X, betaX) + np.random.standard_normal(n)

    groups = ['a', 'a', 'a', 'b', 'b', 'c']
    i = 0
    while True:
        groups.extend([str(i)]*5)
        if len(groups) > p:
            groups = np.asarray(groups)
            groups = groups[:p]
            break
        i += 1
    np.random.shuffle(groups)
    l1weights = np.ones(p)
    sparse_group_lasso1 = sparse_group_lasso.gaussian(X, 
                                                      Y, 
                                                      groups,
                                                      l1weights)
    lagrange_sequence = sparse_group_lasso.default_lagrange_sequence(sparse_group_lasso1.penalty,
                                                                     sparse_group_lasso1.grad_solution, # initialized at "null" model
                                                                     nstep=nstep) 

    cross_validate.cross_validate(sparse_group_lasso1,
                                  lagrange_sequence,
                                  inner_tol=1.e-10,
                                  cv=3)

@set_seed_for_test()
def test_sparse_group_multiresponse(n=1000, p=100, q=3, nstep=20):
    '''
    CV for a LASSO path

    '''
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
                                                                     nstep=nstep) # initialized at "null" model

    cross_validate.cross_validate(sparse_group_block1,
                                  lagrange_sequence,
                                  inner_tol=1.e-10,
                                  cv=3)

@set_seed_for_test()
def test_sparse_group_block(n=1000, p=100, q=3, nstep=20):
    '''
    CV for a LASSO path

    '''
    n1, n2, n3, p, q = 1190, 1200, 1210, 100, 3
    X1 = np.random.standard_normal((n1,p))
    X2 = np.random.standard_normal((n2,p))
    X3 = np.random.standard_normal((n3,p))
    betaX = np.zeros((p,q))
    betaX[:3] = [3,4,5]
    betaX *= 0.2
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
                                                                     nstep=nstep) # initialized at "null" model

    cross_validate.cross_validate(sparse_group_block1,
                                  lagrange_sequence,
                                  inner_tol=1.e-10,
                                  cv=3)



