import numpy as np, regreg.api as rr, regreg.affine as ra
import nose.tools as nt
import regreg.paths.lasso as lasso

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

    lasso1 = lasso.lasso_path.gaussian(X, Y, np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = lasso1.main(lagrange_sequence, inner_tol=1.e-5)
    beta1 = sol1['beta']

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
    lasso1 = lasso.lasso_path.gaussian(X, Y, np.ones(X.shape[1]))
    lasso1 = lasso1.subsample(cases)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = lasso1.main(lagrange_sequence, inner_tol=1.e-10)
    beta1 = sol1['beta']

    lasso2 = lasso.lasso_path.gaussian(X[cases], Y[cases], np.ones(X.shape[1]))
    sol2 = lasso2.main(lagrange_sequence, inner_tol=1.e-10)
    beta2 = sol2['beta']

    np.testing.assert_allclose(beta1, beta2)

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
    lasso1 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = lasso1.main(lagrange_sequence, inner_tol=1.e-5)
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
    lasso1 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights,
                                       alpha=0.5,
                                       elastic_net_param=np.ones(X.shape[1]))
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = lasso1.main(lagrange_sequence, inner_tol=1.e-5)
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

    lasso1 = lasso.lasso_path.gaussian(X, 
                                       Y, 
                                       weights,
                                       alpha=0.5,
                                       elastic_net_param=enet)
    lagrange_sequence = lasso.default_lagrange_sequence(lasso1.penalty,
                                                        lasso1.grad_solution,
                                                        nstep=23) # initialized at "null" model
    sol1 = lasso1.main(lagrange_sequence, inner_tol=1.e-5)
    beta1 = sol1['beta']

